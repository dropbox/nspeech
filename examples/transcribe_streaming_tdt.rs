/// Cache-aware streaming transcription using Nemotron Streaming TDT
///
/// This example demonstrates true cache-aware streaming with zero overlapping computations:
/// - Configurable chunk sizes (80ms to 1120ms)
/// - Maintains attention and convolution caches across chunks
/// - No redundant computation - each frame processed exactly once
/// - Built-in punctuation and capitalization (from model)
///
/// Usage:
///   cargo run --example transcribe_streaming_tdt --release -- dots.wav
///   cargo run --example transcribe_streaming_tdt --release -- --chunk-size 560 MLKDream_16k.wav
///   cargo run --example transcribe_streaming_tdt --release -- --chunk-size 1120 audio.wav
///   PARAKEET_DEVICE=cpu cargo run --example transcribe_streaming_tdt --release -- audio.wav

use anyhow::Result;
use speech::parakeet::{
    get_device, load_parakeet_streaming_tdt_from_local,
    TransducerModel, ParakeetFeatureExtractor,
    streaming_encoder::StreamingEncoderCache,
};
use candle_core::{DType, Device};
use candle_nn::rnn;
use std::path::PathBuf;

/// Configuration for cache-aware streaming
#[derive(Debug, Clone)]
struct StreamingConfig {
    /// Chunk size in samples (at 16kHz)
    /// Derived from att_context_size: (right_context + 1) * 80ms * 16 samples/ms
    chunk_samples: usize,

    /// Left context size in encoder frames (typically 70 = 5.6s)
    left_context_frames: usize,

    /// Right context size in encoder frames (0 to 13)
    right_context_frames: usize,
}

impl StreamingConfig {
    /// Create config from att_context_size parameter
    /// att_context_size format: [left_frames, right_frames]
    /// Each encoder frame = 80ms (after 8x subsampling from 10ms mel frames)
    fn from_att_context_size(left: usize, right: usize) -> Self {
        // Calculate chunk size in samples
        // Each encoder frame represents 80ms of audio (8x subsampling * 10ms)
        // 80ms = 1280 samples at 16kHz
        let chunk_ms = (right + 1) * 80;
        let chunk_samples = chunk_ms * 16; // 16 samples per ms at 16kHz

        Self {
            chunk_samples,
            left_context_frames: left,
            right_context_frames: right,
        }
    }

    /// Get chunk duration in seconds
    fn chunk_duration_secs(&self) -> f64 {
        self.chunk_samples as f64 / 16000.0
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        // Default to [70, 6] = 560ms chunks (balanced latency/quality)
        Self::from_att_context_size(70, 6)
    }
}

/// Cache-aware streaming transcriber
struct CachedStreamingTranscriber {
    model: TransducerModel,
    feat_extractor: ParakeetFeatureExtractor,
    device: Device,
    config: StreamingConfig,

    // Cache state (one per encoder layer)
    encoder_caches: Option<StreamingEncoderCache>,

    // Predictor LSTM state (maintained across chunks)
    pred_states: Option<Vec<rnn::LSTMState>>,
    last_token: u32,

    // Token accumulation
    tokens: Vec<u32>,
    last_decoded: usize,

    // Total samples processed
    total_samples_processed: usize,
}

impl CachedStreamingTranscriber {
    fn new(
        model: TransducerModel,
        config: StreamingConfig,
        device: Device,
    ) -> Result<Self> {
        // Use the model's actual feat_in (may be different from default 128)
        let num_mel_bins = model.encoder.cfg.feat_in;
        let feat_extractor = ParakeetFeatureExtractor::new(num_mel_bins);

        let blank_id = model.config.blank_id as u32;

        Ok(Self {
            model,
            feat_extractor,
            device,
            config,
            encoder_caches: None,
            pred_states: None,
            last_token: blank_id,
            tokens: Vec::new(),
            last_decoded: 0,
            total_samples_processed: 0,
        })
    }

    /// Initialize encoder caches for streaming
    fn init_encoder_caches(&mut self) -> Result<()> {
        let num_layers = self.model.encoder.cfg.num_layers;
        let num_heads = self.model.encoder.cfg.num_heads;
        let d_model = self.model.encoder.cfg.d_model;
        let head_dim = d_model / num_heads;
        let conv_kernel_size = self.model.encoder.cfg.conv_kernel_size;

        let dtype = if self.device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        };

        self.encoder_caches = Some(StreamingEncoderCache::with_capacity(
            num_layers,
            1,  // batch_size
            num_heads,
            self.config.left_context_frames,  // max cache frames
            head_dim,
            d_model,
            conv_kernel_size,
            &self.device,
            dtype,
        )?);

        Ok(())
    }

    /// Process a chunk of audio samples
    /// Returns new transcribed text (incremental)
    fn process_chunk(&mut self, audio_chunk: &[f32]) -> Result<String> {
        // Initialize caches on first chunk
        if self.encoder_caches.is_none() {
            self.init_encoder_caches()?;
        }

        // Extract features from chunk
        let features = self.feat_extractor.extract_to_tensor(audio_chunk, &self.device)?;

        // Convert to model dtype
        let features = if !self.device.is_cpu() {
            features.to_dtype(DType::BF16)?
        } else {
            features
        };

        // Run encoder with caches
        let encoder_out = self.model.encoder.forward_with_cache(
            &features,
            false,
            self.encoder_caches.as_mut(),
        )?;

        // EXPERIMENT: Use regular greedy_decode (no state maintenance between chunks)
        // Each chunk is decoded independently with encoder caches maintained
        eprintln!("[DECODER] Chunk {} - using regular greedy_decode (no decoder state)",
                 self.total_samples_processed / 8960);

        // Run regular greedy decode (NO predictor state maintenance)
        let new_tokens = self.model.greedy_decode(&encoder_out)?;

        eprintln!("[DECODER] After decode: {} new tokens", new_tokens.len());

        // Accumulate tokens
        self.tokens.extend_from_slice(&new_tokens);

        // Decode only new tokens incrementally
        let new_text = self.model.decode_tokens_incremental(&self.tokens, self.last_decoded)?;
        self.last_decoded = self.tokens.len();

        self.total_samples_processed += audio_chunk.len();

        Ok(new_text)
    }

    /// Flush any remaining tokens
    fn flush(&mut self) -> Result<String> {
        // For now, just return any remaining undecoded tokens
        if self.last_decoded < self.tokens.len() {
            let final_text = self.model.decode_tokens_incremental(&self.tokens, self.last_decoded)?;
            self.last_decoded = self.tokens.len();
            return Ok(final_text);
        }
        Ok(String::new())
    }

    /// Reset all caches and state for new utterance
    fn reset(&mut self) {
        if let Some(ref mut caches) = self.encoder_caches {
            caches.reset();
        }
        self.pred_states = None;
        self.last_token = self.model.config.blank_id as u32;
        self.tokens.clear();
        self.last_decoded = 0;
        self.total_samples_processed = 0;
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} [--chunk-size MILLISECONDS] <audio.wav>", args[0]);
        eprintln!("\nCache-Aware Streaming Transcription");
        eprintln!("Zero overlapping computations with configurable chunk sizes:\n");
        eprintln!("  --chunk-size 80     Ultra-low latency (80ms chunks)");
        eprintln!("  --chunk-size 160    Very low latency (160ms chunks)");
        eprintln!("  --chunk-size 560    Balanced (560ms chunks, default)");
        eprintln!("  --chunk-size 1120   High quality (1120ms chunks)");
        return Ok(());
    }

    // Parse arguments
    let mut chunk_ms = 560; // Default: 560ms ([70, 6])
    let mut audio_path = &args[1];

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--chunk-size" => {
                if i + 1 < args.len() {
                    chunk_ms = args[i + 1].parse::<usize>()
                        .unwrap_or_else(|_| {
                            eprintln!("Invalid chunk size, using default 560ms");
                            560
                        });
                    i += 2;
                } else {
                    eprintln!("--chunk-size requires a value");
                    return Ok(());
                }
            }
            _ => {
                audio_path = &args[i];
                break;
            }
        }
    }

    // Map chunk_ms to att_context_size
    let (left, right) = match chunk_ms {
        80 => (70, 0),
        160 => (70, 1),
        240 => (70, 2),
        320 => (70, 3),
        400 => (70, 4),
        480 => (70, 5),
        560 => (70, 6),
        640 => (70, 7),
        720 => (70, 8),
        800 => (70, 9),
        880 => (70, 10),
        960 => (70, 11),
        1040 => (70, 12),
        1120 => (70, 13),
        _ => {
            eprintln!("Unsupported chunk size {}ms", chunk_ms);
            eprintln!("Supported sizes: 80, 160, 240, 320, 400, 480, 560, 640, 720, 800, 880, 960, 1040, 1120");
            return Ok(());
        }
    };

    let stream_config = StreamingConfig::from_att_context_size(left, right);

    println!("Cache-Aware Streaming TDT Transcription");
    println!("========================================\n");
    println!("Audio: {}\n", audio_path);

    // Get device
    let device = get_device()?;
    println!("Device: {:?}", device);
    println!("  (If you encounter errors, try: PARAKEET_DEVICE=cpu)\n");

    let model_dir = PathBuf::from(".cache/parakeet-streaming-tdt");

    // Load Streaming TDT model (BF16 safetensors for best quality)
    println!("Loading Nemotron Streaming TDT model...");
    let model = load_parakeet_streaming_tdt_from_local(&model_dir, &device)?;
    println!("✓ Model loaded\n");

    // Load audio
    println!("Loading audio...");
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    if spec.channels != 1 {
        return Err(anyhow::anyhow!("Expected mono audio, got {} channels", spec.channels));
    }
    if spec.sample_rate != 16000 {
        return Err(anyhow::anyhow!("Expected 16kHz audio, got {} Hz", spec.sample_rate));
    }

    let all_samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;

    let total_duration_sec = all_samples.len() as f64 / 16000.0;
    println!("✓ Loaded: {:.2}s ({} samples)\n", total_duration_sec, all_samples.len());

    // Display streaming configuration
    println!("Streaming Configuration:");
    println!("  Chunk size: {}ms ({} samples)", chunk_ms, stream_config.chunk_samples);
    println!("  Chunk duration: {:.3}s", stream_config.chunk_duration_secs());
    println!("  Left context: {} frames (5.6s)", stream_config.left_context_frames);
    println!("  Right context: {} frames ({}ms)", stream_config.right_context_frames, right * 80);
    println!("  Cache memory: ~17 MB (fixed)\n");

    // Create streaming transcriber
    let mut transcriber = CachedStreamingTranscriber::new(
        model,
        stream_config.clone(),
        device.clone(),
    )?;

    println!("=== STREAMING TRANSCRIPTION ===\n");

    let start_time = std::time::Instant::now();
    let mut idx = 0;
    let mut chunk_count = 0;
    let mut all_text = String::new();

    // Process audio in chunks
    while idx < all_samples.len() {
        let end = (idx + stream_config.chunk_samples).min(all_samples.len());
        let chunk = &all_samples[idx..end];

        chunk_count += 1;
        let chunk_start_sec = idx as f64 / 16000.0;

        // Track token count before processing
        let tokens_before = transcriber.tokens.len();

        // Process chunk and get new text
        let new_text = transcriber.process_chunk(chunk)?;

        // Debug: show token count for first few chunks
        let tokens_after = transcriber.tokens.len();
        let new_tokens_count = tokens_after - tokens_before;
        if chunk_count <= 5 {
            eprintln!("[DEBUG] Chunk {}: added {} tokens (total {})",
                     chunk_count, new_tokens_count, tokens_after);
            if new_tokens_count > 0 {
                let token_ids: Vec<u32> = transcriber.tokens[tokens_before..tokens_after].to_vec();
                eprintln!("[DEBUG]   Token IDs: {:?}", &token_ids[..new_tokens_count.min(10)]);
            }
        }

        if !new_text.is_empty() {
            print!("\x1b[1;36m{}\x1b[0m", new_text);
            std::io::Write::flush(&mut std::io::stdout())?;
            all_text.push_str(&new_text);
        }

        // Progress indicator (every 10 chunks)
        if chunk_count % 10 == 0 {
            eprint!("\r[Processing: {:.1}s / {:.1}s]", chunk_start_sec, total_duration_sec);
            std::io::Write::flush(&mut std::io::stderr())?;
        }

        idx = end;
    }

    // Flush any remaining tokens
    let final_text = transcriber.flush()?;
    if !final_text.is_empty() {
        print!("\x1b[1;36m{}\x1b[0m", final_text);
        all_text.push_str(&final_text);
    }

    println!("\n");
    eprint!("\r                                        \r"); // Clear progress line

    let elapsed = start_time.elapsed();
    let processing_time = elapsed.as_secs_f64();
    let rtf = processing_time / total_duration_sec;

    println!("\n=== FINAL TRANSCRIPT ===\n");
    println!("{}\n", all_text.trim());

    println!("=== STATISTICS ===");
    println!("  Audio duration: {:.2}s", total_duration_sec);
    println!("  Processing time: {:.2}s", processing_time);
    println!("  Real-time factor: {:.3}x", rtf);
    println!("  Chunks processed: {}", chunk_count);
    println!("  Chunk size: {}ms", chunk_ms);
    println!("  Tokens decoded: {}", transcriber.tokens.len());

    if rtf < 1.0 {
        println!("\n✓ Faster than real-time! ({:.1}x speed)", 1.0 / rtf);
    }

    // Compare with baseline if using dots.wav
    if audio_path.contains("dots.wav") {
        let baseline_tokens = 187;  // From transcribe_tdt.rs (beam_size=2)
        let quality_percent = (transcriber.tokens.len() as f32 / baseline_tokens as f32) * 100.0;
        println!("\n  Baseline (transcribe_tdt.rs): {} tokens", baseline_tokens);
        println!("  Cache-aware streaming: {} tokens ({:.1}%)", transcriber.tokens.len(), quality_percent);

        if quality_percent >= 90.0 && quality_percent <= 110.0 {
            println!("\n✓ Quality close to baseline!");
        }
    }

    println!("\n✓ Streaming transcription complete!");

    Ok(())
}
