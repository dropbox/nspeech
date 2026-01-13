/// Streaming transcription example using Parakeet TDT (Transducer) model
///
/// This example demonstrates pseudo-streaming transcription by:
/// 1. Processing audio in overlapping chunks
/// 2. Emitting results as chunks complete
/// 3. Showing progressive transcription output
///
/// Note: True streaming with FastConformer requires attention caching, which is
/// not yet implemented. This example shows a practical approach for incremental
/// transcription of buffered audio.
///
/// Usage:
///   cargo run --example transcribe_tdt_streaming --release -- dots.wav
///   cargo run --example transcribe_tdt_streaming --release -- MLKDream_16k.wav

use anyhow::Result;
use speech::parakeet::{
    get_device, load_parakeet_tdt_from_local,
    ParakeetFeatureExtractor,
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        eprintln!("\nThis example demonstrates streaming-style transcription with Parakeet TDT.");
        eprintln!("Audio is processed in chunks, with results emitted incrementally.");
        return Ok(());
    }

    let audio_path = &args[1];

    println!("Parakeet TDT Streaming Transcription");
    println!("=====================================\n");
    println!("Audio: {}\n", audio_path);

    // Get device
    let device = get_device()?;

    // Load TDT model
    println!("Loading TDT model...");
    let mut model = load_parakeet_tdt_from_local(".cache/parakeet-tdt", &device)?;
    println!("  Model: nvidia/parakeet-tdt-0.6b-v3");
    println!("  Architecture: Transducer (RNN-T)\n");

    // Load tokenizer
    println!("Loading tokenizer...");
    model.load_tokenizer(".cache/parakeet-tdt")?;
    println!("  Tokenizer loaded\n");

    // Load audio
    println!("Loading audio...");
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    println!("  Sample rate: {} Hz", spec.sample_rate);
    println!("  Channels: {}", spec.channels);
    println!("  Bits per sample: {}\n", spec.bits_per_sample);

    // Read all samples
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    // Convert to mono if stereo
    let samples = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect()
    } else {
        samples
    };

    println!("  Audio samples: {}", samples.len());
    println!("  Duration: {:.2}s\n", samples.len() as f32 / spec.sample_rate as f32);

    // Feature extractor
    let feat_extractor = ParakeetFeatureExtractor::new(128);  // TDT uses 128 mel bins

    // Configure streaming chunks
    // Note: Overlap disabled - proper overlap handling requires frame-level masking
    // (see NeMo's delay parameter and middle token alignment)
    const CHUNK_SECONDS: f32 = 3.0;
    const OVERLAP_SECONDS: f32 = 0.0; // Non-overlapping for now
    const SAMPLES_PER_CHUNK: usize = (16000.0 * CHUNK_SECONDS) as usize;
    const OVERLAP_SAMPLES: usize = (16000.0 * OVERLAP_SECONDS) as usize;

    let stride = SAMPLES_PER_CHUNK - OVERLAP_SAMPLES;
    let total_chunks = (samples.len() + stride - 1) / stride;

    println!("Streaming configuration:");
    println!("  Chunk size: {:.1}s ({} samples)", CHUNK_SECONDS, SAMPLES_PER_CHUNK);
    println!("  Overlap: {:.1}s ({} samples)", OVERLAP_SECONDS, OVERLAP_SAMPLES);
    println!("  Stride: {} samples", stride);
    println!("  Total chunks: {}\n", total_chunks);
    println!("=== STREAMING TRANSCRIPTION ===\n");

    let start_time = std::time::Instant::now();

    // Use StreamingTransducer for state management
    let streaming_config = speech::parakeet::StreamingConfig {
        chunk_samples: SAMPLES_PER_CHUNK,
        overlap_samples: OVERLAP_SAMPLES,
        emit_partial: true,
    };
    let mut transcriber = speech::parakeet::StreamingTransducer::new(model, streaming_config);

    // Accumulate text as we go
    let mut accumulated_text = String::new();

    for chunk_idx in 0..total_chunks {
        let chunk_start_idx = chunk_idx * stride;
        let chunk_end_idx = (chunk_start_idx + SAMPLES_PER_CHUNK).min(samples.len());
        let chunk_samples = &samples[chunk_start_idx..chunk_end_idx];

        let chunk_start = std::time::Instant::now();

        // Extract features for this chunk
        let features = feat_extractor.extract_to_tensor(chunk_samples, &device)?;

        // Convert to BF16 for GPU
        let features = if !device.is_cpu() {
            features.to_dtype(candle_core::DType::BF16)?
        } else {
            features
        };

        // Process features through streaming transcriber
        let new_tokens = transcriber.process_features(&features)?;

        let chunk_time = chunk_start.elapsed();

        // Print progress
        print!("[Chunk {}/{}] ", chunk_idx + 1, total_chunks);
        print!("{:.1}s processed in {:.2}s ", chunk_samples.len() as f32 / 16000.0, chunk_time.as_secs_f32());
        println!("({} new tokens, {} total)", new_tokens.len(), transcriber.tokens().len());

        // Show NEW text incrementally if we got new tokens
        if !new_tokens.is_empty() {
            match transcriber.decode_text_incremental() {
                Ok((new_text, _total_decoded)) => {
                    if !new_text.is_empty() {
                        // Print only the NEW text that was decoded
                        print!("  + \"{}\"", new_text.trim());

                        // Accumulate for full text display
                        accumulated_text.push_str(&new_text);

                        // Show current full text so far
                        println!("\n  → Full: {}\n", accumulated_text.trim());
                    } else {
                        println!();
                    }
                }
                Err(e) => {
                    println!("  → (decode error: {})\n", e);
                }
            }
        } else {
            println!();
        }
    }

    let total_time = start_time.elapsed();

    println!("================================\n");

    // Compare accumulated text with final decode
    let final_text = transcriber.decode_text()?;

    println!("=== FINAL TRANSCRIPTION ===");
    println!("{}", final_text.trim());
    println!("===========================\n");

    if accumulated_text.trim() != final_text.trim() {
        println!("Note: Incremental decode differs slightly from final decode");
        println!("(This is normal for subword tokenization)\n");
    }

    println!("✓ Streaming transcription complete!");
    println!("\nPerformance:");
    println!("  Total time: {:.2}s", total_time.as_secs_f32());
    println!("  Audio duration: {:.2}s", samples.len() as f32 / spec.sample_rate as f32);
    println!("  Real-time factor: {:.2}x", total_time.as_secs_f32() / (samples.len() as f32 / spec.sample_rate as f32));
    println!("\nStreaming approach:");
    println!("  - Processes audio in {:.1}s chunks with {:.1}s overlap", CHUNK_SECONDS, OVERLAP_SECONDS);
    println!("  - Maintains predictor (LSTM) state across chunks");
    println!("  - Emits results incrementally as each chunk completes");
    println!("  - Suitable for buffered/near-realtime applications");
    println!("\nLatency characteristics:");
    println!("  - Chunk latency: ~{:.1}s (buffering + processing)", CHUNK_SECONDS + 0.5);
    println!("  - For true frame-level streaming:");
    println!("    * Requires attention caching (in development)");
    println!("    * Would process ~40-80ms chunks with <100ms latency");

    Ok(())
}
