//! Moonshine V2 full model orchestration and decoding.
//!
//! Combines frontend + encoder + decoder into an end-to-end transcription pipeline.
//! Supports greedy decoding with KV cache.
//!
//! Loads from GGUF Q8_0 quantized format only. Encoder/decoder weights are kept
//! quantized and dequantized on-the-fly during matmul for reduced memory usage.

use std::path::Path;

use anyhow::Result;
use candle_core::{Device, IndexOp, Module, Tensor};
use candle_transformers::models::with_tracing::QMatMul;

use super::config::MoonshineConfig;
use super::decoder::{DecoderCache, MoonshineDecoder};
use super::encoder::MoonshineEncoder;
use super::frontend::MoonshineFrontend;

/// Streaming transcription state.
///
/// Tracks audio accumulation and controls when to emit partial results.
/// Uses incremental encoding: caches committed encoder output and only
/// re-encodes new audio plus a small overlap on each update.
pub struct MoonshineStream {
    /// Total audio samples at last transcription
    samples_at_last_update: usize,
    /// Minimum samples between transcription updates
    update_interval_samples: usize,
    /// Minimum audio length before first transcription attempt
    min_audio_samples: usize,

    // Encoder cache
    /// Committed (stable) encoder output: [1, num_committed, encoder_dim]
    committed_encoder: Option<Tensor>,
    /// Number of committed encoder output frames
    num_committed: usize,
    /// Total frontend feature frames from last encoder run
    total_features_at_last_encode: usize,

    // Derived from config (set once in stream_new)
    /// Effective right context = sum of right windows across all encoder layers
    encoder_right_context: usize,
    /// Number of committed feature frames to re-include as left context
    encoder_overlap: usize,
}

impl MoonshineStream {
    /// Reset streaming state for a new utterance.
    pub fn reset(&mut self) {
        self.samples_at_last_update = 0;
        self.committed_encoder = None;
        self.num_committed = 0;
        self.total_features_at_last_encode = 0;
    }
}

/// Full Moonshine V2 model with quantized inference.
pub struct MoonshineModel {
    pub cfg: MoonshineConfig,
    frontend: MoonshineFrontend,
    encoder: MoonshineEncoder,
    decoder: MoonshineDecoder,
    proj_out: QMatMul,
    tokenizer: Option<tokenizers::Tokenizer>,
}

impl MoonshineModel {
    /// Load model from memory-mapped GGUF Q8_0 quantized format.
    ///
    /// Encoder/decoder weights stay quantized (Q8_0) and are dequantized on-the-fly
    /// during matrix multiplication. Frontend conv weights and embeddings are
    /// dequantized on load (Candle has no quantized conv1d or index_select).
    pub fn load_from_gguf_mmap<P: AsRef<Path>>(assets: P, device: &Device) -> Result<Self> {
        use super::{MOONSHINE_CONFIG, MOONSHINE_MODEL_Q8_0_GGUF_MMAP, MOONSHINE_TOKENIZER};

        let assets = assets.as_ref().to_path_buf();

        // Load config from embedded/file asset
        let cfg_bytes = MOONSHINE_CONFIG.bytes(&assets).map_err(|_| {
            anyhow::anyhow!("failed to load Moonshine config from assets")
        })?;
        let cfg: MoonshineConfig = serde_json::from_slice(cfg_bytes)?;

        println!(
            "Moonshine config: encoder_dim={}, decoder_dim={}, depth={}, vocab_size={}",
            cfg.encoder_dim, cfg.decoder_dim, cfg.encoder_num_layers, cfg.vocab_size
        );

        // Load tokenizer from embedded/file asset
        let tok_bytes = MOONSHINE_TOKENIZER.bytes(&assets).map_err(|_| {
            anyhow::anyhow!("failed to load Moonshine tokenizer from assets")
        })?;
        let tokenizer = match tokenizers::Tokenizer::from_bytes(tok_bytes) {
            Ok(t) => {
                println!("Loaded tokenizer from assets");
                Some(t)
            }
            Err(e) => {
                println!("Warning: Failed to load tokenizer: {}", e);
                None
            }
        };

        // Memory-map GGUF file
        let gguf_bytes = MOONSHINE_MODEL_Q8_0_GGUF_MMAP.bytes(&assets).map_err(|_| {
            anyhow::anyhow!("failed to mmap Moonshine GGUF from assets")
        })?;

        // Create quantized VarBuilder — keeps weights as QTensor (Q8_0)
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
            gguf_bytes,
            device,
        )?;

        println!("Building model (quantized weights stay in Q8_0 format)...");

        // Build model components
        let frontend = MoonshineFrontend::new(&cfg, vb.pp("model.encoder.embedder"))?;
        let encoder = MoonshineEncoder::new(&cfg, vb.pp("model.encoder"))?;
        let decoder = MoonshineDecoder::new(&cfg, device, vb.pp("model.decoder"))?;

        // Output projection: decoder_dim -> vocab_size (quantized, no bias)
        let proj_out = QMatMul::new(cfg.decoder_dim, cfg.vocab_size, vb.pp("proj_out"))?;

        Ok(Self {
            cfg,
            frontend,
            encoder,
            decoder,
            proj_out,
            tokenizer,
        })
    }

    /// Run the full encoder pipeline: audio → features.
    ///
    /// Input: raw audio samples `[1, audio_len]` (padded to multiple of frame_len).
    /// Output: `[1, enc_seq_len, encoder_dim]`.
    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        let features = self.frontend.forward(audio)?;
        self.encoder.forward(&features)
    }

    /// Greedy decode from encoder output.
    ///
    /// Returns vector of token IDs (excluding BOS, including EOS if generated).
    pub fn greedy_decode(
        &self,
        encoder_hidden: &Tensor,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        let device = encoder_hidden.device();
        let mut cache = DecoderCache::new(self.cfg.decoder_num_layers);
        let mut generated = Vec::new();

        // First step with BOS
        let input_ids = Tensor::from_vec(vec![self.cfg.bos_id as u32], (1, 1), device)?;
        let hidden = self.decoder.forward(&input_ids, encoder_hidden, &mut cache)?;
        let logits = self.proj_out.forward(&hidden)?;

        // Get first token
        let mut next_token = logits.i((0, 0))?.argmax(0)?.to_scalar::<u32>()?;
        generated.push(next_token);

        if next_token == self.cfg.eos_id as u32 {
            return Ok(generated);
        }

        // Continue generation
        for _step in 0..max_tokens - 1 {
            let input_ids = Tensor::from_vec(vec![next_token], (1, 1), device)?;
            let hidden = self.decoder.forward(&input_ids, encoder_hidden, &mut cache)?;
            let logits = self.proj_out.forward(&hidden)?;

            next_token = logits.i((0, 0))?.argmax(0)?.to_scalar::<u32>()?;
            generated.push(next_token);

            if next_token == self.cfg.eos_id as u32 {
                break;
            }
        }

        Ok(generated)
    }

    /// Decode token IDs to text using the tokenizer.
    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String> {
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;

        let text = tokenizer.decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {}", e))?;
        Ok(text)
    }

    /// Create a new streaming state.
    ///
    /// `update_interval_ms`: minimum ms of new audio between partial updates.
    /// `min_audio_ms`: minimum audio duration before first partial.
    pub fn stream_new(&self, update_interval_ms: usize, min_audio_ms: usize) -> MoonshineStream {
        // Sum of right windows across all encoder layers = effective right context
        let encoder_right_context: usize = self.cfg.sliding_windows.iter()
            .map(|[_, right]| right)
            .sum();
        let max_left: usize = self.cfg.sliding_windows.iter()
            .map(|[left, _]| *left)
            .max()
            .unwrap_or(0);
        // 3x max left window provides sufficient overlap for attention context
        let encoder_overlap = max_left * 3;

        MoonshineStream {
            samples_at_last_update: 0,
            update_interval_samples: update_interval_ms * 16, // 16 samples per ms at 16kHz
            min_audio_samples: min_audio_ms * 16,
            committed_encoder: None,
            num_committed: 0,
            total_features_at_last_encode: 0,
            encoder_right_context,
            encoder_overlap,
        }
    }

    /// Incrementally encode audio using cached encoder output for committed frames.
    ///
    /// 1. Run frontend on full audio (cheap, ~10ms for 35s)
    /// 2. Determine new feature frames vs cached
    /// 3. Re-encode: overlap of committed frames + new frames
    /// 4. Commit stable frames (all except last right_context)
    /// 5. Append to committed cache, return committed encoder output
    fn incremental_encode(
        &self,
        audio_samples: &[f32],
        stream: &mut MoonshineStream,
        device: &Device,
    ) -> Result<Tensor> {
        // 1. Frontend on full audio
        let frame_len = self.cfg.frame_len;
        let pad_len = (frame_len - audio_samples.len() % frame_len) % frame_len;
        let mut padded = audio_samples.to_vec();
        padded.extend(std::iter::repeat(0.0f32).take(pad_len));
        let audio = Tensor::from_vec(padded, (1, audio_samples.len() + pad_len), device)?;
        let all_features = self.frontend.forward(&audio)?;
        let total_features = all_features.dim(1)?;

        // 2. No new features? Return cached
        if total_features <= stream.total_features_at_last_encode
            && stream.committed_encoder.is_some()
        {
            return Ok(stream.committed_encoder.clone().unwrap());
        }

        // 3. First encode (no cache): run full encoder
        if stream.committed_encoder.is_none() {
            let encoded = self.encoder.forward(&all_features)?;
            let committable = total_features.saturating_sub(stream.encoder_right_context);
            if committable > 0 {
                let committed = encoded.i((.., ..committable, ..))?;
                stream.committed_encoder = Some(committed);
                stream.num_committed = committable;
            }
            stream.total_features_at_last_encode = total_features;
            return Ok(stream.committed_encoder.as_ref()
                .cloned()
                .unwrap_or(encoded));
        }

        // 4. Incremental: re-encode overlap + new frames
        let chunk_start = stream.num_committed.saturating_sub(stream.encoder_overlap);
        let chunk_features = all_features.i((.., chunk_start.., ..))?;
        let chunk_encoded = self.encoder.forward(&chunk_features)?;
        let chunk_len = chunk_encoded.dim(1)?;

        // 5. Extract new committed frames from chunk
        let new_committed_start = stream.num_committed - chunk_start;
        let new_committed_end = chunk_len.saturating_sub(stream.encoder_right_context);

        if new_committed_end > new_committed_start {
            let new_frames = chunk_encoded.i((.., new_committed_start..new_committed_end, ..))?;
            let prev = stream.committed_encoder.as_ref().unwrap();
            let full = Tensor::cat(&[prev, &new_frames], 1)?;
            stream.num_committed = chunk_start + new_committed_end;
            stream.committed_encoder = Some(full);
        }

        stream.total_features_at_last_encode = total_features;
        Ok(stream.committed_encoder.clone().unwrap())
    }

    /// Check if enough new audio has accumulated and transcribe if so.
    ///
    /// Returns `Some(partial_text)` when a new partial is available.
    /// Uses incremental encoding to avoid re-encoding already-committed frames.
    pub fn stream_try_update(
        &self,
        stream: &mut MoonshineStream,
        audio: &[f32],
        device: &Device,
    ) -> Result<Option<String>> {
        if audio.len() < stream.min_audio_samples {
            return Ok(None);
        }
        if audio.len() - stream.samples_at_last_update < stream.update_interval_samples {
            return Ok(None);
        }

        let encoder_out = self.incremental_encode(audio, stream, device)?;
        let enc_frames = encoder_out.dim(1)?;
        if enc_frames == 0 {
            return Ok(None);
        }

        // ~0.02s per encoder frame (frontend 4x reduction of 80-sample frames at 16kHz)
        let max_tokens = ((enc_frames as f64 * 0.02) * 6.5).ceil() as usize + 10;
        let tokens = self.greedy_decode(&encoder_out, max_tokens)?;

        // Remove EOS token if present
        let tokens: Vec<u32> = tokens
            .into_iter()
            .filter(|&t| t != self.cfg.eos_id as u32)
            .collect();

        stream.samples_at_last_update = audio.len();
        let text = self.decode_tokens(&tokens)?;
        Ok(Some(text))
    }

    /// Final transcription of all accumulated audio. Resets stream state.
    ///
    /// Uses full encode (not incremental) to ensure all frames including
    /// right-context are included in the final result.
    pub fn stream_finalize(
        &self,
        stream: &mut MoonshineStream,
        audio: &[f32],
        device: &Device,
    ) -> Result<String> {
        let text = self.transcribe(audio, device)?;
        stream.reset();
        Ok(text)
    }

    /// Full transcription pipeline: audio → text.
    ///
    /// Input: raw 16kHz mono audio samples.
    /// Output: transcribed text.
    pub fn transcribe(&self, audio_samples: &[f32], device: &Device) -> Result<String> {
        // Pad to multiple of frame_len
        let frame_len = self.cfg.frame_len;
        let pad_len = (frame_len - audio_samples.len() % frame_len) % frame_len;
        let mut padded = audio_samples.to_vec();
        padded.extend(std::iter::repeat(0.0f32).take(pad_len));

        let audio = Tensor::from_vec(padded, (1, audio_samples.len() + pad_len), device)?;

        // Encode
        let encoder_hidden = self.encode(&audio)?;

        // Compute max tokens based on audio duration
        let duration_sec = audio_samples.len() as f64 / self.cfg.sample_rate as f64;
        let max_tokens = (duration_sec * 6.5).ceil() as usize + 10; // 6.5 tokens/sec + margin

        // Decode
        let tokens = self.greedy_decode(&encoder_hidden, max_tokens)?;

        // Remove EOS token if present
        let tokens: Vec<u32> = tokens
            .into_iter()
            .filter(|&t| t != self.cfg.eos_id as u32)
            .collect();

        self.decode_tokens(&tokens)
    }
}
