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

use super::config::MoonshineConfig;

// Conditional matmul type: fast-cpu dequantizes to F32 for Accelerate BLAS
#[cfg(feature = "fast-cpu")]
type MM = crate::fast_matmul::MatMul;
#[cfg(not(feature = "fast-cpu"))]
type MM = candle_transformers::models::with_tracing::QMatMul;

type QVarBuilder = candle_transformers::quantized_var_builder::VarBuilder;

fn new_mm(in_dim: usize, out_dim: usize, vb: QVarBuilder) -> Result<MM> {
    #[cfg(feature = "fast-cpu")]
    {
        let qt = vb.get((out_dim, in_dim), "weight")?;
        let t = qt.dequantize(vb.device())?;
        Ok(MM::from_tensor(t))
    }
    #[cfg(not(feature = "fast-cpu"))]
    {
        Ok(MM::new(in_dim, out_dim, vb)?)
    }
}
use super::decoder::{DecoderCache, KVCache, MoonshineDecoder};
use super::encoder::MoonshineEncoder;
use super::frontend::MoonshineFrontend;
// GPU backend type aliases — unify Metal and D3D12 behind common names
#[cfg(feature = "triton-metal")]
type GpuEnc = super::gpu_encoder::GpuEncoder<super::gpu_encoder_metal::MetalEncoderBackend>;
#[cfg(feature = "triton-metal")]
type GpuDec = super::gpu_decoder::GpuDecoder<super::gpu_decoder_metal::MetalBackend>;
#[cfg(feature = "triton-d3d12")]
type GpuEnc = super::gpu_encoder::GpuEncoder<super::gpu_encoder_d3d12::D3D12EncoderBackend>;
#[cfg(feature = "triton-d3d12")]
type GpuDec = super::gpu_decoder::GpuDecoder<super::gpu_decoder_d3d12::D3D12Backend>;

/// Dispatch GPU encoder (returns F32 on CPU).
#[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
fn gpu_encode(enc: &GpuEnc, features: &Tensor) -> Result<Tensor> {
    enc.forward(features)
}

/// Streaming transcription state.
///
/// Tracks audio accumulation and controls when to emit partial results.
/// Uses incremental encoding: caches committed encoder output and only
/// re-encodes new audio plus a small overlap on each update.
///
/// Decoder KV cache is preserved across streaming updates for efficiency.
/// Self-attention K/V entries for previously generated tokens are reused,
/// avoiding the O(N²) cost of re-decoding from scratch on each partial.
/// Cross-attention K/V is invalidated when the encoder output grows (new
/// committed frames), then recomputed once from the new encoder output.
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

    // Persistent decoder state (reused across streaming updates)
    /// Decoder KV cache persisted across partial updates
    decoder_cache: Option<DecoderCache>,
    /// Tokens generated so far in this streaming session (excluding BOS/EOS)
    cached_tokens: Vec<u32>,
    /// Next token to feed to the decoder. None = need BOS (fresh) or hit EOS.
    pending_input: Option<u32>,
    /// Encoder frame count when decoder cache was last used
    encoder_frames_at_last_decode: usize,
}

impl MoonshineStream {
    /// Reset streaming state for a new utterance.
    pub fn reset(&mut self) {
        self.samples_at_last_update = 0;
        self.committed_encoder = None;
        self.num_committed = 0;
        self.total_features_at_last_encode = 0;
        self.decoder_cache = None;
        self.cached_tokens = Vec::new();
        self.pending_input = None;
        self.encoder_frames_at_last_decode = 0;
    }
}

/// Full Moonshine V2 model with quantized inference.
pub struct MoonshineModel {
    pub cfg: MoonshineConfig,
    frontend: MoonshineFrontend,
    #[allow(dead_code)] // fallback when triton_encoder is None
    encoder: MoonshineEncoder,
    #[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
    gpu_encoder: Option<GpuEnc>,
    #[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
    gpu_decoder: Option<GpuDec>,
    decoder: MoonshineDecoder,
    proj_out: MM,
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

        #[cfg(feature = "triton-metal")]
        let (gpu_encoder, gpu_decoder): (Option<GpuEnc>, Option<GpuDec>) = {
            // Get the Metal device
            let metal_candle_dev = match device {
                Device::Metal(_) => device.clone(),
                _ => Device::new_metal(0)
                    .unwrap_or_else(|_| device.clone()),
            };
            let metal_dev: Option<candle_core::MetalDevice> = match &metal_candle_dev {
                Device::Metal(md) => Some(md.clone()),
                _ => None,
            };
            let enc = metal_dev.as_ref().and_then(|md| {
                use super::gpu_encoder_metal::MetalEncoderBackend;
                let backend = match MetalEncoderBackend::new(md) {
                    Ok(b) => b,
                    Err(e) => {
                        println!("  Metal encoder backend unavailable: {e}");
                        return None;
                    }
                };
                // Max seq len: 2048 is generous for Moonshine encoder
                match GpuEnc::new(backend, &cfg, vb.pp("model.encoder"), 2048) {
                    Ok(enc) => {
                        println!("  Triton encoder loaded");
                        Some(enc)
                    }
                    Err(e) => {
                        println!("  Triton encoder unavailable: {e}");
                        None
                    }
                }
            });
            let dec = metal_dev.as_ref().and_then(|md| {
                use super::gpu_decoder_metal::MetalBackend;
                let backend = match MetalBackend::new(md,
                    cfg.decoder_num_heads, cfg.decoder_intermediate_size) {
                    Ok(b) => b,
                    Err(e) => {
                        println!("  Metal backend unavailable: {e}");
                        return None;
                    }
                };
                match GpuDec::new(backend, &cfg, vb.pp("model.decoder"), vb.pp("proj_out")) {
                    Ok(dec) => {
                        println!("  Triton Metal decoder loaded");
                        Some(dec)
                    }
                    Err(e) => {
                        println!("  Triton Metal decoder unavailable: {e}");
                        None
                    }
                }
            });
            (enc, dec)
        };

        #[cfg(feature = "triton-d3d12")]
        let (gpu_encoder, gpu_decoder): (Option<GpuEnc>, Option<GpuDec>) = {
            use std::sync::Arc;
            match candle_d3d12_kernels::Gpu::new(0) {
                Ok(gpu) => {
                    let gpu = Arc::new(gpu);
                    let enc = {
                        use super::gpu_encoder_d3d12::D3D12EncoderBackend;
                        let use_fp16_acc = std::env::var("USE_FP16_ACC").map_or(false, |v| v == "1");
                        println!("  Loading Triton DXIL kernels (fp16_acc={})...", use_fp16_acc);
                        match D3D12EncoderBackend::new(&gpu, use_fp16_acc, cfg.encoder_dim) {
                            Ok(backend) => {
                                match GpuEnc::new(backend, &cfg, vb.pp("model.encoder"), 2048) {
                                    Ok(enc) => {
                                        println!("  Triton D3D12 encoder loaded");
                                        Some(enc)
                                    }
                                    Err(e) => {
                                        println!("  Triton D3D12 encoder unavailable: {e}");
                                        None
                                    }
                                }
                            }
                            Err(e) => {
                                println!("  D3D12 encoder backend unavailable: {e}");
                                None
                            }
                        }
                    };
                    let dec = {
                        use super::gpu_decoder_d3d12::D3D12Backend;
                        match D3D12Backend::new(&gpu, cfg.vocab_size, cfg.decoder_dim) {
                            Ok(backend) => {
                                match GpuDec::new(backend, &cfg, vb.pp("model.decoder"), vb.pp("proj_out")) {
                                    Ok(dec) => {
                                        println!("  Triton D3D12 decoder loaded");
                                        Some(dec)
                                    }
                                    Err(e) => {
                                        println!("  Triton D3D12 decoder unavailable: {e}");
                                        None
                                    }
                                }
                            }
                            Err(e) => {
                                println!("  D3D12 backend unavailable: {e}");
                                None
                            }
                        }
                    };
                    (enc, dec)
                }
                Err(e) => {
                    println!("  D3D12 GPU unavailable: {e}");
                    (None, None)
                }
            }
        };

        let decoder = MoonshineDecoder::new(&cfg, device, vb.pp("model.decoder"))?;

        // Output projection: decoder_dim -> vocab_size (quantized, no bias)
        let proj_out = new_mm(cfg.decoder_dim, cfg.vocab_size, vb.pp("proj_out"))?;

        Ok(Self {
            cfg,
            frontend,
            encoder,
            #[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
            gpu_encoder,
            #[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
            gpu_decoder,
            decoder,
            proj_out,
            tokenizer,
        })
    }

    /// Run the full encoder pipeline: audio → features.
    ///
    /// Input: raw audio samples `[1, audio_len]` (padded to multiple of frame_len).
    /// Output: `[1, enc_seq_len, encoder_dim]`.
    /// Run encoder on features (dispatches to GPU when available).
    fn run_encoder(&self, features: &Tensor) -> Result<Tensor> {
        #[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
        if let Some(enc) = &self.gpu_encoder {
            return gpu_encode(enc, features);
        }
        self.encoder.forward(features)
    }

    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        let features = self.frontend.forward(audio)?;
        self.run_encoder(&features)
    }

    /// Greedy decode from encoder output.
    ///
    /// Returns vector of token IDs (excluding BOS, including EOS if generated).
    pub fn greedy_decode(
        &self,
        encoder_hidden: &Tensor,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        #[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
        if let Some(dec) = &self.gpu_decoder {
            return dec.greedy_decode(encoder_hidden, max_tokens);
        }

        let device = encoder_hidden.device();
        let mut cache = DecoderCache::new(self.cfg.decoder_num_layers);
        let mut generated = Vec::new();

        // First step with BOS
        let input_ids = Tensor::from_vec(vec![self.cfg.bos_id as u32], (1, 1), device)?;
        let hidden = self.decoder.forward(&input_ids, encoder_hidden, &mut cache)?;
        let logits = self.proj_out.forward(&hidden)?;

        // Get first token
        {
            let l = logits.i((0, 0))?.to_device(&candle_core::Device::Cpu)?.to_vec1::<f32>()?;
            let top5: Vec<(usize, f32)> = {
                let mut indexed: Vec<(usize, f32)> = l.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                indexed.into_iter().take(5).collect()
            };
            eprintln!("  [cpu-dec] step0 logits first5={:.4?} top5={:?}", &l[..5], top5);
        }
        let mut next_token = logits.i((0, 0))?.argmax(0)?.to_scalar::<u32>()?;
        generated.push(next_token);
        eprint!("  [cpu-dec] tokens: {}", next_token);

        if next_token == self.cfg.eos_id as u32 {
            eprintln!();
            return Ok(generated);
        }

        // Continue generation
        for _step in 0..max_tokens - 1 {
            let input_ids = Tensor::from_vec(vec![next_token], (1, 1), device)?;
            let hidden = self.decoder.forward(&input_ids, encoder_hidden, &mut cache)?;
            let logits = self.proj_out.forward(&hidden)?;

            if _step < 5 {
                let l = logits.i((0, 0))?.to_device(&candle_core::Device::Cpu)?.to_vec1::<f32>()?;
                let top5: Vec<(usize, f32)> = {
                    let mut indexed: Vec<(usize, f32)> = l.iter().copied().enumerate().collect();
                    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    indexed.into_iter().take(5).collect()
                };
                eprintln!("  [cpu-dec] step{} top5={:?}", _step + 1, top5);
            }

            next_token = logits.i((0, 0))?.argmax(0)?.to_scalar::<u32>()?;
            generated.push(next_token);
            eprint!(" {}", next_token);

            if next_token == self.cfg.eos_id as u32 {
                break;
            }
        }
        eprintln!();

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
            decoder_cache: None,
            cached_tokens: Vec::new(),
            pending_input: None,
            encoder_frames_at_last_decode: 0,
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
            let encoded = self.run_encoder(&all_features)?;
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
        let chunk_encoded = self.run_encoder(&chunk_features)?;
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
    /// Uses incremental encoding and persistent decoder KV cache to avoid
    /// redundant work across streaming updates.
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

        let num_layers = self.cfg.decoder_num_layers;
        let cache = stream.decoder_cache.get_or_insert_with(|| DecoderCache::new(num_layers));
        let bos = self.cfg.bos_id as u32;
        let eos = self.cfg.eos_id as u32;

        // When encoder output grows, invalidate cross-attention K/V cache
        // (encoder projections change) but keep self-attention K/V cache
        // (previously generated token representations are approximately stable).
        if enc_frames != stream.encoder_frames_at_last_decode {
            for cc in &mut cache.cross_caches {
                *cc = KVCache::new();
            }
            cache.encoder_proj = None;
            stream.encoder_frames_at_last_decode = enc_frames;

            // If we previously hit EOS, the last token was already consumed
            // by the decoder but produced EOS. With more encoder context, it
            // might not be EOS anymore. Truncate to re-evaluate.
            if stream.pending_input.is_none() && !stream.cached_tokens.is_empty() {
                let last = stream.cached_tokens.pop().unwrap();
                cache.truncate(cache.seq_len.saturating_sub(1));
                stream.pending_input = Some(last);
            }
        }

        // ~0.02s per encoder frame (frontend 4x reduction of 80-sample frames at 16kHz)
        let max_tokens = ((enc_frames as f64 * 0.02) * 6.5).ceil() as usize + 10;

        // Bootstrap: feed BOS token to start decoding
        if stream.pending_input.is_none() && stream.cached_tokens.is_empty() {
            let input_ids = Tensor::from_vec(vec![bos], (1, 1), device)?;
            let hidden = self.decoder.forward(&input_ids, &encoder_out, cache)?;
            let logits = self.proj_out.forward(&hidden)?;
            let next_token = logits.i((0, 0))?.argmax(0)?.to_scalar::<u32>()?;
            if next_token == eos {
                stream.samples_at_last_update = audio.len();
                return Ok(Some(String::new()));
            }
            stream.cached_tokens.push(next_token);
            stream.pending_input = Some(next_token);
        }

        // Continue generating from where we left off
        while let Some(token) = stream.pending_input {
            if stream.cached_tokens.len() >= max_tokens {
                break;
            }
            let input_ids = Tensor::from_vec(vec![token], (1, 1), device)?;
            let hidden = self.decoder.forward(&input_ids, &encoder_out, cache)?;
            let logits = self.proj_out.forward(&hidden)?;
            let next_token = logits.i((0, 0))?.argmax(0)?.to_scalar::<u32>()?;
            if next_token == eos {
                stream.pending_input = None;
                break;
            }
            stream.cached_tokens.push(next_token);
            stream.pending_input = Some(next_token);
        }

        stream.samples_at_last_update = audio.len();
        let text = self.decode_tokens(&stream.cached_tokens)?;
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
        let t_enc = std::time::Instant::now();
        let encoder_hidden = self.encode(&audio)?;
        let enc_ms = t_enc.elapsed().as_millis();

        // Compute max tokens based on audio duration
        let duration_sec = audio_samples.len() as f64 / self.cfg.sample_rate as f64;
        let max_tokens = (duration_sec * 6.5).ceil() as usize + 10; // 6.5 tokens/sec + margin

        // Decode
        let t_dec = std::time::Instant::now();
        let tokens = self.greedy_decode(&encoder_hidden, max_tokens)?;
        let dec_ms = t_dec.elapsed().as_millis();
        eprintln!("  Encoder: {enc_ms}ms, Decoder: {dec_ms}ms ({} tokens, {:.0}ms/token)",
            tokens.len(), dec_ms as f64 / tokens.len().max(1) as f64);

        // Remove EOS token if present
        let tokens: Vec<u32> = tokens
            .into_iter()
            .filter(|&t| t != self.cfg.eos_id as u32)
            .collect();

        self.decode_tokens(&tokens)
    }
}
