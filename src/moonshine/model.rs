//! Moonshine V2 full model orchestration and decoding.
//!
//! Combines frontend + encoder + decoder into an end-to-end transcription pipeline.
//! Supports greedy decoding with KV cache.

use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;

use super::config::MoonshineConfig;
use super::decoder::{DecoderCache, MoonshineDecoder};
use super::encoder::MoonshineEncoder;
use super::frontend::MoonshineFrontend;

/// Full Moonshine V2 model.
pub struct MoonshineModel {
    pub cfg: MoonshineConfig,
    frontend: MoonshineFrontend,
    encoder: MoonshineEncoder,
    decoder: MoonshineDecoder,
    proj_out: candle_nn::Linear,
    tokenizer: Option<tokenizers::Tokenizer>,
}

impl MoonshineModel {
    /// Load model from safetensors file.
    pub fn load<P: AsRef<Path>>(model_dir: P, device: &Device) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config_path = model_dir.join("streaming_config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", config_path.display(), e))?;
        let cfg: MoonshineConfig = serde_json::from_str(&config_str)?;

        println!("Moonshine config: encoder_dim={}, decoder_dim={}, depth={}, vocab_size={}",
            cfg.encoder_dim, cfg.decoder_dim, cfg.encoder_num_layers, cfg.vocab_size);

        // Load weights
        let safetensors_path = model_dir.join("model.safetensors");
        let tensors = candle_core::safetensors::load(&safetensors_path, device)?;

        println!("Loaded {} tensors from {}", tensors.len(), safetensors_path.display());

        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);

        // Build model components
        // HF weight prefix: model.encoder.embedder.* -> embedder.*
        let frontend = MoonshineFrontend::new(&cfg, vb.pp("model.encoder.embedder"))?;
        let encoder = MoonshineEncoder::new(&cfg, vb.pp("model.encoder"))?;
        let decoder = MoonshineDecoder::new(&cfg, device, vb.pp("model.decoder"))?;

        // Output projection: decoder_dim -> vocab_size
        let proj_out_w = vb.pp("proj_out").get((cfg.vocab_size, cfg.decoder_dim), "weight")?;
        let proj_out = candle_nn::Linear::new(proj_out_w, None);

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            match tokenizers::Tokenizer::from_file(&tokenizer_path) {
                Ok(t) => {
                    println!("Loaded tokenizer from {}", tokenizer_path.display());
                    Some(t)
                }
                Err(e) => {
                    println!("Warning: Failed to load tokenizer: {}", e);
                    None
                }
            }
        } else {
            println!("Warning: tokenizer.json not found at {}", tokenizer_path.display());
            None
        };

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
