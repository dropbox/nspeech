//! KokoroModel — top-level inference orchestration.
//!
//! Loads all submodules from safetensors by exact tensor names,
//! runs the full TTS pipeline: phoneme tokens + style → waveform at 24kHz.

use anyhow::{Result, Context};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use std::path::Path;

use super::albert::Albert;
use super::config::KokoroConfig;
use super::decoder::ISTFTNetDecoder;
use super::prosody::{self, ProsodyPredictor};
use super::text_encoder::TextEncoder;

pub struct KokoroModel {
    albert: Albert,
    bert_encoder: Linear,
    text_encoder: TextEncoder,
    prosody: ProsodyPredictor,
    decoder: ISTFTNetDecoder,
    config: KokoroConfig,
    device: Device,
}

impl KokoroModel {
    pub fn load(model_path: &Path, config: KokoroConfig, device: &Device) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_path.to_path_buf()],
                DType::F32,
                device,
            )?
        };

        let albert = Albert::load(vb.pp("bert"), config.plbert.num_hidden_layers)
            .context("loading ALBERT")?;
        let bert_encoder = candle_nn::linear(
            config.plbert.hidden_size, config.hidden_dim,
            vb.pp("bert_encoder"),
        ).context("loading bert_encoder")?;
        let text_encoder = TextEncoder::load(
            vb.pp("text_encoder"), config.n_token, config.hidden_dim,
        ).context("loading text_encoder")?;
        let prosody = ProsodyPredictor::load(
            vb.pp("predictor"), &config,
        ).context("loading predictor")?;
        let decoder = ISTFTNetDecoder::load(
            vb.pp("decoder"), &config,
        ).context("loading decoder")?;

        Ok(Self {
            albert,
            bert_encoder,
            text_encoder,
            prosody,
            decoder,
            config,
            device: device.clone(),
        })
    }

    /// Run TTS inference.
    ///
    /// - `input_ids`: phoneme token IDs [T] (no batch dim)
    /// - `style`: style vector [256] from voice pack
    /// - `speed`: speaking rate multiplier (1.0 = normal)
    ///
    /// Returns audio samples at 24kHz.
    pub fn synthesize(&self, input_ids: &[u32], style: &Tensor, speed: f32) -> Result<Vec<f32>> {
        // Wrap tokens with padding token 0 at start and end (matches Python reference)
        let mut padded: Vec<u32> = Vec::with_capacity(input_ids.len() + 2);
        padded.push(0);
        padded.extend_from_slice(input_ids);
        padded.push(0);

        let tokens = Tensor::new(padded, &self.device)?.unsqueeze(0)?.contiguous()?; // [1, T+2]

        // Split style: first 128 = acoustic, last 128 = prosody
        let acoustic_style = style.narrow(0, 0, self.config.style_dim)?.unsqueeze(0)?;
        let prosody_style = style.narrow(0, self.config.style_dim, self.config.style_dim)?.unsqueeze(0)?;

        let bert_out = self.albert.forward(&tokens)?;
        let d_en = self.bert_encoder.forward(&bert_out)?.transpose(1, 2)?;
        let text_enc = self.text_encoder.forward(&tokens)?;

        let (durations_raw, dur_enc_out) = self.prosody.predict_duration(&d_en, &prosody_style)?;
        let durations_raw = durations_raw.squeeze(0)?;
        let dur_vec: Vec<f32> = durations_raw.to_vec1()?;

        let durations: Vec<usize> = dur_vec.iter()
            .map(|&d| ((d / speed).round().max(1.0)) as usize)
            .collect();

        let total_frames: usize = durations.iter().sum();
        if total_frames == 0 {
            return Ok(Vec::new());
        }

        let expanded_enc = prosody::duration_expand(&dur_enc_out, &durations)?;
        let expanded_text = prosody::duration_expand(&text_enc, &durations)?;

        let f0 = self.prosody.predict_f0(&expanded_enc, &prosody_style)?;
        let n = self.prosody.predict_n(&expanded_enc, &prosody_style)?;

        let audio = self.decoder.forward(&expanded_text, &f0, &n, &acoustic_style)?;
        let audio = audio.squeeze(0)?;

        audio.to_vec1().map_err(Into::into)
    }

    /// Load a voice pack (.safetensors) and select style for given sequence length.
    pub fn load_voice(voice_path: &Path, seq_len: usize, device: &Device) -> Result<Tensor> {
        let tensors = candle_core::safetensors::load(voice_path, device)?;
        let voice_tensor = tensors.into_values().next()
            .ok_or_else(|| anyhow::anyhow!("Empty voice pack"))?;

        let (n, _dim) = voice_tensor.dims2()?;
        let idx = (seq_len - 1).min(n - 1);
        voice_tensor.get(idx).map_err(Into::into)
    }

    pub fn config(&self) -> &KokoroConfig {
        &self.config
    }

    pub fn sample_rate(&self) -> u32 {
        24000
    }
}
