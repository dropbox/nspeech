//! KokoroModel — top-level inference orchestration.
//!
//! Loads all submodules from safetensors by exact tensor names,
//! runs the full TTS pipeline: phoneme tokens + style → waveform at 24kHz.

use anyhow::{Result, Context};
use candle_core::{DType, Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_nn::{Linear, Module, VarBuilder};
use std::collections::HashMap;
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

    pub fn load_gguf(model_path: &Path, config: KokoroConfig, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(model_path)?;
        let gguf = gguf_file::Content::read(&mut file)?;

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (name, info) in &gguf.tensor_infos {
            let qtensor = info.read(&mut file, gguf.tensor_data_offset, device)?;
            let tensor = qtensor.dequantize(device)?;
            // Restore 3D conv shape: if name contains conv/ups/noise_convs weight and
            // the original model has 3D weights, reshape from [out, in*kernel] -> [out, in, kernel]
            let tensor = Self::maybe_reshape_conv(name, tensor, &config);
            tensors.insert(name.clone(), tensor);
        }

        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);

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

    fn maybe_reshape_conv(name: &str, tensor: Tensor, config: &KokoroConfig) -> Tensor {
        if tensor.rank() != 2 {
            return tensor;
        }
        // Known 3D conv weights and their kernel sizes
        let (out_ch, flat) = match tensor.dims2() {
            Ok(dims) => dims,
            Err(_) => return tensor,
        };

        let kernel = if name.contains("text_encoder.cnn") && name.contains(".0.weight") {
            5
        } else if name.contains("decoder.encode.conv") || name.contains("decoder.decode.") {
            if name.contains("conv1x1") { 1 } else { 3 }
        } else if name.contains("decoder.generator.ups.") {
            if name.contains("ups.0") { 20 } else { 12 }
        } else if name.contains("decoder.generator.noise_convs.0") {
            12
        } else if name.contains("decoder.generator.noise_convs.1") {
            1
        } else if name.contains("decoder.generator.resblocks.") || name.contains("noise_res.") {
            if name.contains("convs") {
                // Kernel varies: check from config resblock_kernel_sizes
                let idx: usize = name.split("resblocks.").nth(1)
                    .or_else(|| name.split("noise_res.").nth(1))
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                if name.contains("noise_res") {
                    if idx == 0 { 7 } else { 11 }
                } else {
                    config.istftnet.resblock_kernel_sizes[idx % config.istftnet.resblock_kernel_sizes.len()]
                }
            } else { 0 }
        } else if name.contains("conv_post") {
            7
        } else if name.contains("F0_conv") || name.contains("N_conv") || name.contains("pool.weight") {
            3
        } else if name.contains("asr_res") {
            1
        } else if name.contains("predictor.F0_proj") || name.contains("predictor.N_proj") {
            1
        } else if (name.contains("predictor.F0.") || name.contains("predictor.N.")) && name.contains(".weight") {
            if name.contains("conv1x1") { 1 } else { 3 }
        } else {
            0
        };

        if kernel > 0 && flat % kernel == 0 {
            let in_ch = flat / kernel;
            tensor.reshape((out_ch, in_ch, kernel)).unwrap_or(tensor)
        } else {
            tensor
        }
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
