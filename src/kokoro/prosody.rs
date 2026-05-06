//! ProsodyPredictor — duration, F0, and energy prediction with AdaIN style conditioning.
//!
//! Tensor name prefix: "predictor."
//!
//! Structure:
//!   predictor.text_encoder.lstms.{0,2,4} — BiLSTMs (input=640, hidden=256)
//!   predictor.text_encoder.lstms.{1,3,5} — AdaIN FCs (fc.weight: [1024, 128])
//!   predictor.shared — BiLSTM (input=640, hidden=256)
//!   predictor.lstm — BiLSTM (input=640, hidden=256)
//!   predictor.duration_proj.linear_layer — Linear(512, 50)
//!   predictor.F0.{0,1,2} — AdaINResBlocks for F0 prediction
//!   predictor.N.{0,1,2} — AdaINResBlocks for energy prediction
//!   predictor.F0_proj — Conv1d(256, 1, 1) projection
//!   predictor.N_proj — Conv1d(256, 1, 1) projection

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{self as nn, Linear, Module, VarBuilder};

use super::config::KokoroConfig;

pub struct ProsodyPredictor {
    // Duration path
    dur_lstms: Vec<BiLSTMWithAdaIN>,
    shared_lstm: BiLSTM,
    duration_lstm: BiLSTM,
    duration_proj: Linear,
    // F0/N path
    f0_blocks: Vec<AdaINResBlock>,
    n_blocks: Vec<AdaINResBlock>,
    f0_proj: Tensor, // [1, 256, 1] conv weight
    f0_proj_bias: Tensor,
    n_proj: Tensor,
    n_proj_bias: Tensor,
}

struct BiLSTMWithAdaIN {
    lstm: BiLSTM,
    adain_fc_weight: Tensor,
    adain_fc_bias: Tensor,
}

struct BiLSTM {
    forward_ih: Tensor,
    forward_hh: Tensor,
    forward_bias_ih: Tensor,
    forward_bias_hh: Tensor,
    reverse_ih: Tensor,
    reverse_hh: Tensor,
    reverse_bias_ih: Tensor,
    reverse_bias_hh: Tensor,
    hidden_size: usize,
}

struct AdaINResBlock {
    conv1_weight: Tensor,
    conv1_bias: Tensor,
    conv2_weight: Tensor,
    conv2_bias: Tensor,
    norm1_fc: Linear,
    norm2_fc: Linear,
    conv1x1: Option<Tensor>,
    pool_weight: Option<Tensor>,
    pool_bias: Option<Tensor>,
}

impl ProsodyPredictor {
    pub fn load(vb: VarBuilder, cfg: &KokoroConfig) -> Result<Self> {
        // Duration text encoder: 3 BiLSTMs interleaved with AdaIN
        let te_vb = vb.pp("text_encoder").pp("lstms");
        let mut dur_lstms = Vec::new();
        for i in 0..3 {
            let lstm_idx = i * 2;
            let adain_idx = i * 2 + 1;
            let lstm = BiLSTM::load(te_vb.pp(format!("{}", lstm_idx)), 640, 256)?;
            let adain_vb = te_vb.pp(format!("{}.fc", adain_idx));
            let adain_fc_weight = adain_vb.get((1024, 128), "weight")?;
            let adain_fc_bias = adain_vb.get(1024, "bias")?;
            dur_lstms.push(BiLSTMWithAdaIN { lstm, adain_fc_weight, adain_fc_bias });
        }

        let shared_lstm = BiLSTM::load(vb.pp("shared"), 640, 256)?;
        let duration_lstm = BiLSTM::load(vb.pp("lstm"), 640, 256)?;

        let dp_vb = vb.pp("duration_proj").pp("linear_layer");
        let duration_proj = candle_nn::linear(512, cfg.max_dur, dp_vb)?;

        // F0 predictor blocks
        let mut f0_blocks = Vec::new();
        for i in 0..3 {
            f0_blocks.push(AdaINResBlock::load(vb.pp(format!("F0.{}", i)), cfg, i)?);
        }
        let f0_proj = vb.get((1, 256, 1), "F0_proj.weight")?;
        let f0_proj_bias = vb.get(1, "F0_proj.bias")?;

        // N predictor blocks
        let mut n_blocks = Vec::new();
        for i in 0..3 {
            n_blocks.push(AdaINResBlock::load(vb.pp(format!("N.{}", i)), cfg, i)?);
        }
        let n_proj = vb.get((1, 256, 1), "N_proj.weight")?;
        let n_proj_bias = vb.get(1, "N_proj.bias")?;

        Ok(Self {
            dur_lstms,
            shared_lstm,
            duration_lstm,
            duration_proj,
            f0_blocks,
            n_blocks,
            f0_proj,
            f0_proj_bias,
            n_proj,
            n_proj_bias,
        })
    }

    /// Run DurationEncoder + duration LSTM.
    /// Returns (durations [B, T], dur_encoder_output [B, 512, T] for F0/N expansion).
    pub fn predict_duration(&self, d_en: &Tensor, style: &Tensor) -> Result<(Tensor, Tensor)> {
        // d_en: [B, 512, T], style: [B, 128] (prosody style)
        // DurationEncoder: 3x (BiLSTM + AdaIN)
        let (batch, _hidden, t) = d_en.dims3()?;

        // Transpose to [B, T, 512] for LSTM input
        let mut x = d_en.transpose(1, 2)?.contiguous()?;

        // Expand style for concatenation: [B, T, 128]
        let style_expanded = style.unsqueeze(1)?.expand((batch, t, 128))?;

        for lstm_adain in &self.dur_lstms {
            // Concat style → [B, T, 640], run BiLSTM → [B, T, 512]
            let lstm_input = Tensor::cat(&[&x, &style_expanded], 2)?;
            let lstm_out = lstm_adain.lstm.forward(&lstm_input)?;

            // AdaIN: style → fc → gamma/beta, instance-normalize lstm_out
            let params = style.matmul(&lstm_adain.adain_fc_weight.t()?)?
                .broadcast_add(&lstm_adain.adain_fc_bias)?; // [B, 1024]
            let gamma = params.narrow(1, 0, 512)?.unsqueeze(1)?;
            let beta = params.narrow(1, 512, 512)?.unsqueeze(1)?;
            let mean = lstm_out.mean_keepdim(2)?;
            let diff = lstm_out.broadcast_sub(&mean)?;
            let var = diff.sqr()?.mean_keepdim(2)?;
            let norm = diff.broadcast_div(&(var + 1e-5)?.sqrt()?)?;
            x = norm.broadcast_mul(&(gamma + 1.0)?)?.broadcast_add(&beta)?;
        }

        // Concat style → [B, T, 640] (this is the DurationEncoder output in Python)
        let dur_enc_with_style = Tensor::cat(&[&x, &style_expanded], 2)?;

        // Transpose to [B, 640, T] for later duration_expand
        let dur_enc_out = dur_enc_with_style.transpose(1, 2)?;

        // predictor.lstm input is this same 640-dim output
        let dur_input = dur_enc_with_style;

        // predictor.lstm (NOT shared — shared is for F0/N)
        let dur_out = self.duration_lstm.forward(&dur_input)?; // [B, T, 512]

        // Project to max_dur and sigmoid-sum
        let logits = self.duration_proj.forward(&dur_out)?;
        let dur_probs = nn::ops::sigmoid(&logits)?;
        let durations = dur_probs.sum(candle_core::D::Minus1)?;

        Ok((durations, dur_enc_out))
    }

    pub fn predict_f0(&self, expanded_enc: &Tensor, style: &Tensor) -> Result<Tensor> {
        // Python F0Ntrain: shared LSTM first, then AdaIN blocks
        let shared_out = self.run_shared_lstm(expanded_enc, style)?;
        self.predict_prosody_feature(&shared_out, style, &self.f0_blocks, &self.f0_proj, &self.f0_proj_bias)
    }

    pub fn predict_n(&self, expanded_enc: &Tensor, style: &Tensor) -> Result<Tensor> {
        let shared_out = self.run_shared_lstm(expanded_enc, style)?;
        self.predict_prosody_feature(&shared_out, style, &self.n_blocks, &self.n_proj, &self.n_proj_bias)
    }

    fn run_shared_lstm(&self, x: &Tensor, _style: &Tensor) -> Result<Tensor> {
        // x: [B, 640, T_frames] (DurationEncoder output already has style)
        // Transpose to [B, T, 640] for LSTM
        let xt = x.transpose(1, 2)?.contiguous()?;
        // Run shared LSTM (input=640, hidden=256) → [B, T, 512]
        let out = self.shared_lstm.forward(&xt)?;
        // Transpose back to [B, 512, T] for conv processing
        out.transpose(1, 2).map_err(Into::into)
    }

    fn predict_prosody_feature(
        &self, x: &Tensor, style: &Tensor,
        blocks: &[AdaINResBlock], proj_w: &Tensor, proj_b: &Tensor,
    ) -> Result<Tensor> {
        // x: [B, 512, T_frames], style: [B, 128]
        let mut h = x.clone();
        for block in blocks {
            h = block.forward(&h, style)?;
        }
        // Project to scalar: conv1d with [1, 256, 1]
        let out = h.conv1d(proj_w, 0, 1, 1, 1)?;
        let out = out.broadcast_add(&proj_b.unsqueeze(0)?.unsqueeze(2)?)?;
        out.squeeze(1).map_err(Into::into)
    }
}

impl BiLSTM {
    fn load(vb: VarBuilder, input_size: usize, hidden_size: usize) -> Result<Self> {
        let gate_size = 4 * hidden_size;
        Ok(Self {
            forward_ih: vb.get((gate_size, input_size), "weight_ih_l0")?,
            forward_hh: vb.get((gate_size, hidden_size), "weight_hh_l0")?,
            forward_bias_ih: vb.get(gate_size, "bias_ih_l0")?,
            forward_bias_hh: vb.get(gate_size, "bias_hh_l0")?,
            reverse_ih: vb.get((gate_size, input_size), "weight_ih_l0_reverse")?,
            reverse_hh: vb.get((gate_size, hidden_size), "weight_hh_l0_reverse")?,
            reverse_bias_ih: vb.get(gate_size, "bias_ih_l0_reverse")?,
            reverse_bias_hh: vb.get(gate_size, "bias_hh_l0_reverse")?,
            hidden_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fwd = self.run_direction(
            x, &self.forward_ih, &self.forward_hh,
            &self.forward_bias_ih, &self.forward_bias_hh, false,
        )?;
        let rev = self.run_direction(
            x, &self.reverse_ih, &self.reverse_hh,
            &self.reverse_bias_ih, &self.reverse_bias_hh, true,
        )?;
        Tensor::cat(&[&fwd, &rev], 2).map_err(Into::into)
    }

    fn run_direction(
        &self, x: &Tensor,
        w_ih: &Tensor, w_hh: &Tensor,
        b_ih: &Tensor, b_hh: &Tensor,
        reverse: bool,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();
        let hs = self.hidden_size;

        let mut h = Tensor::zeros((batch, hs), dtype, device)?;
        let mut c = Tensor::zeros((batch, hs), dtype, device)?;
        let mut outputs = Vec::with_capacity(seq_len);

        let indices: Vec<usize> = if reverse {
            (0..seq_len).rev().collect()
        } else {
            (0..seq_len).collect()
        };

        for &t in &indices {
            let xt = x.narrow(1, t, 1)?.squeeze(1)?.contiguous()?;
            let gates = (xt.matmul(&w_ih.t()?)?.broadcast_add(b_ih)?
                + h.matmul(&w_hh.t()?)?.broadcast_add(b_hh)?)?;

            let i = nn::ops::sigmoid(&gates.narrow(1, 0, hs)?)?;
            let f = nn::ops::sigmoid(&gates.narrow(1, hs, hs)?)?;
            let g = gates.narrow(1, 2 * hs, hs)?.tanh()?;
            let o = nn::ops::sigmoid(&gates.narrow(1, 3 * hs, hs)?)?;

            c = ((&f * &c)? + (&i * &g)?)?;
            h = (&o * &c.tanh()?)?;
            outputs.push(h.unsqueeze(1)?);
        }

        if reverse {
            outputs.reverse();
        }
        Tensor::cat(&outputs, 1).map_err(Into::into)
    }
}

impl AdaINResBlock {
    fn load(vb: VarBuilder, cfg: &KokoroConfig, block_idx: usize) -> Result<Self> {
        // Block 0: in=512, out=512
        // Block 1: in=512, out=256, upsample=True (has pool)
        // Block 2: in=256, out=256
        let (in_ch, out_ch) = match block_idx {
            0 => (512, 512),
            1 => (512, 256),
            _ => (256, 256),
        };

        let conv1_weight = vb.get((out_ch, in_ch, 3), "conv1.weight")?;
        let conv1_bias = vb.get(out_ch, "conv1.bias")?;
        let conv2_weight = vb.get((out_ch, out_ch, 3), "conv2.weight")?;
        let conv2_bias = vb.get(out_ch, "conv2.bias")?;

        let norm1_fc = candle_nn::linear(cfg.style_dim, in_ch * 2, vb.pp("norm1").pp("fc"))?;
        let norm2_fc = candle_nn::linear(cfg.style_dim, out_ch * 2, vb.pp("norm2").pp("fc"))?;

        let conv1x1 = if in_ch != out_ch {
            Some(vb.get((out_ch, in_ch, 1), "conv1x1.weight")?)
        } else {
            None
        };

        // Block 1 has upsample (pool = depthwise ConvTranspose1d)
        let (pool_weight, pool_bias) = if block_idx == 1 {
            (
                Some(vb.get((in_ch, 1, 3), "pool.weight")?),
                Some(vb.get(in_ch, "pool.bias")?),
            )
        } else {
            (None, None)
        };

        Ok(Self { conv1_weight, conv1_bias, conv2_weight, conv2_bias, norm1_fc, norm2_fc, conv1x1, pool_weight, pool_bias })
    }

    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let (_, in_ch, _) = x.dims3()?;

        // Residual path: AdaIN1 → LeakyReLU → pool → conv1
        let h = adain(x, style, &self.norm1_fc, in_ch)?;
        let h = leaky_relu(&h, 0.2)?;
        let h = match (&self.pool_weight, &self.pool_bias) {
            (Some(pw), Some(pb)) => {
                let h = h.conv_transpose1d(pw, 1, 1, 2, 1, in_ch)?;
                h.broadcast_add(&pb.unsqueeze(0)?.unsqueeze(2)?)?
            }
            _ => h,
        };
        let h = h.conv1d(&self.conv1_weight, 1, 1, 1, 1)?;
        let h = h.broadcast_add(&self.conv1_bias.unsqueeze(0)?.unsqueeze(2)?)?;

        let (_, out_ch, _) = h.dims3()?;

        // AdaIN2 → LeakyReLU → conv2
        let h = adain(&h, style, &self.norm2_fc, out_ch)?;
        let h = leaky_relu(&h, 0.2)?;
        let h = h.conv1d(&self.conv2_weight, 1, 1, 1, 1)?;
        let h = h.broadcast_add(&self.conv2_bias.unsqueeze(0)?.unsqueeze(2)?)?;

        // Shortcut: upsample 2x (if pool) → conv1x1
        let shortcut = if self.pool_weight.is_some() {
            upsample_nearest_2x(x)?
        } else {
            x.clone()
        };
        let shortcut = match &self.conv1x1 {
            Some(w) => shortcut.conv1d(w, 0, 1, 1, 1)?,
            None => shortcut,
        };

        let rsqrt2 = 1.0 / 2.0f64.sqrt();
        ((h + shortcut)? * rsqrt2).map_err(Into::into)
    }
}

fn leaky_relu(x: &Tensor, negative_slope: f64) -> Result<Tensor> {
    let zeros = x.zeros_like()?;
    let pos = x.maximum(&zeros)?;
    let neg = x.minimum(&zeros)?;
    (pos + neg * negative_slope).map_err(Into::into)
}

fn upsample_nearest_2x(x: &Tensor) -> Result<Tensor> {
    let (batch, channels, len) = x.dims3()?;
    let expanded = x.unsqueeze(3)?;
    let expanded = expanded.expand((batch, channels, len, 2))?;
    expanded.reshape((batch, channels, len * 2)).map_err(Into::into)
}

fn adain(x: &Tensor, style: &Tensor, fc: &Linear, channels: usize) -> Result<Tensor> {
    let params = fc.forward(style)?; // [B, 2*C]
    let gamma = params.narrow(1, 0, channels)?.unsqueeze(2)?;
    let beta = params.narrow(1, channels, channels)?.unsqueeze(2)?;

    let mean = x.mean_keepdim(2)?;
    let diff = x.broadcast_sub(&mean)?;
    let var = diff.sqr()?.mean_keepdim(2)?;
    let norm = diff.broadcast_div(&(var + 1e-5)?.sqrt()?)?;

    let scale = (gamma + 1.0)?;
    norm.broadcast_mul(&scale)?.broadcast_add(&beta).map_err(Into::into)
}

/// Expand phoneme-level features to frame-level using predicted durations.
pub fn duration_expand(features: &Tensor, durations: &[usize]) -> Result<Tensor> {
    let (batch, channels, _t_phonemes) = features.dims3()?;

    let mut expanded_slices = Vec::new();
    for (i, &dur) in durations.iter().enumerate() {
        if dur > 0 {
            let slice = features.narrow(2, i, 1)?; // [B, C, 1]
            let expanded = slice.expand((batch, channels, dur))?;
            expanded_slices.push(expanded.contiguous()?);
        }
    }

    if expanded_slices.is_empty() {
        let device = features.device();
        return Ok(Tensor::zeros((batch, channels, 0), features.dtype(), device)?);
    }

    Tensor::cat(&expanded_slices, 2).map_err(Into::into)
}
