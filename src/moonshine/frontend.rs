//! Moonshine V2 convolutional audio frontend (embedder).
//!
//! Processes raw audio waveform → feature sequence:
//! 1. Reshape to frames of `frame_len` samples (80 = 5ms at 16kHz)
//! 2. Frame-level CMVN normalization
//! 3. Asinh compression with learnable scale
//! 4. Linear projection + SiLU activation
//! 5. Causal Conv1d (stride 2) + SiLU → 2x temporal reduction
//! 6. Causal Conv1d (stride 2) → 2x temporal reduction
//!
//! Total: 4x temporal reduction. For 1s of audio (16000 samples = 200 frames),
//! output is ~50 time steps of `encoder_dim`-dimensional features.
//!
//! Conv and linear weights are dequantized on load (no quantized conv1d in Candle).

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear};

use super::config::MoonshineConfig;

type QVarBuilder = candle_transformers::quantized_var_builder::VarBuilder;

/// Causal Conv1d: left-pads input so convolution is causal.
struct CausalConv1d {
    conv: Conv1d,
    left_pad: usize,
}

impl CausalConv1d {
    /// Build from pre-dequantized weight and bias tensors.
    fn from_tensors(
        weight: Tensor,
        bias: Tensor,
        kernel_size: usize,
        stride: usize,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            stride,
            padding: 0,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let conv = Conv1d::new(weight, Some(bias), cfg);
        let left_pad = kernel_size - 1; // dilation = 1
        Ok(Self { conv, left_pad })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, channels, time]
        let x = if self.left_pad > 0 {
            x.pad_with_zeros(2, self.left_pad, 0)?
        } else {
            x.clone()
        };
        Ok(self.conv.forward(&x)?)
    }
}

/// Frame-level cepstral mean and variance normalization.
fn frame_cmvn(x: &Tensor) -> Result<Tensor> {
    // x: [batch, frames, frame_len]
    let eps = 1e-6;
    let mean = x.mean_keepdim(2)?; // mean over frame_len dimension
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(2)?;
    let rms = (var + eps)?.sqrt()?;
    Ok(centered.broadcast_div(&rms)?)
}

/// Asinh compression with learnable scale: asinh(exp(log_k) * x).
fn asinh_compression(x: &Tensor, log_k: &Tensor) -> Result<Tensor> {
    let k = log_k.exp()?;
    let scaled = x.broadcast_mul(&k)?;
    // asinh(x) = ln(x + sqrt(x^2 + 1))
    let x_sq = scaled.sqr()?;
    let sqrt_term = (x_sq + 1.0)?.sqrt()?;
    let result = (&scaled + sqrt_term)?.log()?;
    Ok(result)
}

/// Moonshine V2 audio frontend / embedder.
///
/// All weights are dequantized on load (conv ops require dense tensors).
pub struct MoonshineFrontend {
    frame_len: usize,
    linear: Linear,
    log_k: Tensor,
    conv1: CausalConv1d,
    conv2: CausalConv1d,
}

impl MoonshineFrontend {
    pub fn new(cfg: &MoonshineConfig, vb: QVarBuilder) -> Result<Self> {
        let fe = &cfg.frontend;
        let device = vb.device();

        // Linear: frame_len -> d_model (no bias)
        // May be stored as FP32 in GGUF (Q8_0 quantization failed for small 768x80 matrix)
        let linear_w = vb.pp("linear").get((fe.d_model, cfg.frame_len), "weight")?
            .dequantize(device)?;
        let linear = Linear::new(linear_w, None);

        // Asinh compression parameter (scalar stored as [1] in GGUF)
        let log_k = vb.pp("comp").get(1, "log_k")?.dequantize(device)?
            .reshape(&[] as &[usize])?;

        // Conv1: weight stored as flattened [c1, d_model*kernel] in GGUF, reshape to [c1, d_model, kernel]
        let conv1_w = vb.pp("conv1").get((fe.c1, fe.d_model * fe.kernel_size), "weight")?
            .dequantize(device)?
            .reshape((fe.c1, fe.d_model, fe.kernel_size))?;
        let conv1_b = vb.pp("conv1").get(fe.c1, "bias")?.dequantize(device)?;
        let conv1 = CausalConv1d::from_tensors(conv1_w, conv1_b, fe.kernel_size, fe.stride)?;

        // Conv2: weight stored as flattened [c2, c1*kernel] in GGUF, reshape to [c2, c1, kernel]
        let conv2_w = vb.pp("conv2").get((fe.c2, fe.c1 * fe.kernel_size), "weight")?
            .dequantize(device)?
            .reshape((fe.c2, fe.c1, fe.kernel_size))?;
        let conv2_b = vb.pp("conv2").get(fe.c2, "bias")?.dequantize(device)?;
        let conv2 = CausalConv1d::from_tensors(conv2_w, conv2_b, fe.kernel_size, fe.stride)?;

        Ok(Self {
            frame_len: cfg.frame_len,
            linear,
            log_k,
            conv1,
            conv2,
        })
    }

    /// Process raw audio waveform to feature sequence.
    ///
    /// Input: `[batch, audio_len]` raw 16kHz audio (padded to multiple of frame_len).
    /// Output: `[batch, num_frames/4, encoder_dim]` feature sequence.
    pub fn forward(&self, audio: &Tensor) -> Result<Tensor> {
        let (batch, audio_len) = audio.dims2()?;
        let num_frames = audio_len / self.frame_len;

        // Reshape to frames: [batch, num_frames, frame_len]
        let frames = audio.reshape((batch, num_frames, self.frame_len))?;

        // Frame CMVN
        let x = frame_cmvn(&frames)?;

        // Asinh compression
        let x = asinh_compression(&x, &self.log_k)?;

        // Linear + SiLU: [batch, num_frames, d_model]
        let x = self.linear.forward(&x)?;
        let x = candle_nn::ops::silu(&x)?;

        // Transpose for conv: [batch, d_model, num_frames]
        let x = x.transpose(1, 2)?;

        // Conv1 + SiLU: temporal reduction 2x
        let x = self.conv1.forward(&x)?;
        let x = candle_nn::ops::silu(&x)?;

        // Conv2: temporal reduction 2x
        let x = self.conv2.forward(&x)?;

        // Transpose back: [batch, time, d_model]
        let x = x.transpose(1, 2)?;

        Ok(x)
    }
}
