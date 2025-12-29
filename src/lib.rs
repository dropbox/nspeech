use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Module, ModuleT, Tensor, D};
use candle_nn::{
    batch_norm, conv1d, conv1d_no_bias, conv2d, layer_norm, linear, BatchNorm, BatchNormConfig,
    Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Dropout, LayerNorm, LayerNormConfig, Linear,
    VarBuilder,
};
use hf_hub::api::sync::Api;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;

// Native Rust/Candle implementation of Parakeet CTC
pub mod parakeet_ctc;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct FastConformerConfig {
    pub feat_in: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub ff_mult: usize,
    pub num_layers: usize,
    pub conv_kernel_size: usize,
    pub dropout: f64,
    pub dropout_positions: f64,
    pub subsampling_channels: usize,
    pub subsampling_stride: usize,
    pub subsampling_factor: usize,
    pub scale_input: bool,
    pub vocab_size: usize,
    pub blank_id: usize,
}

impl Default for FastConformerConfig {
    fn default() -> Self {
        Self {
            feat_in: 80,
            d_model: 512,
            num_heads: 8,
            ff_mult: 4,
            num_layers: 16,
            conv_kernel_size: 31,
            dropout: 0.1,
            dropout_positions: 0.0,
            subsampling_channels: 256,
            subsampling_stride: 2,
            subsampling_factor: 8,
            scale_input: true,
            vocab_size: 1024,
            blank_id: 0,
        }
    }
}

// Sinusoidal positional encoding (kept for reference; not used in HF path).
#[allow(dead_code)]
fn sinusoidal_positional_encoding(length: usize, dim: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; length * dim];
    for pos in 0..length {
        for i in 0..(dim / 2) {
            let idx = 2 * i;
            let div_term = (pos as f32) / (10000_f32.powf(2.0 * i as f32 / dim as f32));
            data[pos * dim + idx] = div_term.sin();
            if idx + 1 < dim {
                data[pos * dim + idx + 1] = div_term.cos();
            }
        }
    }
    Ok(Tensor::from_slice(&data, (1, length, dim), device)?)
}

fn relative_positional_encoding(batch: usize, seq: usize, dim: usize, device: &Device) -> Result<Tensor> {
    // Relative positional encoding needs 2*seq-1 positions for all relative distances
    let pos_len = 2 * seq - 1;
    let mut data = vec![0f32; pos_len * dim];
    for pos in 0..pos_len {
        for i in 0..(dim / 2) {
            let idx = 2 * i;
            let div_term = (pos as f32) / (10000_f32.powf(2.0 * i as f32 / dim as f32));
            data[pos * dim + idx] = div_term.sin();
            if idx + 1 < dim {
                data[pos * dim + idx + 1] = div_term.cos();
            }
        }
    }
    let pos = Tensor::from_slice(&data, (1, pos_len, dim), device)?;
    Ok(pos.broadcast_as((batch, pos_len, dim))?)
}

/// HF-compatible 2D subsampling front-end (matches Parakeet checkpoint).
/// input: [B, T, F] -> output: [B, T/8, D_model]
pub struct ConvSubsampling {
    layers0: Conv2d,
    layers2: Conv2d,
    layers3: Conv2d,
    layers5: Conv2d,
    layers6: Conv2d,
    linear: Linear,
}

impl ConvSubsampling {
    pub fn new(cfg: &FastConformerConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let mut c = Conv2dConfig::default();
        c.stride = cfg.subsampling_stride;
        c.padding = 1;
        let conv0 = conv2d(1, cfg.subsampling_channels, 3, c, vb.pp("layers.0"))?;

        let mut c2 = Conv2dConfig::default();
        c2.stride = cfg.subsampling_stride;
        c2.padding = 1;
        c2.groups = cfg.subsampling_channels; // depthwise
        let conv2 = conv2d(
            cfg.subsampling_channels,
            cfg.subsampling_channels,
            3,
            c2,
            vb.pp("layers.2"),
        )?;

        let mut c3 = Conv2dConfig::default();
        c3.stride = 1;
        c3.padding = 0;
        let conv3 = conv2d(
            cfg.subsampling_channels,
            cfg.subsampling_channels,
            1,
            c3,
            vb.pp("layers.3"),
        )?;

        let mut c5 = Conv2dConfig::default();
        c5.stride = cfg.subsampling_stride;
        c5.padding = 1;
        c5.groups = cfg.subsampling_channels; // depthwise
        let conv5 = conv2d(
            cfg.subsampling_channels,
            cfg.subsampling_channels,
            3,
            c5,
            vb.pp("layers.5"),
        )?;

        let mut c6 = Conv2dConfig::default();
        c6.stride = 1;
        c6.padding = 0;
        let conv6 = conv2d(
            cfg.subsampling_channels,
            cfg.subsampling_channels,
            1,
            c6,
            vb.pp("layers.6"),
        )?;

        let flatten_dim = cfg.subsampling_channels * (cfg.feat_in / cfg.subsampling_factor);
        let linear = linear(flatten_dim, cfg.d_model, vb.pp("linear"))?;

        Ok(Self {
            layers0: conv0,
            layers2: conv2,
            layers3: conv3,
            layers5: conv5,
            layers6: conv6,
            linear,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: [B, T, F] -> [B, 1, T, F]
        let (b, t, f) = xs.dims3()?;
        let xs_input_flat = xs.flatten_all()?.to_vec1::<f32>()?;
        println!("    Subsampling input: [{}, {}, {}], mean={:.6}, min={:.6}, max={:.6}",
            b, t, f,
            xs_input_flat.iter().sum::<f32>() / xs_input_flat.len() as f32,
            xs_input_flat.iter().cloned().fold(f32::INFINITY, f32::min),
            xs_input_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        let xs = xs.reshape((b, t, f, 1))?.transpose(1, 3)?.transpose(2, 3)?;

        let xs = self.layers0.forward(&xs)?.relu()?;
        let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;
        println!("    After conv0+relu: shape={:?}, mean={:.6}, min={:.6}, max={:.6}",
            xs.dims(), xs_flat.iter().sum::<f32>() / xs_flat.len() as f32,
            xs_flat.iter().cloned().fold(f32::INFINITY, f32::min),
            xs_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        let xs = self.layers2.forward(&xs)?.relu()?;
        let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;
        println!("    After conv2+relu: shape={:?}, mean={:.6}, min={:.6}, max={:.6}",
            xs.dims(), xs_flat.iter().sum::<f32>() / xs_flat.len() as f32,
            xs_flat.iter().cloned().fold(f32::INFINITY, f32::min),
            xs_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        let xs = self.layers3.forward(&xs)?.relu()?;
        let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;
        println!("    After conv3+relu: shape={:?}, mean={:.6}, min={:.6}, max={:.6}",
            xs.dims(), xs_flat.iter().sum::<f32>() / xs_flat.len() as f32,
            xs_flat.iter().cloned().fold(f32::INFINITY, f32::min),
            xs_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        let xs = self.layers5.forward(&xs)?.relu()?;
        let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;
        println!("    After conv5+relu: shape={:?}, mean={:.6}, min={:.6}, max={:.6}",
            xs.dims(), xs_flat.iter().sum::<f32>() / xs_flat.len() as f32,
            xs_flat.iter().cloned().fold(f32::INFINITY, f32::min),
            xs_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        let xs = self.layers6.forward(&xs)?.relu()?;
        let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;
        println!("    After conv6+relu: shape={:?}, mean={:.6}, min={:.6}, max={:.6}",
            xs.dims(), xs_flat.iter().sum::<f32>() / xs_flat.len() as f32,
            xs_flat.iter().cloned().fold(f32::INFINITY, f32::min),
            xs_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        let (b, c, h, w) = xs.dims4()?;
        let xs = xs.transpose(1, 2)?.reshape((b, h, c * w))?;
        println!("    After flatten: shape={:?}", xs.dims());

        let xs = self.linear.forward(&xs)?;
        let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;
        println!("    After linear: shape={:?}, mean={:.6}, min={:.6}, max={:.6}",
            xs.dims(), xs_flat.iter().sum::<f32>() / xs_flat.len() as f32,
            xs_flat.iter().cloned().fold(f32::INFINITY, f32::min),
            xs_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        Ok(xs)
    }
}

pub struct FeedForward {
    w1: Linear,
    w2: Linear,
    dropout: Dropout,
}

impl FeedForward {
    pub fn new(d_model: usize, ff_mult: usize, drop: f64, vb: VarBuilder<'_>) -> Result<Self> {
        let hidden = d_model * ff_mult;
        let w1 = linear(d_model, hidden, vb.pp("linear1"))?;
        let w2 = linear(hidden, d_model, vb.pp("linear2"))?;
        let dropout = Dropout::new(drop as f32);
        Ok(Self { w1, w2, dropout })
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let xs = self.w1.forward(xs)?.silu()?;
        let xs = self.dropout.forward(&xs, train)?;
        let xs = self.w2.forward(&xs)?;
        Ok(xs)
    }
}

pub struct MultiHeadSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rel_k_weight: Tensor,
    bias_u: Tensor,
    bias_v: Tensor,
    num_heads: usize,
    head_dim: usize,
    dropout: Dropout,
    scaling: f64,
}

impl MultiHeadSelfAttention {
    pub fn new(d_model: usize, num_heads: usize, drop: f64, vb: VarBuilder<'_>) -> Result<Self> {
        if d_model % num_heads != 0 {
            return Err(anyhow!(
                "d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            ));
        }
        let head_dim = d_model / num_heads;
        let q_proj = linear(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = linear(d_model, d_model, vb.pp("k_proj"))?;
        let v_proj = linear(d_model, d_model, vb.pp("v_proj"))?;
        let o_proj = linear(d_model, d_model, vb.pp("o_proj"))?;
        let rel_k_weight = vb.get((d_model, d_model), "relative_k_proj.weight")?;
        let bias_u = vb.get((num_heads, head_dim), "bias_u")?;
        let bias_v = vb.get((num_heads, head_dim), "bias_v")?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rel_k_weight,
            bias_u,
            bias_v,
            num_heads,
            head_dim,
            dropout: Dropout::new(drop as f32),
            scaling: (head_dim as f64).powf(-0.5),
        })
    }

    pub fn forward(&self, xs: &Tensor, pos: &Tensor, attn_mask: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let (b, t, d) = xs.dims3()?;
        let debug = t == 26;  // Debug for silence.wav

        if debug {
            let xs_vec = xs.flatten_all()?.to_vec1::<f32>()?;
            println!("      Attention input: mean={:.6}", xs_vec.iter().sum::<f32>() / xs_vec.len() as f32);
        }

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        if debug {
            let q_vec = q.flatten_all()?.to_vec1::<f32>()?;
            let k_vec = k.flatten_all()?.to_vec1::<f32>()?;
            let v_vec = v.flatten_all()?.to_vec1::<f32>()?;
            println!("      Q: mean={:.6}, K: mean={:.6}, V: mean={:.6}",
                q_vec.iter().sum::<f32>() / q_vec.len() as f32,
                k_vec.iter().sum::<f32>() / k_vec.len() as f32,
                v_vec.iter().sum::<f32>() / v_vec.len() as f32);
        }
        let pos2 = pos.reshape((b * pos.dims()[1], d))?;
        let k_rel = pos2
            .matmul(&self.rel_k_weight.transpose(D::Minus2, D::Minus1)?)?
            .reshape((b, pos.dims()[1], d))?;

        if debug {
            let k_rel_vec = k_rel.flatten_all()?.to_vec1::<f32>()?;
            println!("      k_rel: mean={:.6}", k_rel_vec.iter().sum::<f32>() / k_rel_vec.len() as f32);
        }

        let q = q.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k_rel = k_rel
            .reshape((b, pos.dims()[1], self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let bu = self.bias_u.unsqueeze(0)?.unsqueeze(2)?; // [1,H,1,Dh]
        let bv = self.bias_v.unsqueeze(0)?.unsqueeze(2)?;
        let q_bias_u = q.broadcast_add(&bu)?;
        let q_bias_v = q.broadcast_add(&bv)?;
        let attn_scores_c = q_bias_u.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;

        if debug {
            let ac_vec = attn_scores_c.flatten_all()?.to_vec1::<f32>()?;
            println!("      attn_c: mean={:.6}", ac_vec.iter().sum::<f32>() / ac_vec.len() as f32);
        }

        let mut attn_scores_r = q_bias_v.matmul(&k_rel.transpose(D::Minus2, D::Minus1)?)?;

        if debug {
            let ar_vec = attn_scores_r.flatten_all()?.to_vec1::<f32>()?;
            println!("      attn_r (before shift): shape={:?}, mean={:.6}", attn_scores_r.dims(), ar_vec.iter().sum::<f32>() / ar_vec.len() as f32);
        }

        attn_scores_r = self.rel_shift(&attn_scores_r)?;

        if debug {
            let ar_vec = attn_scores_r.flatten_all()?.to_vec1::<f32>()?;
            println!("      attn_r (after shift): shape={:?}, mean={:.6}", attn_scores_r.dims(), ar_vec.iter().sum::<f32>() / ar_vec.len() as f32);
        }

        let last = attn_scores_r.dims4()?.3;
        let take = last.min(t);
        attn_scores_r = attn_scores_r.narrow(D::Minus1, 0, take)?;

        if debug {
            println!("      after narrow: last={}, take={}, final shape={:?}", last, take, attn_scores_r.dims());
        }

        let mut attn_scores = (attn_scores_c + attn_scores_r)?;

        if debug {
            let as_vec = attn_scores.flatten_all()?.to_vec1::<f32>()?;
            println!("      combined attn_scores: mean={:.6}", as_vec.iter().sum::<f32>() / as_vec.len() as f32);
        }

        if let Some(mask) = attn_mask {
            attn_scores = (attn_scores + mask)?;
        }
        let scale = (self.head_dim as f64).sqrt() as f32;
        let scale_t = Tensor::from_slice(&[scale], (), xs.device())?;
        let scale_t = scale_t.broadcast_as(attn_scores.shape())?;
        attn_scores = (attn_scores / scale_t)?;
        let attn_weights = candle_nn::ops::softmax(&attn_scores, D::Minus1)?;
        let attn_weights = self.dropout.forward(&attn_weights, train)?;
        let context = attn_weights.matmul(&v)?;
        let context = context.transpose(1, 2)?.reshape((b, t, d))?;
        let out = self.o_proj.forward(&context)?;
        Ok(out)
    }

    fn rel_shift(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B,H,T,2T-1] -> [B,H,T,T] by shifting relative positions
        let (b, h, t, p) = x.dims4()?;
        // Pad with zeros on the left: [B,H,T,2T-1] -> [B,H,T,2T]
        let zeros = Tensor::zeros((b, h, t, 1), x.dtype(), x.device())?;
        let x = Tensor::cat(&[zeros, x.clone()], 3)?;  // [B,H,T,2T]
        // Reshape to [B,H,2T,T]
        let x = x.reshape((b, h, p + 1, t))?;
        // Remove first row: [B,H,2T,T] -> [B,H,2T-1,T]
        let x = x.narrow(D::Minus2, 1, p)?;
        // Reshape back to [B,H,T,2T-1]
        let x = x.reshape((b, h, t, p))?;
        // Take only first T columns: [B,H,T,2T-1] -> [B,H,T,T]
        let x = x.narrow(D::Minus1, 0, t)?;
        Ok(x)
    }
}

pub struct ConvModule {
    pw1: Conv1d,
    dw: Conv1d,
    pw2: Conv1d,
    bn: BatchNorm,
    dropout: Dropout,
    d_model: usize,
}

impl ConvModule {
    pub fn new(d_model: usize, kernel_size: usize, drop: f64, vb: VarBuilder<'_>) -> Result<Self> {
        println!("DEBUG ConvModule::new: d_model={}, kernel_size={}", d_model, kernel_size);

        let mut cfg_pw = Conv1dConfig::default();
        cfg_pw.stride = 1;
        cfg_pw.padding = 0;
        let pw1 = conv1d(d_model, 2 * d_model, 1, cfg_pw, vb.pp("pointwise_conv1"))?;

        let mut cfg_dw = Conv1dConfig::default();
        cfg_dw.stride = 1;
        cfg_dw.padding = kernel_size / 2;
        cfg_dw.groups = d_model;
        // IMPORTANT: Depthwise conv MUST have bias (Python uses bias=True)
        let dw = conv1d(d_model, d_model, kernel_size, cfg_dw, vb.pp("depthwise_conv"))?;
        println!("DEBUG ConvModule: Created depthwise conv with kernel_size={}, padding={}, groups={}, WITH BIAS",
            kernel_size, kernel_size / 2, d_model);

        let bn_cfg = BatchNormConfig {
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            remove_mean: true,  // Must be true to match PyTorch BatchNorm (which always centers)
        };
        let bn = batch_norm(d_model, bn_cfg, vb.pp("norm"))?;

        // Debug: Check if BatchNorm loaded running stats
        // Try to access running_mean and running_var to verify they're loaded
        if let Ok(running_mean) = vb.pp("norm").get(d_model, "running_mean") {
            let rm_vec = running_mean.flatten_all()?.to_vec1::<f32>()?;
            println!("DEBUG ConvModule BatchNorm: running_mean mean={:.6}, min={:.6}, max={:.6}",
                rm_vec.iter().sum::<f32>() / rm_vec.len() as f32,
                rm_vec.iter().cloned().fold(f32::INFINITY, f32::min),
                rm_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        } else {
            println!("WARNING: BatchNorm running_mean not found!");
        }

        if let Ok(running_var) = vb.pp("norm").get(d_model, "running_var") {
            let rv_vec = running_var.flatten_all()?.to_vec1::<f32>()?;
            println!("DEBUG ConvModule BatchNorm: running_var mean={:.6}, min={:.6}, max={:.6}",
                rv_vec.iter().sum::<f32>() / rv_vec.len() as f32,
                rv_vec.iter().cloned().fold(f32::INFINITY, f32::min),
                rv_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        } else {
            println!("WARNING: BatchNorm running_var not found!");
        }

        let pw2 = conv1d(d_model, d_model, 1, cfg_pw, vb.pp("pointwise_conv2"))?;

        Ok(Self {
            pw1,
            dw,
            pw2,
            bn,
            dropout: Dropout::new(drop as f32),
            d_model,
        })
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (_b, _t, d) = xs.dims3()?;
        if d != self.d_model {
            return Err(anyhow!(
                "conv module expected d_model {}, got {}",
                self.d_model,
                d
            ));
        }
        let debug = _t == 442;  // Debug for dots.wav (442 frames)

        if debug {
            println!("    ConvModule DEBUG:");
            let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;
            println!("      Input [B,T,D]: mean={:.6}", xs_flat.iter().sum::<f32>() / xs_flat.len() as f32);
        }

        let xs = xs.transpose(1, 2)?;
        if debug {
            let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;
            println!("      After transpose [B,D,T]: mean={:.6}", xs_flat.iter().sum::<f32>() / xs_flat.len() as f32);
        }

        let xs = self.pw1.forward(&xs)?;
        if debug {
            let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;
            println!("      After pw1 [B,2D,T]: mean={:.6}", xs_flat.iter().sum::<f32>() / xs_flat.len() as f32);
        }

        let a = xs.narrow(1, 0, d)?;
        let b = xs.narrow(1, d, d)?;
        if debug {
            let a_flat = a.flatten_all()?.to_vec1::<f32>()?;
            let b_flat = b.flatten_all()?.to_vec1::<f32>()?;
            println!("      a: mean={:.6}, b: mean={:.6}",
                a_flat.iter().sum::<f32>() / a_flat.len() as f32,
                b_flat.iter().sum::<f32>() / b_flat.len() as f32);
        }

        let gated = candle_nn::ops::sigmoid(&b)?;
        if debug {
            let gated_flat = gated.flatten_all()?.to_vec1::<f32>()?;
            println!("      sigmoid(b): mean={:.6}", gated_flat.iter().sum::<f32>() / gated_flat.len() as f32);
        }

        let xs = (a * gated)?;
        if debug {
            let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;
            println!("      After gating (a*sigmoid(b)): mean={:.6}", xs_flat.iter().sum::<f32>() / xs_flat.len() as f32);
        }

        let xs = self.dw.forward(&xs)?;
        if debug {
            let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;
            println!("      After depthwise conv: mean={:.6}", xs_flat.iter().sum::<f32>() / xs_flat.len() as f32);
        }

        // Debug BatchNorm - save input for manual computation (only once for layer 0)
        static BN_SAVED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if debug && !BN_SAVED.load(std::sync::atomic::Ordering::SeqCst) {
            let xs_before_bn = xs.flatten_all()?.to_vec1::<f32>()?;
            let bn_input_mean = xs_before_bn.iter().sum::<f32>() / xs_before_bn.len() as f32;
            println!("      ConvModule [LAYER 0] train={}, input to BN: mean={:.6}, min={:.6}, max={:.6}",
                train,
                bn_input_mean,
                xs_before_bn.iter().cloned().fold(f32::INFINITY, f32::min),
                xs_before_bn.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

            // Save input to BatchNorm for Python comparison
            std::fs::write("rust_bn_input_layer0.bin",
                unsafe { std::slice::from_raw_parts(xs_before_bn.as_ptr() as *const u8, xs_before_bn.len() * 4) })?;
            println!("      → Saved BatchNorm input to rust_bn_input_layer0.bin");

            BN_SAVED.store(true, std::sync::atomic::Ordering::SeqCst);
        } else if debug {
            let xs_before_bn = xs.flatten_all()?.to_vec1::<f32>()?;
            let bn_input_mean = xs_before_bn.iter().sum::<f32>() / xs_before_bn.len() as f32;
            println!("      ConvModule train={}, input to BN: mean={:.6}, min={:.6}, max={:.6}",
                train,
                bn_input_mean,
                xs_before_bn.iter().cloned().fold(f32::INFINITY, f32::min),
                xs_before_bn.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        }

        let xs = self.bn.forward_t(&xs, train)?;

        static BN_OUTPUT_SAVED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if debug && !BN_OUTPUT_SAVED.load(std::sync::atomic::Ordering::SeqCst) {
            let xs_after_bn = xs.flatten_all()?.to_vec1::<f32>()?;
            let bn_output_mean = xs_after_bn.iter().sum::<f32>() / xs_after_bn.len() as f32;
            println!("      After BatchNorm: mean={:.6}, min={:.6}, max={:.6}",
                bn_output_mean,
                xs_after_bn.iter().cloned().fold(f32::INFINITY, f32::min),
                xs_after_bn.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

            // Save output from BatchNorm for Python comparison
            std::fs::write("rust_bn_output_layer0.bin",
                unsafe { std::slice::from_raw_parts(xs_after_bn.as_ptr() as *const u8, xs_after_bn.len() * 4) })?;
            println!("      → Saved BatchNorm output to rust_bn_output_layer0.bin");

            BN_OUTPUT_SAVED.store(true, std::sync::atomic::Ordering::SeqCst);
        } else if debug {
            let xs_after_bn = xs.flatten_all()?.to_vec1::<f32>()?;
            let bn_output_mean = xs_after_bn.iter().sum::<f32>() / xs_after_bn.len() as f32;
            println!("      After BatchNorm: mean={:.6}, min={:.6}, max={:.6}",
                bn_output_mean,
                xs_after_bn.iter().cloned().fold(f32::INFINITY, f32::min),
                xs_after_bn.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        }

        let xs = xs.silu()?;
        if debug {
            let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;
            println!("      After SiLU: mean={:.6}", xs_flat.iter().sum::<f32>() / xs_flat.len() as f32);
        }

        let xs = self.pw2.forward(&xs)?;
        if debug {
            let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;
            println!("      After pw2: mean={:.6}", xs_flat.iter().sum::<f32>() / xs_flat.len() as f32);
        }

        let xs = self.dropout.forward(&xs, train)?;
        if debug {
            let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;
            println!("      After dropout: mean={:.6}", xs_flat.iter().sum::<f32>() / xs_flat.len() as f32);
        }

        let xs = xs.transpose(1, 2)?;
        if debug {
            let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;
            println!("      After transpose back [B,T,D]: mean={:.6}", xs_flat.iter().sum::<f32>() / xs_flat.len() as f32);
        }

        Ok(xs)
    }
}

pub struct FastConformerBlock {
    ff1: FeedForward,
    ff2: FeedForward,
    self_attn: MultiHeadSelfAttention,
    conv_module: ConvModule,
    ln_ff1: LayerNorm,
    ln_mha: LayerNorm,
    ln_conv: LayerNorm,
    ln_ff2: LayerNorm,
    ln_out: LayerNorm,
}

impl FastConformerBlock {
    pub fn new(cfg: &FastConformerConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let d_model = cfg.d_model;
        let ln_cfg = LayerNormConfig {
            eps: 1e-5,
            affine: true,
            remove_mean: true,
        };
        Ok(Self {
            ff1: FeedForward::new(d_model, cfg.ff_mult, cfg.dropout, vb.pp("feed_forward1"))?,
            ff2: FeedForward::new(d_model, cfg.ff_mult, cfg.dropout, vb.pp("feed_forward2"))?,
            self_attn: MultiHeadSelfAttention::new(
                d_model,
                cfg.num_heads,
                cfg.dropout,
                vb.pp("self_attn"),
            )?,
            conv_module: ConvModule::new(d_model, cfg.conv_kernel_size, cfg.dropout, vb.pp("conv"))?,
            ln_ff1: layer_norm(d_model, ln_cfg, vb.pp("norm_feed_forward1"))?,
            ln_mha: layer_norm(d_model, ln_cfg, vb.pp("norm_self_att"))?,
            ln_conv: layer_norm(d_model, ln_cfg, vb.pp("norm_conv"))?,
            ln_ff2: layer_norm(d_model, ln_cfg, vb.pp("norm_feed_forward2"))?,
            ln_out: layer_norm(d_model, ln_cfg, vb.pp("norm_out"))?,
        })
    }

    pub fn forward(&self, _xs: &Tensor, _train: bool) -> Result<Tensor> {
        unreachable!("FastConformerBlock::forward without position embeddings is not used");
    }

    pub fn forward_with_pos_debug(
        &self,
        xs: &Tensor,
        pos: &Tensor,
        attn_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        println!("\n=== LAYER 0 DETAILED DEBUG ===");

        let ln_ff1_out = self.ln_ff1.forward(xs)?;
        let ln_ff1_flat = ln_ff1_out.flatten_all()?.to_vec1::<f32>()?;
        std::fs::write("rust_layer0_ln_ff1.bin",
            unsafe { std::slice::from_raw_parts(ln_ff1_flat.as_ptr() as *const u8, ln_ff1_flat.len() * 4) })?;
        println!("  LN_FF1: mean={:.6}, min={:.6}, max={:.6}",
            ln_ff1_flat.iter().sum::<f32>() / ln_ff1_flat.len() as f32,
            ln_ff1_flat.iter().cloned().fold(f32::INFINITY, f32::min),
            ln_ff1_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        let y_ff1 = self.ff1.forward(&ln_ff1_out, train)?;
        let y_ff1_flat = y_ff1.flatten_all()?.to_vec1::<f32>()?;
        std::fs::write("rust_layer0_ff1.bin",
            unsafe { std::slice::from_raw_parts(y_ff1_flat.as_ptr() as *const u8, y_ff1_flat.len() * 4) })?;
        println!("  FF1: mean={:.6}, min={:.6}, max={:.6}",
            y_ff1_flat.iter().sum::<f32>() / y_ff1_flat.len() as f32,
            y_ff1_flat.iter().cloned().fold(f32::INFINITY, f32::min),
            y_ff1_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        let mut y = (xs + &y_ff1)?;
        let y_after_ff1_flat = y.flatten_all()?.to_vec1::<f32>()?;
        std::fs::write("rust_layer0_after_ff1.bin",
            unsafe { std::slice::from_raw_parts(y_after_ff1_flat.as_ptr() as *const u8, y_after_ff1_flat.len() * 4) })?;
        println!("  After FF1 residual: mean={:.6}",
            y_after_ff1_flat.iter().sum::<f32>() / y_after_ff1_flat.len() as f32);

        let ln_mha_out = self.ln_mha.forward(&y)?;
        let ln_mha_flat = ln_mha_out.flatten_all()?.to_vec1::<f32>()?;
        std::fs::write("rust_layer0_ln_mha.bin",
            unsafe { std::slice::from_raw_parts(ln_mha_flat.as_ptr() as *const u8, ln_mha_flat.len() * 4) })?;
        println!("  LN_MHA: mean={:.6}",
            ln_mha_flat.iter().sum::<f32>() / ln_mha_flat.len() as f32);

        let y_attn = self.self_attn.forward(&ln_mha_out, pos, attn_mask, train)?;
        let y_attn_flat = y_attn.flatten_all()?.to_vec1::<f32>()?;
        std::fs::write("rust_layer0_attn.bin",
            unsafe { std::slice::from_raw_parts(y_attn_flat.as_ptr() as *const u8, y_attn_flat.len() * 4) })?;
        println!("  ATTN: mean={:.6}",
            y_attn_flat.iter().sum::<f32>() / y_attn_flat.len() as f32);

        y = (&y + &y_attn)?;
        let ln_conv_out = self.ln_conv.forward(&y)?;
        let ln_conv_flat = ln_conv_out.flatten_all()?.to_vec1::<f32>()?;
        std::fs::write("rust_layer0_ln_conv.bin",
            unsafe { std::slice::from_raw_parts(ln_conv_flat.as_ptr() as *const u8, ln_conv_flat.len() * 4) })?;
        println!("  LN_CONV (input to conv module): mean={:.6}",
            ln_conv_flat.iter().sum::<f32>() / ln_conv_flat.len() as f32);

        let y_conv = self.conv_module.forward(&ln_conv_out, train)?;
        let y_conv_flat = y_conv.flatten_all()?.to_vec1::<f32>()?;
        std::fs::write("rust_layer0_conv.bin",
            unsafe { std::slice::from_raw_parts(y_conv_flat.as_ptr() as *const u8, y_conv_flat.len() * 4) })?;
        println!("  CONV (output): mean={:.6}",
            y_conv_flat.iter().sum::<f32>() / y_conv_flat.len() as f32);

        y = (&y + &y_conv)?;
        let y_ff2 = self.ff2.forward(&self.ln_ff2.forward(&y)?, train)?;
        y = (&y + &y_ff2)?;
        let y_out = self.ln_out.forward(&y)?;

        println!("=== END LAYER 0 DEBUG ===\n");
        Ok(y_out)
    }

    pub fn forward_with_pos(
        &self,
        xs: &Tensor,
        pos: &Tensor,
        attn_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        // Detailed debug for first few values
        let debug = xs.dims()[1] == 26;  // Only debug when we have 26 frames (silence.wav)

        let ln_ff1_out = self.ln_ff1.forward(xs)?;
        if debug {
            let ln_vec = ln_ff1_out.flatten_all()?.to_vec1::<f32>()?;
            println!("    After LN_FF1: mean={:.6}, min={:.6}, max={:.6}",
                ln_vec.iter().sum::<f32>() / ln_vec.len() as f32,
                ln_vec.iter().cloned().fold(f32::INFINITY, f32::min),
                ln_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

            // Save for comparison - only save layer 0 by checking if we haven't saved before
            static SAVED_LAYER0: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAVED_LAYER0.swap(true, std::sync::atomic::Ordering::SeqCst) {
                std::fs::write("rust_ln_ff1_layer0_output.bin",
                    unsafe { std::slice::from_raw_parts(ln_vec.as_ptr() as *const u8, ln_vec.len() * 4) }).ok();
                println!("    SAVED layer 0 LN_FF1 output to file");
            }

            // Print first timestep
            let first_ts = ln_ff1_out.i((0, 0))?.to_vec1::<f32>()?;
            println!("    First timestep[0:10]: {:?}", &first_ts[..10]);
        }
        let y_ff1 = self.ff1.forward(&ln_ff1_out, train)?;
        if debug {
            let y_ff1_vec = y_ff1.flatten_all()?.to_vec1::<f32>()?;
            println!("    After FF1: mean={:.6}, min={:.6}, max={:.6}",
                y_ff1_vec.iter().sum::<f32>() / y_ff1_vec.len() as f32,
                y_ff1_vec.iter().cloned().fold(f32::INFINITY, f32::min),
                y_ff1_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        }

        let mut y = (xs + &y_ff1)?;

        if debug {
            let y_vec = y.flatten_all()?.to_vec1::<f32>()?;
            println!("    After add (before LN_MHA): mean={:.6}", y_vec.iter().sum::<f32>() / y_vec.len() as f32);
        }

        let ln_mha_out = self.ln_mha.forward(&y)?;

        if debug {
            let ln_vec = ln_mha_out.flatten_all()?.to_vec1::<f32>()?;
            println!("    After LN_MHA: mean={:.6}, min={:.6}, max={:.6}",
                ln_vec.iter().sum::<f32>() / ln_vec.len() as f32,
                ln_vec.iter().cloned().fold(f32::INFINITY, f32::min),
                ln_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        }

        let y_attn = self
            .self_attn
            .forward(&ln_mha_out, pos, attn_mask, train)?;
        if debug {
            let y_attn_vec = y_attn.flatten_all()?.to_vec1::<f32>()?;
            println!("    After Attn: mean={:.6}, min={:.6}, max={:.6}",
                y_attn_vec.iter().sum::<f32>() / y_attn_vec.len() as f32,
                y_attn_vec.iter().cloned().fold(f32::INFINITY, f32::min),
                y_attn_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        }

        y = (&y + &y_attn)?;
        let y_conv = self.conv_module.forward(&self.ln_conv.forward(&y)?, train)?;
        if debug {
            let y_conv_vec = y_conv.flatten_all()?.to_vec1::<f32>()?;
            println!("    After Conv: mean={:.6}, min={:.6}, max={:.6}",
                y_conv_vec.iter().sum::<f32>() / y_conv_vec.len() as f32,
                y_conv_vec.iter().cloned().fold(f32::INFINITY, f32::min),
                y_conv_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        }

        y = (&y + &y_conv)?;
        let y_ff2 = self.ff2.forward(&self.ln_ff2.forward(&y)?, train)?;
        if debug {
            let y_ff2_vec = y_ff2.flatten_all()?.to_vec1::<f32>()?;
            println!("    After FF2: mean={:.6}, min={:.6}, max={:.6}",
                y_ff2_vec.iter().sum::<f32>() / y_ff2_vec.len() as f32,
                y_ff2_vec.iter().cloned().fold(f32::INFINITY, f32::min),
                y_ff2_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        }

        y = (&y + &y_ff2)?;
        let y_out = self.ln_out.forward(&y)?;
        Ok(y_out)
    }
}

pub struct FastConformerEncoder {
    subsampling: ConvSubsampling,
    blocks: Vec<FastConformerBlock>,
    pos_dropout: Dropout,
    pos_dropout_positions: Dropout,
    cfg: FastConformerConfig,
}

impl FastConformerEncoder {
    pub fn new(cfg: FastConformerConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let subsampling = ConvSubsampling::new(&cfg, vb.pp("subsampling"))?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(FastConformerBlock::new(
                &cfg,
                vb.pp(format!("layers.{i}")),
            )?);
        }
        Ok(Self {
            subsampling,
            blocks,
            pos_dropout: Dropout::new(cfg.dropout as f32),
            pos_dropout_positions: Dropout::new(cfg.dropout_positions as f32),
            cfg,
        })
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let device = xs.device();
        println!("  DEBUG: Encoder forward - train={}", train);

        // Check if input is already subsampled (d_model dimension)
        let (_, _, input_dim) = xs.dims3()?;
        let xs = if input_dim == self.cfg.d_model {
            println!("  DEBUG: Input already has d_model dimensions, skipping subsampling");
            xs.clone()
        } else {
            println!("  DEBUG: Running subsampling ({} -> {} dims)", input_dim, self.cfg.d_model);
            self.subsampling.forward(xs)?
        };

        // Check subsampling output
        let xs_flat = xs.flatten_all()?.to_vec1::<f32>()?;
        let xs_mean = xs_flat.iter().sum::<f32>() / xs_flat.len() as f32;
        let xs_min = xs_flat.iter().cloned().fold(f32::INFINITY, f32::min);
        let xs_max = xs_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("  DEBUG: After subsampling - mean={:.6}, min={:.6}, max={:.6}", xs_mean, xs_min, xs_max);

        let (b, t, d) = xs.dims3()?;
        if d != self.cfg.d_model {
            return Err(anyhow!(
                "encoder expected d_model {}, got {}",
                self.cfg.d_model,
                d
            ));
        }

        let xs = if self.cfg.scale_input {
            let scale = (self.cfg.d_model as f64).sqrt() as f32;
            println!("  DEBUG: Scaling input by sqrt(d_model) = sqrt({}) = {:.6}", self.cfg.d_model, scale);
            let scale_t = Tensor::from_slice(&[scale], (), device)?;
            let scale_t = scale_t.broadcast_as(xs.shape())?;
            let xs_scaled = (xs * scale_t)?;

            // Check scaled values
            let xs_scaled_flat = xs_scaled.flatten_all()?.to_vec1::<f32>()?;
            let xs_scaled_mean = xs_scaled_flat.iter().sum::<f32>() / xs_scaled_flat.len() as f32;
            let xs_scaled_min = xs_scaled_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let xs_scaled_max = xs_scaled_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("  DEBUG: After scaling - mean={:.6}, min={:.6}, max={:.6}", xs_scaled_mean, xs_scaled_min, xs_scaled_max);

            xs_scaled
        } else {
            println!("  DEBUG: NOT scaling input (scale_input=false)");
            xs
        };
        // Relative positional encodings (2T-1, d_model)
        let pos = relative_positional_encoding(b, t, d, device)?;
        let pos = self.pos_dropout_positions.forward(&pos, train)?;
        let mut h = self.pos_dropout.forward(&xs, train)?;

        // Check if dropout modified the input
        let xs_after_dropout = h.clone();
        let diff = ((xs_after_dropout - xs.clone())?.abs()?.sum_all()?).to_scalar::<f32>()?;
        println!("  DEBUG: Dropout diff (should be 0 if train=false): {:.6}", diff);

        // Debug: check input to first encoder layer
        let h_first_layer_input = h.flatten_all()?.to_vec1::<f32>()?;
        println!("  DEBUG: Input to encoder layer 0:");
        println!("    Stats: min={:.6}, max={:.6}, mean={:.6}",
            h_first_layer_input.iter().cloned().fold(f32::INFINITY, f32::min),
            h_first_layer_input.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            h_first_layer_input.iter().sum::<f32>() / h_first_layer_input.len() as f32);
        let h_first_timestep = h.i((0, 0))?.to_vec1::<f32>()?;
        println!("    First timestep[0:10]: {:?}", &h_first_timestep[..10]);

        // Save Rust encoder input for precise comparison
        std::fs::write("rust_encoder_input.bin",
            unsafe { std::slice::from_raw_parts(h_first_layer_input.as_ptr() as *const u8, h_first_layer_input.len() * 4) })?;
        println!("    Saved encoder input to rust_encoder_input.bin");

        for (i, blk) in self.blocks.iter().enumerate() {
            // For layer 0, use special debugging version
            if i == 0 {
                h = blk.forward_with_pos_debug(&h, &pos, None, train)?;
            } else {
                h = blk.forward_with_pos(&h, &pos, None, train)?;
            }

            // Save all layer outputs for debugging
            let h_flat = h.flatten_all()?.to_vec1::<f32>()?;
            std::fs::write(format!("rust_layer{}_output.bin", i),
                unsafe { std::slice::from_raw_parts(h_flat.as_ptr() as *const u8, h_flat.len() * 4) })?;

            let mean = h_flat.iter().sum::<f32>() / h_flat.len() as f32;
            let min = h_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = h_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            println!("  Layer {:2}: mean={:8.6}, min={:8.3}, max={:8.3}", i, mean, min, max);
        }
        Ok(h)
    }
}

pub struct ParakeetFastConformerCtc {
    pub encoder: FastConformerEncoder,
    pub proj: Linear,
    pub cfg: FastConformerConfig,
    tokenizer: Option<Tokenizer>,
    id2token: Option<Vec<String>>,
}

impl ParakeetFastConformerCtc {
    pub fn new(cfg: FastConformerConfig, vb: VarBuilder<'_>, id2token: Vec<String>) -> Result<Self> {
        if id2token.len() != cfg.vocab_size {
            return Err(anyhow!(
                "id2token length {} must equal vocab_size {}",
                id2token.len(),
                cfg.vocab_size
            ));
        }
        let encoder = FastConformerEncoder::new(cfg.clone(), vb.pp("encoder"))?;

        // Load CTC head - weights may be in Conv1d format [V, D, 1], need to reshape to [V, D]
        let ctc_vb = vb.pp("ctc_head");
        let proj = if let Ok(weight_3d) = ctc_vb.get((cfg.vocab_size, cfg.d_model, 1), "weight") {
            // Conv1d format - squeeze last dimension
            let weight_2d = weight_3d.squeeze(2)?;
            let bias = ctc_vb.get(cfg.vocab_size, "bias")?;
            Linear::new(weight_2d, Some(bias))
        } else {
            // Try Linear format directly
            linear(cfg.d_model, cfg.vocab_size, ctc_vb)?
        };

        Ok(Self {
            encoder,
            proj,
            cfg,
            tokenizer: None,
            id2token: Some(id2token),
        })
    }

    pub fn new_with_tokenizer(
        cfg: FastConformerConfig,
        vb: VarBuilder<'_>,
        tokenizer: Tokenizer,
    ) -> Result<Self> {
        let encoder = FastConformerEncoder::new(cfg.clone(), vb.pp("encoder"))?;

        // Load CTC head - manually load and verify weights
        let ctc_vb = vb.pp("ctc_head");

        // Check what format the weights are in and load manually
        let weight = if let Ok(w) = ctc_vb.get((cfg.vocab_size, cfg.d_model), "weight") {
            println!("DEBUG: Found Linear format weights [vocab={}, d_model={}]", cfg.vocab_size, cfg.d_model);
            w
        } else if let Ok(w) = ctc_vb.get((cfg.vocab_size, cfg.d_model, 1), "weight") {
            println!("DEBUG: Found Conv1d format weights [vocab={}, d_model={}, 1], squeezing...", cfg.vocab_size, cfg.d_model);
            let w_squeezed = w.squeeze(2)?;
            println!("DEBUG: After squeeze, shape={:?}", w_squeezed.dims());
            w_squeezed
        } else {
            return Err(anyhow!("Could not find ctc_head.weight in any expected format"));
        };

        // Verify weight shape
        println!("DEBUG: Final weight shape={:?}", weight.dims());
        let weight_vec = weight.flatten_all()?.to_vec1::<f32>()?;
        println!("DEBUG: Weight stats: mean={:.6}, min={:.6}, max={:.6}",
            weight_vec.iter().sum::<f32>() / weight_vec.len() as f32,
            weight_vec.iter().cloned().fold(f32::INFINITY, f32::min),
            weight_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        let bias = ctc_vb.get(cfg.vocab_size, "bias")?;

        // Create Linear layer with the squeezed weights
        let proj = Linear::new(weight, Some(bias));
        println!("DEBUG: Created Linear layer");

        Ok(Self {
            encoder,
            proj,
            cfg,
            tokenizer: Some(tokenizer),
            id2token: None,
        })
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let h = self.encoder.forward(xs, train)?; // [B,T,D]
        let (b, t, d) = h.dims3()?;

        // Debug: check encoder output
        let h_flat = h.flatten_all()?;
        let h_vec = h_flat.to_vec1::<f32>()?;
        let h_min = h_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let h_max = h_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let h_mean = h_vec.iter().sum::<f32>() / h_vec.len() as f32;
        println!("  Encoder output stats: shape=[{},{},{}], min={:.4}, max={:.4}, mean={:.4}",
            b, t, d, h_min, h_max, h_mean);

        // Debug: Check first feature value of encoder output
        let h_first = h.i((0, 0))?;  // First timestep
        let h_first_vec = h_first.to_vec1::<f32>()?;
        println!("  Encoder first timestep features[0:10]: {:?}", &h_first_vec[..10]);

        // Save encoder output for Python comparison
        let h_flat_for_save = h.flatten_all()?.to_vec1::<f32>()?;
        std::fs::write("rust_encoder_output.bin",
            unsafe { std::slice::from_raw_parts(h_flat_for_save.as_ptr() as *const u8,
                h_flat_for_save.len() * 4) })?;
        println!("  Saved encoder output to rust_encoder_output.bin");

        // Test Linear.forward()
        let logits = self.proj.forward(&h)?;
        let logits_first = logits.i((0, 0))?.to_vec1::<f32>()?;
        println!("  Linear.forward result[0,0,0:5]: {:?}", &logits_first[..5]);

        println!("  Python expects logit[0] ≈ 0.309, logit[1] ≈ -1.382");
        println!("  Rust got:      logit[0] = {:.3}, logit[1] = {:.3}",
            logits_first[0], logits_first[1]);

        Ok(logits)
    }

    pub fn greedy_decode(&self, logits: &Tensor) -> Result<Vec<String>> {
        let (b, t, v) = logits.dims3()?;

        // Debug: check logit statistics
        let logits_flat = logits.flatten_all()?;
        let logits_vec = logits_flat.to_vec1::<f32>()?;
        let min_val = logits_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_val = logits_vec.iter().sum::<f32>() / logits_vec.len() as f32;
        println!("  Logit stats: min={:.4}, max={:.4}, mean={:.4}, vocab_size={}", min_val, max_val, mean_val, v);

        // Check first timestep logits
        let first_logits = logits.i((0, 0))?;
        let first_vec = first_logits.to_vec1::<f32>()?;
        let blank_logit = first_vec[self.cfg.blank_id];
        let non_blank_sample: Vec<f32> = first_vec[..10].to_vec();
        println!("  First frame - blank_logit[{}]={:.4}, first_10={:?}",
            self.cfg.blank_id, blank_logit, &non_blank_sample[..10.min(non_blank_sample.len())]);

        let pred_ids = logits.argmax(D::Minus1)?;
        let pred_ids = pred_ids.to_vec2::<u32>()?;
        let mut transcripts = Vec::with_capacity(b);
        for bidx in 0..b {
            let mut prev = self.cfg.blank_id as u32;
            let mut tokens = Vec::new();

            // Debug: count predictions
            let mut blank_count = 0;
            let mut non_blank_count = 0;
            let mut unique_tokens = std::collections::HashSet::new();

            for tidx in 0..t {
                let cur = pred_ids[bidx][tidx];
                if cur == self.cfg.blank_id as u32 {
                    blank_count += 1;
                    prev = cur;
                    continue;
                }
                non_blank_count += 1;
                unique_tokens.insert(cur);
                if cur == prev {
                    continue;
                }
                tokens.push(cur);
                prev = cur;
            }

            println!("  Debug: blank_id={}, blanks={}/{}, non-blanks={}, unique={}, after_dedup={}",
                self.cfg.blank_id, blank_count, t, non_blank_count, unique_tokens.len(), tokens.len());
            if tokens.len() > 0 && tokens.len() <= 20 {
                println!("  First tokens: {:?}", &tokens[..tokens.len().min(20)]);
            }

            let text = self.decode_tokens(&tokens)?;
            transcripts.push(text);
        }
        Ok(transcripts)
    }

    pub fn decode_tokens(&self, ids: &[u32]) -> Result<String> {
        if let Some(ref tok) = self.tokenizer {
            return tok.decode(ids, true).map_err(|e| anyhow!("decode error: {e}"));
        }
        if let Some(ref vocab) = self.id2token {
            let mut pieces = Vec::with_capacity(ids.len());
            for &id in ids {
                let idx = id as usize;
                if idx < vocab.len() {
                    pieces.push(vocab[idx].clone());
                }
            }
            return Ok(pieces.join(""));
        }
        Err(anyhow!("no tokenizer or id2token available for decoding"))
    }
}

#[derive(Debug, Deserialize)]
pub struct HfEncoderConfig {
    pub activation_dropout: f64,
    pub attention_dropout: f64,
    pub conv_kernel_size: usize,
    pub dropout: f64,
    pub dropout_positions: f64,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_mel_bins: usize,
    pub subsampling_conv_channels: usize,
    pub subsampling_conv_stride: usize,
    pub subsampling_factor: usize,
    pub scale_input: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct HfParakeetCtcConfig {
    pub encoder_config: HfEncoderConfig,
    pub vocab_size: usize,
    pub pad_token_id: usize,
}

impl FastConformerConfig {
    pub fn from_hf(hf: &HfParakeetCtcConfig) -> Self {
        let enc = &hf.encoder_config;
        println!("DEBUG: Loading config from HF:");
        println!("  conv_kernel_size: {}", enc.conv_kernel_size);
        println!("  d_model: {}", enc.hidden_size);
        Self {
            feat_in: enc.num_mel_bins,
            d_model: enc.hidden_size,
            num_heads: enc.num_attention_heads,
            ff_mult: enc.intermediate_size / enc.hidden_size,
            num_layers: enc.num_hidden_layers,
            conv_kernel_size: enc.conv_kernel_size,
            dropout: enc.dropout,
            dropout_positions: enc.dropout_positions,
            subsampling_channels: enc.subsampling_conv_channels,
            subsampling_stride: enc.subsampling_conv_stride,
            subsampling_factor: enc.subsampling_factor,
            scale_input: enc.scale_input.unwrap_or(true),
            vocab_size: hf.vocab_size,
            blank_id: hf.pad_token_id,
        }
    }
}

pub fn load_parakeet_ctc_from_hf(repo_id: &str, device: &Device) -> Result<ParakeetFastConformerCtc> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    let config_path = repo.get("config.json")?;
    let weights_path = repo.get("model.safetensors")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let cfg_json = std::fs::read_to_string(config_path)?;
    let hf_cfg: HfParakeetCtcConfig = serde_json::from_str(&cfg_json)?;
    let cfg = FastConformerConfig::from_hf(&hf_cfg);
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow!("tokenizer load error: {e}"))?;
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };
    ParakeetFastConformerCtc::new_with_tokenizer(cfg, vb, tokenizer)
}

pub fn load_parakeet_ctc_from_local<P: AsRef<Path>>(
    dir: P,
    device: &Device,
) -> Result<ParakeetFastConformerCtc> {
    let dir = dir.as_ref();
    let config_path = dir.join("config.json");
    // Try model_fixed.safetensors first (with reshaped CTC weights), fall back to model.safetensors
    let weights_path = if dir.join("model_fixed.safetensors").exists() {
        dir.join("model_fixed.safetensors")
    } else {
        dir.join("model.safetensors")
    };
    let tokenizer_path = dir.join("tokenizer.json");
    if !config_path.exists() || !weights_path.exists() || !tokenizer_path.exists() {
        return Err(anyhow!(
            "missing files in {:?}, need config.json, model*.safetensors, tokenizer.json",
            dir
        ));
    }
    let cfg_json = fs::read_to_string(&config_path)?;
    let hf_cfg: HfParakeetCtcConfig = serde_json::from_str(&cfg_json)?;
    let cfg = FastConformerConfig::from_hf(&hf_cfg);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("tokenizer load error: {e}"))?;
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };
    ParakeetFastConformerCtc::new_with_tokenizer(cfg, vb, tokenizer)
}

// ----------------- Audio / log-mel frontend -----------------
const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const EPS: f32 = 1e-6;

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let phase = std::f32::consts::PI * i as f32 / (n as f32);
            (phase.sin()).powi(2)
        })
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

fn build_mel_filterbank(num_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sample_rate as f32 / 2.0);
    let mel_points: Vec<f32> = (0..(num_mels + 2))
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (num_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|m| mel_to_hz(*m)).collect();
    let mut fb = vec![0f32; num_mels * n_freqs];
    for m in 0..num_mels {
        let f_m_left = hz_points[m];
        let f_m_center = hz_points[m + 1];
        let f_m_right = hz_points[m + 2];
        for (k, f) in (0..n_freqs).map(|k| (k, k as f32 * sample_rate as f32 / n_fft as f32)) {
            let weight = if f < f_m_left || f > f_m_right {
                0.0
            } else if f <= f_m_center {
                (f - f_m_left) / (f_m_center - f_m_left + EPS)
            } else {
                (f_m_right - f) / (f_m_right - f_m_center + EPS)
            };
            fb[m * n_freqs + k] = weight.max(0.0);
        }
    }
    fb
}

fn process_frame_to_mel(
    frame: &[f32],
    window: &[f32],
    fft: &std::sync::Arc<dyn Fft<f32>>,
    fb: &[f32],
    num_mels: usize,
    n_freqs: usize,
    buffer: &mut [Complex32],
    out: &mut Vec<f32>,
) {
    buffer.iter_mut().for_each(|c| *c = Complex32::new(0.0, 0.0));
    for (i, sample) in frame.iter().enumerate() {
        buffer[i] = Complex32::new(sample * window[i], 0.0);
    }
    fft.process(buffer);
    let mut power = vec![0f32; n_freqs];
    for i in 0..n_freqs {
        let c = buffer[i];
        power[i] = c.re * c.re + c.im * c.im;
    }
    for m in 0..num_mels {
        let base = m * n_freqs;
        let mut acc = 0f32;
        for k in 0..n_freqs {
            acc += fb[base + k] * power[k];
        }
        out.push((acc.max(EPS)).ln());
    }
}

pub fn log_mel_spectrogram(samples: &[f32], sample_rate: u32, num_mels: usize) -> Result<Vec<f32>> {
    if sample_rate != SAMPLE_RATE {
        return Err(anyhow!(
            "expected sample_rate {} got {}",
            SAMPLE_RATE,
            sample_rate
        ));
    }
    // Apply pre-emphasis BEFORE padding (as per preprocessor_config.json: preemphasis=0.97)
    let preemph = 0.97f32;
    let mut preemph_samples = Vec::with_capacity(samples.len());
    for (i, &s) in samples.iter().enumerate() {
        if i == 0 {
            preemph_samples.push(s);
        } else {
            preemph_samples.push(s - preemph * samples[i - 1]);
        }
    }

    // Apply reflection padding to match torchaudio center=True behavior
    // Pad by N_FFT/2 on each side
    let pad_len = N_FFT / 2;
    let mut padded = vec![0f32; preemph_samples.len() + 2 * pad_len];
    // Reflect left side
    for i in 0..pad_len {
        padded[pad_len - 1 - i] = preemph_samples[i + 1];
    }
    // Copy center
    padded[pad_len..pad_len + preemph_samples.len()].copy_from_slice(&preemph_samples);
    // Reflect right side
    let last_idx = preemph_samples.len() - 1;
    for i in 0..pad_len {
        padded[pad_len + preemph_samples.len() + i] = preemph_samples[last_idx - 1 - i];
    }

    let window = hann_window(WIN_LENGTH);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let fb = build_mel_filterbank(num_mels, N_FFT, sample_rate);
    let n_freqs = N_FFT / 2 + 1;

    // Calculate frames with center padding
    let num_frames = (samples.len() + HOP_LENGTH - 1) / HOP_LENGTH;
    let mut mel_out = Vec::with_capacity(num_frames * num_mels);
    let mut buffer = vec![Complex32::default(); N_FFT];
    let mut frame_buffer = vec![0f32; WIN_LENGTH];

    for frame_idx in 0..num_frames {
        let center = frame_idx * HOP_LENGTH + pad_len;
        let start = center.saturating_sub(WIN_LENGTH / 2);
        let end = (start + WIN_LENGTH).min(padded.len());

        frame_buffer.fill(0.0);
        let copy_len = end - start;
        frame_buffer[..copy_len].copy_from_slice(&padded[start..end]);

        process_frame_to_mel(
            &frame_buffer,
            &window,
            &fft,
            &fb,
            num_mels,
            n_freqs,
            &mut buffer,
            &mut mel_out,
        );
    }

    // Apply per-utterance mean normalization (required by Parakeet)
    // Python's ParakeetFeatureExtractor normalizes features to zero mean
    let mean = mel_out.iter().sum::<f32>() / mel_out.len() as f32;
    for val in mel_out.iter_mut() {
        *val -= mean;
    }
    Ok(mel_out)
}

/// Load pre-computed encoder input from Python (bypasses mel+subsampling)
pub fn load_python_encoder_input<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<Tensor> {
    let path_str = path.as_ref().to_str().unwrap();

    // Map audio file to pre-computed subsampling output
    let subsamp_file = if path_str.contains("dots.wav") {
        "python_subsamp_dots.bin"
    } else {
        return Err(anyhow!("Pre-computed subsampling not available for this file. Use dots.wav"));
    };

    println!("  Loading pre-computed subsampling output from: {}", subsamp_file);
    println!("  [Bypassing both Rust mel computation AND subsampling]");

    let data = std::fs::read(subsamp_file)?;
    let n_floats = data.len() / 4;
    let mut feats = Vec::with_capacity(n_floats);

    for chunk in data.chunks_exact(4) {
        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
        feats.push(f32::from_le_bytes(bytes));
    }

    // Subsampling output is [T, 1024]
    let n_frames = feats.len() / 1024;
    println!("  Subsampling output (UNSCALED): {} frames x 1024 dims", n_frames);
    println!("  Stats: mean={:.6}, min={:.6}, max={:.6}",
        feats.iter().sum::<f32>() / feats.len() as f32,
        feats.iter().cloned().fold(f32::INFINITY, f32::min),
        feats.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // NOTE: Do NOT scale here - encoder.forward() will apply sqrt(d_model) scaling
    println!("  (Scaling will be applied by encoder.forward())");

    let tensor = Tensor::from_slice(&feats, (1, n_frames, 1024), device)?;
    Ok(tensor)
}

/// Load pre-computed mel features from Python (temporary workaround)
pub fn load_python_mel_features<P: AsRef<Path>>(
    path: P,
    feat_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let path_str = path.as_ref().to_str().unwrap();

    // Map audio file to pre-computed mel features
    let mel_file = if path_str.contains("dots.wav") {
        "python_mel_dots.bin"
    } else if path_str.contains("silence.wav") {
        "python_mel_silence.bin"
    } else {
        return Err(anyhow!("Pre-computed mel features not available for this file. Use dots.wav or silence.wav"));
    };

    println!("  Loading pre-computed mel features from: {}", mel_file);

    let mel_data = std::fs::read(mel_file)?;
    let n_floats = mel_data.len() / 4;
    let mut feats = Vec::with_capacity(n_floats);

    for chunk in mel_data.chunks_exact(4) {
        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
        feats.push(f32::from_le_bytes(bytes));
    }

    let n_frames = feats.len() / feat_dim;
    println!("  Mel features: {} frames x {} dims", n_frames, feat_dim);
    println!("  Stats: mean={:.6}, min={:.6}, max={:.6}",
        feats.iter().sum::<f32>() / feats.len() as f32,
        feats.iter().cloned().fold(f32::INFINITY, f32::min),
        feats.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    let tensor = Tensor::from_slice(&feats, (1, n_frames, feat_dim), device)?;
    Ok(tensor)
}

pub fn load_wav_as_features<P: AsRef<Path>>(
    path: P,
    feat_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut reader = hound::WavReader::open(&path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(anyhow!("expected mono wav, got {} channels", spec.channels));
    }
    if spec.sample_rate != SAMPLE_RATE {
        return Err(anyhow!(
            "expected {} Hz audio, got {} Hz",
            SAMPLE_RATE,
            spec.sample_rate
        ));
    }
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Int, 24) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / 8_388_608.0))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()?,
        _ => return Err(anyhow!("unsupported WAV format")),
    };
    if samples.is_empty() {
        return Err(anyhow!("wav contains no samples"));
    }
    let feats = log_mel_spectrogram(&samples, SAMPLE_RATE, feat_dim)?;

    // Debug: save mel features for comparison
    std::fs::write("rust_mel_features.bin",
        unsafe { std::slice::from_raw_parts(feats.as_ptr() as *const u8, feats.len() * 4) })?;
    println!("  Saved mel features to rust_mel_features.bin ({} frames)", feats.len() / feat_dim);

    let tensor = Tensor::from_slice(&feats, (1, feats.len() / feat_dim, feat_dim), device)?;
    Ok(tensor)
}

pub fn stream_wav_as_feature_chunks<P: AsRef<Path>>(
    path: P,
    feat_dim: usize,
    chunk_frames: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let mut reader = hound::WavReader::open(&path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(anyhow!("expected mono wav, got {} channels", spec.channels));
    }
    if spec.sample_rate != SAMPLE_RATE {
        return Err(anyhow!(
            "expected {} Hz audio, got {} Hz",
            SAMPLE_RATE,
            spec.sample_rate
        ));
    }
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Int, 24) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / 8_388_608.0))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()?,
        _ => return Err(anyhow!("unsupported WAV format")),
    };
    if samples.is_empty() {
        return Err(anyhow!("wav contains no samples"));
    }
    // pre-emphasis
    let preemph = 0.97f32;
    let mut preemph_samples = Vec::with_capacity(samples.len());
    for (i, &s) in samples.iter().enumerate() {
        if i == 0 {
            preemph_samples.push(s);
        } else {
            preemph_samples.push(s - preemph * samples[i - 1]);
        }
    }
    let window = hann_window(WIN_LENGTH);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let fb = build_mel_filterbank(feat_dim, N_FFT, SAMPLE_RATE);
    let n_freqs = N_FFT / 2 + 1;
    let mut chunks = Vec::new();
    let mut chunk_buf: Vec<f32> = Vec::with_capacity(chunk_frames * feat_dim);
    let mut buffer = vec![Complex32::default(); N_FFT];
    let mut offset = 0usize;
    while offset + WIN_LENGTH <= samples.len() {
        process_frame_to_mel(
            &preemph_samples[offset..offset + WIN_LENGTH],
            &window,
            &fft,
            &fb,
            feat_dim,
            n_freqs,
            &mut buffer,
            &mut chunk_buf,
        );
        if chunk_buf.len() >= chunk_frames * feat_dim {
            let frames = chunk_buf.len() / feat_dim;
            let t = Tensor::from_slice(&chunk_buf, (1, frames, feat_dim), device)?;
            chunks.push(t);
            chunk_buf.clear();
        }
        offset += HOP_LENGTH;
    }
    if offset < samples.len() {
        let mut last = vec![0f32; WIN_LENGTH];
        let rem = samples.len() - offset;
        last[..rem].copy_from_slice(&preemph_samples[offset..]);
        process_frame_to_mel(
            &last,
            &window,
            &fft,
            &fb,
            feat_dim,
            n_freqs,
            &mut buffer,
            &mut chunk_buf,
        );
    }
    if !chunk_buf.is_empty() {
        let frames = chunk_buf.len() / feat_dim;
        let t = Tensor::from_slice(&chunk_buf, (1, frames, feat_dim), device)?;
        chunks.push(t);
    }
    Ok(chunks)
}
