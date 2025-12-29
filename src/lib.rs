use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Module, ModuleT, Tensor, D};
use candle_nn::{
    batch_norm, conv1d, conv2d, layer_norm, linear, BatchNorm, BatchNormConfig,
    Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Dropout, LayerNorm, LayerNormConfig, Linear,
    VarBuilder,
};
use hf_hub::api::sync::Api;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;
use std::f32::consts::PI;
use std::sync::Arc;


// Native Rust/Candle implementation of Parakeet CTC
pub mod parakeet_ctc;

// Quantized weight loader
pub mod quantized_loader;

/// Select the best available device for inference
/// Prefers Metal on macOS if PARAKEET_DEVICE env var is not set to "cpu"
/// Falls back to CPU with Accelerate framework
pub fn get_device() -> Result<Device> {
    // Allow forcing CPU mode via environment variable
    if std::env::var("PARAKEET_DEVICE").as_deref() == Ok("cpu") {
        println!("Using CPU (forced by PARAKEET_DEVICE=cpu)");
        return Ok(Device::Cpu);
    }

    #[cfg(target_os = "macos")]
    {
        // Note: Metal acceleration has some known issues with certain tensor operations
        // in Candle. If you encounter errors, set PARAKEET_DEVICE=cpu
        match Device::new_metal(0) {
            Ok(device) => {
                println!("Using Metal GPU acceleration");
                println!("  (If you encounter errors, try: PARAKEET_DEVICE=cpu)");
                return Ok(device);
            }
            Err(e) => {
                println!("Metal not available ({}), using CPU with Accelerate", e);
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    println!("Using CPU");

    #[cfg(target_os = "macos")]
    println!("Using CPU with Accelerate framework");

    Ok(Device::Cpu)
}

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

fn relative_positional_encoding(batch: usize, seq: usize, dim: usize, device: &Device) -> Result<Tensor> {
    // Relative positional encoding needs 2*seq-1 positions for all relative distances
    // Positions range from -(seq-1) to +(seq-1), centered at 0
    // Python uses symmetric encoding: sin(abs(pos)) with sign flip for negative positions
    let pos_len = 2 * seq - 1;
    let mut data = vec![0f32; pos_len * dim];
    for idx in 0..pos_len {
        // Convert index to relative position: -(seq-1), ..., -1, 0, 1, ..., +(seq-1)
        let pos = (idx as isize) - (seq as isize - 1);
        let abs_pos = pos.abs() as f32;
        // Python uses NEGATIVE sign for positive positions!
        let sign = if pos > 0 { -1.0f32 } else { 1.0f32 };

        for i in 0..(dim / 2) {
            let col_idx = 2 * i;
            let div_term = abs_pos / (10000_f32.powf(2.0 * i as f32 / dim as f32));
            // Sine: use abs(pos) but flip sign for negative positions
            data[idx * dim + col_idx] = sign * div_term.sin();
            // Cosine: use abs(pos) directly
            if col_idx + 1 < dim {
                data[idx * dim + col_idx + 1] = div_term.cos();
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
        let xs = xs.reshape((b, t, f, 1))?.transpose(1, 3)?.transpose(2, 3)?;
        // Python applies ReLU after conv0, after (conv2+conv3), after (conv5+conv6)
        let xs = self.layers0.forward(&xs)?.relu()?;
        let xs = self.layers2.forward(&xs)?;
        let xs = self.layers3.forward(&xs)?.relu()?;
        let xs = self.layers5.forward(&xs)?;
        let xs = self.layers6.forward(&xs)?.relu()?;
        let (b, c, h, w) = xs.dims4()?;
        let xs = xs.transpose(1, 2)?.reshape((b, h, c * w))?;
        let xs = self.linear.forward(&xs)?;
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
        })
    }

    pub fn forward(&self, xs: &Tensor, pos: &Tensor, attn_mask: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let (b, t, d) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;
        let pos2 = pos.reshape((b * pos.dims()[1], d))?;
        let k_rel = pos2
            .matmul(&self.rel_k_weight.transpose(D::Minus2, D::Minus1)?)?
            .reshape((b, pos.dims()[1], d))?;
        let q = q.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k_rel = k_rel
            .reshape((b, pos.dims()[1], self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let bu = self.bias_u.unsqueeze(0)?.unsqueeze(2)?; // [1,H,1,Dh]
        let bv = self.bias_v.unsqueeze(0)?.unsqueeze(2)?;
        let q_bias_u = q.broadcast_add(&bu)?;
        let q_bias_v = q.broadcast_add(&bv)?;
        let attn_scores_c = q_bias_u.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
        let mut attn_scores_r = q_bias_v.matmul(&k_rel.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
        attn_scores_r = self.rel_shift(&attn_scores_r)?;
        let last = attn_scores_r.dims4()?.3;
        let take = last.min(t);
        attn_scores_r = attn_scores_r.narrow(D::Minus1, 0, take)?;
        let mut attn_scores = (attn_scores_c + attn_scores_r)?;
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

        let bn_cfg = BatchNormConfig {
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            remove_mean: true,  // Must be true to match PyTorch BatchNorm (which always centers)
        };
        let bn = batch_norm(d_model, bn_cfg, vb.pp("norm"))?;
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
        let xs = xs.transpose(1, 2)?;
        let xs = self.pw1.forward(&xs)?;
        let a = xs.narrow(1, 0, d)?;
        let b = xs.narrow(1, d, d)?;
        let gated = candle_nn::ops::sigmoid(&b)?;
        let xs = (a * gated)?;
        let xs = self.dw.forward(&xs)?;
        let xs = self.bn.forward_t(&xs, train)?;
        let xs = xs.silu()?;
        let xs = self.pw2.forward(&xs)?;
        let xs = self.dropout.forward(&xs, train)?;
        let xs = xs.transpose(1, 2)?;
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

    pub fn forward_with_pos(
        &self,
        xs: &Tensor,
        pos: &Tensor,
        attn_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        // FF1 with 0.5 scaling factor (Conformer architecture)
        let ln_ff1_out = self.ln_ff1.forward(xs)?;
        let y_ff1 = self.ff1.forward(&ln_ff1_out, train)?;
        let y_ff1_scaled = (y_ff1 * 0.5)?;
        let mut y = (xs + &y_ff1_scaled)?;

        // Attention (no scaling)
        let ln_mha_out = self.ln_mha.forward(&y)?;
        let y_attn = self
            .self_attn
            .forward(&ln_mha_out, pos, attn_mask, train)?;
        y = (&y + &y_attn)?;

        // Conv (no scaling)
        let y_conv = self.conv_module.forward(&self.ln_conv.forward(&y)?, train)?;
        y = (&y + &y_conv)?;

        // FF2 with 0.5 scaling factor (Conformer architecture)
        let y_ff2 = self.ff2.forward(&self.ln_ff2.forward(&y)?, train)?;
        let y_ff2_scaled = (y_ff2 * 0.5)?;
        y = (&y + &y_ff2_scaled)?;

        let y_out = self.ln_out.forward(&y)?;
        Ok(y_out)
    }

}

pub struct FastConformerEncoder {
    pub subsampling: ConvSubsampling,
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
        let (_, _, input_dim) = xs.dims3()?;
        let xs = if input_dim == self.cfg.d_model {
            xs.clone()
        } else {
            self.subsampling.forward(xs)?
        };
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
            let scale_t = Tensor::from_slice(&[scale], (), device)?;
            let scale_t = scale_t.broadcast_as(xs.shape())?;
            (xs * scale_t)?
        } else {
            xs
        };
        let pos = relative_positional_encoding(b, t, d, device)?;
        let pos = self.pos_dropout_positions.forward(&pos, train)?;
        let mut h = self.pos_dropout.forward(&xs, train)?;
        for blk in self.blocks.iter() {
            h = blk.forward_with_pos(&h, &pos, None, train)?;
        }
        Ok(h)
    }

    /// Debug version that saves layer outputs for comparison with Python
    pub fn forward_debug(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let device = xs.device();
        let (_, _, input_dim) = xs.dims3()?;
        let xs = if input_dim == self.cfg.d_model {
            xs.clone()
        } else {
            self.subsampling.forward(xs)?
        };
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
            let scale_t = Tensor::from_slice(&[scale], (), device)?;
            let scale_t = scale_t.broadcast_as(xs.shape())?;
            (xs * scale_t)?
        } else {
            xs
        };
        let pos = relative_positional_encoding(b, t, d, device)?;
        let pos = self.pos_dropout_positions.forward(&pos, train)?;
        let mut h = self.pos_dropout.forward(&xs, train)?;

        // Save scaled encoder input (for comparison with Python)
        let scaled_flat = h.flatten_all()?.to_vec1::<f32>()?;
        std::fs::write("rust_encoder_input_v2.bin",
            unsafe { std::slice::from_raw_parts(scaled_flat.as_ptr() as *const u8, scaled_flat.len() * 4) })?;

        // Save layer outputs
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward_with_pos(&h, &pos, None, train)?;
            let layer_flat = h.flatten_all()?.to_vec1::<f32>()?;
            let filename = format!("rust_layer{}_output_v2.bin", i);
            std::fs::write(&filename,
                unsafe { std::slice::from_raw_parts(layer_flat.as_ptr() as *const u8, layer_flat.len() * 4) })?;
            let mean = layer_flat.iter().sum::<f32>() / layer_flat.len() as f32;
            let min = layer_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = layer_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("  Layer {:2}: mean={:.6}, min={:8.3}, max={:8.3}", i, mean, min, max);
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
        let ctc_vb = vb.pp("ctc_head");
        let weight = if let Ok(w) = ctc_vb.get((cfg.vocab_size, cfg.d_model), "weight") {
            w
        } else if let Ok(w) = ctc_vb.get((cfg.vocab_size, cfg.d_model, 1), "weight") {
            w.squeeze(2)?
        } else {
            return Err(anyhow!("Could not find ctc_head.weight in any expected format"));
        };
        let bias = ctc_vb.get(cfg.vocab_size, "bias")?;
        let proj = Linear::new(weight, Some(bias));
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
        let logits = self.proj.forward(&h)?;
        Ok(logits)
    }

    pub fn greedy_decode(&self, logits: &Tensor) -> Result<Vec<String>> {
        let (b, t, _v) = logits.dims3()?;
        let pred_ids = logits.argmax(D::Minus1)?;
        let pred_ids = pred_ids.to_vec2::<u32>()?;
        let mut transcripts = Vec::with_capacity(b);
        for bidx in 0..b {
            let mut prev = self.cfg.blank_id as u32;
            let mut tokens = Vec::new();
            for tidx in 0..t {
                let cur = pred_ids[bidx][tidx];
                if cur == self.cfg.blank_id as u32 {
                    prev = cur;
                    continue;
                }
                if cur == prev {
                    continue;
                }
                tokens.push(cur);
                prev = cur;
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

/// Load Parakeet CTC model from GGUF quantized weights stored on Hugging Face Hub
///
/// # Arguments
/// * `repo_id` - Hugging Face repository (e.g., "nvidia/parakeet-ctc-0.6b")
/// * `gguf_filename` - GGUF file name (e.g., "model_q8_0.gguf" or "model_q4k.gguf")
/// * `device` - Device to load model on
///
/// # Example
/// ```no_run
/// use parakeet::{load_parakeet_ctc_from_gguf_hf, get_device};
/// let device = get_device()?;
/// let model = load_parakeet_ctc_from_gguf_hf("nvidia/parakeet-ctc-0.6b", "model_q8_0.gguf", &device)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn load_parakeet_ctc_from_gguf_hf(
    repo_id: &str,
    gguf_filename: &str,
    device: &Device,
) -> Result<ParakeetFastConformerCtc> {
    println!("Loading quantized model from Hugging Face Hub");
    println!("  Repository: {}", repo_id);
    println!("  GGUF file: {}", gguf_filename);

    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    let config_path = repo.get("config.json")?;
    let gguf_path = repo.get(gguf_filename)?;
    let tokenizer_path = repo.get("tokenizer.json")?;

    load_gguf_model_common(config_path, gguf_path, tokenizer_path, device)
}

/// Load Parakeet CTC model from GGUF quantized weights stored locally
///
/// # Arguments
/// * `dir` - Directory containing config.json, *.gguf, and tokenizer.json
/// * `device` - Device to load model on
///
/// Expected files in directory:
/// - `config.json` - Model configuration
/// - `model_q8_0.gguf` or `model_q4k.gguf` - Quantized weights (tries q8_0 first)
/// - `tokenizer.json` - Tokenizer
///
/// # Example
/// ```no_run
/// use parakeet::{load_parakeet_ctc_from_gguf_local, get_device};
/// let device = get_device()?;
/// let model = load_parakeet_ctc_from_gguf_local("hf_parakeet", &device)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn load_parakeet_ctc_from_gguf_local<P: AsRef<Path>>(
    dir: P,
    device: &Device,
) -> Result<ParakeetFastConformerCtc> {
    let dir = dir.as_ref();
    let config_path = dir.join("config.json");
    let tokenizer_path = dir.join("tokenizer.json");

    // Try Q8_0 first (recommended), then Q4K
    let gguf_path = if dir.join("model_q8_0.gguf").exists() {
        println!("Loading Q8_0 quantized model (recommended)");
        dir.join("model_q8_0.gguf")
    } else if dir.join("model_q4k.gguf").exists() {
        println!("Loading Q4K quantized model (high compression)");
        dir.join("model_q4k.gguf")
    } else {
        return Err(anyhow!(
            "No GGUF file found in {:?}. Expected model_q8_0.gguf or model_q4k.gguf",
            dir
        ));
    };

    if !config_path.exists() || !tokenizer_path.exists() {
        return Err(anyhow!(
            "missing files in {:?}, need config.json, *.gguf, tokenizer.json",
            dir
        ));
    }

    load_gguf_model_common(config_path, gguf_path, tokenizer_path, device)
}

/// Common helper to load GGUF model from paths
fn load_gguf_model_common<P: AsRef<Path>>(
    config_path: P,
    gguf_path: P,
    tokenizer_path: P,
    device: &Device,
) -> Result<ParakeetFastConformerCtc> {
    use candle_core::quantized::gguf_file;

    // Load config
    let cfg_json = fs::read_to_string(&config_path)?;
    let hf_cfg: HfParakeetCtcConfig = serde_json::from_str(&cfg_json)?;
    let cfg = FastConformerConfig::from_hf(&hf_cfg);

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("tokenizer load error: {e}"))?;

    // Load GGUF file
    println!("  Loading GGUF file...");
    let mut file = fs::File::open(&gguf_path)?;
    let gguf_content = gguf_file::Content::read(&mut file)?;
    println!("  Loaded {} tensors from GGUF", gguf_content.tensor_infos.len());

    // Dequantize all tensors to FP32
    println!("  Dequantizing tensors to FP32...");
    let mut tensors = HashMap::new();
    for (name, _tensor_info) in gguf_content.tensor_infos.iter() {
        let qtensor = gguf_content.tensor(&mut file, name, device)?;
        let tensor = qtensor.dequantize(device)?;
        tensors.insert(name.clone(), tensor);
    }
    println!("  ✓ All tensors dequantized");

    // Create VarBuilder from dequantized tensors
    let vb = VarBuilder::from_tensors(tensors, DType::F32, device);

    println!("  Building model...");
    let model = ParakeetFastConformerCtc::new_with_tokenizer(cfg, vb, tokenizer)?;
    println!("✓ Quantized model loaded successfully\n");

    Ok(model)
}

// ----------------- Audio / log-mel frontend -----------------
const SAMPLE_RATE: u32 = 16_000;

//const LOG_ZERO_GUARD_VALUE: f32 = 2.0_f32.powi(-24);
const LOG_ZERO_GUARD_VALUE: f32 = 5.9604645e-08; // 2^-24

pub struct ParakeetFeatureExtractor {
    pub feature_size: usize,  // 80
    pub sampling_rate: usize, // 16000
    pub hop_length: usize,    // 160
    pub n_fft: usize,         // 512
    pub win_length: usize,    // 400
    pub preemphasis: f32,     // 0.97
    pub padding_value: f32,   // 0.0 (only for later padding if batching)

    window: Vec<f32>,
    mel_filters: Vec<Vec<f32>>, // [feature_size][n_fft/2+1]
    fft: Arc<dyn Fft<f32>>,
}

impl ParakeetFeatureExtractor {
    pub fn new(feature_size: usize) -> Self {
        let sampling_rate = 16_000usize;
        let hop_length = 160usize;
        let n_fft = 512usize;
        let win_length = 400usize;
        let preemphasis = 0.97f32;
        let padding_value = 0.0f32;

        let window = hann_window2(win_length);
        let mel_filters =
            mel_filterbank_slaney_norm(feature_size, sampling_rate, n_fft, 0.0, sampling_rate as f32 / 2.0);

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        Self {
            feature_size,
            sampling_rate,
            hop_length,
            n_fft,
            win_length,
            preemphasis,
            padding_value,
            window,
            mel_filters,
            fft,
        }
    }

    /// Input: 16kHz mono f32 PCM
    /// Output: Candle tensor [1, T, feature_size]
    pub fn extract_to_tensor(&self, pcm16k: &[f32], device: &Device) -> Result<Tensor> {
        let (frames, feats) = self.extract_flat(pcm16k);
        Ok(Tensor::from_vec(feats, (1, frames, self.feature_size), device)?)
    }

    /// Output flattened row-major [T, F]
    pub fn extract_flat(&self, x: &[f32]) -> (usize, Vec<f32>) {
        // 1) preemphasis
        let x = if self.preemphasis != 0.0 {
            preemphasis(x, self.preemphasis)
        } else {
            x.to_vec()
        };

        // 2) torch.stft center padding: reflection padding, pad = n_fft/2 on both sides
        let pad = self.n_fft / 2;
        let mut padded = Vec::with_capacity(x.len() + 2 * pad);

        // Reflect left side
        for i in 0..pad {
            let idx = (i + 1).min(x.len() - 1);
            padded.push(x[idx]);
        }
        padded.extend_from_slice(&x);
        // Reflect right side
        for i in 0..pad {
            let idx = x.len().saturating_sub(2 + i);
            padded.push(x[idx]);
        }

        // 3) number of frames
        let frames = if padded.len() >= self.n_fft {
            1 + (padded.len() - self.n_fft) / self.hop_length
        } else {
            0
        };

        let n_freq = self.n_fft / 2 + 1;
        let mut feats = Vec::with_capacity(frames * self.feature_size);

        let mut fft_in = vec![Complex32::new(0.0, 0.0); self.n_fft];

        for t in 0..frames {
            let start = t * self.hop_length;

            // zero buffer
            for v in fft_in.iter_mut() {
                *v = Complex32::new(0.0, 0.0);
            }

            // windowed frame in first win_length, then zero pad to n_fft
            for i in 0..self.win_length {
                fft_in[i].re = padded[start + i] * self.window[i];
            }

            // FFT
            self.fft.process(&mut fft_in);

            // power spectrum
            let mut power = vec![0.0f32; n_freq];
            for k in 0..n_freq {
                let c = fft_in[k];
                power[k] = c.re * c.re + c.im * c.im;
            }

            // mel filterbank + log10
            for m in 0..self.feature_size {
                let filt = &self.mel_filters[m];
                let mut acc = 0.0f32;
                for k in 0..n_freq {
                    acc += filt[k] * power[k];
                }
                feats.push((acc + LOG_ZERO_GUARD_VALUE).log10());
            }
        }

        // Apply per-utterance mean normalization (required by Parakeet)
        let mean = feats.iter().sum::<f32>() / feats.len() as f32;
        for val in feats.iter_mut() {
            *val -= mean;
        }

        (frames, feats)
    }
}

/* ------------------------- helpers ------------------------- */

fn preemphasis(x: &[f32], coef: f32) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }
    let mut y = Vec::with_capacity(x.len());
    y.push(x[0]);
    for i in 1..x.len() {
        y.push(x[i] - coef * x[i - 1]);
    }
    y
}

/// Hann window, periodic=true: w[n]=0.5-0.5*cos(2*pi*n/N)
/// This matches PyTorch's torch.hann_window() default (periodic=True)
fn hann_window2(n: usize) -> Vec<f32> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = n as f32;
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * (i as f32) / denom).cos())
        .collect()
}

/// Slaney mel scale
fn hz_to_mel_slaney(f: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp; // 15
    let logstep = (6.4_f32).ln() / 27.0;
    if f < min_log_hz {
        f / f_sp
    } else {
        min_log_mel + (f / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz_slaney(m: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp; // 15
    let logstep = (6.4_f32).ln() / 27.0;
    if m < min_log_mel {
        f_sp * m
    } else {
        min_log_hz * ((m - min_log_mel) * logstep).exp()
    }
}

/// librosa mel with norm="slaney" (area norm), fmin=0, fmax=sr/2
fn mel_filterbank_slaney_norm(
    n_mels: usize,
    sr: usize,
    n_fft: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<Vec<f32>> {
    let n_freq = n_fft / 2 + 1;

    let fft_freqs: Vec<f32> = (0..n_freq)
        .map(|k| (k as f32) * (sr as f32) / (n_fft as f32))
        .collect();

    let mel_min = hz_to_mel_slaney(fmin);
    let mel_max = hz_to_mel_slaney(fmax);

    // n_mels + 2 points
    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..(n_mels + 2) {
        let t = i as f32 / (n_mels + 1) as f32;
        mel_points.push(mel_min + t * (mel_max - mel_min));
    }
    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz_slaney).collect();

    let mut filters = vec![vec![0.0f32; n_freq]; n_mels];

    for m in 0..n_mels {
        let f_left = hz_points[m];
        let f_center = hz_points[m + 1];
        let f_right = hz_points[m + 2];

        // Slaney area normalization
        let denom = (f_right - f_left).max(1e-12);
        let enorm = 2.0 / denom;

        for (k, &f) in fft_freqs.iter().enumerate() {
            let w = if f < f_left || f > f_right {
                0.0
            } else if f <= f_center {
                (f - f_left) / (f_center - f_left).max(1e-12)
            } else {
                (f_right - f) / (f_right - f_center).max(1e-12)
            };
            filters[m][k] = w * enorm;
        }
    }

    filters
}

/// Load pre-computed encoder input from Python (bypasses mel+subsampling)
pub fn load_python_encoder_input<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<Tensor> {
    let path_str = path.as_ref().to_str().unwrap();
    let subsamp_file = if path_str.contains("dots.wav") {
        "python_subsamp_dots.bin"
    } else {
        return Err(anyhow!("Pre-computed subsampling not available for this file. Use dots.wav"));
    };
    let data = std::fs::read(subsamp_file)?;
    let n_floats = data.len() / 4;
    let mut feats = Vec::with_capacity(n_floats);
    for chunk in data.chunks_exact(4) {
        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
        feats.push(f32::from_le_bytes(bytes));
    }
    let n_frames = feats.len() / 1024;
    let tensor = Tensor::from_slice(&feats, (1, n_frames, 1024), device)?;
    Ok(tensor)
}

pub fn load_wav_as_features<P: AsRef<Path>>(
    path: P,
    _feat_dim: usize,
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

    let fe = ParakeetFeatureExtractor::new(80);
    let tensor: Tensor = fe.extract_to_tensor(&samples, device)?;

    //let feats = log_mel_spectrogram(&samples, SAMPLE_RATE, feat_dim)?;
    //let tensor = Tensor::from_slice(&feats, (1, feats.len() / feat_dim, feat_dim), device)?;
    Ok(tensor)
}
