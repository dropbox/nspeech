use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Module, ModuleT, Tensor, D};
use candle_nn::{
    batch_norm, conv1d, conv2d, layer_norm, linear, BatchNorm, BatchNormConfig,
    Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Dropout, LayerNorm, LayerNormConfig, Linear,
    VarBuilder,
};
use candle_transformers::models::with_tracing::QMatMul;
use hf_hub::api::sync::Api;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
#[cfg(feature = "quantized")]
use std::io::Seek;
use std::path::Path;
use tokenizers::Tokenizer;

// Embedded assets (configs/tokenizer/models)
use crate::embed_zst_asset;
embed_zst_asset!(pub PARAKEET_CONFIG,                "config.json.zst");
embed_zst_asset!(pub PARAKEET_TOKENIZER,             "tokenizer.json.zst");
embed_zst_asset!(pub PARAKEET_SPECIAL_TOKENS_MAP,    "special_tokens_map.json.zst");
embed_zst_asset!(pub PARAKEET_TOKENIZER_CONFIG,      "tokenizer_config.json.zst");
embed_zst_asset!(pub VAD_CONFIG,                     "vad16.config.json.zst");
embed_zst_asset!(pub VAD_MODEL,                      "vad16.safetensors.zst");

// Model weights (large files)
embed_zst_asset!(pub PARAKEET_MODEL_SAFETENSORS,     "model.safetensors.zst");
embed_zst_asset!(pub PARAKEET_MODEL_Q8_0_GGUF,       "model_q8_0.gguf.zst");
embed_zst_asset!(pub PARAKEET_MODEL_Q4K_GGUF,        "model_q4k.gguf.zst");

// Qwen3-0.6B-Instruct for text correction (only when "qwen" feature enabled)
#[cfg(feature = "qwen")]
embed_zst_asset!(pub QWEN_CONFIG,                    "qwen3-0.6b-instruct-config.json.zst");
#[cfg(feature = "qwen")]
embed_zst_asset!(pub QWEN_TOKENIZER,                 "qwen3-0.6b-instruct-tokenizer.json.zst");
#[cfg(feature = "qwen")]
embed_zst_asset!(pub QWEN_MODEL_Q4,                  "qwen3-0.6b-instruct-q4_k_m.gguf.zst");

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

// ============================================================================
// Quantized versions using QMatMul for faster inference with GGUF models
// ============================================================================

pub struct QFeedForward {
    w1: QMatMul,
    w2: QMatMul,
    dropout: Dropout,
}

impl QFeedForward {
    pub fn new(
        d_model: usize,
        ff_mult: usize,
        drop: f64,
        vb: candle_transformers::quantized_var_builder::VarBuilder,
    ) -> Result<Self> {
        let hidden = d_model * ff_mult;
        let w1 = QMatMul::new(d_model, hidden, vb.pp("linear1"))?;
        let w2 = QMatMul::new(hidden, d_model, vb.pp("linear2"))?;
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

pub struct QMultiHeadSelfAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    rel_k_weight: Tensor,
    bias_u: Tensor,
    bias_v: Tensor,
    num_heads: usize,
    head_dim: usize,
    dropout: Dropout,
}

impl QMultiHeadSelfAttention {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        drop: f64,
        vb: candle_transformers::quantized_var_builder::VarBuilder,
    ) -> Result<Self> {
        if d_model % num_heads != 0 {
            return Err(anyhow!(
                "d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            ));
        }
        let head_dim = d_model / num_heads;
        let q_proj = QMatMul::new(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = QMatMul::new(d_model, d_model, vb.pp("k_proj"))?;
        let v_proj = QMatMul::new(d_model, d_model, vb.pp("v_proj"))?;
        let o_proj = QMatMul::new(d_model, d_model, vb.pp("o_proj"))?;

        // These tensors are used in regular matmul ops, so dequantize them
        let rel_k_weight_q = vb.get((d_model, d_model), "relative_k_proj.weight")?;
        let rel_k_weight = rel_k_weight_q.dequantize(vb.device())?;
        let bias_u_q = vb.get((num_heads, head_dim), "bias_u")?;
        let bias_u = bias_u_q.dequantize(vb.device())?;
        let bias_v_q = vb.get((num_heads, head_dim), "bias_v")?;
        let bias_v = bias_v_q.dequantize(vb.device())?;
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
        let bu = self.bias_u.unsqueeze(0)?.unsqueeze(2)?;
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
        let (b, h, t, p) = x.dims4()?;
        let zeros = Tensor::zeros((b, h, t, 1), x.dtype(), x.device())?;
        let x = Tensor::cat(&[zeros, x.clone()], 3)?;
        let x = x.reshape((b, h, p + 1, t))?;
        let x = x.narrow(D::Minus2, 1, p)?;
        let x = x.reshape((b, h, t, p))?;
        let x = x.narrow(D::Minus1, 0, t)?;
        Ok(x)
    }
}

pub struct QFastConformerBlock {
    ff1: QFeedForward,
    ff2: QFeedForward,
    self_attn: QMultiHeadSelfAttention,
    conv_module: ConvModule,
    ln_ff1: LayerNorm,
    ln_mha: LayerNorm,
    ln_conv: LayerNorm,
    ln_ff2: LayerNorm,
    ln_out: LayerNorm,
}

impl QFastConformerBlock {
    pub fn new(
        cfg: &FastConformerConfig,
        vb: candle_transformers::quantized_var_builder::VarBuilder,
    ) -> Result<Self> {
        let d_model = cfg.d_model;
        let ln_cfg = LayerNormConfig {
            eps: 1e-5,
            affine: true,
            remove_mean: true,
        };

        // LayerNorm weights need to be dequantized since LayerNorm doesn't support quantized ops
        let ln_ff1_weight = vb.get(d_model, "norm_feed_forward1.weight")?.dequantize(vb.device())?;
        let ln_ff1_bias = vb.get(d_model, "norm_feed_forward1.bias")?.dequantize(vb.device())?;
        let ln_mha_weight = vb.get(d_model, "norm_self_att.weight")?.dequantize(vb.device())?;
        let ln_mha_bias = vb.get(d_model, "norm_self_att.bias")?.dequantize(vb.device())?;
        let ln_conv_weight = vb.get(d_model, "norm_conv.weight")?.dequantize(vb.device())?;
        let ln_conv_bias = vb.get(d_model, "norm_conv.bias")?.dequantize(vb.device())?;
        let ln_ff2_weight = vb.get(d_model, "norm_feed_forward2.weight")?.dequantize(vb.device())?;
        let ln_ff2_bias = vb.get(d_model, "norm_feed_forward2.bias")?.dequantize(vb.device())?;
        let ln_out_weight = vb.get(d_model, "norm_out.weight")?.dequantize(vb.device())?;
        let ln_out_bias = vb.get(d_model, "norm_out.bias")?.dequantize(vb.device())?;

        // Conv module weights need to be dequantized and loaded into a regular VarBuilder
        let kernel_size = cfg.conv_kernel_size;
        let device = vb.device();
        let mut conv_weights = HashMap::new();

        // Helper to load and dequantize a weight tensor
        macro_rules! load_weight {
            ($name:expr, $shape:expr) => {{
                let full_name = format!("conv.{}", $name);
                let qtensor = vb.get($shape, &full_name)?;
                let tensor = qtensor.dequantize(&device)?;
                conv_weights.insert($name.to_string(), tensor);
            }};
        }

        load_weight!("pointwise_conv1.weight", (2 * d_model, d_model, 1));
        load_weight!("pointwise_conv1.bias", 2 * d_model);
        load_weight!("depthwise_conv.weight", (d_model, 1, kernel_size));
        load_weight!("depthwise_conv.bias", d_model);
        load_weight!("norm.weight", d_model);
        load_weight!("norm.bias", d_model);
        load_weight!("norm.running_mean", d_model);
        load_weight!("norm.running_var", d_model);
        // num_batches_tracked is optional and might not be in GGUF
        if let Ok(qtensor) = vb.get((), "conv.norm.num_batches_tracked") {
            if let Ok(tensor) = qtensor.dequantize(&device) {
                conv_weights.insert("norm.num_batches_tracked".to_string(), tensor);
            }
        }
        load_weight!("pointwise_conv2.weight", (d_model, d_model, 1));
        load_weight!("pointwise_conv2.bias", d_model);

        let conv_vb = VarBuilder::from_tensors(conv_weights, DType::F32, vb.device());

        Ok(Self {
            ff1: QFeedForward::new(d_model, cfg.ff_mult, cfg.dropout, vb.pp("feed_forward1"))?,
            ff2: QFeedForward::new(d_model, cfg.ff_mult, cfg.dropout, vb.pp("feed_forward2"))?,
            self_attn: QMultiHeadSelfAttention::new(d_model, cfg.num_heads, cfg.dropout, vb.pp("self_attn"))?,
            conv_module: ConvModule::new(d_model, cfg.conv_kernel_size, cfg.dropout, conv_vb)?,
            ln_ff1: LayerNorm::new(ln_ff1_weight, ln_ff1_bias, ln_cfg.eps),
            ln_mha: LayerNorm::new(ln_mha_weight, ln_mha_bias, ln_cfg.eps),
            ln_conv: LayerNorm::new(ln_conv_weight, ln_conv_bias, ln_cfg.eps),
            ln_ff2: LayerNorm::new(ln_ff2_weight, ln_ff2_bias, ln_cfg.eps),
            ln_out: LayerNorm::new(ln_out_weight, ln_out_bias, ln_cfg.eps),
        })
    }

    pub fn forward_with_pos(
        &self,
        xs: &Tensor,
        pos: &Tensor,
        attn_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        let ln_ff1_out = self.ln_ff1.forward(xs)?;
        let y_ff1 = self.ff1.forward(&ln_ff1_out, train)?;
        let y_ff1_scaled = (y_ff1 * 0.5)?;
        let mut y = (xs + &y_ff1_scaled)?;

        let ln_mha_out = self.ln_mha.forward(&y)?;
        let y_attn = self
            .self_attn
            .forward(&ln_mha_out, pos, attn_mask, train)?;
        y = (&y + &y_attn)?;

        let y_conv = self.conv_module.forward(&self.ln_conv.forward(&y)?, train)?;
        y = (&y + &y_conv)?;

        let y_ff2 = self.ff2.forward(&self.ln_ff2.forward(&y)?, train)?;
        let y_ff2_scaled = (y_ff2 * 0.5)?;
        y = (&y + &y_ff2_scaled)?;

        let y_out = self.ln_out.forward(&y)?;
        Ok(y_out)
    }
}

pub struct QFastConformerEncoder {
    pub subsampling: ConvSubsampling,
    blocks: Vec<QFastConformerBlock>,
    pos_dropout: Dropout,
    pos_dropout_positions: Dropout,
    cfg: FastConformerConfig,
}

impl QFastConformerEncoder {
    pub fn new(
        cfg: FastConformerConfig,
        vb: candle_transformers::quantized_var_builder::VarBuilder,
    ) -> Result<Self> {
        // Subsampling uses Conv2D, need to dequantize weights
        let feat_in = cfg.feat_in;
        let sub_channels = cfg.subsampling_channels;
        let sub_factor = cfg.subsampling_factor;
        let flatten_dim = sub_channels * (feat_in / sub_factor);
        let device = vb.device();
        let mut subsampling_weights = HashMap::new();

        // Helper to load and dequantize subsampling weights
        macro_rules! load_sub_weight {
            ($name:expr, $shape:expr) => {{
                let path = format!("subsampling.{}", $name);
                let qtensor = vb.get($shape, &path)?;
                let tensor = qtensor.dequantize(&device)?;
                subsampling_weights.insert($name.to_string(), tensor);
            }};
        }

        load_sub_weight!("layers.0.weight", (sub_channels, 1, 3, 3));
        load_sub_weight!("layers.0.bias", sub_channels);
        load_sub_weight!("layers.2.weight", (sub_channels, 1, 3, 3));
        load_sub_weight!("layers.2.bias", sub_channels);
        load_sub_weight!("layers.3.weight", (sub_channels, sub_channels, 1, 1));
        load_sub_weight!("layers.3.bias", sub_channels);
        load_sub_weight!("layers.5.weight", (sub_channels, 1, 3, 3));
        load_sub_weight!("layers.5.bias", sub_channels);
        load_sub_weight!("layers.6.weight", (sub_channels, sub_channels, 1, 1));
        load_sub_weight!("layers.6.bias", sub_channels);
        load_sub_weight!("linear.weight", (cfg.d_model, flatten_dim));
        load_sub_weight!("linear.bias", cfg.d_model);

        let subsampling_vb = VarBuilder::from_tensors(
            subsampling_weights,
            DType::F32,
            vb.device(),
        );
        let subsampling = ConvSubsampling::new(&cfg, subsampling_vb)?;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(QFastConformerBlock::new(
                &cfg,
                vb.pp(&format!("layers.{i}")),
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
}

// ============================================================================
// End of quantized versions
// ============================================================================

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

// ============================================================================
// Quantized Parakeet model using QMatMul for faster inference
// ============================================================================

pub struct QParakeetFastConformerCtc {
    pub encoder: QFastConformerEncoder,
    pub proj: Linear,  // Keep CTC head as regular Linear (small, so dequantizing is fine)
    pub cfg: FastConformerConfig,
    tokenizer: Option<Tokenizer>,
    id2token: Option<Vec<String>>,
}

impl QParakeetFastConformerCtc {
    pub fn new_with_tokenizer(
        cfg: FastConformerConfig,
        vb: candle_transformers::quantized_var_builder::VarBuilder,
        tokenizer: Tokenizer,
    ) -> Result<Self> {
        let encoder = QFastConformerEncoder::new(cfg.clone(), vb.pp("encoder"))?;

        // CTC head is small, so we dequantize it for compatibility
        // The bulk of the speed improvement comes from the quantized encoder anyway
        let device = vb.device();
        let weight_q = vb.get((cfg.vocab_size, cfg.d_model), "ctc_head.weight")
            .or_else(|_| {
                // Try Conv1d format [V, D, 1] and squeeze
                vb.get((cfg.vocab_size, cfg.d_model, 1), "ctc_head.weight")
            })?;
        let weight = weight_q.dequantize(&device)?;
        // Squeeze if it's 3D
        let weight = if weight.dims().len() == 3 {
            weight.squeeze(2)?
        } else {
            weight
        };
        let bias_q = vb.get(cfg.vocab_size, "ctc_head.bias")?;
        let bias = bias_q.dequantize(&device)?;
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
        let h = self.encoder.forward(xs, train)?;
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

// ============================================================================
// End of quantized model
// ============================================================================

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
#[cfg(feature = "quantized")]
pub fn load_parakeet_ctc_from_gguf_hf(
    repo_id: &str,
    gguf_filename: &str,
    device: &Device,
) -> Result<QParakeetFastConformerCtc> {
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

/// Load Parakeet CTC model from safetensors (FP32 full-precision inference)
#[cfg(not(feature = "quantized"))]
pub fn load_parakeet_ctc_from_gguf_hf(
    repo_id: &str,
    _gguf_filename: &str,  // Ignored in FP32 mode
    device: &Device,
) -> Result<ParakeetFastConformerCtc> {
    println!("Loading model from Hugging Face Hub (FP32 full-precision from safetensors)");
    println!("  Repository: {}", repo_id);

    // Use the existing safetensors loader for true FP32 weights
    load_parakeet_ctc_from_hf(repo_id, device)
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
/// let model = load_parakeet_ctc_from_gguf_local("assets", &device)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
#[cfg(feature = "quantized")]
pub fn load_parakeet_ctc_from_gguf_local<P: AsRef<Path>>(
    dir: P,
    device: &Device,
) -> Result<QParakeetFastConformerCtc> {
    use std::io::{Error, ErrorKind};

    let assets = dir.as_ref().to_path_buf();

    // Load config from embedded asset
    let cfg_bytes = PARAKEET_CONFIG.bytes(&assets).map_err(|_| {
        Error::new(
            ErrorKind::Other,
            "failed to get decompressed bytes for PARAKEET_CONFIG",
        )
    })?;
    let hf_cfg: HfParakeetCtcConfig = serde_json::from_slice(cfg_bytes).map_err(|e| {
        Error::new(
            ErrorKind::Other,
            format!("failed to parse PARAKEET_CONFIG as JSON: {e}"),
        )
    })?;
    let cfg = FastConformerConfig::from_hf(&hf_cfg);

    // Load tokenizer from embedded asset
    let tok_bytes = PARAKEET_TOKENIZER.bytes(&assets).map_err(|_| {
        Error::new(
            ErrorKind::Other,
            "failed to get decompressed bytes for PARAKEET_TOKENIZER",
        )
    })?;
    let tokenizer = Tokenizer::from_bytes(tok_bytes)
        .map_err(|e| Error::new(ErrorKind::Other, format!("failed to parse PARAKEET_TOKENIZER: {e}")))?;

    // Load GGUF from embedded asset (already decompressed by embed_zst_asset macro)
    println!("Loading Q8_0 quantized model (recommended, compressed)");
    println!("  Loading GGUF file from assets...");
    let gguf_bytes = PARAKEET_MODEL_Q8_0_GGUF.bytes(&assets).map_err(|_| {
        Error::new(
            ErrorKind::Other,
            "failed to load PARAKEET_MODEL_Q8_0_GGUF",
        )
    })?;

    // Use quantized VarBuilder to keep weights quantized for faster inference
    println!("  Creating quantized VarBuilder (keeps weights in Q8_0 format)...");

    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
        gguf_bytes,
        device,
    )?;
    println!("  ✓ Quantized VarBuilder created (weights stay quantized for speed)");

    println!("  Building model...");
    let model = QParakeetFastConformerCtc::new_with_tokenizer(cfg, vb, tokenizer)?;
    println!("✓ Quantized model loaded successfully\n");

    Ok(model)
}

/// Load Parakeet CTC model from safetensors (FP32 full-precision inference)
#[cfg(not(feature = "quantized"))]
pub fn load_parakeet_ctc_from_gguf_local<P: AsRef<Path>>(
    dir: P,
    device: &Device,
) -> Result<ParakeetFastConformerCtc> {
    use std::io::{Error, ErrorKind};
    let assets = dir.as_ref().to_path_buf();

    println!("Loading model with FP32 full-precision inference (from safetensors)");

    // Load config from embedded asset
    let cfg_bytes = PARAKEET_CONFIG.bytes(&assets).map_err(|_| {
        Error::new(ErrorKind::Other, "failed to load PARAKEET_CONFIG")
    })?;
    let hf_cfg: HfParakeetCtcConfig = serde_json::from_slice(cfg_bytes).map_err(|e| {
        Error::new(ErrorKind::Other, format!("failed to parse config: {e}"))
    })?;
    let cfg = FastConformerConfig::from_hf(&hf_cfg);

    // Load tokenizer from embedded asset
    let tok_bytes = PARAKEET_TOKENIZER.bytes(&assets).map_err(|_| {
        Error::new(ErrorKind::Other, "failed to load PARAKEET_TOKENIZER")
    })?;
    let tokenizer = Tokenizer::from_bytes(tok_bytes)
        .map_err(|e| Error::new(ErrorKind::Other, format!("failed to parse tokenizer: {e}")))?;

    // Load safetensors from embedded asset (already decompressed by embed_zst_asset macro)
    println!("  Loading safetensors file from assets...");
    let safetensors_bytes = PARAKEET_MODEL_SAFETENSORS.bytes(&assets).map_err(|_| {
        Error::new(ErrorKind::Other, "failed to load PARAKEET_MODEL_SAFETENSORS")
    })?;

    // Create VarBuilder from safetensors bytes
    println!("  Creating FP32 VarBuilder...");
    let vb = VarBuilder::from_buffered_safetensors(safetensors_bytes.to_vec(), DType::F32, device)?;
    println!("  ✓ FP32 VarBuilder created");

    println!("  Building model...");
    let model = ParakeetFastConformerCtc::new_with_tokenizer(cfg, vb, tokenizer)?;
    println!("✓ Model loaded successfully (FP32 inference)\n");

    Ok(model)
}

/// Common helper to load GGUF model from paths (using quantized inference)
#[cfg(feature = "quantized")]
fn load_gguf_model_common<P: AsRef<Path>>(
    config_path: P,
    gguf_path: P,
    tokenizer_path: P,
    device: &Device,
) -> Result<QParakeetFastConformerCtc> {
    // Load config
    let cfg_json = fs::read_to_string(&config_path)?;
    let hf_cfg: HfParakeetCtcConfig = serde_json::from_str(&cfg_json)?;
    let cfg = FastConformerConfig::from_hf(&hf_cfg);

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("tokenizer load error: {e}"))?;

    // Load GGUF file and keep quantized
    println!("  Loading GGUF file...");
    let mut gguf_file = fs::File::open(&gguf_path)?;

    // Read GGUF content into memory for quantized_var_builder
    gguf_file.seek(std::io::SeekFrom::Start(0))?;
    let mut gguf_bytes = Vec::new();
    std::io::Read::read_to_end(&mut gguf_file, &mut gguf_bytes)?;

    // Use quantized VarBuilder to keep weights quantized for faster inference
    println!("  Creating quantized VarBuilder (keeps weights in Q8_0/Q4K format)...");
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
        &gguf_bytes,
        device,
    )?;
    println!("  ✓ Quantized VarBuilder created (weights stay quantized for speed)");

    println!("  Building model...");
    let model = QParakeetFastConformerCtc::new_with_tokenizer(cfg, vb, tokenizer)?;
    println!("✓ Quantized model loaded successfully\n");

    Ok(model)
}

