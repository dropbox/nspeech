//! Moonshine V2 Streaming Transformer Encoder.
//!
//! Architecture per layer:
//! - Pre-norm (custom LayerNorm with unit offset gamma)
//! - Multi-head self-attention with sliding window (no RoPE, no bias on projections)
//! - Residual connection
//! - Post-norm
//! - FFN (GELU activation, with bias on fc1/fc2)
//! - Residual connection
//!
//! Final norm after all layers.

use anyhow::Result;
use candle_core::{Device, Module, Tensor, D};
use candle_nn::{Linear, VarBuilder};

use super::config::MoonshineConfig;

/// Custom LayerNorm with unit-offset gamma: output = LN(x) * (gamma + 1.0).
struct UnitOffsetLayerNorm {
    gamma: Tensor,
    eps: f64,
}

impl UnitOffsetLayerNorm {
    fn new(dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self { gamma, eps: 1e-5 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(D::Minus1)?;
        let std = (var + self.eps)?.sqrt()?;
        let normed = x_centered.broadcast_div(&std)?;
        let gamma_plus_one = (&self.gamma + 1.0)?;
        Ok(normed.broadcast_mul(&gamma_plus_one)?)
    }
}

/// Encoder MLP: fc1 (with bias) -> GELU -> fc2 (with bias).
struct EncoderMLP {
    fc1: Linear,
    fc2: Linear,
}

impl EncoderMLP {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let fc1_w = vb.pp("fc1").get((intermediate_size, hidden_size), "weight")?;
        let fc1_b = vb.pp("fc1").get(intermediate_size, "bias")?;
        let fc1 = Linear::new(fc1_w, Some(fc1_b));

        let fc2_w = vb.pp("fc2").get((hidden_size, intermediate_size), "weight")?;
        let fc2_b = vb.pp("fc2").get(hidden_size, "bias")?;
        let fc2 = Linear::new(fc2_w, Some(fc2_b));

        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu_erf()?;
        Ok(self.fc2.forward(&x)?)
    }
}

/// Create a bidirectional sliding window attention mask.
///
/// Returns a mask of shape `[1, 1, seq_len, seq_len]` with 0.0 for attend and -inf for mask.
/// Position i can attend to positions in range `[i - left, i + right]`.
fn sliding_window_mask(
    seq_len: usize,
    left: usize,
    right: usize,
    device: &Device,
) -> Result<Tensor> {
    let neg_inf = f32::NEG_INFINITY;
    let mut mask_data = vec![neg_inf; seq_len * seq_len];
    for i in 0..seq_len {
        let start = if i >= left { i - left } else { 0 };
        let end = (i + right + 1).min(seq_len);
        for j in start..end {
            mask_data[i * seq_len + j] = 0.0;
        }
    }
    Ok(Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)?)
}

/// Encoder self-attention with sliding window mask.
struct EncoderAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl EncoderAttention {
    fn new(cfg: &MoonshineConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let h = cfg.encoder_dim;
        let kv_dim = cfg.encoder_num_kv_heads * cfg.encoder_head_dim;

        let q_proj = Linear::new(vb.pp("q_proj").get((kv_dim, h), "weight")?, None);
        let k_proj = Linear::new(vb.pp("k_proj").get((kv_dim, h), "weight")?, None);
        let v_proj = Linear::new(vb.pp("v_proj").get((kv_dim, h), "weight")?, None);
        let o_proj = Linear::new(vb.pp("o_proj").get((h, kv_dim), "weight")?, None);

        let scale = (cfg.encoder_head_dim as f64).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.encoder_num_heads,
            num_kv_heads: cfg.encoder_num_kv_heads,
            head_dim: cfg.encoder_head_dim,
            scale,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [batch, heads, seq, head_dim]
        let q = q.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, t, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, t, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        // GQA repeat if needed
        let (k, v) = if self.num_kv_heads != self.num_heads {
            let repeats = self.num_heads / self.num_kv_heads;
            let k = k.unsqueeze(2)?
                .expand((b, self.num_kv_heads, repeats, t, self.head_dim))?
                .reshape((b, self.num_heads, t, self.head_dim))?;
            let v = v.unsqueeze(2)?
                .expand((b, self.num_kv_heads, repeats, t, self.head_dim))?
                .reshape((b, self.num_heads, t, self.head_dim))?;
            (k, v)
        } else {
            (k, v)
        };

        // Attention: softmax((Q K^T / sqrt(d)) + mask) V
        let attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?;
        let attn_weights = if let Some(mask) = mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, seq, heads * head_dim]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((b, t, ()))?;
        Ok(self.o_proj.forward(&attn_output)?)
    }
}

/// Single encoder transformer layer.
struct EncoderLayer {
    self_attn: EncoderAttention,
    mlp: EncoderMLP,
    input_layernorm: UnitOffsetLayerNorm,
    post_attention_layernorm: UnitOffsetLayerNorm,
}

impl EncoderLayer {
    fn new(cfg: &MoonshineConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            self_attn: EncoderAttention::new(cfg, vb.pp("self_attn"))?,
            mlp: EncoderMLP::new(cfg.encoder_dim, cfg.encoder_intermediate_size, vb.pp("mlp"))?,
            input_layernorm: UnitOffsetLayerNorm::new(cfg.encoder_dim, vb.pp("input_layernorm"))?,
            post_attention_layernorm: UnitOffsetLayerNorm::new(cfg.encoder_dim, vb.pp("post_attention_layernorm"))?,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Pre-norm self-attention + residual
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, mask)?;
        let x = (residual + h)?;

        // Pre-norm FFN + residual
        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        Ok((residual + h)?)
    }
}

/// Full Moonshine encoder: transformer layers + final norm.
pub struct MoonshineEncoder {
    layers: Vec<EncoderLayer>,
    final_norm: UnitOffsetLayerNorm,
    sliding_windows: Vec<[usize; 2]>,
}

impl MoonshineEncoder {
    pub fn new(cfg: &MoonshineConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.encoder_num_layers);
        for i in 0..cfg.encoder_num_layers {
            layers.push(EncoderLayer::new(cfg, vb.pp(&format!("layers.{i}")))?);
        }
        let final_norm = UnitOffsetLayerNorm::new(cfg.encoder_dim, vb.pp("final_norm"))?;
        Ok(Self {
            layers,
            final_norm,
            sliding_windows: cfg.sliding_windows.clone(),
        })
    }

    /// Run encoder on embedder output.
    ///
    /// Input: `[batch, seq_len, encoder_dim]` from frontend.
    /// Output: `[batch, seq_len, encoder_dim]` encoded features.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let device = x.device();

        // Pre-compute per-layer sliding window masks
        let masks: Vec<Option<Tensor>> = self.sliding_windows.iter()
            .map(|[left, right]| {
                sliding_window_mask(seq_len, *left, *right, device).ok()
            })
            .collect();

        let mut hidden = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let mask = masks.get(i).and_then(|m| m.as_ref());
            hidden = layer.forward(&hidden, mask)?;
        }
        self.final_norm.forward(&hidden)
    }
}
