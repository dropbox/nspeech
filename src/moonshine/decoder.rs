//! Moonshine V2 Streaming Transformer Decoder (quantized inference).
//!
//! Architecture per layer:
//! - Pre-norm (standard LayerNorm, no bias)
//! - Causal self-attention with RoPE (partial rotary, interleaved)
//! - Residual
//! - Post-norm + Cross-attention to encoder hidden states
//! - Residual
//! - Final norm + GLU MLP: fc1 -> chunk(2) -> silu(gate) * x -> fc2
//! - Residual
//!
//! Weights are kept quantized (Q8_0) and dequantized on-the-fly during matmul.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_transformers::models::with_tracing::QMatMul;

use super::config::MoonshineConfig;

type QVarBuilder = candle_transformers::quantized_var_builder::VarBuilder;

/// Standard LayerNorm (weight only, no bias).
///
/// Weight is dequantized on load (small 1D tensor).
struct LayerNorm {
    weight: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn new(dim: usize, vb: QVarBuilder) -> Result<Self> {
        let weight_q = vb.get(dim, "weight")?;
        let weight = weight_q.dequantize(vb.device())?;
        Ok(Self { weight, eps: 1e-5 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(D::Minus1)?;
        let std = (var + self.eps)?.sqrt()?;
        let normed = x_centered.broadcast_div(&std)?;
        Ok(normed.broadcast_mul(&self.weight)?)
    }
}

/// RoPE with partial rotation and interleaved pattern.
///
/// No learned weights — computed from config parameters.
struct RotaryEmbedding {
    inv_freq: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &MoonshineConfig, device: &Device) -> Result<Self> {
        let dim = cfg.rotary_dim();
        let half_dim = dim / 2;
        let theta = cfg.rope_theta;

        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (half_dim,), device)?;

        Ok(Self { inv_freq })
    }

    /// Compute cos and sin for positions.
    /// Returns (cos, sin) each of shape [batch, seq_len, rotary_dim].
    fn forward(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        // position_ids: [batch, seq_len]
        let inv_freq = self.inv_freq.unsqueeze(0)?.unsqueeze(2)?; // [1, half_dim, 1]
        let pos = position_ids.unsqueeze(1)?.to_dtype(DType::F32)?; // [batch, 1, seq_len]

        let freqs = inv_freq.broadcast_mul(&pos)?; // [batch, half_dim, seq_len]
        let freqs = freqs.transpose(1, 2)?; // [batch, seq_len, half_dim]
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [batch, seq_len, rotary_dim]
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        Ok((cos, sin))
    }
}

/// Interleaved rotate_half: pairs (x0,x1) -> (-x1,x0), (x2,x3) -> (-x3,x2), etc.
fn rotate_half_interleaved(x: &Tensor) -> Result<Tensor> {
    let dims: Vec<usize> = x.shape().dims().to_vec();
    let last = *dims.last().unwrap();

    // Reshape to (..., last/2, 2)
    let mut new_dims = dims[..dims.len() - 1].to_vec();
    new_dims.push(last / 2);
    new_dims.push(2);
    let x_pairs = x.reshape(new_dims.as_slice())?;

    let x1 = x_pairs.narrow(D::Minus1, 0, 1)?;
    let x2 = x_pairs.narrow(D::Minus1, 1, 1)?;
    let neg_x2 = x2.neg()?;

    let rotated = Tensor::cat(&[&neg_x2, &x1], D::Minus1)?;
    Ok(rotated.reshape(dims.as_slice())?)
}

fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rotary_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let head_dim = q.dim(D::Minus1)?;
    let half_rot = rotary_dim / 2;

    // Take first half and repeat_interleave(2)
    let cos_half = cos.narrow(D::Minus1, 0, half_rot)?;
    let sin_half = sin.narrow(D::Minus1, 0, half_rot)?;

    // repeat_interleave(2, dim=-1): [a,b] -> [a,a,b,b]
    let b = cos_half.dim(0)?;
    let s = cos_half.dim(1)?;
    let cos_interleaved = cos_half.unsqueeze(D::Minus1)?
        .broadcast_as((b, s, half_rot, 2))?
        .reshape((b, s, rotary_dim))?;
    let sin_interleaved = sin_half.unsqueeze(D::Minus1)?
        .broadcast_as((b, s, half_rot, 2))?
        .reshape((b, s, rotary_dim))?;

    // [batch, 1, seq_len, rotary_dim]
    let cos_4d = cos_interleaved.unsqueeze(1)?;
    let sin_4d = sin_interleaved.unsqueeze(1)?;

    // Split into rotary and pass-through
    let q_rot = q.narrow(D::Minus1, 0, rotary_dim)?;
    let k_rot = k.narrow(D::Minus1, 0, rotary_dim)?;

    let q_embed = (q_rot.broadcast_mul(&cos_4d)? + rotate_half_interleaved(&q_rot)?.broadcast_mul(&sin_4d)?)?;
    let k_embed = (k_rot.broadcast_mul(&cos_4d)? + rotate_half_interleaved(&k_rot)?.broadcast_mul(&sin_4d)?)?;

    // Concatenate with pass-through part
    let q_out = if rotary_dim < head_dim {
        let q_pass = q.narrow(D::Minus1, rotary_dim, head_dim - rotary_dim)?;
        Tensor::cat(&[&q_embed, &q_pass], D::Minus1)?
    } else {
        q_embed
    };
    let k_out = if rotary_dim < head_dim {
        let k_pass = k.narrow(D::Minus1, rotary_dim, head_dim - rotary_dim)?;
        Tensor::cat(&[&k_embed, &k_pass], D::Minus1)?
    } else {
        k_embed
    };

    Ok((q_out, k_out))
}

/// KV cache for a single attention layer.
pub struct KVCache {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn update(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let (k_full, v_full) = match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
            _ => (k, v),
        };
        self.k = Some(k_full.clone());
        self.v = Some(v_full.clone());
        Ok((k_full, v_full))
    }
}

/// Decoder attention (self or cross).
///
/// Projection weights kept quantized via QMatMul (no bias on projections).
struct DecoderAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    is_causal: bool,
}

impl DecoderAttention {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        is_causal: bool,
        vb: QVarBuilder,
    ) -> Result<Self> {
        let kv_dim = num_kv_heads * head_dim;
        let q_dim = num_heads * head_dim;

        let q_proj = QMatMul::new(hidden_size, q_dim, vb.pp("q_proj"))?;
        let k_proj = QMatMul::new(hidden_size, kv_dim, vb.pp("k_proj"))?;
        let v_proj = QMatMul::new(hidden_size, kv_dim, vb.pp("v_proj"))?;
        let o_proj = QMatMul::new(q_dim, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
            is_causal,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        kv_states: Option<&Tensor>,
        cache: &mut KVCache,
        rope: Option<(&Tensor, &Tensor)>,
        rotary_dim: usize,
    ) -> Result<Tensor> {
        let (b, q_len, _) = hidden_states.dims3()?;
        let is_cross = kv_states.is_some();

        let q = self.q_proj.forward(hidden_states)?
            .reshape((b, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;

        // Cross-attention with cached KV: reuse
        if is_cross && cache.k.is_some() {
            let k_full = cache.k.as_ref().unwrap().clone();
            let v_full = cache.v.as_ref().unwrap().clone();
            let out = self.compute_attention(&q, &k_full, &v_full, b, q_len, None)?;
            return Ok(self.o_proj.forward(&out)?);
        }

        let kv_input = kv_states.unwrap_or(hidden_states);
        let kv_len = kv_input.dim(1)?;

        let mut k = self.k_proj.forward(kv_input)?
            .reshape((b, kv_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;
        let v = self.v_proj.forward(kv_input)?
            .reshape((b, kv_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;

        // RoPE for self-attention
        let q = if let Some((cos, sin)) = rope {
            let (q_rot, k_rot) = apply_rotary_pos_emb(&q, &k, cos, sin, rotary_dim)?;
            k = k_rot;
            q_rot
        } else {
            q
        };

        // Update cache
        let (k_full, v_full) = cache.update(k, v)?;

        // Causal mask
        let mask = if self.is_causal && q_len > 1 {
            let total_len = k_full.dim(2)?;
            Some(Self::causal_mask(q_len, total_len, hidden_states.device())?)
        } else {
            None
        };

        let out = self.compute_attention(&q, &k_full, &v_full, b, q_len, mask.as_ref())?;
        Ok(self.o_proj.forward(&out)?)
    }

    fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        b: usize,
        q_len: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (k, v) = if self.num_kv_heads != self.num_heads {
            let repeats = self.num_heads / self.num_kv_heads;
            let kv_len = k.dim(2)?;
            let k = k.unsqueeze(2)?
                .expand((b, self.num_kv_heads, repeats, kv_len, self.head_dim))?
                .reshape((b, self.num_heads, kv_len, self.head_dim))?;
            let v = v.unsqueeze(2)?
                .expand((b, self.num_kv_heads, repeats, kv_len, self.head_dim))?
                .reshape((b, self.num_heads, kv_len, self.head_dim))?;
            (k, v)
        } else {
            (k.clone(), v.clone())
        };

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?;
        let attn_weights = if let Some(mask) = mask {
            (attn_weights + mask)?
        } else {
            attn_weights
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        Ok(attn_output.transpose(1, 2)?.contiguous()?.reshape((b, q_len, ()))?)
    }

    fn causal_mask(q_len: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
        let offset = kv_len - q_len;
        let neg_inf = f32::NEG_INFINITY;
        let mut mask_data = vec![0.0f32; q_len * kv_len];
        for i in 0..q_len {
            for j in 0..kv_len {
                if j > i + offset {
                    mask_data[i * kv_len + j] = neg_inf;
                }
            }
        }
        Ok(Tensor::from_vec(mask_data, (1, 1, q_len, kv_len), device)?)
    }
}

/// Decoder GLU MLP: fc1 -> chunk(2) -> silu(gate) * x -> fc2.
///
/// Weights kept quantized via QMatMul; biases dequantized on load.
struct DecoderMLP {
    fc1: QMatMul,
    fc1_bias: Tensor,
    fc2: QMatMul,
    fc2_bias: Tensor,
}

impl DecoderMLP {
    fn new(hidden_size: usize, intermediate_size: usize, vb: QVarBuilder) -> Result<Self> {
        let device = vb.device();
        let fc1 = QMatMul::new(hidden_size, intermediate_size * 2, vb.pp("fc1"))?;
        let fc1_bias = vb.pp("fc1").get(intermediate_size * 2, "bias")?.dequantize(device)?;

        let fc2 = QMatMul::new(intermediate_size, hidden_size, vb.pp("fc2"))?;
        let fc2_bias = vb.pp("fc2").get(hidden_size, "bias")?.dequantize(device)?;

        Ok(Self {
            fc1,
            fc1_bias,
            fc2,
            fc2_bias,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?.broadcast_add(&self.fc1_bias)?;
        let dim = h.dim(D::Minus1)?;
        let half = dim / 2;
        let x_part = h.narrow(D::Minus1, 0, half)?;
        let gate = h.narrow(D::Minus1, half, half)?;
        let activated = candle_nn::ops::silu(&gate)?.mul(&x_part)?;
        let x = self.fc2.forward(&activated)?.broadcast_add(&self.fc2_bias)?;
        Ok(x)
    }
}

/// Single decoder transformer layer.
struct DecoderLayer {
    self_attn: DecoderAttention,
    encoder_attn: DecoderAttention,
    mlp: DecoderMLP,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    final_layernorm: LayerNorm,
}

impl DecoderLayer {
    fn new(cfg: &MoonshineConfig, _layer_idx: usize, vb: QVarBuilder) -> Result<Self> {
        let h = cfg.decoder_dim;
        Ok(Self {
            self_attn: DecoderAttention::new(h, cfg.decoder_num_heads, cfg.decoder_num_kv_heads, cfg.decoder_head_dim, true, vb.pp("self_attn"))?,
            encoder_attn: DecoderAttention::new(h, cfg.decoder_num_heads, cfg.decoder_num_kv_heads, cfg.decoder_head_dim, false, vb.pp("encoder_attn"))?,
            mlp: DecoderMLP::new(h, cfg.decoder_intermediate_size, vb.pp("mlp"))?,
            input_layernorm: LayerNorm::new(h, vb.pp("input_layernorm"))?,
            post_attention_layernorm: LayerNorm::new(h, vb.pp("post_attention_layernorm"))?,
            final_layernorm: LayerNorm::new(h, vb.pp("final_layernorm"))?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden: &Tensor,
        self_cache: &mut KVCache,
        cross_cache: &mut KVCache,
        rope: (&Tensor, &Tensor),
        rotary_dim: usize,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let h = self.input_layernorm.forward(hidden_states)?;
        let h = self.self_attn.forward(&h, None, self_cache, Some(rope), rotary_dim)?;
        let x = (residual + h)?;

        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = self.encoder_attn.forward(&h, Some(encoder_hidden), cross_cache, None, 0)?;
        let x = (residual + h)?;

        let residual = x.clone();
        let h = self.final_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        Ok((residual + h)?)
    }
}

/// Full Moonshine decoder.
pub struct MoonshineDecoder {
    embed_tokens: candle_nn::Embedding,
    pos_emb: candle_nn::Embedding,
    proj: Option<QMatMul>,
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
    rotary_emb: RotaryEmbedding,
    rotary_dim: usize,
    num_layers: usize,
}

/// Decoder KV caches for all layers.
pub struct DecoderCache {
    pub self_caches: Vec<KVCache>,
    pub cross_caches: Vec<KVCache>,
    pub seq_len: usize,
}

impl DecoderCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            self_caches: (0..num_layers).map(|_| KVCache::new()).collect(),
            cross_caches: (0..num_layers).map(|_| KVCache::new()).collect(),
            seq_len: 0,
        }
    }
}

impl MoonshineDecoder {
    pub fn new(cfg: &MoonshineConfig, device: &Device, vb: QVarBuilder) -> Result<Self> {
        // Dequantize embeddings on load (index_select requires dense tensors)
        let embed_w = vb.pp("embed_tokens").get((cfg.vocab_size, cfg.decoder_dim), "weight")?
            .dequantize(device)?;
        let embed_tokens = candle_nn::Embedding::new(embed_w, cfg.decoder_dim);

        let pos_w = vb.pp("pos_emb").get((cfg.max_position_embeddings, cfg.encoder_dim), "weight")?
            .dequantize(device)?;
        let pos_emb = candle_nn::Embedding::new(pos_w, cfg.encoder_dim);

        let proj = if cfg.encoder_dim != cfg.decoder_dim {
            Some(QMatMul::new(cfg.encoder_dim, cfg.decoder_dim, vb.pp("proj"))?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.decoder_num_layers);
        for i in 0..cfg.decoder_num_layers {
            layers.push(DecoderLayer::new(cfg, i, vb.pp(&format!("layers.{i}")))?);
        }

        let norm = LayerNorm::new(cfg.decoder_dim, vb.pp("norm"))?;
        let rotary_emb = RotaryEmbedding::new(cfg, device)?;

        Ok(Self {
            embed_tokens,
            pos_emb,
            proj,
            layers,
            norm,
            rotary_emb,
            rotary_dim: cfg.rotary_dim(),
            num_layers: cfg.decoder_num_layers,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        encoder_hidden: &Tensor,
        cache: &mut DecoderCache,
    ) -> Result<Tensor> {
        let enc_len = encoder_hidden.dim(1)?;

        // Positional embedding for encoder states
        let pos_ids = Tensor::arange(0u32, enc_len as u32, encoder_hidden.device())?;
        let pos_emb = self.pos_emb.forward(&pos_ids)?;
        let encoder_with_pos = encoder_hidden.broadcast_add(&pos_emb)?;
        let encoder_proj = if let Some(proj) = &self.proj {
            proj.forward(&encoder_with_pos)?
        } else {
            encoder_with_pos
        };

        // Token embeddings
        let hidden = self.embed_tokens.forward(input_ids)?;
        let seq_len = hidden.dim(1)?;

        // RoPE position ids
        let past_len = cache.seq_len;
        let position_ids = Tensor::arange(
            past_len as u32,
            (past_len + seq_len) as u32,
            hidden.device(),
        )?.unsqueeze(0)?;
        let (cos, sin) = self.rotary_emb.forward(&position_ids)?;

        let mut x = hidden;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(
                &x,
                &encoder_proj,
                &mut cache.self_caches[i],
                &mut cache.cross_caches[i],
                (&cos, &sin),
                self.rotary_dim,
            )?;
        }

        cache.seq_len += seq_len;
        self.norm.forward(&x)
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}
