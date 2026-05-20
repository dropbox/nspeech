//! Shared GPU decoder logic for Metal and D3D12 backends.
//!
//! The `DecoderBackend` trait abstracts per-platform dispatch (Metal enc_* vs D3D12 record_dispatch).
//! `GpuDecoder<B>` implements the full Moonshine decoder layer loop, weight loading, KV cache
//! management, and greedy decode — once, shared across both backends.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::config::MoonshineConfig;

type QVarBuilder = candle_transformers::quantized_var_builder::VarBuilder;

// ── Trait ──

/// Backend-specific GPU operations. Each method corresponds to one logical operation
/// in the decoder forward pass. Backends decide kernel fusion, barriers, split-K, etc.
///
/// All methods take `&self` — backends use interior mutability for command encoder/batch state.
pub trait DecoderBackend: Sized {
    type Buf;

    // ── Buffer management ──
    fn alloc_f16(&self, count: usize) -> Result<Self::Buf>;
    fn alloc_f32(&self, count: usize) -> Result<Self::Buf>;

    /// Convert f32 data to f16 and upload to GPU.
    fn upload_f16_weight(&self, data_f32: &[f32]) -> Result<Self::Buf>;
    /// Upload f32 data directly to GPU (for bias, rope table, etc).
    fn upload_f32_data(&self, data_f32: &[f32]) -> Result<Self::Buf>;
    /// Upload cross-attention K/V projection weight. Metal: f16, D3D12: f32.
    fn upload_cross_kv_weight(&self, data_f32: &[f32]) -> Result<Self::Buf>;

    // ── Pass management ──
    /// Begin a GPU command pass (Metal: create encoder, D3D12: begin_batch).
    fn begin_pass(&self) -> Result<()>;
    /// Upload f32 embedding data to a GPU buffer (Metal: memcpy, D3D12: staging + record_copy).
    /// Called after begin_pass.
    fn upload_embed(&self, dst: &Self::Buf, data: &[f32]) -> Result<()>;
    /// End the GPU command pass and wait for completion.
    fn end_pass(&self) -> Result<()>;

    // ── Readback ──
    /// Read logits from GPU and return argmax token ID.
    fn argmax_logits(&self, logits: &Self::Buf, vocab_size: usize) -> Result<u32>;

    // ── Forward operations ──
    /// LayerNorm on f32 input: out_f16 = LN(x_f32) * weight_f16. (Layer 0 only.)
    fn layernorm_f32in(&self, x: &Self::Buf, w: &Self::Buf, out: &Self::Buf, dim: usize);
    /// Fused or separate Q/K/V projection from normalized input.
    fn qkv_proj(&self, norm: &Self::Buf, attn: &AttentionW<Self::Buf>,
                 q: &Self::Buf, k: &Self::Buf, v: &Self::Buf,
                 q_dim: usize, kv_dim: usize, dim: usize);
    /// Apply RoPE to Q,K and write K,V to self-attention cache.
    fn rope_kv_cache(&self, q: &Self::Buf, k: &Self::Buf, v: &Self::Buf,
                      rope: &Self::Buf, cache_k: &Self::Buf, cache_v: &Self::Buf,
                      p: &RopeCacheParams);
    /// Self-attention decode (single query position against cached K/V).
    fn self_attention(&self, q: &Self::Buf, cache_k: &Self::Buf, cache_v: &Self::Buf,
                       out: &Self::Buf, p: &SelfAttentionParams);
    /// Cross-attention decode (single query against full encoder K/V).
    fn cross_attention(&self, q: &Self::Buf, k: &Self::Buf, v: &Self::Buf,
                        out: &Self::Buf, p: &CrossAttentionParams);
    /// GEMV + residual add + layernorm (Metal: 2 dispatches, D3D12: 1 fused dispatch).
    /// `temp` is a scratch buffer for the GEMV output (Metal uses it; D3D12 ignores it).
    fn gemv_resadd_ln(&self, x: &Self::Buf, w: &Self::Buf,
                       res_in: &Self::Buf, res_out: &Self::Buf,
                       ln_w: &Self::Buf, norm_out: &Self::Buf,
                       temp: &Self::Buf, dim: usize, in_dim: usize);
    /// Cross-attention Q projection GEMV.
    fn cross_q_proj(&self, x: &Self::Buf, w: &Self::Buf, out: &Self::Buf,
                     n: usize, k: usize);
    /// MLP fc1 + GLU activation.
    fn mlp_fc1_glu(&self, x: &Self::Buf, w: &Self::Buf, bias: &Self::Buf,
                    out: &Self::Buf, intermediate: usize, dim: usize);
    /// MLP fc2 + bias.
    fn mlp_fc2_bias(&self, x: &Self::Buf, w: &Self::Buf, bias: &Self::Buf,
                     out: &Self::Buf, dim: usize, intermediate: usize);
    /// Residual add + layernorm (for MLP output → next layer input).
    fn residual_add_ln(&self, proj: &Self::Buf, res_in: &Self::Buf, res_out: &Self::Buf,
                        ln_w: &Self::Buf, norm_out: &Self::Buf, dim: usize);
    /// LM head GEMV: logits = norm @ proj_out_weight.
    fn lm_head(&self, x: &Self::Buf, w: &Self::Buf, out: &Self::Buf,
                vocab: usize, dim: usize);
    /// Tiled matmul for cross-attention K/V projection (enc_proj @ weight → out).
    fn matmul_cross_kv(&self, enc: &Self::Buf, w: &Self::Buf, out: &Self::Buf,
                        m: usize, n: usize, k: usize);
}

// ── Parameter structs ──

pub struct RopeCacheParams {
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub half_rot: usize,
    pub pos: usize,
    pub max_kv_len: usize,
}

pub struct SelfAttentionParams {
    pub kv_len: usize,
    pub head_dim: usize,
    pub n_kv_heads: usize,
    pub n_q_heads: usize,
    pub sm_scale: f32,
    pub max_kv_len: usize,
}

pub struct CrossAttentionParams {
    pub kv_len: usize,
    pub head_dim: usize,
    pub n_kv_heads: usize,
    pub n_q_heads: usize,
    pub sm_scale: f32,
}

// ── Weight structs (generic over buffer type) ──

pub struct LinearW<B> {
    pub weight: B,
}

pub struct LinearBiasW<B> {
    pub weight: B,
    pub bias: B,
}

pub struct AttentionW<B> {
    pub q_proj: LinearW<B>,
    pub k_proj: LinearW<B>,
    pub v_proj: LinearW<B>,
    pub o_proj: LinearW<B>,
}

pub struct MlpW<B> {
    pub fc1: LinearBiasW<B>,
    pub fc2: LinearBiasW<B>,
}

pub struct LayerW<B> {
    pub self_attn: AttentionW<B>,
    pub cross_attn: AttentionW<B>,
    pub mlp: MlpW<B>,
    pub input_ln: B,
    pub post_attn_ln: B,
    pub final_ln: B,
}

// ── Cache ──

pub struct DecoderCache<B> {
    pub self_k: Vec<B>,
    pub self_v: Vec<B>,
    pub self_len: usize,
    pub cross_k: Vec<B>,
    pub cross_v: Vec<B>,
    pub cross_len: usize,
    pub cross_initialized: bool,
    pub encoder_proj: Option<B>,
}

// ── Scratch buffers ──

pub struct Scratch<B> {
    pub f16_norm: B,
    pub f16_q: B,
    pub f16_k: B,
    pub f16_v: B,
    pub f16_attn: B,
    pub f16_act: B,
    pub f32_a: B,
    pub f32_b: B,
    pub f16_logits: B,
}

// ── CPU-side weight dequantization helpers ──

/// Dequantize 2D weight: GGUF → f32, transpose, flatten.
fn dequant_2d(shape: (usize, usize), vb: &QVarBuilder) -> Result<Vec<f32>> {
    let qt = vb.get(shape, "weight")?;
    let t = qt.dequantize(&Device::Cpu)?;
    let t = t.t()?.contiguous()?.flatten_all()?;
    Ok(t.to_vec1::<f32>()?)
}

/// Dequantize 1D weight/bias: GGUF → f32.
fn dequant_1d(dim: usize, name: &str, vb: &QVarBuilder) -> Result<Vec<f32>> {
    let qt = vb.get(dim, name)?;
    let t = qt.dequantize(&Device::Cpu)?;
    Ok(t.to_vec1::<f32>()?)
}

// ── Decoder data (immutable after construction, borrowed separately from backend) ──

struct DecoderData<B: DecoderBackend> {
    layers: Vec<LayerW<B::Buf>>,
    scratch: Scratch<B::Buf>,
    proj_out_weight: B::Buf,
    final_norm_weight: B::Buf,
    rope_table: B::Buf,
    embed_tokens_data: Vec<f32>,
    pos_emb_data: Vec<f32>,
    proj_weight: Option<Vec<f32>>,
    decoder_dim: usize,
    encoder_dim: usize,
    num_layers: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    half_rot: usize,
    vocab_size: usize,
    intermediate_size: usize,
    bos_id: u32,
    eos_id: u32,
    max_kv_len: usize,
    sm_scale: f32,
}

// ── Main decoder struct ──

pub struct GpuDecoder<B: DecoderBackend> {
    pub(crate) backend: B,
    d: DecoderData<B>,
}

impl<B: DecoderBackend> GpuDecoder<B> {
    pub fn new(
        backend: B,
        cfg: &MoonshineConfig,
        dec_vb: QVarBuilder,
        proj_out_vb: QVarBuilder,
    ) -> Result<Self> {
        let decoder_dim = cfg.decoder_dim;
        let encoder_dim = cfg.encoder_dim;
        let n_q_heads = cfg.decoder_num_heads;
        let n_kv_heads = cfg.decoder_num_kv_heads;
        let head_dim = cfg.decoder_head_dim;
        let rotary_dim = cfg.rotary_dim();
        let half_rot = rotary_dim / 2;
        let intermediate_size = cfg.decoder_intermediate_size;
        let q_dim = n_q_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        // Token embedding (CPU for index_select)
        let embed_qt = dec_vb.pp("embed_tokens").get((cfg.vocab_size, decoder_dim), "weight")?;
        let embed_t = embed_qt.dequantize(&Device::Cpu)?;
        let embed_tokens_data: Vec<f32> = embed_t.to_vec2::<f32>()?
            .into_iter().flatten().collect();

        // Position embedding (CPU)
        let pos_qt = dec_vb.pp("pos_emb").get((cfg.max_position_embeddings, encoder_dim), "weight")?;
        let pos_t = pos_qt.dequantize(&Device::Cpu)?;
        let pos_emb_data: Vec<f32> = pos_t.to_vec2::<f32>()?
            .into_iter().flatten().collect();

        // Encoder projection (optional, CPU matmul once per decode)
        let proj_weight = if encoder_dim != decoder_dim {
            let qt = dec_vb.pp("proj").get((decoder_dim, encoder_dim), "weight")?;
            let t = qt.dequantize(&Device::Cpu)?;
            Some(t.to_vec2::<f32>()?.into_iter().flatten().collect())
        } else {
            None
        };

        // LM head
        let proj_out_weight = backend.upload_f16_weight(&dequant_2d((cfg.vocab_size, decoder_dim), &proj_out_vb)?)?;

        // Decoder layers
        let mut layers = Vec::with_capacity(cfg.decoder_num_layers);
        for i in 0..cfg.decoder_num_layers {
            let lvb = dec_vb.pp(&format!("layers.{i}"));

            let avb = lvb.pp("self_attn");
            let self_attn = AttentionW {
                q_proj: LinearW { weight: backend.upload_f16_weight(&dequant_2d((q_dim, decoder_dim), &avb.pp("q_proj"))?)? },
                k_proj: LinearW { weight: backend.upload_f16_weight(&dequant_2d((kv_dim, decoder_dim), &avb.pp("k_proj"))?)? },
                v_proj: LinearW { weight: backend.upload_f16_weight(&dequant_2d((kv_dim, decoder_dim), &avb.pp("v_proj"))?)? },
                o_proj: LinearW { weight: backend.upload_f16_weight(&dequant_2d((decoder_dim, q_dim), &avb.pp("o_proj"))?)? },
            };

            let cavb = lvb.pp("encoder_attn");
            let cross_attn = AttentionW {
                q_proj: LinearW { weight: backend.upload_f16_weight(&dequant_2d((q_dim, decoder_dim), &cavb.pp("q_proj"))?)? },
                k_proj: LinearW { weight: backend.upload_cross_kv_weight(&dequant_2d((kv_dim, decoder_dim), &cavb.pp("k_proj"))?)? },
                v_proj: LinearW { weight: backend.upload_cross_kv_weight(&dequant_2d((kv_dim, decoder_dim), &cavb.pp("v_proj"))?)? },
                o_proj: LinearW { weight: backend.upload_f16_weight(&dequant_2d((decoder_dim, q_dim), &cavb.pp("o_proj"))?)? },
            };

            let mvb = lvb.pp("mlp");
            let mlp = MlpW {
                fc1: LinearBiasW {
                    weight: backend.upload_f16_weight(&dequant_2d((intermediate_size * 2, decoder_dim), &mvb.pp("fc1"))?)?,
                    bias: backend.upload_f32_data(&dequant_1d(intermediate_size * 2, "bias", &mvb.pp("fc1"))?)?,
                },
                fc2: LinearBiasW {
                    weight: backend.upload_f16_weight(&dequant_2d((decoder_dim, intermediate_size), &mvb.pp("fc2"))?)?,
                    bias: backend.upload_f32_data(&dequant_1d(decoder_dim, "bias", &mvb.pp("fc2"))?)?,
                },
            };

            layers.push(LayerW {
                self_attn,
                cross_attn,
                mlp,
                input_ln: backend.upload_f16_weight(&dequant_1d(decoder_dim, "weight", &lvb.pp("input_layernorm"))?)?,
                post_attn_ln: backend.upload_f16_weight(&dequant_1d(decoder_dim, "weight", &lvb.pp("post_attention_layernorm"))?)?,
                final_ln: backend.upload_f16_weight(&dequant_1d(decoder_dim, "weight", &lvb.pp("final_layernorm"))?)?,
            });
        }

        let final_norm_weight = backend.upload_f16_weight(&dequant_1d(decoder_dim, "weight", &dec_vb.pp("norm"))?)?;

        // Pre-allocate scratch buffers
        let scratch = Scratch {
            f16_norm: backend.alloc_f16(decoder_dim.max(q_dim))?,
            f16_q: backend.alloc_f16(q_dim)?,
            f16_k: backend.alloc_f16(kv_dim)?,
            f16_v: backend.alloc_f16(kv_dim)?,
            f16_attn: backend.alloc_f16(q_dim)?,
            f16_act: backend.alloc_f16(intermediate_size)?,
            f32_a: backend.alloc_f32(decoder_dim)?,
            f32_b: backend.alloc_f32(decoder_dim)?,
            f16_logits: backend.alloc_f16(cfg.vocab_size)?,
        };

        // Precompute RoPE table
        let max_pos = cfg.max_position_embeddings.min(512);
        let theta = cfg.rope_theta as f32;
        let inv_freq: Vec<f32> = (0..half_rot)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / rotary_dim as f32))
            .collect();
        let mut rope_data = vec![0.0f32; max_pos * half_rot * 2];
        for pos in 0..max_pos {
            for i in 0..half_rot {
                let angle = pos as f32 * inv_freq[i];
                rope_data[pos * half_rot * 2 + i] = angle.cos();
                rope_data[pos * half_rot * 2 + half_rot + i] = angle.sin();
            }
        }
        let rope_table = backend.upload_f32_data(&rope_data)?;

        Ok(Self {
            backend,
            d: DecoderData {
                layers,
                scratch,
                proj_out_weight,
                final_norm_weight,
                rope_table,
                embed_tokens_data,
                pos_emb_data,
                proj_weight,
                decoder_dim,
                encoder_dim,
                num_layers: cfg.decoder_num_layers,
                n_q_heads,
                n_kv_heads,
                head_dim,
                half_rot,
                vocab_size: cfg.vocab_size,
                intermediate_size,
                bos_id: cfg.bos_id as u32,
                eos_id: cfg.eos_id as u32,
                max_kv_len: max_pos,
                sm_scale: (head_dim as f32).powf(-0.5),
            },
        })
    }

    fn new_cache(&self) -> Result<DecoderCache<B::Buf>> {
        let kv_cache_size = self.d.n_kv_heads * self.d.max_kv_len * self.d.head_dim;
        let mut self_k = Vec::with_capacity(self.d.num_layers);
        let mut self_v = Vec::with_capacity(self.d.num_layers);
        for _ in 0..self.d.num_layers {
            self_k.push(self.backend.alloc_f16(kv_cache_size)?);
            self_v.push(self.backend.alloc_f16(kv_cache_size)?);
        }
        Ok(DecoderCache {
            self_k,
            self_v,
            self_len: 0,
            cross_k: Vec::new(),
            cross_v: Vec::new(),
            cross_len: 0,
            cross_initialized: false,
            encoder_proj: None,
        })
    }

    /// Compute encoder projection on CPU, upload as f16 to GPU. Called once per decode run.
    fn prepare_encoder_proj(
        &self, encoder_hidden: &Tensor, cache: &mut DecoderCache<B::Buf>,
    ) -> Result<()> {
        if cache.encoder_proj.is_some() { return Ok(()); }

        let enc_seq = encoder_hidden.dim(1)?;
        let enc_hidden = encoder_hidden.squeeze(0)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let enc_data = enc_hidden.to_vec2::<f32>()?;

        let max_pos_emb = self.d.pos_emb_data.len() / self.d.encoder_dim;

        let mut proj_data = vec![0.0f32; enc_seq * self.d.encoder_dim];
        for s in 0..enc_seq {
            for d in 0..self.d.encoder_dim {
                let pos_val = if s < max_pos_emb { self.d.pos_emb_data[s * self.d.encoder_dim + d] } else { 0.0 };
                proj_data[s * self.d.encoder_dim + d] = enc_data[s][d] + pos_val;
            }
        }

        let final_data = if let Some(proj_w) = &self.d.proj_weight {
            let mut out = vec![0.0f32; enc_seq * self.d.decoder_dim];
            for s in 0..enc_seq {
                for dd in 0..self.d.decoder_dim {
                    let mut sum = 0.0f32;
                    for k in 0..self.d.encoder_dim {
                        sum += proj_data[s * self.d.encoder_dim + k] * proj_w[dd * self.d.encoder_dim + k];
                    }
                    out[s * self.d.decoder_dim + dd] = sum;
                }
            }
            out
        } else {
            proj_data
        };

        // Upload as f16 (both backends use f16 encoder projection)
        cache.encoder_proj = Some(self.backend.upload_f16_weight(&final_data)?);
        cache.cross_len = enc_seq;
        Ok(())
    }

    /// Compute cross-attention K/V for all layers using tiled matmul. Called once per decode run.
    fn initialize_cross_attention(&self, cache: &mut DecoderCache<B::Buf>) -> Result<()> {
        if cache.cross_initialized { return Ok(()); }

        let enc_seq = cache.cross_len;
        let kv_dim = self.d.n_kv_heads * self.d.head_dim;

        cache.cross_k.clear();
        cache.cross_v.clear();
        for _ in 0..self.d.num_layers {
            cache.cross_k.push(self.backend.alloc_f16(enc_seq * kv_dim)?);
            cache.cross_v.push(self.backend.alloc_f16(enc_seq * kv_dim)?);
        }

        let enc_proj = cache.encoder_proj.as_ref()
            .ok_or_else(|| anyhow::anyhow!("encoder projection not initialized"))?;

        self.backend.begin_pass()?;
        for (i, layer) in self.d.layers.iter().enumerate() {
            self.backend.matmul_cross_kv(
                enc_proj, &layer.cross_attn.k_proj.weight, &cache.cross_k[i],
                enc_seq, kv_dim, self.d.decoder_dim);
            self.backend.matmul_cross_kv(
                enc_proj, &layer.cross_attn.v_proj.weight, &cache.cross_v[i],
                enc_seq, kv_dim, self.d.decoder_dim);
        }
        self.backend.end_pass()?;

        cache.cross_initialized = true;
        Ok(())
    }

    /// Run one decoder step. All dispatches batched on a single GPU pass.
    fn forward_one_token(&self, token_id: u32, cache: &mut DecoderCache<B::Buf>) -> Result<()> {
        let d = &self.d;
        let s = &d.scratch;
        let dim = d.decoder_dim;
        let q_dim = d.n_q_heads * d.head_dim;
        let kv_dim = d.n_kv_heads * d.head_dim;
        let pos = cache.self_len;

        // 1. Begin pass + embed upload
        let token_offset = (token_id as usize) * dim;
        let embed_slice = &d.embed_tokens_data[token_offset..token_offset + dim];
        self.backend.begin_pass()?;
        self.backend.upload_embed(&s.f32_a, embed_slice)?;

        // Ping-pong f32 residual stream
        let buffers: [&B::Buf; 2] = [&s.f32_a, &s.f32_b];
        let mut write_idx: usize = 1;

        for (layer_idx, layer) in d.layers.iter().enumerate() {
            let read_f32 = buffers[1 - write_idx];
            let write_f32 = buffers[write_idx];

            // ── Pre-norm (layer 0 only; layers 1+ fused into prev MLP residual) ──
            if layer_idx == 0 {
                self.backend.layernorm_f32in(read_f32, &layer.input_ln, &s.f16_norm, dim);
            }

            // ── Self-attention QKV projection ──
            self.backend.qkv_proj(&s.f16_norm, &layer.self_attn,
                &s.f16_q, &s.f16_k, &s.f16_v, q_dim, kv_dim, dim);

            // ── RoPE + KV cache ──
            self.backend.rope_kv_cache(&s.f16_q, &s.f16_k, &s.f16_v,
                &d.rope_table, &cache.self_k[layer_idx], &cache.self_v[layer_idx],
                &RopeCacheParams {
                    n_q_heads: d.n_q_heads, n_kv_heads: d.n_kv_heads,
                    head_dim: d.head_dim, half_rot: d.half_rot,
                    pos, max_kv_len: d.max_kv_len,
                });

            // ── Self-attention decode ──
            self.backend.self_attention(
                &s.f16_q, &cache.self_k[layer_idx], &cache.self_v[layer_idx], &s.f16_attn,
                &SelfAttentionParams {
                    kv_len: pos + 1, head_dim: d.head_dim,
                    n_kv_heads: d.n_kv_heads, n_q_heads: d.n_q_heads,
                    sm_scale: d.sm_scale, max_kv_len: d.max_kv_len,
                });

            // ── O proj + residual + layernorm ──
            self.backend.gemv_resadd_ln(
                &s.f16_attn, &layer.self_attn.o_proj.weight,
                read_f32, write_f32, &layer.post_attn_ln, &s.f16_norm,
                &s.f16_act, dim, q_dim);
            write_idx = 1 - write_idx;

            // ── Cross-attention ──
            let read_f32 = buffers[1 - write_idx];
            let write_f32 = buffers[write_idx];

            self.backend.cross_q_proj(&s.f16_norm, &layer.cross_attn.q_proj.weight,
                &s.f16_q, q_dim, dim);

            self.backend.cross_attention(
                &s.f16_q, &cache.cross_k[layer_idx], &cache.cross_v[layer_idx], &s.f16_attn,
                &CrossAttentionParams {
                    kv_len: cache.cross_len, head_dim: d.head_dim,
                    n_kv_heads: d.n_kv_heads, n_q_heads: d.n_q_heads,
                    sm_scale: d.sm_scale,
                });

            self.backend.gemv_resadd_ln(
                &s.f16_attn, &layer.cross_attn.o_proj.weight,
                read_f32, write_f32, &layer.final_ln, &s.f16_norm,
                &s.f16_act, dim, q_dim);
            write_idx = 1 - write_idx;

            // ── MLP ──
            let read_f32 = buffers[1 - write_idx];
            let write_f32 = buffers[write_idx];

            self.backend.mlp_fc1_glu(
                &s.f16_norm, &layer.mlp.fc1.weight, &layer.mlp.fc1.bias,
                &s.f16_act, d.intermediate_size, dim);

            self.backend.mlp_fc2_bias(
                &s.f16_act, &layer.mlp.fc2.weight, &layer.mlp.fc2.bias,
                &s.f16_norm, dim, d.intermediate_size);

            let next_ln_weight = if layer_idx + 1 < d.num_layers {
                &d.layers[layer_idx + 1].input_ln
            } else {
                &d.final_norm_weight
            };
            self.backend.residual_add_ln(
                &s.f16_norm, read_f32, write_f32,
                next_ln_weight, &s.f16_norm, dim);
            write_idx = 1 - write_idx;
        }

        // ── LM head ──
        self.backend.lm_head(&s.f16_norm, &d.proj_out_weight, &s.f16_logits,
            d.vocab_size, dim);

        self.backend.end_pass()?;
        cache.self_len += 1;
        Ok(())
    }

    /// Full greedy decode.
    pub fn greedy_decode(
        &self, encoder_hidden: &Tensor, max_tokens: usize,
    ) -> Result<Vec<u32>> {
        let mut cache = self.new_cache()?;

        self.prepare_encoder_proj(encoder_hidden, &mut cache)?;
        self.initialize_cross_attention(&mut cache)?;

        let mut generated = Vec::new();

        // First step with BOS
        self.forward_one_token(self.d.bos_id, &mut cache)?;
        let mut next_token = self.backend.argmax_logits(&self.d.scratch.f16_logits, self.d.vocab_size)?;
        generated.push(next_token);
        if next_token == self.d.eos_id { return Ok(generated); }

        for _step in 0..max_tokens - 1 {
            self.forward_one_token(next_token, &mut cache)?;
            next_token = self.backend.argmax_logits(&self.d.scratch.f16_logits, self.d.vocab_size)?;
            generated.push(next_token);
            if next_token == self.d.eos_id { break; }
        }

        Ok(generated)
    }
}
