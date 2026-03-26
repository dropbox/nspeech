//! Moonshine V2 Decoder using Triton-compiled Metal kernels.
//!
//! All decoder operations run on the Metal GPU. Weights stored as F16 on GPU.
//! Mixed precision: F32 residual stream, F16 within-layer computation.
//! Only GPU→CPU transfers: final logits download for argmax (once per token).
//!
//! Architecture per layer:
//!  - Pre-norm (standard LayerNorm) → self-attention with RoPE → residual
//!  - Post-norm → cross-attention → residual
//!  - Final-norm → GLU MLP (fc1 → chunk → silu·x → fc2) → residual

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};

use super::config::MoonshineConfig;
use crate::triton_kernels::{
    DecoderKernels, empty_f16, empty_f32, triton_matmul,
    enc_gemv_f16w, enc_gemv_bias_f16w,
    enc_layernorm_std_f32in, enc_attention_decode, enc_attention_splitkv,
    enc_rope_qk_cache_fused, enc_kv_cache_append,
    enc_gemv_bias_glu, enc_gemv_resadd_ln,
    enc_residual_add_layernorm,
    enc_gemv_splitk, enc_gemv_splitk_bias, enc_gemv_glu_splitk,
};

type QVarBuilder = candle_transformers::quantized_var_builder::VarBuilder;

/// Load 2D weight from GGUF → dequantize → F16 → Metal tensor [K, N] (transposed).
fn load_f16_weight(shape: (usize, usize), vb: &QVarBuilder, metal: &Device) -> Result<Tensor> {
    let qt = vb.get(shape, "weight")?;
    let t = qt.dequantize(&Device::Cpu)?;
    Ok(t.to_dtype(DType::F16)?.to_device(metal)?.t()?.contiguous()?)
}

/// Load 1D from GGUF → dequantize → F16 → Metal tensor.
fn load_f16_1d(dim: usize, name: &str, vb: &QVarBuilder, metal: &Device) -> Result<Tensor> {
    let qt = vb.get(dim, name)?;
    let t = qt.dequantize(&Device::Cpu)?;
    Ok(t.to_dtype(DType::F16)?.to_device(metal)?)
}

/// Load 1D from GGUF → dequantize → F32 → Metal tensor.
fn load_f32_1d(dim: usize, name: &str, vb: &QVarBuilder, metal: &Device) -> Result<Tensor> {
    let qt = vb.get(dim, name)?;
    let t = qt.dequantize(&Device::Cpu)?;
    Ok(t.to_device(metal)?)
}

// ── Weight structures ──

struct LinearWeights {
    weight: Tensor, // f16 [K, N] transposed (for GEMV)
}

struct LinearBiasWeights {
    weight: Tensor, // f16 [K, N] transposed
    bias: Tensor,   // f32 [out_dim]
}

struct AttentionWeights {
    q_proj: LinearWeights,
    k_proj: LinearWeights,
    v_proj: LinearWeights,
    o_proj: LinearWeights,
}

struct MLPWeights {
    fc1: LinearBiasWeights,  // [decoder_dim, intermediate_size * 2]
    fc2: LinearBiasWeights,  // [intermediate_size, decoder_dim]
}

struct DecoderLayerWeights {
    self_attn: AttentionWeights,
    cross_attn: AttentionWeights,
    mlp: MLPWeights,
    input_layernorm: Tensor,          // f16 [decoder_dim]
    post_attention_layernorm: Tensor,  // f16 [decoder_dim]
    final_layernorm: Tensor,          // f16 [decoder_dim]
}

// ── KV Cache ──

struct MetalDecoderCache {
    // Self-attention KV: [n_kv_heads * max_kv_len * head_dim] per layer
    self_k: Vec<Tensor>,
    self_v: Vec<Tensor>,
    self_len: usize,

    // Cross-attention KV: [enc_seq, kv_dim] per layer
    cross_k: Vec<Tensor>,
    cross_v: Vec<Tensor>,
    cross_len: usize,
    cross_initialized: bool,

    // Encoder projection (computed once per decode run)
    encoder_proj_f16: Option<Tensor>,
}

// ── Scratch buffers ──

struct DecoderScratch {
    f16_norm: Tensor,       // [dim] F16 — layernorm output → GEMV input
    f16_q: Tensor,          // [q_dim]
    f16_k: Tensor,          // [kv_dim]
    f16_v: Tensor,          // [kv_dim]
    f16_attn: Tensor,       // [q_dim] F16 — attention output
    f16_act: Tensor,        // [intermediate_size]
    f32_a: Tensor,          // [dim] F32 — residual stream ping
    f32_b: Tensor,          // [dim] F32 — residual stream pong
    f16_logits: Tensor,     // [vocab_size]
    f32_splitkv_partial: Tensor, // [n_q_heads * n_splits * 3 * BLOCK_D] F32 — split-KV partials
    f32_mlp_partial: Tensor,    // F32 scratch for K-split GEMV partials (fc1 and fc2)
}

// ── Main decoder struct ──

pub struct TritonMetalDecoder {
    device: candle_core::MetalDevice,
    metal_device: Device,
    kernels: DecoderKernels,
    encoder_kernels: crate::triton_kernels::TritonKernels,
    scratch: DecoderScratch,

    // Token embedding (CPU side for lookup)
    embed_tokens_data: Vec<f32>,

    // Position embedding (CPU side)
    pos_emb_data: Vec<f32>,

    // Encoder projection (optional, CPU matmul once per decode)
    proj_weight: Option<Vec<f32>>,

    // LM head
    proj_out_weight: Tensor, // f16 [decoder_dim, vocab_size]

    // Decoder layers
    layers: Vec<DecoderLayerWeights>,

    // Final norm
    final_norm_weight: Tensor, // f16 [decoder_dim]

    // RoPE table
    rope_table: Tensor, // f32 [max_pos, half_rot * 2]

    // Config
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

impl TritonMetalDecoder {
    pub fn new(
        cfg: &MoonshineConfig,
        dec_vb: QVarBuilder,
        proj_out_vb: QVarBuilder,
        metal_device: &Device,
        kernel_dir: &std::path::Path,
    ) -> Result<Self> {
        let md = match metal_device {
            Device::Metal(md) => md.clone(),
            _ => return Err(anyhow::anyhow!("TritonMetalDecoder requires Metal device")),
        };

        println!("  Compiling decoder Metal kernels...");
        let kernels = DecoderKernels::load(&md, kernel_dir)?;
        let encoder_kernels = crate::triton_kernels::TritonKernels::load(&md, kernel_dir)?;

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
        let metal = metal_device;

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
        let proj_out_weight = load_f16_weight((cfg.vocab_size, decoder_dim), &proj_out_vb, metal)?;

        // Decoder layers
        let mut layers = Vec::with_capacity(cfg.decoder_num_layers);
        for i in 0..cfg.decoder_num_layers {
            let lvb = dec_vb.pp(&format!("layers.{i}"));
            let avb = lvb.pp("self_attn");
            let self_attn = AttentionWeights {
                q_proj: LinearWeights { weight: load_f16_weight((q_dim, decoder_dim), &avb.pp("q_proj"), metal)? },
                k_proj: LinearWeights { weight: load_f16_weight((kv_dim, decoder_dim), &avb.pp("k_proj"), metal)? },
                v_proj: LinearWeights { weight: load_f16_weight((kv_dim, decoder_dim), &avb.pp("v_proj"), metal)? },
                o_proj: LinearWeights { weight: load_f16_weight((decoder_dim, q_dim), &avb.pp("o_proj"), metal)? },
            };

            let cavb = lvb.pp("encoder_attn");
            let cross_attn = AttentionWeights {
                q_proj: LinearWeights { weight: load_f16_weight((q_dim, decoder_dim), &cavb.pp("q_proj"), metal)? },
                k_proj: LinearWeights { weight: load_f16_weight((kv_dim, decoder_dim), &cavb.pp("k_proj"), metal)? },
                v_proj: LinearWeights { weight: load_f16_weight((kv_dim, decoder_dim), &cavb.pp("v_proj"), metal)? },
                o_proj: LinearWeights { weight: load_f16_weight((decoder_dim, q_dim), &cavb.pp("o_proj"), metal)? },
            };

            let mvb = lvb.pp("mlp");
            let mlp = MLPWeights {
                fc1: LinearBiasWeights {
                    weight: load_f16_weight((intermediate_size * 2, decoder_dim), &mvb.pp("fc1"), metal)?,
                    bias: load_f32_1d(intermediate_size * 2, "bias", &mvb.pp("fc1"), metal)?,
                },
                fc2: LinearBiasWeights {
                    weight: load_f16_weight((decoder_dim, intermediate_size), &mvb.pp("fc2"), metal)?,
                    bias: load_f32_1d(decoder_dim, "bias", &mvb.pp("fc2"), metal)?,
                },
            };

            layers.push(DecoderLayerWeights {
                self_attn,
                cross_attn,
                mlp,
                input_layernorm: load_f16_1d(decoder_dim, "weight", &lvb.pp("input_layernorm"), metal)?,
                post_attention_layernorm: load_f16_1d(decoder_dim, "weight", &lvb.pp("post_attention_layernorm"), metal)?,
                final_layernorm: load_f16_1d(decoder_dim, "weight", &lvb.pp("final_layernorm"), metal)?,
            });
        }

        let final_norm_weight = load_f16_1d(decoder_dim, "weight", &dec_vb.pp("norm"), metal)?;

        // Pre-allocate scratch buffers
        let scratch = DecoderScratch {
            f16_norm: empty_f16(&md, (decoder_dim.max(q_dim),))?,
            f16_q: empty_f16(&md, (q_dim,))?,
            f16_k: empty_f16(&md, (kv_dim,))?,
            f16_v: empty_f16(&md, (kv_dim,))?,
            f16_attn: empty_f16(&md, (q_dim,))?,
            f16_act: empty_f16(&md, (intermediate_size,))?,
            f32_a: empty_f32(&md, (decoder_dim,))?,
            f32_b: empty_f32(&md, (decoder_dim,))?,
            f16_logits: empty_f16(&md, (cfg.vocab_size,))?,
            // n_splits=32, BLOCK_D=128, 3 arrays (m, l, acc) per partial
            f32_splitkv_partial: empty_f32(&md, (n_q_heads * 32 * 3 * 128,))?,
            // K-split MLP partial: supports up to 16 splits × 2 × intermediate_size
            f32_mlp_partial: empty_f32(&md, (16 * 2 * intermediate_size,))?,
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
        let rope_table = Tensor::from_vec(rope_data, (max_pos, half_rot * 2), &Device::Cpu)?
            .to_device(metal)?;

        Ok(Self {
            device: md,
            metal_device: metal_device.clone(),
            kernels,
            encoder_kernels,
            scratch,
            embed_tokens_data,
            pos_emb_data,
            proj_weight,
            proj_out_weight,
            layers,
            final_norm_weight,
            rope_table,
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
        })
    }

    fn new_cache(&self) -> Result<MetalDecoderCache> {
        let mut self_k = Vec::with_capacity(self.num_layers);
        let mut self_v = Vec::with_capacity(self.num_layers);
        for _ in 0..self.num_layers {
            self_k.push(empty_f16(&self.device, (self.n_kv_heads * self.max_kv_len * self.head_dim,))?);
            self_v.push(empty_f16(&self.device, (self.n_kv_heads * self.max_kv_len * self.head_dim,))?);
        }
        Ok(MetalDecoderCache {
            self_k,
            self_v,
            self_len: 0,
            cross_k: Vec::new(),
            cross_v: Vec::new(),
            cross_len: 0,
            cross_initialized: false,
            encoder_proj_f16: None,
        })
    }

    /// Compute encoder projection on CPU, upload as F16 to GPU.
    fn prepare_encoder_proj(
        &self, encoder_hidden: &Tensor, cache: &mut MetalDecoderCache,
    ) -> Result<()> {
        if cache.encoder_proj_f16.is_some() { return Ok(()); }

        let enc_seq = encoder_hidden.dim(1)?;
        let enc_hidden = encoder_hidden.squeeze(0)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let enc_data = enc_hidden.to_vec2::<f32>()?;

        // Debug: check encoder output
        let enc_nan = enc_data.iter().flatten().filter(|v| v.is_nan()).count();
        eprintln!("  [enc_proj] encoder_hidden: seq={} dim={} nan={} first5={:?}",
            enc_seq, self.encoder_dim, enc_nan, &enc_data[0][..5]);

        let max_pos_emb = self.pos_emb_data.len() / self.encoder_dim;
        eprintln!("  [enc_proj] max_pos_emb={} pos_emb first5={:?}", max_pos_emb, &self.pos_emb_data[..5]);

        let mut proj_data = vec![0.0f32; enc_seq * self.encoder_dim];
        for s in 0..enc_seq {
            for d in 0..self.encoder_dim {
                let pos_val = if s < max_pos_emb { self.pos_emb_data[s * self.encoder_dim + d] } else { 0.0 };
                proj_data[s * self.encoder_dim + d] = enc_data[s][d] + pos_val;
            }
        }

        let proj_nan = proj_data.iter().filter(|v| v.is_nan()).count();
        eprintln!("  [enc_proj] after pos_emb: nan={} first5={:?}", proj_nan, &proj_data[..5]);

        let final_data = if let Some(proj_w) = &self.proj_weight {
            let pw_nan = proj_w.iter().filter(|v| v.is_nan()).count();
            eprintln!("  [enc_proj] proj_weight: len={} nan={} first5={:?}", proj_w.len(), pw_nan, &proj_w[..5]);

            let mut out = vec![0.0f32; enc_seq * self.decoder_dim];
            for s in 0..enc_seq {
                for d in 0..self.decoder_dim {
                    let mut sum = 0.0f32;
                    for k in 0..self.encoder_dim {
                        sum += proj_data[s * self.encoder_dim + k] * proj_w[d * self.encoder_dim + k];
                    }
                    out[s * self.decoder_dim + d] = sum;
                }
            }

            let out_nan = out.iter().filter(|v| v.is_nan()).count();
            let out_max = out.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
            eprintln!("  [enc_proj] after proj: nan={} max_abs={:.1} first5={:?}", out_nan, out_max, &out[..5]);
            out
        } else {
            proj_data
        };

        let dim = if self.proj_weight.is_some() { self.decoder_dim } else { self.encoder_dim };
        let t = Tensor::from_vec(final_data, (enc_seq, dim), &Device::Cpu)?
            .to_dtype(DType::F16)?.to_device(&self.metal_device)?;

        // Debug: verify after F16 conversion
        let t_check = t.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let t_data = t_check.to_vec1::<f32>()?;
        let t_nan = t_data.iter().filter(|v| v.is_nan()).count();
        eprintln!("  [enc_proj] after f16 upload: nan={}/{} first5={:?}", t_nan, t_data.len(), &t_data[..5]);

        cache.encoder_proj_f16 = Some(t);
        cache.cross_len = enc_seq;
        Ok(())
    }

    /// Compute cross-attention K/V for all layers using tiled matmul.
    fn initialize_cross_attention(&self, cache: &mut MetalDecoderCache) -> Result<()> {
        if cache.cross_initialized { return Ok(()); }

        let enc_proj = cache.encoder_proj_f16.as_ref()
            .ok_or_else(|| anyhow::anyhow!("encoder projection not initialized"))?;
        let enc_seq = cache.cross_len;
        let kv_dim = self.n_kv_heads * self.head_dim;

        let enc_f16 = enc_proj;

        // Debug: check enc_proj
        {
            let ep = enc_f16.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
            let ep_data = ep.to_vec1::<f32>()?;
            let nan_count = ep_data.iter().filter(|v| v.is_nan()).count();
            eprintln!("  [dbg] enc_proj shape={:?} dtype={:?} nan={}/{} first5={:?}",
                enc_f16.shape(), enc_f16.dtype(), nan_count, ep_data.len(), &ep_data[..5]);
        }

        cache.cross_k.clear();
        cache.cross_v.clear();

        // Use 64x64 matmul for cross-attention K/V projections
        let (block_m, pipeline) = if let Some(ref p128) = self.encoder_kernels.matmul_128x128 {
            (128, p128)
        } else {
            (64, &self.encoder_kernels.matmul_64x64)
        };

        for (i, layer) in self.layers.iter().enumerate() {
            // Cross-attention K/V weights are f32 → convert to f16 for matmul
            let kw_f16 = layer.cross_attn.k_proj.weight.to_dtype(DType::F16)?;
            let vw_f16 = layer.cross_attn.v_proj.weight.to_dtype(DType::F16)?;

            if i == 0 {
                let kw = kw_f16.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
                let kw_data = kw.to_vec1::<f32>()?;
                let nan_count = kw_data.iter().filter(|v| v.is_nan()).count();
                eprintln!("  [dbg] cross_k_weight shape={:?} dtype={:?} nan={}/{} first5={:?}",
                    layer.cross_attn.k_proj.weight.shape(), layer.cross_attn.k_proj.weight.dtype(),
                    nan_count, kw_data.len(), &kw_data[..5]);
            }

            let cross_k = triton_matmul(&self.device, pipeline,
                enc_f16, &kw_f16, enc_seq, kv_dim, self.decoder_dim, block_m, block_m)?;
            let cross_v = triton_matmul(&self.device, pipeline,
                enc_f16, &vw_f16, enc_seq, kv_dim, self.decoder_dim, block_m, block_m)?;

            if i == 0 {
                self.device.wait_until_completed()?;
                let ck = cross_k.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
                let ck_data = ck.to_vec1::<f32>()?;
                let nan_count = ck_data.iter().filter(|v| v.is_nan()).count();
                eprintln!("  [dbg] cross_k result shape={:?} nan={}/{} first5={:?}",
                    cross_k.shape(), nan_count, ck_data.len(), &ck_data[..5]);
            }

            cache.cross_k.push(cross_k);
            cache.cross_v.push(cross_v);
        }

        self.device.wait_until_completed()?;
        cache.cross_initialized = true;
        Ok(())
    }

    /// Run one decoder step. All dispatches batched on a single encoder.
    fn forward_one_token(
        &self, token_id: u32, cache: &mut MetalDecoderCache, timing: bool,
    ) -> Result<()> {
        self.forward_one_token_inner(token_id, cache, timing, false)
    }

    /// Profile mode: separate command buffer per phase to get per-phase GPU timing.
    fn forward_one_token_profile(
        &self, token_id: u32, cache: &mut MetalDecoderCache,
    ) -> Result<()> {
        self.forward_one_token_inner(token_id, cache, true, true)
    }

    fn forward_one_token_inner(
        &self, token_id: u32, cache: &mut MetalDecoderCache, timing: bool, profile: bool,
    ) -> Result<()> {
        let k = &self.kernels;
        let s = &self.scratch;
        let dim = self.decoder_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let q_dim = self.n_q_heads * self.head_dim;
        let pos = cache.self_len;
        let dev = &self.device;

        let t0 = std::time::Instant::now();

        // 1. Token embedding: CPU lookup → F32 tensor → upload to Metal
        let token_offset = (token_id as usize) * dim;
        let embed_slice = &self.embed_tokens_data[token_offset..token_offset + dim];
        let embed_f32 = Tensor::from_slice(embed_slice, (dim,), &Device::Cpu)?
            .to_device(&self.metal_device)?;

        let t1 = std::time::Instant::now();

        // Profiling helper: sync GPU and return elapsed time
        macro_rules! gpu_sync {
            ($enc:ident, $dev:ident) => {{
                drop($enc);
                $dev.wait_until_completed()?;
                $dev.command_encoder()?
            }};
        }

        let mut enc = dev.command_encoder()?;

        let buffers: [&Tensor; 2] = [&s.f32_a, &s.f32_b];
        let mut write_idx: usize = 1;

        // Profiling accumulators (per-phase, summed across layers)
        let mut t_gemv = 0.0f64;
        let mut t_self_attn = 0.0f64;
        let mut t_cross_attn = 0.0f64;
        let mut t_mlp = 0.0f64;
        let mut t_misc = 0.0f64;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let read_f32 = if layer_idx == 0 { &embed_f32 } else { buffers[1 - write_idx] };
            let write_f32 = buffers[write_idx];

            // ── Self-attention QKV GEMVs (split-K for parallelism) ──
            let qkv_splits = 8usize;
            let tp = std::time::Instant::now();
            if layer_idx == 0 {
                enc_layernorm_std_f32in(&enc, &k.layernorm_std_f32in,
                    read_f32, &layer.input_layernorm, &s.f16_norm, 1, dim)?;
            }

            enc_gemv_splitk(&enc,
                &k.gemv_splitk_partial, &k.gemv_splitk_reduce,
                &s.f16_norm, &layer.self_attn.q_proj.weight, &s.f16_q, &s.f32_mlp_partial,
                q_dim, dim, qkv_splits)?;
            if profile && layer_idx == 0 {
                enc = gpu_sync!(enc, dev);
                let t_q = tp.elapsed().as_secs_f64() * 1000.0;
                let tp2 = std::time::Instant::now();
                enc_gemv_splitk(&enc,
                    &k.gemv_splitk_partial, &k.gemv_splitk_reduce,
                    &s.f16_norm, &layer.self_attn.k_proj.weight, &s.f16_k, &s.f32_mlp_partial,
                    kv_dim, dim, qkv_splits)?;
                enc = gpu_sync!(enc, dev);
                let t_k = tp2.elapsed().as_secs_f64() * 1000.0;
                let tp2 = std::time::Instant::now();
                enc_gemv_splitk(&enc,
                    &k.gemv_splitk_partial, &k.gemv_splitk_reduce,
                    &s.f16_norm, &layer.self_attn.v_proj.weight, &s.f16_v, &s.f32_mlp_partial,
                    kv_dim, dim, qkv_splits)?;
                enc = gpu_sync!(enc, dev);
                let t_v = tp2.elapsed().as_secs_f64() * 1000.0;
                eprintln!("  [layer0] LN+Q={:.3}ms K={:.3}ms V={:.3}ms", t_q, t_k, t_v);
                t_gemv += t_q + t_k + t_v;
            } else {
                enc_gemv_splitk(&enc,
                    &k.gemv_splitk_partial, &k.gemv_splitk_reduce,
                    &s.f16_norm, &layer.self_attn.k_proj.weight, &s.f16_k, &s.f32_mlp_partial,
                    kv_dim, dim, qkv_splits)?;
                enc_gemv_splitk(&enc,
                    &k.gemv_splitk_partial, &k.gemv_splitk_reduce,
                    &s.f16_norm, &layer.self_attn.v_proj.weight, &s.f16_v, &s.f32_mlp_partial,
                    kv_dim, dim, qkv_splits)?;
                if profile { enc = gpu_sync!(enc, dev); t_gemv += tp.elapsed().as_secs_f64() * 1000.0; }
            }

            let tp = std::time::Instant::now();
            enc_rope_qk_cache_fused(&enc, &k.rope_qk_cache_fused,
                &s.f16_q, &s.f16_k, &self.rope_table, &cache.self_k[layer_idx],
                self.n_q_heads, self.n_kv_heads, self.head_dim, self.half_rot,
                pos, self.max_kv_len)?;

            enc_kv_cache_append(&enc, &k.kv_cache_append,
                &s.f16_v, &cache.self_v[layer_idx],
                self.n_kv_heads, self.head_dim, self.max_kv_len, pos)?;
            if profile { enc = gpu_sync!(enc, dev); t_misc += tp.elapsed().as_secs_f64() * 1000.0; }

            // ── Self-attention decode ──
            let tp = std::time::Instant::now();
            let self_kv_len = pos + 1;
            enc_attention_decode(&enc, &k.attention_decode,
                &s.f16_q, &cache.self_k[layer_idx], &cache.self_v[layer_idx], &s.f16_attn,
                self_kv_len, self.head_dim, self.n_kv_heads, self.n_q_heads,
                self.sm_scale,
                self.max_kv_len * self.head_dim,
                self.head_dim)?;
            if profile { enc = gpu_sync!(enc, dev); t_self_attn += tp.elapsed().as_secs_f64() * 1000.0; }

            let tp = std::time::Instant::now();
            enc_gemv_splitk(&enc,
                &k.gemv_splitk_partial, &k.gemv_splitk_reduce,
                &s.f16_attn, &layer.self_attn.o_proj.weight, &s.f16_act, &s.f32_mlp_partial,
                dim, q_dim, qkv_splits)?;
            enc_residual_add_layernorm(&enc, &k.residual_add_layernorm,
                &s.f16_act, read_f32, write_f32,
                &layer.post_attention_layernorm, &s.f16_norm,
                1, dim)?;
            if profile { enc = gpu_sync!(enc, dev); t_gemv += tp.elapsed().as_secs_f64() * 1000.0; }
            write_idx = 1 - write_idx;

            // ── Cross-attention ──
            let read_f32 = buffers[1 - write_idx];
            let write_f32 = buffers[write_idx];

            let tp = std::time::Instant::now();
            enc_gemv_splitk(&enc,
                &k.gemv_splitk_partial, &k.gemv_splitk_reduce,
                &s.f16_norm, &layer.cross_attn.q_proj.weight, &s.f16_q, &s.f32_mlp_partial,
                q_dim, dim, qkv_splits)?;
            if profile { enc = gpu_sync!(enc, dev); t_gemv += tp.elapsed().as_secs_f64() * 1000.0;
                if layer_idx == 0 { eprintln!("  [layer0] cross_Q={:.3}ms", tp.elapsed().as_secs_f64() * 1000.0); }
            }

            let tp = std::time::Instant::now();
            let cross_n_splits = 32usize.min(cache.cross_len);
            enc_attention_splitkv(&enc,
                &k.attention_splitkv_partial, &k.attention_splitkv_reduce,
                &s.f16_q, &cache.cross_k[layer_idx], &cache.cross_v[layer_idx],
                &s.f16_attn, &s.f32_splitkv_partial,
                cache.cross_len, self.head_dim, self.n_kv_heads, self.n_q_heads,
                self.sm_scale,
                self.head_dim,
                self.n_kv_heads * self.head_dim,
                cross_n_splits)?;
            if profile { enc = gpu_sync!(enc, dev); t_cross_attn += tp.elapsed().as_secs_f64() * 1000.0;
                if layer_idx == 0 { eprintln!("  [layer0] cross_attn={:.3}ms (kv_len={}, splits={})", tp.elapsed().as_secs_f64() * 1000.0, cache.cross_len, cross_n_splits); }
            }

            let tp = std::time::Instant::now();
            enc_gemv_splitk(&enc,
                &k.gemv_splitk_partial, &k.gemv_splitk_reduce,
                &s.f16_attn, &layer.cross_attn.o_proj.weight, &s.f16_act, &s.f32_mlp_partial,
                dim, q_dim, qkv_splits)?;
            enc_residual_add_layernorm(&enc, &k.residual_add_layernorm,
                &s.f16_act, read_f32, write_f32,
                &layer.final_layernorm, &s.f16_norm,
                1, dim)?;
            if profile { enc = gpu_sync!(enc, dev); t_gemv += tp.elapsed().as_secs_f64() * 1000.0;
                if layer_idx == 0 { eprintln!("  [layer0] cross_O_resadd_ln={:.3}ms", tp.elapsed().as_secs_f64() * 1000.0); }
            }
            write_idx = 1 - write_idx;

            // ── MLP ──
            let read_f32 = buffers[1 - write_idx];
            let write_f32 = buffers[write_idx];

            // fc1_glu: split-K with 8 splits (640/8=80 K per split, 160 TGs)
            // fc2: split-K with 16 splits (2560/16=160 K per split, 80 TGs)
            let fc1_splits = 8usize;
            let fc2_splits = 16usize;

            if profile && layer_idx == 0 {
                let tp2 = std::time::Instant::now();
                enc_gemv_glu_splitk(&enc,
                    &k.gemv_glu_splitk_partial, &k.gemv_glu_splitk_reduce,
                    &s.f16_norm, &layer.mlp.fc1.weight, &layer.mlp.fc1.bias, &s.f16_act,
                    &s.f32_mlp_partial,
                    self.intermediate_size, dim, fc1_splits)?;
                enc = gpu_sync!(enc, dev);
                let t_fc1 = tp2.elapsed().as_secs_f64() * 1000.0;
                let tp2 = std::time::Instant::now();
                enc_gemv_splitk_bias(&enc,
                    &k.gemv_splitk_partial, &k.gemv_splitk_bias_reduce,
                    &s.f16_act, &layer.mlp.fc2.weight, &layer.mlp.fc2.bias, &s.f16_norm,
                    &s.f32_mlp_partial,
                    dim, self.intermediate_size, fc2_splits)?;
                enc = gpu_sync!(enc, dev);
                let t_fc2 = tp2.elapsed().as_secs_f64() * 1000.0;
                let tp2 = std::time::Instant::now();
                let next_ln_weight = if layer_idx + 1 < self.num_layers {
                    &self.layers[layer_idx + 1].input_layernorm
                } else {
                    &self.final_norm_weight
                };
                enc_residual_add_layernorm(&enc, &k.residual_add_layernorm,
                    &s.f16_norm, read_f32, write_f32,
                    next_ln_weight, &s.f16_norm,
                    1, dim)?;
                enc = gpu_sync!(enc, dev);
                let t_res = tp2.elapsed().as_secs_f64() * 1000.0;
                eprintln!("  [layer0] fc1_glu={:.3}ms fc2={:.3}ms resadd_ln={:.3}ms", t_fc1, t_fc2, t_res);
                t_mlp += t_fc1 + t_fc2 + t_res;
            } else {
                let tp = std::time::Instant::now();
                enc_gemv_glu_splitk(&enc,
                    &k.gemv_glu_splitk_partial, &k.gemv_glu_splitk_reduce,
                    &s.f16_norm, &layer.mlp.fc1.weight, &layer.mlp.fc1.bias, &s.f16_act,
                    &s.f32_mlp_partial,
                    self.intermediate_size, dim, fc1_splits)?;

                enc_gemv_splitk_bias(&enc,
                    &k.gemv_splitk_partial, &k.gemv_splitk_bias_reduce,
                    &s.f16_act, &layer.mlp.fc2.weight, &layer.mlp.fc2.bias, &s.f16_norm,
                    &s.f32_mlp_partial,
                    dim, self.intermediate_size, fc2_splits)?;

                let next_ln_weight = if layer_idx + 1 < self.num_layers {
                    &self.layers[layer_idx + 1].input_layernorm
                } else {
                    &self.final_norm_weight
                };
                enc_residual_add_layernorm(&enc, &k.residual_add_layernorm,
                    &s.f16_norm, read_f32, write_f32,
                    next_ln_weight, &s.f16_norm,
                    1, dim)?;
                if profile { enc = gpu_sync!(enc, dev); t_mlp += tp.elapsed().as_secs_f64() * 1000.0; }
            }
            write_idx = 1 - write_idx;
        }

        // LM head: GEMV f16_norm @ proj_out → f16_logits
        let tp = std::time::Instant::now();
        enc_gemv_f16w(&enc, &k.gemv_f16w,
            &s.f16_norm, &self.proj_out_weight, &s.f16_logits,
            self.vocab_size, dim)?;

        let t2 = std::time::Instant::now();

        // End encoding and sync GPU
        drop(enc);
        dev.wait_until_completed()?;
        if profile {
            let t_lm = tp.elapsed().as_secs_f64() * 1000.0;
            eprintln!("  [layer0] lm_head={:.3}ms (640→{})", t_lm, self.vocab_size);
            t_gemv += t_lm;
        }

        let t3 = std::time::Instant::now();

        if timing {
            eprintln!("  [timing] embed={:.2}ms dispatch={:.2}ms gpu={:.2}ms total={:.2}ms",
                t1.duration_since(t0).as_secs_f64() * 1000.0,
                t2.duration_since(t1).as_secs_f64() * 1000.0,
                t3.duration_since(t2).as_secs_f64() * 1000.0,
                t3.duration_since(t0).as_secs_f64() * 1000.0);
        }
        if profile {
            let total = t_gemv + t_self_attn + t_cross_attn + t_mlp + t_misc;
            let n = self.num_layers as f64;
            eprintln!("  [profile] TOTAL={:.1}ms over {} layers + LM head", total, self.num_layers);
            eprintln!("    gemv={:.2}ms ({:.2}ms/layer)  self_attn={:.2}ms ({:.2}ms/layer)",
                t_gemv, t_gemv / n, t_self_attn, t_self_attn / n);
            eprintln!("    cross_attn={:.2}ms ({:.2}ms/layer)  mlp={:.2}ms ({:.2}ms/layer)",
                t_cross_attn, t_cross_attn / n, t_mlp, t_mlp / n);
            eprintln!("    misc={:.2}ms ({:.2}ms/layer)", t_misc, t_misc / n);
        }

        cache.self_len += 1;
        Ok(())
    }

    /// Read f16 logits from GPU and return argmax token ID.
    fn argmax_logits(&self, debug: bool) -> Result<u32> {
        let logits = self.scratch.f16_logits.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let data = logits.to_vec1::<f32>()?;
        if debug {
            let nan_count = data.iter().filter(|v| v.is_nan()).count();
            let zero_count = data.iter().filter(|&&v| v == 0.0).count();
            let nonzero: Vec<(usize, f32)> = data.iter().enumerate()
                .filter(|&(_, &v)| v != 0.0 && !v.is_nan())
                .take(10)
                .map(|(i, &v)| (i, v))
                .collect();
            eprintln!("  [logits] len={} nan={} zero={} nonzero_sample={:?} first5={:?}",
                data.len(), nan_count, zero_count, nonzero, &data[..5.min(data.len())]);
        }
        let mut indexed: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if debug {
            eprintln!("  [metal-logits] top5={:?}", &indexed[..5]);
        }
        Ok(indexed[0].0 as u32)
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
        self.forward_one_token(self.bos_id, &mut cache, true)?;
        let ta0 = std::time::Instant::now();
        let mut next_token = self.argmax_logits(true)?;
        let ta1 = std::time::Instant::now();
        eprintln!("  [timing] argmax={:.2}ms", ta1.duration_since(ta0).as_secs_f64() * 1000.0);
        generated.push(next_token);
        if next_token == self.eos_id { return Ok(generated); }

        eprint!("  [metal-dec] tokens: {}", next_token);
        let decode_start = std::time::Instant::now();
        for step in 0..max_tokens - 1 {
            self.forward_one_token(next_token, &mut cache, step < 5)?;
            let ta0 = std::time::Instant::now();
            next_token = self.argmax_logits(step < 5)?;
            if step < 5 {
                eprintln!("  [timing] argmax={:.2}ms",
                    ta0.elapsed().as_secs_f64() * 1000.0);
            }
            eprint!(" {}", next_token);
            generated.push(next_token);
            if next_token == self.eos_id { break; }
        }
        let decode_elapsed = decode_start.elapsed();
        let n_tokens = generated.len().saturating_sub(1).max(1);
        eprintln!("\n  [perf] {} tokens in {:.1}ms = {:.2}ms/token",
            n_tokens, decode_elapsed.as_secs_f64() * 1000.0,
            decode_elapsed.as_secs_f64() * 1000.0 / n_tokens as f64);

        Ok(generated)
    }
}
