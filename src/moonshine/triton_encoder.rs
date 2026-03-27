//! Moonshine V2 Encoder using Triton-compiled Metal kernels — all-GPU pipeline.
//!
//! All operations (matmul, layernorm, GELU, bias add, residual add, flash attention)
//! run on the Metal GPU. Weights and activations stay in F16 on GPU throughout.
//! Only a single CPU↔GPU sync occurs at the very end of forward().

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::config::MoonshineConfig;
use crate::triton_kernels::{
    TritonKernels, empty_f16, triton_bias_add, triton_flash_attention, triton_gelu,
    triton_layernorm_bare, triton_layernorm_unit_offset, triton_matmul, triton_matmul_bias,
    triton_matmul_bias_gelu, triton_residual_add,
};

fn cdiv(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

type QVarBuilder = candle_transformers::quantized_var_builder::VarBuilder;

/// Load weight from GGUF, dequantize to F16 on Metal.
fn load_f16_weight(shape: (usize, usize), vb: &QVarBuilder, metal: &Device) -> Result<Tensor> {
    let qt = vb.get(shape, "weight")?;
    let t = qt.dequantize(&Device::Cpu)?;
    Ok(t.to_dtype(DType::F16)?.to_device(metal)?.t()?.contiguous()?)
}

/// Load 1D parameter from GGUF, dequantize to F16 on Metal.
fn load_f16_1d(dim: usize, name: &str, vb: &QVarBuilder, metal: &Device) -> Result<Tensor> {
    let qt = vb.get(dim, name)?;
    let t = qt.dequantize(&Device::Cpu)?;
    Ok(t.to_dtype(DType::F16)?.to_device(metal)?)
}

struct LayerNormWeights {
    gamma: Tensor, // [dim] F16 on Metal
}

impl LayerNormWeights {
    fn new(dim: usize, vb: &QVarBuilder, metal: &Device) -> Result<Self> {
        Ok(Self { gamma: load_f16_1d(dim, "gamma", vb, metal)? })
    }
}

struct LinearWeights {
    weight: Tensor, // [in_dim, out_dim] F16 on Metal (transposed for Triton)
    #[allow(dead_code)]
    in_dim: usize,
    #[allow(dead_code)]
    out_dim: usize,
}

impl LinearWeights {
    fn new(in_dim: usize, out_dim: usize, vb: &QVarBuilder, metal: &Device) -> Result<Self> {
        Ok(Self {
            weight: load_f16_weight((out_dim, in_dim), vb, metal)?,
            in_dim,
            out_dim,
        })
    }
}

struct LinearBiasWeights {
    weight: Tensor, // [in_dim, out_dim] F16 on Metal
    bias: Tensor,   // [out_dim] F16 on Metal
    in_dim: usize,
    out_dim: usize,
}

impl LinearBiasWeights {
    fn new(in_dim: usize, out_dim: usize, vb: &QVarBuilder, metal: &Device) -> Result<Self> {
        Ok(Self {
            weight: load_f16_weight((out_dim, in_dim), vb, metal)?,
            bias: load_f16_1d(out_dim, "bias", vb, metal)?,
            in_dim,
            out_dim,
        })
    }
}

struct AttentionWeights {
    qkv_weight: Tensor, // [in_dim, 3*kv_dim] F16 on Metal — fused Q/K/V weight
    o_proj: LinearWeights,
    num_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    scale: f64,
}

struct MLPWeights {
    fc1: LinearBiasWeights,
    fc2: LinearBiasWeights,
}

struct EncoderLayerWeights {
    self_attn: AttentionWeights,
    mlp: MLPWeights,
}

/// Triton-accelerated Moonshine encoder — all-GPU pipeline.
///
/// Every operation dispatches on Metal GPU. Weights are F16 on GPU.
/// Activations stay on GPU throughout. Single CPU sync at the end.
pub struct TritonEncoder {
    kernels: TritonKernels,
    layers: Vec<EncoderLayerWeights>,
    final_norm: LayerNormWeights,
    sliding_windows: Vec<[usize; 2]>,
    metal_device: candle_core::MetalDevice,
    metal_candle_device: Device,
    encoder_dim: usize,
}

impl TritonEncoder {
    pub fn new(
        cfg: &MoonshineConfig,
        vb: QVarBuilder,
        device: &Device,
    ) -> Result<Self> {
        // Use existing Metal device, or create one if model is on CPU
        let metal_candle_device = match device {
            Device::Metal(_) => device.clone(),
            _ => Device::new_metal(0)
                .map_err(|e| anyhow::anyhow!("TritonEncoder needs Metal GPU: {e}"))?,
        };
        let metal_device = match &metal_candle_device {
            Device::Metal(md) => md.clone(),
            _ => unreachable!(),
        };

        println!("  Loading Triton kernel pipelines...");
        let kernels = TritonKernels::load(&metal_device)?;

        let kv_dim = cfg.encoder_num_kv_heads * cfg.encoder_head_dim;
        let metal = &metal_candle_device;

        let mut layers = Vec::with_capacity(cfg.encoder_num_layers);
        for i in 0..cfg.encoder_num_layers {
            let lvb = vb.pp(&format!("layers.{i}"));
            let avb = lvb.pp("self_attn");

            // Load layernorm gammas for baking into weights
            let input_ln_gamma = load_f16_1d(cfg.encoder_dim, "gamma", &lvb.pp("input_layernorm"), metal)?;
            let post_ln_gamma = load_f16_1d(cfg.encoder_dim, "gamma", &lvb.pp("post_attention_layernorm"), metal)?;

            // Bake (1 + gamma) from input_layernorm into QKV weights:
            //   W_baked[i, j] = (1 + gamma[i]) * W[i, j]
            // This lets us use a bare layernorm (no gamma) followed by matmul with baked weights.
            let input_scale = (&input_ln_gamma.to_dtype(DType::F32)? + 1.0)?
                .to_dtype(DType::F16)?
                .unsqueeze(1)?; // [768, 1]
            let w_q = load_f16_weight((kv_dim, cfg.encoder_dim), &avb.pp("q_proj"), metal)?;
            let w_k = load_f16_weight((kv_dim, cfg.encoder_dim), &avb.pp("k_proj"), metal)?;
            let w_v = load_f16_weight((kv_dim, cfg.encoder_dim), &avb.pp("v_proj"), metal)?;
            let qkv_weight = Tensor::cat(&[&w_q, &w_k, &w_v], 1)?; // [768, 1920]
            let qkv_weight = qkv_weight.broadcast_mul(&input_scale)?; // bake input LN gamma

            let self_attn = AttentionWeights {
                qkv_weight,
                o_proj: LinearWeights::new(kv_dim, cfg.encoder_dim, &avb.pp("o_proj"), metal)?,
                num_heads: cfg.encoder_num_heads,
                head_dim: cfg.encoder_head_dim,
                kv_dim,
                scale: (cfg.encoder_head_dim as f64).powf(-0.5),
            };

            // Bake (1 + gamma) from post_attention_layernorm into fc1 weights
            let post_scale = (&post_ln_gamma.to_dtype(DType::F32)? + 1.0)?
                .to_dtype(DType::F16)?
                .unsqueeze(1)?; // [768, 1]
            let mvb = lvb.pp("mlp");
            let mut fc1 = LinearBiasWeights::new(cfg.encoder_dim, cfg.encoder_intermediate_size, &mvb.pp("fc1"), metal)?;
            fc1.weight = fc1.weight.broadcast_mul(&post_scale)?; // bake post-attn LN gamma
            let mlp = MLPWeights {
                fc1,
                fc2: LinearBiasWeights::new(cfg.encoder_intermediate_size, cfg.encoder_dim, &mvb.pp("fc2"), metal)?,
            };

            layers.push(EncoderLayerWeights {
                self_attn,
                mlp,
            });
        }

        let final_norm = LayerNormWeights::new(cfg.encoder_dim, &vb.pp("final_norm"), metal)?;

        Ok(Self {
            kernels,
            layers,
            final_norm,
            sliding_windows: cfg.sliding_windows.clone(),
            metal_device,
            metal_candle_device,
            encoder_dim: cfg.encoder_dim,
        })
    }

    pub fn metal_device(&self) -> &Device {
        &self.metal_candle_device
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, dim) = x.dims3()?;
        assert_eq!(batch, 1, "TritonEncoder only supports batch=1");
        assert_eq!(dim, self.encoder_dim);

        // Use 128×128 tiles if the kernel is available, otherwise fall back to 64×64
        let (block_m, matmul_pipeline) = if let Some(ref p128) = self.kernels.matmul_128x128 {
            (128, p128)
        } else {
            (64, &self.kernels.matmul_64x64)
        };
        let padded_seq = cdiv(seq_len, block_m) * block_m;

        // Move input to Metal F16
        let x_metal = x.reshape((seq_len, dim))?
            .to_dtype(DType::F16)?
            .to_device(&self.metal_candle_device)?;

        // Pad to multiple of block_m if needed
        let mut hidden = if padded_seq > seq_len {
            let pad = Tensor::zeros(
                (padded_seq - seq_len, dim),
                DType::F16,
                &self.metal_candle_device,
            )?;
            Tensor::cat(&[&x_metal, &pad], 0)?
        } else {
            x_metal
        };

        let dev = &self.metal_device;
        let k = &self.kernels;

        // Select fused pipelines (128→64→32 fallback)
        let (fc_block, matmul_bias_pipeline) =
            if let Some(ref p) = k.matmul_bias_128x128 { (128, p) }
            else if let Some(ref p) = k.matmul_bias_64x64 { (64, p) }
            else { (32, &k.matmul_bias_32x32) };
        let (fc_gelu_block, matmul_bias_gelu_pipeline) =
            if let Some(ref p) = k.matmul_bias_gelu_128x128 { (128, p) }
            else if let Some(ref p) = k.matmul_bias_gelu_64x64 { (64, p) }
            else { (32, &k.matmul_bias_gelu_32x32) };

        for (_i, layer) in self.layers.iter().enumerate() {
            let [left, right] = self.sliding_windows[_i];

            let kv_dim = layer.self_attn.kv_dim;
            let nh = layer.self_attn.num_heads;
            let hd = layer.self_attn.head_dim;
            let n_elem = padded_seq * dim;

            // ── Pre-norm (bare LN, gamma baked into QKV weights) ──
            let residual = hidden.clone();
            let normed = if let Some(ref bare_pipeline) = k.layernorm_bare {
                triton_layernorm_bare(dev, bare_pipeline, &hidden, padded_seq, dim)?
            } else {
                // Fallback: this path should not be reached when bare kernel is compiled
                unreachable!("layernorm_bare kernel not compiled")
            };

            // ── Fused Q/K/V projection: single matmul [T,768] @ [768,1920] → [T,1920] ──
            let qkv = triton_matmul(dev, matmul_pipeline, &normed, &layer.self_attn.qkv_weight, padded_seq, 3 * kv_dim, dim, block_m, block_m)?;
            let q = qkv.narrow(1, 0, kv_dim)?;
            let kk = qkv.narrow(1, kv_dim, kv_dim)?;
            let v = qkv.narrow(1, 2 * kv_dim, kv_dim)?;

            // ── Flash Attention (GPU) ──
            // Q/K/V are views into qkv with stride_m = 3*kv_dim (row stride of the fused output)
            let attn_out = empty_f16(dev, (padded_seq, kv_dim))?;
            let sm_scale = layer.self_attn.scale as f32;
            triton_flash_attention(
                dev, &k.flash_attention,
                &q, &kk, &v, &attn_out,
                nh, padded_seq, hd,
                hd as i32,
                (3 * kv_dim) as i32,   // stride_qkv = row stride of fused QKV views
                kv_dim as i32,         // stride_o = row stride of contiguous output
                sm_scale,
                left as i32, right as i32,
            )?;

            // ── O projection (GPU matmul) ──
            let attn_proj = triton_matmul(dev, matmul_pipeline, &attn_out, &layer.self_attn.o_proj.weight, padded_seq, dim, kv_dim, block_m, block_m)?;

            // ── Residual add (GPU) ──
            hidden = triton_residual_add(dev, &k.residual_add, &attn_proj, &residual, n_elem)?;

            // ── Post-norm (bare LN, gamma baked into fc1 weights) ──
            let residual = hidden.clone();
            let normed = if let Some(ref bare_pipeline) = k.layernorm_bare {
                triton_layernorm_bare(dev, bare_pipeline, &hidden, padded_seq, dim)?
            } else {
                unreachable!("layernorm_bare kernel not compiled")
            };

            // ── FFN fc1: matmul + bias + GELU ──
            // Separate ops (matmul 128×128 + bias_add + gelu) are faster than fused 32×32
            let fc1_dim = layer.mlp.fc1.out_dim;
            let fc1 = if fc_gelu_block < 64 && block_m >= 64 {
                let mm = triton_matmul(dev, matmul_pipeline, &normed, &layer.mlp.fc1.weight, padded_seq, fc1_dim, layer.mlp.fc1.in_dim, block_m, block_m)?;
                let biased = triton_bias_add(dev, &k.bias_add, &mm, &layer.mlp.fc1.bias, padded_seq * fc1_dim, fc1_dim)?;
                triton_gelu(dev, &k.gelu, &biased, padded_seq * fc1_dim)?
            } else {
                triton_matmul_bias_gelu(dev, matmul_bias_gelu_pipeline, &normed, &layer.mlp.fc1.weight, &layer.mlp.fc1.bias, padded_seq, fc1_dim, layer.mlp.fc1.in_dim, fc_gelu_block, fc_gelu_block)?
            };

            // ── FFN fc2: matmul + bias ──
            let fc2 = if fc_block < 64 && block_m >= 64 {
                let mm = triton_matmul(dev, matmul_pipeline, &fc1, &layer.mlp.fc2.weight, padded_seq, layer.mlp.fc2.out_dim, layer.mlp.fc2.in_dim, block_m, block_m)?;
                triton_bias_add(dev, &k.bias_add, &mm, &layer.mlp.fc2.bias, padded_seq * layer.mlp.fc2.out_dim, layer.mlp.fc2.out_dim)?
            } else {
                triton_matmul_bias(dev, matmul_bias_pipeline, &fc1, &layer.mlp.fc2.weight, &layer.mlp.fc2.bias, padded_seq, layer.mlp.fc2.out_dim, layer.mlp.fc2.in_dim, fc_block, fc_block)?
            };

            // ── Residual add (GPU) ──
            hidden = triton_residual_add(dev, &k.residual_add, &fc2, &residual, n_elem)?;
        }

        // Profile: measure per-operation GPU time (syncs after each op)
        if std::env::var("TRITON_ENCODER_PROFILE").is_ok() {
            return self.forward_profile(&hidden, padded_seq, dim, dev, k, matmul_pipeline, block_m,
                matmul_bias_pipeline, fc_block, matmul_bias_gelu_pipeline, fc_gelu_block, seq_len);
        }

        // Final layernorm (GPU)
        let out = triton_layernorm_unit_offset(
            dev, &k.layernorm_unit_offset,
            &hidden, &self.final_norm.gamma,
            padded_seq, dim,
        )?;

        // Single sync: GPU → CPU, F16 → F32
        let out = out.narrow(0, 0, seq_len)?;
        let out = out.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        Ok(out.reshape((1, seq_len, dim))?)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_profile(
        &self,
        hidden_in: &Tensor,
        padded_seq: usize,
        dim: usize,
        dev: &candle_core::MetalDevice,
        k: &TritonKernels,
        matmul_pipeline: &candle_metal_kernels::metal::ComputePipeline,
        block_m: usize,
        matmul_bias_pipeline: &candle_metal_kernels::metal::ComputePipeline,
        fc_block: usize,
        matmul_bias_gelu_pipeline: &candle_metal_kernels::metal::ComputePipeline,
        fc_gelu_block: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        use std::time::Instant;
        let mut hidden = hidden_in.clone();

        let mut totals = std::collections::HashMap::<&str, f64>::new();
        let mut sync = || -> Result<()> {
            dev.wait_until_completed()?;
            Ok(())
        };

        for (i, layer) in self.layers.iter().enumerate() {
            let [left, right] = self.sliding_windows[i];
            let kv_dim = layer.self_attn.kv_dim;
            let nh = layer.self_attn.num_heads;
            let hd = layer.self_attn.head_dim;
            let n_elem = padded_seq * dim;

            let bare_ln = k.layernorm_bare.as_ref().unwrap();

            let t = Instant::now();
            let residual = hidden.clone();
            let normed = triton_layernorm_bare(dev, bare_ln, &hidden, padded_seq, dim)?;
            sync()?;
            *totals.entry("layernorm_bare").or_default() += t.elapsed().as_secs_f64();

            let t = Instant::now();
            let qkv = triton_matmul(dev, matmul_pipeline, &normed, &layer.self_attn.qkv_weight, padded_seq, 3 * kv_dim, dim, block_m, block_m)?;
            let q = qkv.narrow(1, 0, kv_dim)?;
            let kk = qkv.narrow(1, kv_dim, kv_dim)?;
            let v = qkv.narrow(1, 2 * kv_dim, kv_dim)?;
            sync()?;
            *totals.entry("qkv_matmul").or_default() += t.elapsed().as_secs_f64();

            let t = Instant::now();
            let attn_out = empty_f16(dev, (padded_seq, kv_dim))?;
            let sm_scale = layer.self_attn.scale as f32;
            triton_flash_attention(
                dev, &k.flash_attention,
                &q, &kk, &v, &attn_out,
                nh, padded_seq, hd,
                hd as i32,
                (3 * kv_dim) as i32,
                kv_dim as i32,
                sm_scale,
                left as i32, right as i32,
            )?;
            sync()?;
            *totals.entry("flash_attn").or_default() += t.elapsed().as_secs_f64();

            let t = Instant::now();
            let attn_proj = triton_matmul(dev, matmul_pipeline, &attn_out, &layer.self_attn.o_proj.weight, padded_seq, dim, kv_dim, block_m, block_m)?;
            sync()?;
            *totals.entry("o_matmul").or_default() += t.elapsed().as_secs_f64();

            let t = Instant::now();
            hidden = triton_residual_add(dev, &k.residual_add, &attn_proj, &residual, n_elem)?;
            sync()?;
            *totals.entry("residual").or_default() += t.elapsed().as_secs_f64();

            let t = Instant::now();
            let residual = hidden.clone();
            let normed = triton_layernorm_bare(dev, bare_ln, &hidden, padded_seq, dim)?;
            sync()?;
            *totals.entry("layernorm_bare").or_default() += t.elapsed().as_secs_f64();

            let fc1_dim = layer.mlp.fc1.out_dim;
            let fc1 = if fc_gelu_block >= 64 {
                let t = Instant::now();
                let fc1 = triton_matmul_bias_gelu(dev, matmul_bias_gelu_pipeline, &normed, &layer.mlp.fc1.weight, &layer.mlp.fc1.bias, padded_seq, fc1_dim, layer.mlp.fc1.in_dim, fc_gelu_block, fc_gelu_block)?;
                sync()?;
                *totals.entry("fc1_fused").or_default() += t.elapsed().as_secs_f64();
                fc1
            } else {
                let t = Instant::now();
                let mm = triton_matmul(dev, matmul_pipeline, &normed, &layer.mlp.fc1.weight, padded_seq, fc1_dim, layer.mlp.fc1.in_dim, block_m, block_m)?;
                sync()?;
                *totals.entry("fc1_matmul").or_default() += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let biased = triton_bias_add(dev, &k.bias_add, &mm, &layer.mlp.fc1.bias, padded_seq * fc1_dim, fc1_dim)?;
                sync()?;
                *totals.entry("fc1_bias").or_default() += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let fc1 = triton_gelu(dev, &k.gelu, &biased, padded_seq * fc1_dim)?;
                sync()?;
                *totals.entry("fc1_gelu").or_default() += t.elapsed().as_secs_f64();
                fc1
            };

            let fc2 = if fc_block >= 64 {
                let t = Instant::now();
                let fc2 = triton_matmul_bias(dev, matmul_bias_pipeline, &fc1, &layer.mlp.fc2.weight, &layer.mlp.fc2.bias, padded_seq, layer.mlp.fc2.out_dim, layer.mlp.fc2.in_dim, fc_block, fc_block)?;
                sync()?;
                *totals.entry("fc2_fused").or_default() += t.elapsed().as_secs_f64();
                fc2
            } else {
                let t = Instant::now();
                let mm = triton_matmul(dev, matmul_pipeline, &fc1, &layer.mlp.fc2.weight, padded_seq, layer.mlp.fc2.out_dim, layer.mlp.fc2.in_dim, block_m, block_m)?;
                sync()?;
                *totals.entry("fc2_matmul").or_default() += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let fc2 = triton_bias_add(dev, &k.bias_add, &mm, &layer.mlp.fc2.bias, padded_seq * layer.mlp.fc2.out_dim, layer.mlp.fc2.out_dim)?;
                sync()?;
                *totals.entry("fc2_bias").or_default() += t.elapsed().as_secs_f64();
                fc2
            };

            let t = Instant::now();
            hidden = triton_residual_add(dev, &k.residual_add, &fc2, &residual, n_elem)?;
            sync()?;
            *totals.entry("residual").or_default() += t.elapsed().as_secs_f64();
        }

        let t = Instant::now();
        let out = triton_layernorm_unit_offset(
            dev, &k.layernorm_unit_offset,
            &hidden, &self.final_norm.gamma,
            padded_seq, dim,
        )?;
        sync()?;
        *totals.entry("final_ln").or_default() += t.elapsed().as_secs_f64();

        // Print profile
        let total: f64 = totals.values().sum();
        eprintln!("\n  ── Encoder Profile (14 layers, T={padded_seq}) ──");
        let mut items: Vec<_> = totals.iter().collect();
        items.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (name, ms) in &items {
            eprintln!("    {:<20} {:7.1}ms  ({:4.1}%)", name, *ms * 1000.0, *ms / total * 100.0);
        }
        eprintln!("    {:<20} {:7.1}ms", "TOTAL", total * 1000.0);

        let out = out.narrow(0, 0, seq_len)?;
        let out = out.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        Ok(out.reshape((1, seq_len, dim))?)
    }
}
