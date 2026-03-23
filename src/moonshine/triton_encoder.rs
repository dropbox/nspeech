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
    triton_layernorm_unit_offset, triton_matmul, triton_residual_add,
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
    q_proj: LinearWeights,
    k_proj: LinearWeights,
    v_proj: LinearWeights,
    o_proj: LinearWeights,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

struct MLPWeights {
    fc1: LinearBiasWeights,
    fc2: LinearBiasWeights,
}

struct EncoderLayerWeights {
    self_attn: AttentionWeights,
    mlp: MLPWeights,
    input_layernorm: LayerNormWeights,
    post_attention_layernorm: LayerNormWeights,
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
        kernel_dir: &std::path::Path,
    ) -> Result<Self> {
        let metal_candle_device = Device::new_metal(0)
            .map_err(|e| anyhow::anyhow!("Metal device not available: {e}"))?;
        let metal_device = match &metal_candle_device {
            Device::Metal(md) => md.clone(),
            _ => unreachable!(),
        };

        println!("  Loading Triton kernel pipelines...");
        let kernels = TritonKernels::load(&metal_device, kernel_dir)?;

        let kv_dim = cfg.encoder_num_kv_heads * cfg.encoder_head_dim;
        let metal = &metal_candle_device;

        let mut layers = Vec::with_capacity(cfg.encoder_num_layers);
        for i in 0..cfg.encoder_num_layers {
            let lvb = vb.pp(&format!("layers.{i}"));
            let avb = lvb.pp("self_attn");

            let self_attn = AttentionWeights {
                q_proj: LinearWeights::new(cfg.encoder_dim, kv_dim, &avb.pp("q_proj"), metal)?,
                k_proj: LinearWeights::new(cfg.encoder_dim, kv_dim, &avb.pp("k_proj"), metal)?,
                v_proj: LinearWeights::new(cfg.encoder_dim, kv_dim, &avb.pp("v_proj"), metal)?,
                o_proj: LinearWeights::new(kv_dim, cfg.encoder_dim, &avb.pp("o_proj"), metal)?,
                num_heads: cfg.encoder_num_heads,
                num_kv_heads: cfg.encoder_num_kv_heads,
                head_dim: cfg.encoder_head_dim,
                scale: (cfg.encoder_head_dim as f64).powf(-0.5),
            };

            let mvb = lvb.pp("mlp");
            let mlp = MLPWeights {
                fc1: LinearBiasWeights::new(cfg.encoder_dim, cfg.encoder_intermediate_size, &mvb.pp("fc1"), metal)?,
                fc2: LinearBiasWeights::new(cfg.encoder_intermediate_size, cfg.encoder_dim, &mvb.pp("fc2"), metal)?,
            };

            layers.push(EncoderLayerWeights {
                self_attn,
                mlp,
                input_layernorm: LayerNormWeights::new(cfg.encoder_dim, &lvb.pp("input_layernorm"), metal)?,
                post_attention_layernorm: LayerNormWeights::new(cfg.encoder_dim, &lvb.pp("post_attention_layernorm"), metal)?,
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

        for (i, layer) in self.layers.iter().enumerate() {
            let [left, right] = self.sliding_windows[i];

            let kv_dim = layer.self_attn.num_kv_heads * layer.self_attn.head_dim;
            let nh = layer.self_attn.num_heads;
            let hd = layer.self_attn.head_dim;
            let n_elem = padded_seq * dim;

            // ── Pre-norm (GPU) ──
            let residual = hidden.clone();
            let normed = triton_layernorm_unit_offset(
                dev, &k.layernorm_unit_offset,
                &hidden, &layer.input_layernorm.gamma,
                padded_seq, dim,
            )?;

            // ── Q/K/V projections (GPU matmul) ──
            let q = triton_matmul(dev, matmul_pipeline, &normed, &layer.self_attn.q_proj.weight, padded_seq, kv_dim, dim, block_m, block_m)?;
            let kk = triton_matmul(dev, matmul_pipeline, &normed, &layer.self_attn.k_proj.weight, padded_seq, kv_dim, dim, block_m, block_m)?;
            let v = triton_matmul(dev, matmul_pipeline, &normed, &layer.self_attn.v_proj.weight, padded_seq, kv_dim, dim, block_m, block_m)?;

            // ── Flash Attention (GPU) ──
            // Layout: [padded_seq, kv_dim] = [padded_seq, nh*hd], interpreted as [nh, padded_seq, hd]
            // stride_h = padded_seq * hd (skipping hd elements between consecutive head blocks? No...)
            // Actually layout is [padded_seq, nh, hd] with stride_m = nh*hd = kv_dim, stride_h = hd
            let attn_out = empty_f16(dev, (padded_seq, kv_dim))?;
            let sm_scale = layer.self_attn.scale as f32;
            triton_flash_attention(
                dev, &k.flash_attention,
                &q, &kk, &v, &attn_out,
                nh, padded_seq, hd,
                hd as i32,      // stride_h: offset between heads = hd
                kv_dim as i32,  // stride_m: offset between rows = nh * hd
                sm_scale,
                left as i32, right as i32,
            )?;

            // ── O projection (GPU matmul) ──
            let attn_proj = triton_matmul(dev, matmul_pipeline, &attn_out, &layer.self_attn.o_proj.weight, padded_seq, dim, kv_dim, block_m, block_m)?;

            // ── Residual add (GPU) ──
            hidden = triton_residual_add(dev, &k.residual_add, &attn_proj, &residual, n_elem)?;

            // ── Post-norm (GPU) ──
            let residual = hidden.clone();
            let normed = triton_layernorm_unit_offset(
                dev, &k.layernorm_unit_offset,
                &hidden, &layer.post_attention_layernorm.gamma,
                padded_seq, dim,
            )?;

            // ── FFN: matmul → bias_add → gelu (GPU) ──
            let fc1_dim = layer.mlp.fc1.out_dim;
            let fc1 = triton_matmul(dev, matmul_pipeline, &normed, &layer.mlp.fc1.weight, padded_seq, fc1_dim, layer.mlp.fc1.in_dim, block_m, block_m)?;
            let fc1 = triton_bias_add(dev, &k.bias_add, &fc1, &layer.mlp.fc1.bias, padded_seq * fc1_dim, fc1_dim)?;
            let fc1 = triton_gelu(dev, &k.gelu, &fc1, padded_seq * fc1_dim)?;

            // ── FC2: matmul → bias_add (GPU) ──
            let fc2 = triton_matmul(dev, matmul_pipeline, &fc1, &layer.mlp.fc2.weight, padded_seq, layer.mlp.fc2.out_dim, layer.mlp.fc2.in_dim, block_m, block_m)?;
            let fc2 = triton_bias_add(dev, &k.bias_add, &fc2, &layer.mlp.fc2.bias, n_elem, dim)?;

            // ── Residual add (GPU) ──
            hidden = triton_residual_add(dev, &k.residual_add, &fc2, &residual, n_elem)?;
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
}
