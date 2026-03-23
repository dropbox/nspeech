//! Moonshine V2 Encoder using Triton-compiled Metal kernels.
//!
//! Matmul and Flash Attention run on Metal GPU via Triton kernels.
//! All other ops (layernorm, GELU, residual add, bias) run on CPU for correctness
//! on Intel GPUs where candle's Metal backend has limited op support.
//!
//! Weights are stored as F16 tensors on the Metal device (dequantized from GGUF
//! at load time). Activations shuttle between CPU (F32) and Metal (F16).

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::config::MoonshineConfig;
use crate::triton_kernels::{
    TritonKernels, triton_matmul, triton_flash_attention, empty_f16,
};

fn cdiv(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Unit-offset layernorm in F32 on CPU: LN(x) * (gamma + 1.0)
fn unit_offset_layernorm(x: &Tensor, gamma: &Tensor) -> Result<Tensor> {
    let eps = 1e-5f64;
    let gamma_f32 = gamma.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(1)?;
    let x_centered = x.broadcast_sub(&mean)?;
    let var = (&x_centered * &x_centered)?.mean_keepdim(1)?;
    let inv_std = (var + eps)?.sqrt()?.recip()?;
    let normed = x_centered.broadcast_mul(&inv_std)?;
    let scale = (gamma_f32 + 1.0f64)?;
    Ok(normed.broadcast_mul(&scale)?)
}

type QVarBuilder = candle_transformers::quantized_var_builder::VarBuilder;

/// Load weight from GGUF, dequantize to F16 on Metal.
fn load_f16_weight(shape: (usize, usize), vb: &QVarBuilder, metal: &Device) -> Result<Tensor> {
    let qt = vb.get(shape, "weight")?;
    let t = qt.dequantize(&Device::Cpu)?;
    Ok(t.to_dtype(DType::F16)?.to_device(metal)?.t()?.contiguous()?)
}

/// Load 1D parameter from GGUF, keep on CPU as F32.
fn load_f32_1d(dim: usize, name: &str, vb: &QVarBuilder) -> Result<Tensor> {
    let qt = vb.get(dim, name)?;
    let t = qt.dequantize(&Device::Cpu)?;
    Ok(t.to_dtype(DType::F32)?)
}

/// Load 1D parameter from GGUF, dequantize to F16 on Metal.
fn load_f16_1d(dim: usize, name: &str, vb: &QVarBuilder, metal: &Device) -> Result<Tensor> {
    let qt = vb.get(dim, name)?;
    let t = qt.dequantize(&Device::Cpu)?;
    Ok(t.to_dtype(DType::F16)?.to_device(metal)?)
}

struct LayerNormWeights {
    gamma: Tensor, // [dim] F32 on CPU (for CPU layernorm)
}

impl LayerNormWeights {
    fn new(dim: usize, vb: &QVarBuilder) -> Result<Self> {
        Ok(Self { gamma: load_f32_1d(dim, "gamma", vb)? })
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
    bias: Tensor,   // [out_dim] F32 on CPU
    in_dim: usize,
    out_dim: usize,
}

impl LinearBiasWeights {
    fn new(in_dim: usize, out_dim: usize, vb: &QVarBuilder, metal: &Device) -> Result<Self> {
        Ok(Self {
            weight: load_f16_weight((out_dim, in_dim), vb, metal)?,
            bias: load_f32_1d(out_dim, "bias", vb)?,
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

/// Triton-accelerated Moonshine encoder.
///
/// Hybrid execution: matmul/FA2 on Metal GPU, everything else on CPU.
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
                input_layernorm: LayerNormWeights::new(cfg.encoder_dim, &lvb.pp("input_layernorm"))?,
                post_attention_layernorm: LayerNormWeights::new(cfg.encoder_dim, &lvb.pp("post_attention_layernorm"))?,
            });
        }

        let final_norm = LayerNormWeights::new(cfg.encoder_dim, &vb.pp("final_norm"))?;

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

    /// Triton matmul on Metal: CPU F32 input → Metal F16 → matmul → CPU F16 result.
    fn gpu_matmul(&self, input_f32: &Tensor, weight: &Tensor, m: usize, n: usize, k: usize) -> Result<Tensor> {
        let input_f16 = input_f32.to_dtype(DType::F16)?.to_device(&self.metal_candle_device)?;
        let out_f16 = triton_matmul(
            &self.metal_device, &self.kernels.matmul_64x64,
            &input_f16, weight, m, n, k,
        )?;
        // Return F16 on CPU — caller converts to F32 as needed
        Ok(out_f16.to_device(&Device::Cpu)?)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, dim) = x.dims3()?;
        assert_eq!(batch, 1, "TritonEncoder only supports batch=1");
        assert_eq!(dim, self.encoder_dim);

        let block_m = 64;
        let padded_seq = cdiv(seq_len, block_m) * block_m;

        // Everything on CPU in F32
        let mut hidden = x.reshape((seq_len, dim))?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        if padded_seq > seq_len {
            let pad = Tensor::zeros((padded_seq - seq_len, dim), DType::F32, &Device::Cpu)?;
            hidden = Tensor::cat(&[&hidden, &pad], 0)?;
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let [left, right] = self.sliding_windows[i];

            // ── Pre-norm (CPU F32) ──
            let residual = hidden.clone();
            let normed = unit_offset_layernorm(&hidden, &layer.input_layernorm.gamma)?;

            // ── Q/K/V projections (GPU matmul) ──
            let kv_dim = layer.self_attn.num_kv_heads * layer.self_attn.head_dim;
            let nh = layer.self_attn.num_heads;
            let hd = layer.self_attn.head_dim;

            let q_f16 = self.gpu_matmul(&normed, &layer.self_attn.q_proj.weight, padded_seq, kv_dim, dim)?;
            let k_f16 = self.gpu_matmul(&normed, &layer.self_attn.k_proj.weight, padded_seq, kv_dim, dim)?;
            let v_f16 = self.gpu_matmul(&normed, &layer.self_attn.v_proj.weight, padded_seq, kv_dim, dim)?;

            // ── Flash Attention (GPU) ──
            let q_metal = q_f16.to_device(&self.metal_candle_device)?;
            let k_metal = k_f16.to_device(&self.metal_candle_device)?;
            let v_metal = v_f16.to_device(&self.metal_candle_device)?;
            let attn_out = empty_f16(&self.metal_device, (padded_seq, kv_dim))?;
            let sm_scale = layer.self_attn.scale as f32;
            triton_flash_attention(
                &self.metal_device, &self.kernels.flash_attention,
                &q_metal, &k_metal, &v_metal, &attn_out,
                nh, padded_seq, hd,
                hd as i32,
                kv_dim as i32,
                sm_scale,
                left as i32, right as i32,
            )?;
            let attn_cpu = attn_out.to_device(&Device::Cpu)?;

            // ── O projection (GPU matmul) ──
            let attn_proj_f16 = self.gpu_matmul(
                &attn_cpu.to_dtype(DType::F32)?,
                &layer.self_attn.o_proj.weight,
                padded_seq, dim, kv_dim,
            )?;

            // ── Residual (CPU F32) ──
            hidden = (attn_proj_f16.to_dtype(DType::F32)? + residual)?;

            // ── Post-norm (CPU F32) ──
            let residual = hidden.clone();
            let normed = unit_offset_layernorm(&hidden, &layer.post_attention_layernorm.gamma)?;

            // ── FFN: FC1 (GPU) + bias + GELU (CPU) ──
            let fc1_f16 = self.gpu_matmul(&normed, &layer.mlp.fc1.weight, padded_seq, layer.mlp.fc1.out_dim, layer.mlp.fc1.in_dim)?;
            let fc1_out = fc1_f16.to_dtype(DType::F32)?
                .broadcast_add(&layer.mlp.fc1.bias)?
                .gelu_erf()?;

            // ── FC2 (GPU) + bias (CPU) ──
            let fc2_f16 = self.gpu_matmul(&fc1_out, &layer.mlp.fc2.weight, padded_seq, layer.mlp.fc2.out_dim, layer.mlp.fc2.in_dim)?;
            let fc2_out = fc2_f16.to_dtype(DType::F32)?
                .broadcast_add(&layer.mlp.fc2.bias)?;

            // ── Residual (CPU F32) ──
            hidden = (fc2_out + residual)?;
        }

        // Final norm (CPU F32)
        let out = unit_offset_layernorm(&hidden, &self.final_norm.gamma)?;
        let out = out.narrow(0, 0, seq_len)?;
        Ok(out.reshape((1, seq_len, dim))?)
    }
}
