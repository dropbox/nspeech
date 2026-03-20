//! Moonshine V2 Encoder using Triton-compiled Metal kernels.
//!
//! All operations use pre-compiled Triton kernels dispatched directly via Metal:
//! linear projections (matmul), layernorm, residual add, and Flash Attention 2
//! (fused QK^T + softmax + sliding window mask + attn*V in a single kernel).
//!
//! Weights are stored as F16 tensors on the Metal device (dequantized from GGUF
//! at load time).

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::config::MoonshineConfig;
use crate::triton_kernels::{
    TritonKernels, triton_matmul, triton_flash_attention, empty_f16,
};

fn cdiv(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Unit-offset layernorm in F32: LN(x) * (gamma + 1.0)
/// x: [rows, cols] F32, gamma: [cols] F16 → output [rows, cols] F32
fn unit_offset_layernorm(x: &Tensor, gamma: &Tensor) -> Result<Tensor> {
    let eps = 1e-5f64;
    let mean = x.mean_keepdim(1)?;
    let x_centered = x.broadcast_sub(&mean)?;
    let var = (&x_centered * &x_centered)?.mean_keepdim(1)?;
    let inv_std = (var + eps)?.sqrt()?.recip()?;
    let normed = x_centered.broadcast_mul(&inv_std)?;
    let scale = (gamma.to_dtype(DType::F32)? + 1.0f64)?;
    Ok(normed.broadcast_mul(&scale)?)
}

type QVarBuilder = candle_transformers::quantized_var_builder::VarBuilder;

/// Load weight from GGUF, dequantize to F16 on the Metal device.
/// Returns weight transposed to [in_dim, out_dim] for Triton matmul kernels
/// which require B in [K, N] row-major layout (N contiguous for vec4 loads).
fn load_f16_weight(shape: (usize, usize), vb: &QVarBuilder) -> Result<Tensor> {
    let qt = vb.get(shape, "weight")?;
    let t = qt.dequantize(vb.device())?;
    Ok(t.to_dtype(DType::F16)?.t()?.contiguous()?)
}

/// Load 1D parameter (bias, gamma) from GGUF, dequantize to F16.
fn load_f16_1d(dim: usize, name: &str, vb: &QVarBuilder) -> Result<Tensor> {
    let qt = vb.get(dim, name)?;
    let t = qt.dequantize(vb.device())?;
    Ok(t.to_dtype(DType::F16)?)
}

/// Unit-offset LayerNorm weights (gamma only, no bias).
struct LayerNormWeights {
    gamma: Tensor, // [dim] F16
    #[allow(dead_code)]
    dim: usize,
}

impl LayerNormWeights {
    fn new(dim: usize, vb: &QVarBuilder) -> Result<Self> {
        let gamma = load_f16_1d(dim, "gamma", vb)?;
        Ok(Self { gamma, dim })
    }
}

/// Linear projection weights (no bias).
struct LinearWeights {
    weight: Tensor, // [out_dim, in_dim] F16
    #[allow(dead_code)]
    in_dim: usize,
    #[allow(dead_code)]
    out_dim: usize,
}

impl LinearWeights {
    fn new(in_dim: usize, out_dim: usize, vb: &QVarBuilder) -> Result<Self> {
        let weight = load_f16_weight((out_dim, in_dim), vb)?;
        Ok(Self { weight, in_dim, out_dim })
    }
}

/// Linear + bias weights.
struct LinearBiasWeights {
    weight: Tensor, // [out_dim, in_dim] F16
    bias: Tensor,   // [out_dim] F16
    in_dim: usize,
    out_dim: usize,
}

impl LinearBiasWeights {
    fn new(in_dim: usize, out_dim: usize, vb: &QVarBuilder) -> Result<Self> {
        let weight = load_f16_weight((out_dim, in_dim), vb)?;
        let bias = load_f16_1d(out_dim, "bias", vb)?;
        Ok(Self { weight, bias, in_dim, out_dim })
    }
}

/// Attention weights for one encoder layer.
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

/// MLP weights for one encoder layer.
struct MLPWeights {
    fc1: LinearBiasWeights,
    fc2: LinearBiasWeights,
}

/// One encoder layer's weights and config.
struct EncoderLayerWeights {
    self_attn: AttentionWeights,
    mlp: MLPWeights,
    input_layernorm: LayerNormWeights,
    post_attention_layernorm: LayerNormWeights,
}

/// Triton-accelerated Moonshine encoder.
pub struct TritonEncoder {
    kernels: TritonKernels,
    layers: Vec<EncoderLayerWeights>,
    final_norm: LayerNormWeights,
    sliding_windows: Vec<[usize; 2]>,
    metal_device: candle_core::MetalDevice,
    encoder_dim: usize,
}

impl TritonEncoder {
    /// Load encoder weights from GGUF and compile Triton kernel pipelines.
    pub fn new(
        cfg: &MoonshineConfig,
        vb: QVarBuilder,
        kernel_dir: &std::path::Path,
    ) -> Result<Self> {
        let device = vb.device();
        let metal_device = match device {
            Device::Metal(md) => md.clone(),
            _ => anyhow::bail!("TritonEncoder requires Metal device"),
        };

        println!("  Loading Triton kernel pipelines...");
        let kernels = TritonKernels::load(&metal_device, kernel_dir)?;

        let kv_dim = cfg.encoder_num_kv_heads * cfg.encoder_head_dim;

        let mut layers = Vec::with_capacity(cfg.encoder_num_layers);
        for i in 0..cfg.encoder_num_layers {
            let lvb = vb.pp(&format!("layers.{i}"));
            let avb = lvb.pp("self_attn");

            let self_attn = AttentionWeights {
                q_proj: LinearWeights::new(cfg.encoder_dim, kv_dim, &avb.pp("q_proj"))?,
                k_proj: LinearWeights::new(cfg.encoder_dim, kv_dim, &avb.pp("k_proj"))?,
                v_proj: LinearWeights::new(cfg.encoder_dim, kv_dim, &avb.pp("v_proj"))?,
                o_proj: LinearWeights::new(kv_dim, cfg.encoder_dim, &avb.pp("o_proj"))?,
                num_heads: cfg.encoder_num_heads,
                num_kv_heads: cfg.encoder_num_kv_heads,
                head_dim: cfg.encoder_head_dim,
                scale: (cfg.encoder_head_dim as f64).powf(-0.5),
            };

            let mvb = lvb.pp("mlp");
            let mlp = MLPWeights {
                fc1: LinearBiasWeights::new(cfg.encoder_dim, cfg.encoder_intermediate_size, &mvb.pp("fc1"))?,
                fc2: LinearBiasWeights::new(cfg.encoder_intermediate_size, cfg.encoder_dim, &mvb.pp("fc2"))?,
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
            encoder_dim: cfg.encoder_dim,
        })
    }

    /// Run encoder forward pass.
    ///
    /// Input: `[1, seq_len, encoder_dim]` from frontend (any dtype).
    /// Output: `[1, seq_len, encoder_dim]` F32 encoded features.
    ///
    /// Matmul and FA2 run in F16 via Triton kernels (compute-bound).
    /// Residual stream, layernorm, bias, and GELU run in F32 via Candle (bandwidth-bound)
    /// to maintain precision over 14 layers.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, dim) = x.dims3()?;
        assert_eq!(batch, 1, "TritonEncoder only supports batch=1");
        assert_eq!(dim, self.encoder_dim);

        let dev = &self.metal_device;

        // Flatten to [seq_len, dim] for 2D matmul dispatch.
        // Pad seq_len to multiple of 64 for largest tile size.
        let block_m = 64;
        let padded_seq = cdiv(seq_len, block_m) * block_m;

        // Hidden state stays in F32 for precision; converted to F16 before each matmul.
        let mut hidden = x.reshape((seq_len, dim))?.to_dtype(DType::F32)?;
        if padded_seq > seq_len {
            let pad = Tensor::zeros((padded_seq - seq_len, dim), DType::F32, x.device())?;
            hidden = Tensor::cat(&[&hidden, &pad], 0)?;
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let [left, right] = self.sliding_windows[i];

            // ── Pre-norm (F32 via Candle) ──
            let residual = hidden.clone();
            let normed = unit_offset_layernorm(&hidden, &layer.input_layernorm.gamma)?;

            // ── Self-attention (Triton matmul for projections, Candle for attention) ──
            let kv_dim = layer.self_attn.num_kv_heads * layer.self_attn.head_dim;
            let nh = layer.self_attn.num_heads;
            let hd = layer.self_attn.head_dim;
            let normed_f16 = normed.to_dtype(DType::F16)?;

            let q = triton_matmul(
                dev, &self.kernels.matmul_64x64,
                &normed_f16, &layer.self_attn.q_proj.weight,
                padded_seq, kv_dim, dim,
            )?;
            let k = triton_matmul(
                dev, &self.kernels.matmul_64x64,
                &normed_f16, &layer.self_attn.k_proj.weight,
                padded_seq, kv_dim, dim,
            )?;
            let v = triton_matmul(
                dev, &self.kernels.matmul_64x64,
                &normed_f16, &layer.self_attn.v_proj.weight,
                padded_seq, kv_dim, dim,
            )?;

            // FA2 kernel: fused QK^T + softmax + sliding window + attn*V
            // Q, K, V are [padded_seq, kv_dim] F16 with interleaved heads
            let attn_out = empty_f16(dev, (padded_seq, kv_dim))?;
            let sm_scale = layer.self_attn.scale as f32;
            triton_flash_attention(
                dev, &self.kernels.flash_attention,
                &q, &k, &v, &attn_out,
                nh, padded_seq, hd,
                hd as i32,       // stride_h: head_dim
                kv_dim as i32,   // stride_m: row stride
                sm_scale,
                left as i32, right as i32,
            )?;
            let attn_flat = attn_out;

            let attn_proj = triton_matmul(
                dev, &self.kernels.matmul_64x64,
                &attn_flat, &layer.self_attn.o_proj.weight,
                padded_seq, dim, kv_dim,
            )?;

            // ── Residual (F32) ──
            hidden = (attn_proj.to_dtype(DType::F32)? + residual)?;

            // ── Post-norm (F32 via Candle) ──
            let residual = hidden.clone();
            let normed = unit_offset_layernorm(&hidden, &layer.post_attention_layernorm.gamma)?;

            // ── FFN (Triton matmul in F16, bias+GELU in F32) ──
            let normed_f16 = normed.to_dtype(DType::F16)?;
            let fc1_out = triton_matmul(
                dev, &self.kernels.matmul_64x64,
                &normed_f16, &layer.mlp.fc1.weight,
                padded_seq, layer.mlp.fc1.out_dim, layer.mlp.fc1.in_dim,
            )?;
            let fc1_out = fc1_out.to_dtype(DType::F32)?
                .broadcast_add(&layer.mlp.fc1.bias.to_dtype(DType::F32)?)?
                .gelu_erf()?;
            let fc1_f16 = fc1_out.to_dtype(DType::F16)?;
            let fc2_out = triton_matmul(
                dev, &self.kernels.matmul_64x64,
                &fc1_f16, &layer.mlp.fc2.weight,
                padded_seq, layer.mlp.fc2.out_dim, layer.mlp.fc2.in_dim,
            )?;
            let fc2_out = fc2_out.to_dtype(DType::F32)?
                .broadcast_add(&layer.mlp.fc2.bias.to_dtype(DType::F32)?)?;

            // ── Residual (F32) ──
            hidden = (fc2_out + residual)?;
        }

        // Final norm (F32)
        let out = unit_offset_layernorm(&hidden, &self.final_norm.gamma)?;

        // Slice back to seq_len and reshape to [1, seq_len, dim]
        let out = out.narrow(0, 0, seq_len)?;
        Ok(out.reshape((1, seq_len, dim))?)
    }
}
