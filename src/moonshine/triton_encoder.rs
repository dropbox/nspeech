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
    TritonKernels, empty_f16, triton_flash_attention, triton_layernorm_unit_offset, triton_matmul,
    triton_residual_add,
};

fn cdiv(a: usize, b: usize) -> usize {
    (a + b - 1) / b
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
    /// Input: `[1, seq_len, encoder_dim]` F16 from frontend.
    /// Output: `[1, seq_len, encoder_dim]` F16 encoded features.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, dim) = x.dims3()?;
        assert_eq!(batch, 1, "TritonEncoder only supports batch=1");
        assert_eq!(dim, self.encoder_dim);

        let dev = &self.metal_device;

        // Flatten to [seq_len, dim] for 2D matmul dispatch.
        // Pad seq_len to multiple of 64 for largest tile size.
        let block_m = 64;
        let padded_seq = cdiv(seq_len, block_m) * block_m;
        let mut hidden = x.reshape((seq_len, dim))?.to_dtype(DType::F16)?;
        if padded_seq > seq_len {
            let pad = Tensor::zeros((padded_seq - seq_len, dim), DType::F16, x.device())?;
            hidden = Tensor::cat(&[&hidden, &pad], 0)?;
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let [left, right] = self.sliding_windows[i];

            // ── Pre-norm ──
            let residual = hidden.clone();
            let normed = triton_layernorm_unit_offset(
                dev, &self.kernels.layernorm_unit_offset,
                &hidden, &layer.input_layernorm.gamma,
                padded_seq, dim,
            )?;

            // ── Self-attention (Q/K/V projections via Triton, attention via Candle) ──
            let kv_dim = layer.self_attn.num_kv_heads * layer.self_attn.head_dim;

            let q = triton_matmul(
                dev, &self.kernels.matmul_64x64,
                &normed, &layer.self_attn.q_proj.weight,
                padded_seq, kv_dim, dim,
            )?;
            let k = triton_matmul(
                dev, &self.kernels.matmul_64x64,
                &normed, &layer.self_attn.k_proj.weight,
                padded_seq, kv_dim, dim,
            )?;
            let v = triton_matmul(
                dev, &self.kernels.matmul_64x64,
                &normed, &layer.self_attn.v_proj.weight,
                padded_seq, kv_dim, dim,
            )?;

            // Q/K/V are [padded_seq, kv_dim] = [padded_seq, n_heads * head_dim].
            // The FA2 kernel supports arbitrary strides, so use the
            // [seq, n_heads, D] layout directly — no transpose/copy needed.
            let nh = layer.self_attn.num_heads;
            let hd = layer.self_attn.head_dim;

            // stride_h = D (adjacent heads), stride_m = kv_dim (adjacent rows)
            let stride_h = hd as i32;
            let stride_m = kv_dim as i32;
            let sm_scale = layer.self_attn.scale as f32;

            // Allocate output as [padded_seq, kv_dim] — same layout as Q/K/V
            let attn_flat = empty_f16(dev, (padded_seq, kv_dim))?;

            // Flash Attention 2: fused QK^T, softmax+mask, attn*V — all in F16
            triton_flash_attention(
                dev, &self.kernels.flash_attention,
                &q, &k, &v, &attn_flat,
                nh, seq_len, hd,
                stride_h, stride_m,
                sm_scale,
                left as i32, right as i32,
            )?;

            let attn_proj = triton_matmul(
                dev, &self.kernels.matmul_64x64,
                &attn_flat, &layer.self_attn.o_proj.weight,
                padded_seq, dim, kv_dim,
            )?;

            // ── Residual ──
            hidden = triton_residual_add(
                dev, &self.kernels.residual_add,
                &attn_proj, &residual,
                padded_seq * dim,
            )?;

            // ── Post-norm ──
            let residual = hidden.clone();
            let normed = triton_layernorm_unit_offset(
                dev, &self.kernels.layernorm_unit_offset,
                &hidden, &layer.post_attention_layernorm.gamma,
                padded_seq, dim,
            )?;

            // ── FFN ──
            // Use 64×64 matmul (2× faster than 32×32) + separate bias/GELU.
            // The bias-fused matmul exceeds 32KB threadgroup memory at 64×64,
            // but bias add + GELU are O(M*N) vs O(M*N*K) — negligible cost.
            let fc1_out = triton_matmul(
                dev, &self.kernels.matmul_64x64,
                &normed, &layer.mlp.fc1.weight,
                padded_seq, layer.mlp.fc1.out_dim, layer.mlp.fc1.in_dim,
            )?;
            let fc1_out = fc1_out.broadcast_add(&layer.mlp.fc1.bias)?.gelu_erf()?;
            let fc2_out = triton_matmul(
                dev, &self.kernels.matmul_64x64,
                &fc1_out, &layer.mlp.fc2.weight,
                padded_seq, layer.mlp.fc2.out_dim, layer.mlp.fc2.in_dim,
            )?;
            let fc2_out = fc2_out.broadcast_add(&layer.mlp.fc2.bias)?;

            // ── Residual ──
            hidden = triton_residual_add(
                dev, &self.kernels.residual_add,
                &fc2_out, &residual,
                padded_seq * dim,
            )?;
        }

        // Final norm
        let out = triton_layernorm_unit_offset(
            dev, &self.kernels.layernorm_unit_offset,
            &hidden, &self.final_norm.gamma,
            padded_seq, dim,
        )?;

        // Slice back to seq_len and reshape to [1, seq_len, dim]
        let out = out.narrow(0, 0, seq_len)?;
        Ok(out.reshape((1, seq_len, dim))?)
    }
}
