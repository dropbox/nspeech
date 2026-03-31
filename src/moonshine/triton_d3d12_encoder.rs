//! Moonshine V2 Encoder using Triton-compiled HLSL kernels on D3D12.
//!
//! All operations (matmul, layernorm, GELU, residual add) run on the D3D12 GPU.
//! Weights are stored as F16 in GPU buffers. Activations stay on GPU throughout.
//! Only a single GPU→CPU sync occurs at the very end of forward().

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_d3d12_kernels::{Gpu, GpuBuffer};
use std::sync::Arc;

use super::config::MoonshineConfig;
use crate::triton_d3d12_kernels::{
    TritonD3D12Kernels, create_f16_buffer, create_f32_buffer,
    download_f16, upload_f16, upload_f32,
    triton_d3d12_layernorm_f32in, triton_d3d12_matmul_f32w,
    triton_d3d12_matmul_bias_f32w, triton_d3d12_gelu,
    triton_d3d12_residual_add_f32,
    triton_d3d12_flash_attention,
};

fn cdiv(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

type QVarBuilder = candle_transformers::quantized_var_builder::VarBuilder;

/// Upload a 1D tensor from GGUF → dequantize → F16 → GPU buffer.
fn load_f16_1d_to_gpu(gpu: &Gpu, dim: usize, name: &str, vb: &QVarBuilder) -> Result<GpuBuffer> {
    let qt = vb.get(dim, name)?;
    let t = qt.dequantize(&Device::Cpu)?;
    let t = t.to_dtype(DType::F16)?;
    let data = t.to_vec1::<half::f16>()?;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = create_f16_buffer(gpu, dim)?;
    upload_f16(gpu, &bytes, &buf)?;
    Ok(buf)
}

/// Upload a 2D tensor from GGUF → dequantize → F32 → GPU buffer.
/// Preserves full Q8_0 dequantization precision (no f16 truncation).
fn load_f32_to_gpu(
    gpu: &Gpu,
    shape: (usize, usize),
    vb: &QVarBuilder,
    transpose: bool,
) -> Result<GpuBuffer> {
    let qt = vb.get(shape, "weight")?;
    let t = qt.dequantize(&Device::Cpu)?;
    let t = if transpose { t.t()?.contiguous()? } else { t };

    let data = t.to_vec2::<f32>()?;
    let bytes: Vec<u8> = data.iter().flat_map(|row| {
        row.iter().flat_map(|v| v.to_le_bytes())
    }).collect();

    let buf = create_f32_buffer(gpu, shape.0 * shape.1)?;
    upload_f32(gpu, &bytes, &buf)?;
    Ok(buf)
}

/// Upload a 1D tensor from GGUF → dequantize → F32 → GPU buffer.
fn load_f32_1d_to_gpu(gpu: &Gpu, dim: usize, name: &str, vb: &QVarBuilder) -> Result<GpuBuffer> {
    let qt = vb.get(dim, name)?;
    let t = qt.dequantize(&Device::Cpu)?;
    let data = t.to_vec1::<f32>()?;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = create_f32_buffer(gpu, dim)?;
    upload_f32(gpu, &bytes, &buf)?;
    Ok(buf)
}

struct LayerNormWeights {
    gamma: GpuBuffer,
}

struct LinearWeights {
    weight: GpuBuffer, // [in_dim, out_dim] transposed for matmul
    #[allow(dead_code)]
    in_dim: usize,
    #[allow(dead_code)]
    out_dim: usize,
}

struct LinearBiasWeights {
    weight: GpuBuffer,
    bias: GpuBuffer,
    in_dim: usize,
    out_dim: usize,
}

struct AttentionWeights {
    q_proj: LinearWeights,
    k_proj: LinearWeights,
    v_proj: LinearWeights,
    o_proj: LinearWeights,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
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

/// Triton-accelerated Moonshine encoder on D3D12.
pub struct TritonD3D12Encoder {
    gpu: Arc<Gpu>,
    kernels: TritonD3D12Kernels,
    layers: Vec<EncoderLayerWeights>,
    final_norm: LayerNormWeights,
    sliding_windows: Vec<[usize; 2]>,
    encoder_dim: usize,
}

impl TritonD3D12Encoder {
    pub fn new(cfg: &MoonshineConfig, vb: QVarBuilder, gpu: &Arc<Gpu>) -> Result<Self> {
        // Set USE_FP16_ACC=1 to use fp16 accumulation (2x throughput on Iris Xe)
        let use_fp16_acc = std::env::var("USE_FP16_ACC").map_or(false, |v| v == "1");
        println!("  Loading Triton DXIL kernels (fp16_acc={})...", use_fp16_acc);
        let kernels = TritonD3D12Kernels::load(gpu, use_fp16_acc)?;

        let kv_dim = cfg.encoder_num_kv_heads * cfg.encoder_head_dim;

        let mut layers = Vec::with_capacity(cfg.encoder_num_layers);
        for i in 0..cfg.encoder_num_layers {
            let lvb = vb.pp(&format!("layers.{i}"));
            let avb = lvb.pp("self_attn");

            let self_attn = AttentionWeights {
                q_proj: LinearWeights {
                    weight: load_f32_to_gpu(gpu, (kv_dim, cfg.encoder_dim), &avb.pp("q_proj"), true)?,
                    in_dim: cfg.encoder_dim,
                    out_dim: kv_dim,
                },
                k_proj: LinearWeights {
                    weight: load_f32_to_gpu(gpu, (kv_dim, cfg.encoder_dim), &avb.pp("k_proj"), true)?,
                    in_dim: cfg.encoder_dim,
                    out_dim: kv_dim,
                },
                v_proj: LinearWeights {
                    weight: load_f32_to_gpu(gpu, (kv_dim, cfg.encoder_dim), &avb.pp("v_proj"), true)?,
                    in_dim: cfg.encoder_dim,
                    out_dim: kv_dim,
                },
                o_proj: LinearWeights {
                    weight: load_f32_to_gpu(gpu, (cfg.encoder_dim, kv_dim), &avb.pp("o_proj"), true)?,
                    in_dim: kv_dim,
                    out_dim: cfg.encoder_dim,
                },
                num_heads: cfg.encoder_num_heads,
                head_dim: cfg.encoder_head_dim,
                scale: (cfg.encoder_head_dim as f32).powf(-0.5),
            };

            let mvb = lvb.pp("mlp");
            let mlp = MLPWeights {
                fc1: LinearBiasWeights {
                    weight: load_f32_to_gpu(gpu, (cfg.encoder_intermediate_size, cfg.encoder_dim), &mvb.pp("fc1"), true)?,
                    bias: load_f32_1d_to_gpu(gpu, cfg.encoder_intermediate_size, "bias", &mvb.pp("fc1"))?,
                    in_dim: cfg.encoder_dim,
                    out_dim: cfg.encoder_intermediate_size,
                },
                fc2: LinearBiasWeights {
                    weight: load_f32_to_gpu(gpu, (cfg.encoder_dim, cfg.encoder_intermediate_size), &mvb.pp("fc2"), true)?,
                    bias: load_f32_1d_to_gpu(gpu, cfg.encoder_dim, "bias", &mvb.pp("fc2"))?,
                    in_dim: cfg.encoder_intermediate_size,
                    out_dim: cfg.encoder_dim,
                },
            };

            layers.push(EncoderLayerWeights {
                self_attn,
                mlp,
                input_layernorm: LayerNormWeights {
                    gamma: load_f16_1d_to_gpu(gpu, cfg.encoder_dim, "gamma", &lvb.pp("input_layernorm"))?,
                },
                post_attention_layernorm: LayerNormWeights {
                    gamma: load_f16_1d_to_gpu(gpu, cfg.encoder_dim, "gamma", &lvb.pp("post_attention_layernorm"))?,
                },
            });
        }

        let final_norm = LayerNormWeights {
            gamma: load_f16_1d_to_gpu(gpu, cfg.encoder_dim, "gamma", &vb.pp("final_norm"))?,
        };

        Ok(Self {
            gpu: gpu.clone(),
            kernels,
            layers,
            final_norm,
            sliding_windows: cfg.sliding_windows.clone(),
            encoder_dim: cfg.encoder_dim,
        })
    }

    /// Run the encoder forward pass. Input: [1, seq_len, dim] F32 on CPU.
    /// Output: [1, seq_len, dim] F32 on CPU.
    ///
    /// Mixed precision: hidden state (residual stream) stays in f32 on GPU
    /// to prevent precision loss from compounding through 14 layers.
    /// Within-layer computation (matmul, FA2, GELU) uses f16.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, dim) = x.dims3()?;
        assert_eq!(batch, 1, "TritonD3D12Encoder only supports batch=1");
        assert_eq!(dim, self.encoder_dim);

        let block_m = 64;
        let padded_seq = cdiv(seq_len, block_m) * block_m;

        // Upload input: CPU F32 → GPU F32 (hidden state stays in f32)
        let x_flat = x.reshape((seq_len, dim))?;
        let mut f32_data = x_flat.to_vec2::<f32>()?;

        // Pad to multiple of block_m
        if padded_seq > seq_len {
            for _ in 0..(padded_seq - seq_len) {
                f32_data.push(vec![0.0f32; dim]);
            }
        }

        let hidden_bytes: Vec<u8> = f32_data.iter().flat_map(|row| {
            row.iter().flat_map(|v| v.to_le_bytes())
        }).collect();

        let mut hidden = create_f32_buffer(&self.gpu, padded_seq * dim)?;
        upload_f32(&self.gpu, &hidden_bytes, &hidden)?;

        let k = &self.kernels;

        for (_i, layer) in self.layers.iter().enumerate() {
            let kv_dim = layer.self_attn.num_heads * layer.self_attn.head_dim;
            let n_elem = padded_seq * dim;

            // ── Pre-norm: f32 hidden → f16 normed ──
            let normed = create_f16_buffer(&self.gpu, n_elem)?;
            triton_d3d12_layernorm_f32in(k, &hidden, &layer.input_layernorm.gamma, &normed, padded_seq, dim)?;

            // ── Q/K/V projections: f16_activation × f32_weight → f16 ──
            let q = create_f16_buffer(&self.gpu, padded_seq * kv_dim)?;
            let kk = create_f16_buffer(&self.gpu, padded_seq * kv_dim)?;
            let v = create_f16_buffer(&self.gpu, padded_seq * kv_dim)?;
            triton_d3d12_matmul_f32w(k, &normed, &layer.self_attn.q_proj.weight, &q, padded_seq, kv_dim, dim)?;
            triton_d3d12_matmul_f32w(k, &normed, &layer.self_attn.k_proj.weight, &kk, padded_seq, kv_dim, dim)?;
            triton_d3d12_matmul_f32w(k, &normed, &layer.self_attn.v_proj.weight, &v, padded_seq, kv_dim, dim)?;

            // ── Flash Attention 2 (f16) ──
            let [win_left, win_right] = self.sliding_windows[_i];
            let attn_out_buf = create_f16_buffer(&self.gpu, padded_seq * kv_dim)?;
            triton_d3d12_flash_attention(
                k, &q, &kk, &v, &attn_out_buf,
                layer.self_attn.num_heads, padded_seq, seq_len,
                layer.self_attn.head_dim, layer.self_attn.scale,
                win_left as i32, win_right as i32,
            )?;

            // ── O projection: f16_activation × f32_weight → f16 ──
            let attn_proj = create_f16_buffer(&self.gpu, n_elem)?;
            triton_d3d12_matmul_f32w(k, &attn_out_buf, &layer.self_attn.o_proj.weight, &attn_proj, padded_seq, dim, kv_dim)?;

            // ── Residual add: attn_proj_f16 + hidden_f32 → hidden_f32 ──
            let new_hidden = create_f32_buffer(&self.gpu, n_elem)?;
            triton_d3d12_residual_add_f32(k, &attn_proj, &hidden, &new_hidden, n_elem)?;
            hidden = new_hidden;

            // ── Post-norm: f32 hidden → f16 normed ──
            let normed2 = create_f16_buffer(&self.gpu, n_elem)?;
            triton_d3d12_layernorm_f32in(k, &hidden, &layer.post_attention_layernorm.gamma, &normed2, padded_seq, dim)?;

            // ── FFN: fc1 + bias → gelu → fc2 + bias (f16 act × f32 weight) ──
            let fc1_dim = layer.mlp.fc1.out_dim;
            let fc1_linear = create_f16_buffer(&self.gpu, padded_seq * fc1_dim)?;
            triton_d3d12_matmul_bias_f32w(k, &normed2, &layer.mlp.fc1.weight, &layer.mlp.fc1.bias, &fc1_linear, padded_seq, fc1_dim, layer.mlp.fc1.in_dim)?;

            let fc1_out = create_f16_buffer(&self.gpu, padded_seq * fc1_dim)?;
            triton_d3d12_gelu(k, &fc1_linear, &fc1_out, padded_seq * fc1_dim)?;

            let fc2_out = create_f16_buffer(&self.gpu, n_elem)?;
            triton_d3d12_matmul_bias_f32w(k, &fc1_out, &layer.mlp.fc2.weight, &layer.mlp.fc2.bias, &fc2_out, padded_seq, layer.mlp.fc2.out_dim, layer.mlp.fc2.in_dim)?;

            // ── Residual add: fc2_out_f16 + hidden_f32 → hidden_f32 ──
            let new_hidden = create_f32_buffer(&self.gpu, n_elem)?;
            triton_d3d12_residual_add_f32(k, &fc2_out, &hidden, &new_hidden, n_elem)?;
            hidden = new_hidden;
        }

        // ── Final layernorm: f32 hidden → f16 output ──
        let final_out = create_f16_buffer(&self.gpu, padded_seq * dim)?;
        triton_d3d12_layernorm_f32in(k, &hidden, &self.final_norm.gamma, &final_out, padded_seq, dim)?;

        // ── Download GPU → CPU, F16 → F32, trim padding ──
        let bytes = download_f16(&self.gpu, &final_out, padded_seq * dim)?;
        let f16_data: Vec<half::f16> = bytes.chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
            .collect();
        let f32_data: Vec<f32> = f16_data[..seq_len * dim]
            .iter()
            .map(|v| v.to_f32())
            .collect();

        Ok(Tensor::from_vec(f32_data, (1, seq_len, dim), &Device::Cpu)?)
    }

}
