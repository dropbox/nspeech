//! Moonshine V2 Decoder using Triton-compiled HLSL kernels on D3D12.
//!
//! All decoder operations run on the D3D12 GPU. Weights stored as F32 on GPU.
//! Mixed precision: F32 residual stream, F16 within-layer computation.
//! Only GPU→CPU transfers: final logits download for argmax (once per token).
//!
//! Architecture per layer:
//!  - Pre-norm (standard LayerNorm) → self-attention with RoPE → residual
//!  - Post-norm → cross-attention → residual
//!  - Final-norm → GLU MLP (fc1 → chunk → silu·x → fc2) → residual

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_d3d12_kernels::{BufferBinding, Gpu, GpuBuffer, ID3D12PipelineState};
use std::sync::Arc;

use super::config::MoonshineConfig;
use crate::triton_d3d12_kernels::{
    create_f16_buffer, create_f32_buffer, upload_f16, upload_f32,
};

type QVarBuilder = candle_transformers::quantized_var_builder::VarBuilder;

fn cdiv(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}


fn i32_as_u32(v: i32) -> u32 {
    v as u32
}

fn uav_f16<'a>(buf: &'a GpuBuffer, count: u32) -> BufferBinding<'a> {
    BufferBinding::structured_f16(buf, count)
}

fn uav_f32<'a>(buf: &'a GpuBuffer, count: u32) -> BufferBinding<'a> {
    BufferBinding::structured_f32(buf, count)
}

// ── Embedded HLSL sources ──

// Triton-compiled kernels
const HLSL_MATMUL_F32W_64: &str =
    include_str!("../../kernels/out/hlsl/matmul_f16a_f32w_64x64x32.hlsl");
const HLSL_MATMUL_F32W_32: &str =
    include_str!("../../kernels/out/hlsl/matmul_f16a_f32w_32x32x32.hlsl");
const HLSL_MATMUL_BIAS_F32W_32: &str =
    include_str!("../../kernels/out/hlsl/matmul_bias_f16a_f32w_32x32x32.hlsl");
const HLSL_LAYERNORM_STD_F32IN: &str =
    include_str!("../../kernels/out/hlsl/layernorm_standard_f32in_640.hlsl");
const HLSL_RESIDUAL_ADD_F32: &str =
    include_str!("../../kernels/out/hlsl/residual_add_f32.hlsl");
const HLSL_ADD_BIAS_F32: &str =
    include_str!("../../kernels/out/hlsl/add_bias_f32_fp16.hlsl");
const HLSL_CONVERT_F32_TO_F16: &str =
    include_str!("../../kernels/out/hlsl/convert_f32_to_f16.hlsl");

// GEMV kernels — Triton-compiled, f16 weights (2x bandwidth vs f32)
const HLSL_GEMV_F16W: &str =
    include_str!("../../kernels/out/hlsl/gemv_f16w.hlsl");
const HLSL_GEMV_BIAS_F16W: &str =
    include_str!("../../kernels/out/hlsl/gemv_bias_f16w.hlsl");

// Triton-compiled decoder kernels (replacing hand-written HLSL)
const HLSL_ATTENTION_DECODE: &str =
    include_str!("../../kernels/out/hlsl/attention_decode_1d_d80.hlsl");
const HLSL_ROPE_INTERLEAVED: &str =
    include_str!("../../kernels/out/hlsl/rope_interleaved.hlsl");
const HLSL_KV_CACHE_APPEND: &str =
    include_str!("../../kernels/out/hlsl/kv_cache_append.hlsl");
const HLSL_ROPE_CACHE_FUSED: &str =
    include_str!("../../kernels/out/hlsl/rope_cache_fused.hlsl");
const HLSL_ROPE_QK_CACHE_FUSED: &str =
    include_str!("../../kernels/out/hlsl/rope_qk_cache_fused.hlsl");
const HLSL_GLU_SILU: &str =
    include_str!("../../kernels/out/hlsl/glu_silu_fused.hlsl");
const HLSL_RESIDUAL_ADD_LAYERNORM: &str =
    include_str!("../../kernels/out/hlsl/residual_add_layernorm_fused.hlsl");
const HLSL_GEMV_BIAS_GLU: &str =
    include_str!("../../kernels/out/hlsl/gemv_bias_glu_fused.hlsl");
const HLSL_GEMV_RESADD_LN: &str =
    include_str!("../../kernels/out/hlsl/gemv_resadd_ln_fused.hlsl");

// ── Kernel PSOs ──

#[allow(dead_code)]
struct DecoderKernels {
    gpu: Arc<Gpu>,
    matmul_f32w_64: ID3D12PipelineState,
    matmul_f32w_32: ID3D12PipelineState,
    matmul_bias_f32w_32: ID3D12PipelineState,
    gemv_f16w: ID3D12PipelineState,
    gemv_bias_f16w: ID3D12PipelineState,
    layernorm_std_f32in: ID3D12PipelineState,
    residual_add_f32: ID3D12PipelineState,
    add_bias_f32: ID3D12PipelineState,
    convert_f32_to_f16: ID3D12PipelineState,
    attention_decode: ID3D12PipelineState,
    rope_interleaved: ID3D12PipelineState,
    kv_cache_append: ID3D12PipelineState,
    rope_cache_fused: ID3D12PipelineState,
    rope_qk_cache_fused: ID3D12PipelineState,
    glu_silu: ID3D12PipelineState,
    residual_add_layernorm: ID3D12PipelineState,
    gemv_bias_glu: ID3D12PipelineState,
    gemv_resadd_ln: ID3D12PipelineState,
}

impl DecoderKernels {
    fn load(gpu: &Arc<Gpu>) -> Result<Self> {
        let compile = |name: &str, hlsl: &str, entry: &str| -> Result<ID3D12PipelineState> {
            eprint!("    {name}...");
            let bytecode = gpu
                .compile_shader_sm6(hlsl, entry)
                .map_err(|e| anyhow::anyhow!("Failed to compile {name}: {e}"))?;
            let pso = gpu
                .create_compute_pso(&bytecode)
                .map_err(|e| anyhow::anyhow!("Failed to create PSO for {name}: {e}"))?;
            eprintln!(" ok");
            Ok(pso)
        };

        Ok(Self {
            gpu: gpu.clone(),
            matmul_f32w_64: compile("dec_matmul_f32w_64", HLSL_MATMUL_F32W_64, "matmul_f16a_f32w")?,
            matmul_f32w_32: compile("dec_matmul_f32w_32", HLSL_MATMUL_F32W_32, "matmul_f16a_f32w")?,
            matmul_bias_f32w_32: compile("dec_matmul_bias_f32w_32", HLSL_MATMUL_BIAS_F32W_32, "matmul_bias_f16a_f32w")?,
            gemv_f16w: compile("dec_gemv_f16w", HLSL_GEMV_F16W, "gemv_f16w")?,
            gemv_bias_f16w: compile("dec_gemv_bias_f16w", HLSL_GEMV_BIAS_F16W, "gemv_bias_f16w")?,
            layernorm_std_f32in: compile("dec_layernorm_std_f32in", HLSL_LAYERNORM_STD_F32IN, "layernorm")?,
            residual_add_f32: compile("dec_residual_add_f32", HLSL_RESIDUAL_ADD_F32, "residual_add_f32")?,
            add_bias_f32: compile("dec_add_bias_f32", HLSL_ADD_BIAS_F32, "add_bias_f32")?,
            convert_f32_to_f16: compile("dec_convert_f32_to_f16", HLSL_CONVERT_F32_TO_F16, "convert_f32_to_f16")?,
            attention_decode: compile("dec_attention_decode", HLSL_ATTENTION_DECODE, "attention_decode_1d_masked")?,
            rope_interleaved: compile("dec_rope_interleaved", HLSL_ROPE_INTERLEAVED, "rope_interleaved")?,
            kv_cache_append: compile("dec_kv_cache_append", HLSL_KV_CACHE_APPEND, "kv_cache_append")?,
            rope_cache_fused: compile("dec_rope_cache_fused", HLSL_ROPE_CACHE_FUSED, "rope_cache_fused")?,
            rope_qk_cache_fused: compile("dec_rope_qk_cache_fused", HLSL_ROPE_QK_CACHE_FUSED, "rope_qk_cache_fused")?,
            glu_silu: compile("dec_glu_silu", HLSL_GLU_SILU, "glu_silu_fused")?,
            residual_add_layernorm: compile("dec_resadd_ln", HLSL_RESIDUAL_ADD_LAYERNORM, "residual_add_layernorm_fused")?,
            gemv_bias_glu: compile("dec_gemv_bias_glu", HLSL_GEMV_BIAS_GLU, "gemv_bias_glu_fused")?,
            gemv_resadd_ln: compile("dec_gemv_resadd_ln", HLSL_GEMV_RESADD_LN, "gemv_resadd_ln_fused")?,
        })
    }
}

// ── Dispatch helpers ──

/// GEMV: out_f16[N] = x_f16[K] @ W_f16[K,N]  (Triton-compiled, f16 weights)
fn dispatch_gemv_f16w(
    k: &DecoderKernels,
    x: &GpuBuffer, w: &GpuBuffer, out: &GpuBuffer,
    n: usize, kk: usize,
) -> Result<()> {
    let block_n = 128u32;
    let grid_x = cdiv(n, block_n as usize) as u32;
    // cbuffer: N, K, stride_wn, stride_wk, grid_dims — W stored [K,N] (transposed)
    let root_constants: Vec<u32> = vec![
        i32_as_u32(n as i32),
        i32_as_u32(kk as i32),
        1,                      // stride_wn = 1 (W[K,N] row-major, cols adjacent)
        i32_as_u32(n as i32),   // stride_wk = N
        grid_x, 1, 1,
    ];
    let uavs = [
        uav_f16(x, kk as u32),
        uav_f16(w, (kk * n) as u32),
        uav_f16(out, n as u32),
    ];
    k.gpu.record_dispatch(&k.gemv_f16w, &root_constants, &uavs, [grid_x, 1, 1])
        .map_err(|e| anyhow::anyhow!("gemv_f16w dispatch: {e}"))
}

/// Fused GEMV + residual-add + layernorm.
/// Replaces: O_proj + barrier + residual_add_layernorm.
fn dispatch_gemv_resadd_ln(
    k: &DecoderKernels,
    x: &GpuBuffer, w: &GpuBuffer,
    f32_res: &GpuBuffer, f32_out: &GpuBuffer,
    ln_weight: &GpuBuffer, f16_norm: &GpuBuffer,
    dim: usize, gemv_k: usize,
) -> Result<()> {
    let root_constants: Vec<u32> = vec![
        i32_as_u32(dim as i32),
        i32_as_u32(gemv_k as i32),
        1,                           // stride_wn = 1 (W[K,N])
        i32_as_u32(dim as i32),      // stride_wk = dim
        1, 1, 1,                     // grid_dim
    ];
    let uavs = [
        uav_f16(x, gemv_k as u32),
        uav_f16(w, (gemv_k * dim) as u32),
        uav_f32(f32_res, dim as u32),
        uav_f32(f32_out, dim as u32),
        uav_f16(ln_weight, dim as u32),
        uav_f16(f16_norm, dim as u32),
    ];
    k.gpu.record_dispatch(&k.gemv_resadd_ln, &root_constants, &uavs, [1, 1, 1])
        .map_err(|e| anyhow::anyhow!("gemv_resadd_ln dispatch: {e}"))
}

/// GEMV + bias: out_f16[N] = x_f16[K] @ W_f16[N,K] + bias_f32[N]
fn dispatch_gemv_bias_f16w(
    k: &DecoderKernels,
    x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
    n: usize, kk: usize,
) -> Result<()> {
    let block_n = 128u32;
    let grid_x = cdiv(n, block_n as usize) as u32;
    // cbuffer: N, K, stride_wn, stride_wk, grid_dims — W stored [K,N]
    let root_constants: Vec<u32> = vec![
        i32_as_u32(n as i32),
        i32_as_u32(kk as i32),
        1,                      // stride_wn = 1 (W[K,N])
        i32_as_u32(n as i32),   // stride_wk = N
        grid_x, 1, 1,
    ];
    let uavs = [
        uav_f16(x, kk as u32),
        uav_f16(w, (kk * n) as u32),
        uav_f32(bias, n as u32),
        uav_f16(out, n as u32),
    ];
    k.gpu.record_dispatch(&k.gemv_bias_f16w, &root_constants, &uavs, [grid_x, 1, 1])
        .map_err(|e| anyhow::anyhow!("gemv_bias_f16w dispatch: {e}"))
}

/// Matmul: C_f16[M,N] = A_f16[M,K] @ B_f32[K,N]
fn dispatch_matmul_f32w(
    k: &DecoderKernels,
    a: &GpuBuffer, b: &GpuBuffer, out: &GpuBuffer,
    m: usize, n: usize, kk: usize,
) -> Result<()> {
    // Select 32x32 for small M (GEMV), 64x64 for larger
    let (pso, bm, bn) = if m <= 32 {
        (&k.matmul_f32w_32, 32, 32)
    } else {
        (&k.matmul_f32w_64, 64, 64)
    };
    let grid_x = cdiv(m, bm) as u32;
    let grid_y = cdiv(n, bn) as u32;

    let root_constants: Vec<u32> = vec![
        i32_as_u32(m as i32), i32_as_u32(n as i32), i32_as_u32(kk as i32),
        i32_as_u32(kk as i32), 1, // stride_am=K, stride_ak=1
        i32_as_u32(n as i32), 1,  // stride_bk=N, stride_bn=1
        i32_as_u32(n as i32), 1,  // stride_cm=N, stride_cn=1
        grid_x, grid_y, 1,
    ];

    let uavs = [
        uav_f16(a, (m * kk) as u32),
        uav_f32(b, (kk * n) as u32),
        uav_f16(out, (m * n) as u32),
    ];

    k.gpu.record_dispatch(pso, &root_constants, &uavs, [grid_x, grid_y, 1])
        .map_err(|e| anyhow::anyhow!("matmul_f32w dispatch: {e}"))
}

/// Matmul+bias: C_f16[M,N] = A_f16[M,K] @ B_f32[K,N] + bias_f32[N]
#[allow(dead_code)]
fn dispatch_matmul_bias_f32w(
    k: &DecoderKernels,
    a: &GpuBuffer, b: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
    m: usize, n: usize, kk: usize,
) -> Result<()> {
    let (bm, bn) = (32, 32);
    let grid_x = cdiv(m, bm) as u32;
    let grid_y = cdiv(n, bn) as u32;

    let root_constants: Vec<u32> = vec![
        i32_as_u32(m as i32), i32_as_u32(n as i32), i32_as_u32(kk as i32),
        i32_as_u32(kk as i32), 1,
        i32_as_u32(n as i32), 1,
        i32_as_u32(n as i32), 1,
        grid_x, grid_y, 1,
    ];

    let uavs = [
        uav_f16(a, (m * kk) as u32),
        uav_f32(b, (kk * n) as u32),
        uav_f32(bias, n as u32),
        uav_f16(out, (m * n) as u32),
    ];

    k.gpu.record_dispatch(&k.matmul_bias_f32w_32, &root_constants, &uavs, [grid_x, grid_y, 1])
        .map_err(|e| anyhow::anyhow!("matmul_bias_f32w dispatch: {e}"))
}

/// LayerNorm standard with f32 input: out_f16 = LN(x_f32) * weight
fn dispatch_layernorm_std_f32in(
    k: &DecoderKernels,
    x: &GpuBuffer, weight: &GpuBuffer, out: &GpuBuffer,
    n_rows: usize, n_cols: usize,
) -> Result<()> {
    let grid_x = n_rows as u32;
    let root_constants: Vec<u32> = vec![
        i32_as_u32(n_rows as i32), i32_as_u32(n_cols as i32),
        i32_as_u32(n_cols as i32), i32_as_u32(n_cols as i32), // stride_x, stride_out
        grid_x, 1, 1,
    ];
    let uavs = [
        uav_f32(x, (n_rows * n_cols) as u32),
        uav_f16(weight, n_cols as u32),
        uav_f16(out, (n_rows * n_cols) as u32),
    ];
    k.gpu.record_dispatch(&k.layernorm_std_f32in, &root_constants, &uavs, [grid_x, 1, 1])
        .map_err(|e| anyhow::anyhow!("layernorm_std_f32in dispatch: {e}"))
}

/// Fused GEMV+bias+GLU: fc1 output is immediately GLU-reduced. Saves 1 dispatch + 1 barrier.
fn dispatch_gemv_bias_glu(
    k: &DecoderKernels,
    x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
    n_intermediate: usize, kk: usize,
) -> Result<()> {
    let block_n = 128u32;
    let grid_x = cdiv(n_intermediate, block_n as usize) as u32;
    // N = intermediate_size, K = input dim, W stored [K, 2*N]
    let root_constants: Vec<u32> = vec![
        i32_as_u32(n_intermediate as i32),
        i32_as_u32(kk as i32),
        1,                                        // stride_wn = 1 (W[K, 2*N])
        i32_as_u32((n_intermediate * 2) as i32),  // stride_wk = 2*N
        grid_x, 1, 1,
    ];
    let uavs = [
        uav_f16(x, kk as u32),
        uav_f16(w, (kk * n_intermediate * 2) as u32),
        uav_f32(bias, (n_intermediate * 2) as u32),
        uav_f16(out, n_intermediate as u32),
    ];
    k.gpu.record_dispatch(&k.gemv_bias_glu, &root_constants, &uavs, [grid_x, 1, 1])
        .map_err(|e| anyhow::anyhow!("gemv_bias_glu dispatch: {e}"))
}

/// Fused residual-add + layernorm: saves 1 dispatch + 1 barrier per occurrence.
/// f32_out = f16_proj + f32_residual; f16_norm = layernorm(f32_out) * weight
fn dispatch_residual_add_layernorm(
    k: &DecoderKernels,
    f16_proj: &GpuBuffer, f32_residual: &GpuBuffer, f32_out: &GpuBuffer,
    weight: &GpuBuffer, f16_norm: &GpuBuffer,
    n_rows: usize, dim: usize,
) -> Result<()> {
    let grid_x = n_rows as u32;
    let root_constants: Vec<u32> = vec![
        i32_as_u32(n_rows as i32),
        i32_as_u32(dim as i32),
        i32_as_u32(dim as i32),  // stride_in
        i32_as_u32(dim as i32),  // stride_out
        grid_x, 1, 1,
    ];
    let uavs = [
        uav_f16(f16_proj, (n_rows * dim) as u32),
        uav_f32(f32_residual, (n_rows * dim) as u32),
        uav_f32(f32_out, (n_rows * dim) as u32),
        uav_f16(weight, dim as u32),
        uav_f16(f16_norm, (n_rows * dim) as u32),
    ];
    k.gpu.record_dispatch(&k.residual_add_layernorm, &root_constants, &uavs, [grid_x, 1, 1])
        .map_err(|e| anyhow::anyhow!("residual_add_layernorm dispatch: {e}"))
}

/// Add f32 bias row-wise: out_f16[r,c] = x_f16[r,c] + bias_f32[c]
#[allow(dead_code)]
fn dispatch_add_bias_f32(
    k: &DecoderKernels,
    x: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
    n_rows: usize, n_cols: usize,
) -> Result<()> {
    let grid_x = n_rows as u32;
    let root_constants: Vec<u32> = vec![
        i32_as_u32(n_rows as i32), i32_as_u32(n_cols as i32),
        i32_as_u32(n_cols as i32), // stride_row
        grid_x, 1, 1,
    ];
    let uavs = [
        uav_f16(x, (n_rows * n_cols) as u32),
        uav_f32(bias, n_cols as u32),
        uav_f16(out, (n_rows * n_cols) as u32),
    ];
    k.gpu.record_dispatch(&k.add_bias_f32, &root_constants, &uavs, [grid_x, 1, 1])
        .map_err(|e| anyhow::anyhow!("add_bias_f32 dispatch: {e}"))
}

/// Convert f32 → f16
#[allow(dead_code)]
fn dispatch_convert_f32_to_f16(
    k: &DecoderKernels,
    x: &GpuBuffer, out: &GpuBuffer,
    n_elements: usize,
) -> Result<()> {
    let grid_x = cdiv(n_elements, 1024) as u32;
    let root_constants: Vec<u32> = vec![i32_as_u32(n_elements as i32), grid_x, 1, 1];
    let uavs = [
        uav_f32(x, n_elements as u32),
        uav_f16(out, n_elements as u32),
    ];
    k.gpu.record_dispatch(&k.convert_f32_to_f16, &root_constants, &uavs, [grid_x, 1, 1])
        .map_err(|e| anyhow::anyhow!("convert_f32_to_f16 dispatch: {e}"))
}

/// Triton-compiled 1D attention decode with masked head_dim (BLOCK_D=128, head_dim=runtime)
/// cbuffer: kv_len, n_q_heads, n_kv_heads, sm_scale, stride_kv_head, stride_kv_seq, head_dim, grid_dims
fn dispatch_attention_decode(
    k: &DecoderKernels,
    q: &GpuBuffer, kk: &GpuBuffer, v: &GpuBuffer, out: &GpuBuffer,
    kv_len: usize, head_dim: usize, n_kv_heads: usize, n_q_heads: usize,
    sm_scale: f32, _is_causal: bool, _q_pos: usize,
    stride_kv_head: usize, stride_kv_seq: usize,
    kv_buf_elems: usize,
) -> Result<()> {
    let grid_x = n_q_heads as u32;
    let root_constants: Vec<u32> = vec![
        i32_as_u32(kv_len as i32),
        i32_as_u32(n_q_heads as i32),
        i32_as_u32(n_kv_heads as i32),
        sm_scale.to_bits(),
        i32_as_u32(stride_kv_head as i32),
        i32_as_u32(stride_kv_seq as i32),
        i32_as_u32(head_dim as i32),
        grid_x, 1, 1,
    ];
    let q_count = (n_q_heads * head_dim) as u32;
    let kv_count = kv_buf_elems as u32;
    let uavs = [
        uav_f16(q, q_count),
        uav_f16(kk, kv_count),
        uav_f16(v, kv_count),
        uav_f16(out, q_count),
    ];
    k.gpu.record_dispatch(&k.attention_decode, &root_constants, &uavs, [grid_x, 1, 1])
        .map_err(|e| anyhow::anyhow!("attention_decode dispatch: {e}"))
}


/// Fused RoPE(Q) + RoPE(K) + KV cache append(K,V) in one dispatch.
/// Single threadgroup (grid 1,1,1). Replaces separate RoPE Q + rope_cache K + kv_cache V.
/// cbuffer: n_q_heads, n_kv_heads, head_dim, half_rot, pos, max_kv_len, grid_dims
/// Fused Q RoPE + K RoPE + K cache copy (4 UAVs)
/// V cache handled by separate kv_cache_append dispatch
fn dispatch_rope_qk_cache_fused(
    k: &DecoderKernels,
    q: &GpuBuffer, kk: &GpuBuffer,
    rope_table: &GpuBuffer, cache_k: &GpuBuffer,
    n_q_heads: usize, n_kv_heads: usize, head_dim: usize, half_rot: usize,
    pos: usize, max_kv_len: usize,
) -> Result<()> {
    let root_constants: Vec<u32> = vec![
        i32_as_u32(n_q_heads as i32),
        i32_as_u32(n_kv_heads as i32),
        i32_as_u32(head_dim as i32),
        i32_as_u32(half_rot as i32),
        i32_as_u32(pos as i32),
        i32_as_u32(max_kv_len as i32),
        1, 1, 1,
    ];
    let kv_cache_total = (n_kv_heads * max_kv_len * head_dim) as u32;
    let uavs = [
        uav_f16(q, (n_q_heads * head_dim) as u32),
        uav_f16(kk, (n_kv_heads * head_dim) as u32),
        uav_f32(rope_table, (512 * half_rot * 2) as u32),
        uav_f16(cache_k, kv_cache_total),
    ];
    k.gpu.record_dispatch(&k.rope_qk_cache_fused, &root_constants, &uavs, [1, 1, 1])
        .map_err(|e| anyhow::anyhow!("rope_qk_cache_fused dispatch: {e}"))
}

/// Triton-compiled KV cache append: write new K/V at position
/// cbuffer: total_elems, max_kv_len, head_dim, pos, grid_dims
fn dispatch_kv_cache_append(
    k: &DecoderKernels,
    new_kv: &GpuBuffer, cache: &GpuBuffer,
    n_kv_heads: usize, head_dim: usize, max_kv_len: usize, pos: usize,
) -> Result<()> {
    let total_elems = n_kv_heads * head_dim;
    let grid_x = cdiv(total_elems, 256) as u32;
    let root_constants: Vec<u32> = vec![
        i32_as_u32(total_elems as i32),
        i32_as_u32(max_kv_len as i32),
        i32_as_u32(head_dim as i32),
        i32_as_u32(pos as i32),
        grid_x, 1, 1,
    ];
    let cache_total = (n_kv_heads * max_kv_len * head_dim) as u32;
    let uavs = [
        uav_f16(new_kv, total_elems as u32),
        uav_f16(cache, cache_total),
    ];
    k.gpu.record_dispatch(&k.kv_cache_append, &root_constants, &uavs, [grid_x, 1, 1])
        .map_err(|e| anyhow::anyhow!("kv_cache_append dispatch: {e}"))
}

// ── Weight structures ──

struct LinearWeights {
    weight: GpuBuffer, // f16 [K, N] transposed (for GEMV)
}

struct LinearBiasWeights {
    weight: GpuBuffer, // f16 [K, N] transposed (for GEMV)
    bias: GpuBuffer,   // f32 [out_dim]
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
    input_layernorm: GpuBuffer,          // f16 [decoder_dim]
    post_attention_layernorm: GpuBuffer,  // f16 [decoder_dim]
    final_layernorm: GpuBuffer,          // f16 [decoder_dim]
}

// ── KV Cache ──

/// GPU-side KV cache for all decoder layers.
pub struct D3D12DecoderCache {
    // Self-attention: [n_kv_heads, max_kv_len, head_dim] per layer
    self_k: Vec<GpuBuffer>,
    self_v: Vec<GpuBuffer>,
    self_len: usize,

    // Cross-attention: [enc_seq, n_kv_heads * head_dim] per layer (seq-major from matmul)
    cross_k: Vec<GpuBuffer>,
    cross_v: Vec<GpuBuffer>,
    cross_len: usize,
    cross_initialized: bool,

    // Encoder projection (computed once per decode run)
    encoder_proj_f16: Option<GpuBuffer>,
}

impl D3D12DecoderCache {
    fn new(
        gpu: &Gpu,
        num_layers: usize,
        n_kv_heads: usize,
        max_kv_len: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let mut self_k = Vec::with_capacity(num_layers);
        let mut self_v = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            self_k.push(create_f16_buffer(gpu, n_kv_heads * max_kv_len * head_dim)?);
            self_v.push(create_f16_buffer(gpu, n_kv_heads * max_kv_len * head_dim)?);
        }
        Ok(Self {
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
}

// ── Weight loading helpers ──

/// Load 2D weight from GGUF → dequantize → F32 → GPU buffer (optionally transposed).
fn load_f32_weight(
    gpu: &Gpu, shape: (usize, usize), vb: &QVarBuilder, transpose: bool,
) -> Result<GpuBuffer> {
    let qt = vb.get(shape, "weight")?;
    let t = qt.dequantize(&Device::Cpu)?;
    let t = if transpose { t.t()?.contiguous()? } else { t };
    let data = t.to_vec2::<f32>()?;
    let bytes: Vec<u8> = data.iter().flat_map(|row| row.iter().flat_map(|v| v.to_le_bytes())).collect();
    let buf = create_f32_buffer(gpu, shape.0 * shape.1)?;
    upload_f32(gpu, &bytes, &buf)?;
    Ok(buf)
}

/// Load 2D weight from GGUF → dequantize → F16 → GPU buffer.
/// Q8_0 has only 8-bit precision so f16 is lossless. Halves weight bandwidth.
fn load_f16_weight(
    gpu: &Gpu, shape: (usize, usize), vb: &QVarBuilder, transpose: bool,
) -> Result<GpuBuffer> {
    let qt = vb.get(shape, "weight")?;
    let t = qt.dequantize(&Device::Cpu)?;
    let t = if transpose { t.t()?.contiguous()? } else { t };
    let t = t.to_dtype(DType::F16)?;
    let data = t.to_vec2::<half::f16>()?;
    let bytes: Vec<u8> = data.iter().flat_map(|row| row.iter().flat_map(|v| v.to_le_bytes())).collect();
    let buf = create_f16_buffer(gpu, shape.0 * shape.1)?;
    upload_f16(gpu, &bytes, &buf)?;
    Ok(buf)
}

/// Load 1D weight from GGUF → dequantize → F16 → GPU buffer.
fn load_f16_1d(gpu: &Gpu, dim: usize, name: &str, vb: &QVarBuilder) -> Result<GpuBuffer> {
    let qt = vb.get(dim, name)?;
    let t = qt.dequantize(&Device::Cpu)?;
    let t = t.to_dtype(DType::F16)?;
    let data = t.to_vec1::<half::f16>()?;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = create_f16_buffer(gpu, dim)?;
    upload_f16(gpu, &bytes, &buf)?;
    Ok(buf)
}

/// Load 1D bias from GGUF → dequantize → F32 → GPU buffer.
fn load_f32_1d(gpu: &Gpu, dim: usize, name: &str, vb: &QVarBuilder) -> Result<GpuBuffer> {
    let qt = vb.get(dim, name)?;
    let t = qt.dequantize(&Device::Cpu)?;
    let data = t.to_vec1::<f32>()?;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = create_f32_buffer(gpu, dim)?;
    upload_f32(gpu, &bytes, &buf)?;
    Ok(buf)
}

// ── Pre-allocated scratch buffers ──

/// Scratch buffers reused across tokens to eliminate per-token GPU allocation.
/// Liveness analysis ensures no two live values share the same buffer within a layer.
struct DecoderScratch {
    // F16 scratch (reused within each layer)
    f16_norm: GpuBuffer,       // dim - normed/normed2/normed3/fc2_biased
    f16_q: GpuBuffer,          // q_dim - q_buf/cross_q
    f16_k: GpuBuffer,          // kv_dim - k_buf
    f16_v: GpuBuffer,          // kv_dim - v_buf
    f16_attn: GpuBuffer,       // q_dim - attn_out/cross_attn_out
    f16_act: GpuBuffer,        // intermediate_size - activated
    // F32 ping-pong for hidden state
    f32_a: GpuBuffer,          // dim
    f32_b: GpuBuffer,          // dim
    // Output
    f16_logits: GpuBuffer,     // vocab_size
    // Pre-allocated readback buffer (avoid per-token allocation)
    logits_readback: GpuBuffer, // vocab_size * 2 bytes
    // Pre-allocated upload staging buffer (avoid per-token allocation + fence wait)
    embed_upload: GpuBuffer,    // dim * 4 bytes (f32)
}

impl DecoderScratch {
    fn new(
        gpu: &Gpu, dim: usize, q_dim: usize, kv_dim: usize,
        intermediate_size: usize, vocab_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            f16_norm: create_f16_buffer(gpu, dim)?,
            f16_q: create_f16_buffer(gpu, q_dim)?,
            f16_k: create_f16_buffer(gpu, kv_dim)?,
            f16_v: create_f16_buffer(gpu, kv_dim)?,
            f16_attn: create_f16_buffer(gpu, q_dim)?,
            f16_act: create_f16_buffer(gpu, intermediate_size)?,
            f32_a: create_f32_buffer(gpu, dim)?,
            f32_b: create_f32_buffer(gpu, dim)?,
            f16_logits: create_f16_buffer(gpu, vocab_size)?,
            logits_readback: gpu.create_readback_buffer((vocab_size * 2) as u64)
                .map_err(|e| anyhow::anyhow!("create readback: {e}"))?,
            embed_upload: gpu.create_upload_buffer((dim * 4) as u64)
                .map_err(|e| anyhow::anyhow!("create upload: {e}"))?,
        })
    }
}

// ── Main decoder struct ──

/// Triton-accelerated Moonshine decoder on D3D12.
pub struct TritonD3D12Decoder {
    gpu: Arc<Gpu>,
    kernels: DecoderKernels,
    scratch: DecoderScratch,

    // Token embedding (CPU side for lookup)
    embed_tokens_data: Vec<f32>, // [vocab_size, decoder_dim]

    // Position embedding for encoder projection (CPU side)
    pos_emb_data: Vec<f32>, // [max_pos, encoder_dim]

    // Encoder projection weight (if encoder_dim != decoder_dim)
    proj_weight: Option<Vec<f32>>, // [encoder_dim, decoder_dim] transposed as [decoder_dim, encoder_dim] for matmul

    // LM head output projection
    proj_out_weight: GpuBuffer, // f16 [decoder_dim, vocab_size] transposed

    // Decoder layers
    layers: Vec<DecoderLayerWeights>,

    // Final norm
    final_norm_weight: GpuBuffer, // f16 [decoder_dim]

    // RoPE precomputed table
    rope_table: GpuBuffer, // f32 [max_pos, half_rot * 2]

    // Config
    decoder_dim: usize,
    encoder_dim: usize,
    num_layers: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    #[allow(dead_code)]
    rotary_dim: usize,
    half_rot: usize,
    vocab_size: usize,
    intermediate_size: usize,
    bos_id: u32,
    eos_id: u32,
    max_kv_len: usize,
    sm_scale: f32,
}

impl TritonD3D12Decoder {
    /// Create a new D3D12 decoder.
    /// `dec_vb`: VarBuilder for decoder weights (e.g. `vb.pp("model.decoder")`)
    /// `proj_out_vb`: VarBuilder for output projection (e.g. `vb.pp("proj_out")`)
    pub fn new(cfg: &MoonshineConfig, dec_vb: QVarBuilder, proj_out_vb: QVarBuilder, gpu: &Arc<Gpu>) -> Result<Self> {
        let vb = dec_vb;
        println!("  Compiling decoder HLSL kernels...");
        let kernels = DecoderKernels::load(gpu)?;

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

        // Token embedding (keep on CPU for index_select)
        let embed_qt = vb.pp("embed_tokens").get((cfg.vocab_size, decoder_dim), "weight")?;
        let embed_t = embed_qt.dequantize(&Device::Cpu)?;
        let embed_tokens_data = embed_t.to_vec2::<f32>()?
            .into_iter().flatten().collect::<Vec<f32>>();

        // Position embedding for encoder (keep on CPU)
        let pos_qt = vb.pp("pos_emb").get((cfg.max_position_embeddings, encoder_dim), "weight")?;
        let pos_t = pos_qt.dequantize(&Device::Cpu)?;
        let pos_emb_data = pos_t.to_vec2::<f32>()?
            .into_iter().flatten().collect::<Vec<f32>>();

        // Encoder projection (optional, CPU matmul done once per decode run)
        let proj_weight = if encoder_dim != decoder_dim {
            let qt = vb.pp("proj").get((decoder_dim, encoder_dim), "weight")?;
            let t = qt.dequantize(&Device::Cpu)?;
            // Store as [decoder_dim, encoder_dim] for CPU matmul: out = input @ weight.T
            let data = t.to_vec2::<f32>()?.into_iter().flatten().collect();
            Some(data)
        } else {
            None
        };

        // LM head (output projection): [vocab_size, decoder_dim] → transposed [decoder_dim, vocab_size]
        let proj_out_weight = load_f16_weight(gpu, (cfg.vocab_size, decoder_dim), &proj_out_vb, true)?;

        // Decoder layers
        let mut layers = Vec::with_capacity(cfg.decoder_num_layers);
        for i in 0..cfg.decoder_num_layers {
            let lvb = vb.pp(&format!("layers.{i}"));

            let self_attn_vb = lvb.pp("self_attn");
            let self_attn = AttentionWeights {
                q_proj: LinearWeights {
                    weight: load_f16_weight(gpu, (q_dim, decoder_dim), &self_attn_vb.pp("q_proj"), true)?,

                },
                k_proj: LinearWeights {
                    weight: load_f16_weight(gpu, (kv_dim, decoder_dim), &self_attn_vb.pp("k_proj"), true)?,

                },
                v_proj: LinearWeights {
                    weight: load_f16_weight(gpu, (kv_dim, decoder_dim), &self_attn_vb.pp("v_proj"), true)?,

                },
                o_proj: LinearWeights {
                    weight: load_f16_weight(gpu, (decoder_dim, q_dim), &self_attn_vb.pp("o_proj"), true)?,

                },
            };

            let cross_attn_vb = lvb.pp("encoder_attn");
            let cross_attn = AttentionWeights {
                q_proj: LinearWeights {
                    weight: load_f16_weight(gpu, (q_dim, decoder_dim), &cross_attn_vb.pp("q_proj"), true)?,

                },
                // K/V stay f32 — used with tiled matmul in initialize_cross_attention
                k_proj: LinearWeights {
                    weight: load_f32_weight(gpu, (kv_dim, decoder_dim), &cross_attn_vb.pp("k_proj"), true)?,

                },
                v_proj: LinearWeights {
                    weight: load_f32_weight(gpu, (kv_dim, decoder_dim), &cross_attn_vb.pp("v_proj"), true)?,

                },
                o_proj: LinearWeights {
                    weight: load_f16_weight(gpu, (decoder_dim, q_dim), &cross_attn_vb.pp("o_proj"), true)?,

                },
            };

            let mlp_vb = lvb.pp("mlp");
            let mlp = MLPWeights {
                fc1: LinearBiasWeights {
                    weight: load_f16_weight(gpu, (intermediate_size * 2, decoder_dim), &mlp_vb.pp("fc1"), true)?,
                    bias: load_f32_1d(gpu, intermediate_size * 2, "bias", &mlp_vb.pp("fc1"))?,

                },
                fc2: LinearBiasWeights {
                    weight: load_f16_weight(gpu, (decoder_dim, intermediate_size), &mlp_vb.pp("fc2"), true)?,
                    bias: load_f32_1d(gpu, decoder_dim, "bias", &mlp_vb.pp("fc2"))?,

                },
            };

            layers.push(DecoderLayerWeights {
                self_attn,
                cross_attn,
                mlp,
                input_layernorm: load_f16_1d(gpu, decoder_dim, "weight", &lvb.pp("input_layernorm"))?,
                post_attention_layernorm: load_f16_1d(gpu, decoder_dim, "weight", &lvb.pp("post_attention_layernorm"))?,
                final_layernorm: load_f16_1d(gpu, decoder_dim, "weight", &lvb.pp("final_layernorm"))?,
            });
        }

        let final_norm_weight = load_f16_1d(gpu, decoder_dim, "weight", &vb.pp("norm"))?;

        // Pre-allocate scratch buffers (reused across all tokens)
        let scratch = DecoderScratch::new(
            gpu, decoder_dim, q_dim, kv_dim, intermediate_size, cfg.vocab_size,
        )?;

        // Precompute RoPE table: [max_pos, half_rot * 2]
        let max_pos = cfg.max_position_embeddings.min(512); // reasonable limit
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
        let rope_bytes: Vec<u8> = rope_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let rope_table = create_f32_buffer(gpu, max_pos * half_rot * 2)?;
        upload_f32(gpu, &rope_bytes, &rope_table)?;

        Ok(Self {
            gpu: gpu.clone(),
            kernels,
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
            rotary_dim,
            half_rot,
            vocab_size: cfg.vocab_size,
            intermediate_size,
            bos_id: cfg.bos_id as u32,
            eos_id: cfg.eos_id as u32,
            max_kv_len: max_pos,
            sm_scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// Create a new decoder cache (call once per decode run).
    pub fn new_cache(&self) -> Result<D3D12DecoderCache> {
        D3D12DecoderCache::new(
            &self.gpu, self.num_layers,
            self.n_kv_heads, self.max_kv_len, self.head_dim,
        )
    }

    /// Compute encoder projection on CPU (position embedding + optional linear projection).
    /// Upload result as F16 to GPU. Called once per decode run.
    fn prepare_encoder_proj(
        &self, encoder_hidden: &Tensor, cache: &mut D3D12DecoderCache,
    ) -> Result<()> {
        if cache.encoder_proj_f16.is_some() {
            return Ok(());
        }

        let enc_seq = encoder_hidden.dim(1)?;
        let enc_hidden = encoder_hidden.squeeze(0)?.to_dtype(DType::F32)?; // [enc_seq, encoder_dim]
        let enc_data = enc_hidden.to_vec2::<f32>()?;

        // Add position embeddings (clamp to table size)
        let max_pos_emb = self.pos_emb_data.len() / self.encoder_dim;
        let mut proj_data = vec![0.0f32; enc_seq * self.encoder_dim];
        for s in 0..enc_seq {
            for d in 0..self.encoder_dim {
                let pos_val = if s < max_pos_emb {
                    self.pos_emb_data[s * self.encoder_dim + d]
                } else {
                    0.0 // beyond position table — no positional encoding
                };
                proj_data[s * self.encoder_dim + d] = enc_data[s][d] + pos_val;
            }
        }

        // Optional linear projection: [enc_seq, encoder_dim] @ proj_weight.T → [enc_seq, decoder_dim]
        let final_data = if let Some(proj_w) = &self.proj_weight {
            // proj_w is [decoder_dim, encoder_dim]
            let mut out = vec![0.0f32; enc_seq * self.decoder_dim];
            for s in 0..enc_seq {
                for d in 0..self.decoder_dim {
                    let mut sum = 0.0f32;
                    for k in 0..self.encoder_dim {
                        sum += proj_data[s * self.encoder_dim + k]
                            * proj_w[d * self.encoder_dim + k];
                    }
                    out[s * self.decoder_dim + d] = sum;
                }
            }
            out
        } else {
            proj_data
        };

        // Convert to f16 and upload
        let f16_data: Vec<half::f16> = final_data.iter().map(|v| half::f16::from_f32(*v)).collect();
        let bytes: Vec<u8> = f16_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dim = if self.proj_weight.is_some() { self.decoder_dim } else { self.encoder_dim };
        let buf = create_f16_buffer(&self.gpu, enc_seq * dim)?;
        upload_f16(&self.gpu, &bytes, &buf)?;
        cache.encoder_proj_f16 = Some(buf);
        cache.cross_len = enc_seq;

        Ok(())
    }

    /// Compute cross-attention K/V for all layers (called once per decode run).
    fn initialize_cross_attention(
        &self, cache: &mut D3D12DecoderCache,
    ) -> Result<()> {
        if cache.cross_initialized {
            return Ok(());
        }

        let enc_proj = cache.encoder_proj_f16.as_ref()
            .ok_or_else(|| anyhow::anyhow!("encoder projection not initialized"))?;
        let enc_seq = cache.cross_len;
        let kv_dim = self.n_kv_heads * self.head_dim;

        cache.cross_k.clear();
        cache.cross_v.clear();

        // Batch all cross-attention K/V projections into one command list
        self.gpu.begin_batch()
            .map_err(|e| anyhow::anyhow!("begin_batch cross_attn: {e}"))?;

        for layer in &self.layers {
            // K = encoder_proj_f16 @ W_k_f32 → f16 [enc_seq, kv_dim]
            let cross_k = create_f16_buffer(&self.gpu, enc_seq * kv_dim)?;
            dispatch_matmul_f32w(
                &self.kernels, enc_proj, &layer.cross_attn.k_proj.weight, &cross_k,
                enc_seq, kv_dim, self.decoder_dim,
            )?;

            // V = encoder_proj_f16 @ W_v_f32 → f16 [enc_seq, kv_dim]
            let cross_v = create_f16_buffer(&self.gpu, enc_seq * kv_dim)?;
            dispatch_matmul_f32w(
                &self.kernels, enc_proj, &layer.cross_attn.v_proj.weight, &cross_v,
                enc_seq, kv_dim, self.decoder_dim,
            )?;

            cache.cross_k.push(cross_k);
            cache.cross_v.push(cross_v);
        }

        self.gpu.end_batch()
            .map_err(|e| anyhow::anyhow!("end_batch cross_attn: {e}"))?;

        cache.cross_initialized = true;
        Ok(())
    }

    /// Run one decoder step (single token). Logits written to scratch.f16_logits.
    /// All 296 dispatches batched into a single GPU command list submission.
    fn forward_one_token(
        &self,
        token_id: u32,
        cache: &mut D3D12DecoderCache,
    ) -> Result<()> {
        let k = &self.kernels;
        let s = &self.scratch;
        let dim = self.decoder_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let q_dim = self.n_q_heads * self.head_dim;
        let pos = cache.self_len;

        // 1. Token embedding: CPU lookup → write f32 slice directly to upload buffer
        let token_offset = (token_id as usize) * dim;
        let embed_slice = &self.embed_tokens_data[token_offset..token_offset + dim];
        let embed_bytes = unsafe {
            std::slice::from_raw_parts(embed_slice.as_ptr() as *const u8, dim * 4)
        };
        self.gpu.write_upload_buffer(&s.embed_upload, embed_bytes)
            .map_err(|e| anyhow::anyhow!("write upload: {e}"))?;

        // Track which f32 buffer holds current hidden state (ping-pong)
        let mut cur_is_a = true;

        // Batch all dispatches + embedding copy into single command list
        self.gpu.begin_batch()
            .map_err(|e| anyhow::anyhow!("begin_batch token: {e}"))?;

        // Copy embedding from upload staging → f32_a (within the batch, no extra fence wait)
        self.gpu.record_copy(&s.embed_upload, &s.f32_a, (dim * 4) as u64);

        // 2. Process each decoder layer
        // Barriers placed only at read-after-write dependency points.
        // Independent dispatches (Q/K/V, RoPE Q/K, KV appends) run without barriers.
        let b = || self.gpu.record_uav_barrier();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (cur_f32, next_f32) = if cur_is_a {
                (&s.f32_a, &s.f32_b)
            } else {
                (&s.f32_b, &s.f32_a)
            };

            // ── Pre-norm + Self-attention ──
            // Layer 0: standalone layernorm. Layers 1+: already fused into prev MLP residual.
            if layer_idx == 0 {
                b(); // barrier before layernorm reads cur_f32
                dispatch_layernorm_std_f32in(k, cur_f32, &layer.input_layernorm, &s.f16_norm, 1, dim)?;
            }

            // Self-attention Q/K/V projections (3 independent GEMVs, no barriers between)
            b(); // barrier: f16_norm written → Q/K/V read it
            dispatch_gemv_f16w(k, &s.f16_norm, &layer.self_attn.q_proj.weight, &s.f16_q, q_dim, dim)?;
            dispatch_gemv_f16w(k, &s.f16_norm, &layer.self_attn.k_proj.weight, &s.f16_k, kv_dim, dim)?;
            dispatch_gemv_f16w(k, &s.f16_norm, &layer.self_attn.v_proj.weight, &s.f16_v, kv_dim, dim)?;

            // Fused Q RoPE + K RoPE + K cache (1 dispatch) + separate V cache (1 dispatch)
            b(); // barrier: Q/K/V written → RoPE/cache reads them
            dispatch_rope_qk_cache_fused(k, &s.f16_q, &s.f16_k,
                &self.rope_table, &cache.self_k[layer_idx],
                self.n_q_heads, self.n_kv_heads, self.head_dim, self.half_rot,
                pos, self.max_kv_len)?;
            dispatch_kv_cache_append(k, &s.f16_v, &cache.self_v[layer_idx],
                self.n_kv_heads, self.head_dim, self.max_kv_len, pos)?;

            b(); // barrier: cache updated, Q ready → attention reads all
            let self_kv_len = pos + 1;
            let self_kv_buf_elems = self.n_kv_heads * self.max_kv_len * self.head_dim;
            dispatch_attention_decode(
                k, &s.f16_q, &cache.self_k[layer_idx], &cache.self_v[layer_idx], &s.f16_attn,
                self_kv_len, self.head_dim, self.n_kv_heads, self.n_q_heads,
                self.sm_scale, true, pos,
                self.max_kv_len * self.head_dim,
                self.head_dim,
                self_kv_buf_elems,
            )?;

            b(); // barrier: attn output → fused O proj + residual-add + layernorm
            dispatch_gemv_resadd_ln(k, &s.f16_attn, &layer.self_attn.o_proj.weight,
                cur_f32, next_f32, &layer.post_attention_layernorm, &s.f16_norm,
                dim, q_dim)?;
            cur_is_a = !cur_is_a;

            // ── Cross-attention ──
            let (cur_f32, next_f32) = if cur_is_a {
                (&s.f32_a, &s.f32_b)
            } else {
                (&s.f32_b, &s.f32_a)
            };

            b(); // barrier: fused norm → Q proj
            dispatch_gemv_f16w(k, &s.f16_norm, &layer.cross_attn.q_proj.weight, &s.f16_q, q_dim, dim)?;

            b(); // barrier: Q ready → cross attention
            let cross_kv_buf_elems = cache.cross_len * self.n_kv_heads * self.head_dim;
            dispatch_attention_decode(
                k, &s.f16_q, &cache.cross_k[layer_idx], &cache.cross_v[layer_idx], &s.f16_attn,
                cache.cross_len, self.head_dim, self.n_kv_heads, self.n_q_heads,
                self.sm_scale, false, 0,
                self.head_dim,
                self.n_kv_heads * self.head_dim,
                cross_kv_buf_elems,
            )?;

            b(); // barrier: cross attn → fused O proj + residual-add + layernorm
            dispatch_gemv_resadd_ln(k, &s.f16_attn, &layer.cross_attn.o_proj.weight,
                cur_f32, next_f32, &layer.final_layernorm, &s.f16_norm,
                dim, q_dim)?;
            cur_is_a = !cur_is_a;

            // ── MLP ──
            let (cur_f32, next_f32) = if cur_is_a {
                (&s.f32_a, &s.f32_b)
            } else {
                (&s.f32_b, &s.f32_a)
            };

            b(); // barrier: fused norm → fused fc1+GLU
            dispatch_gemv_bias_glu(k, &s.f16_norm, &layer.mlp.fc1.weight, &layer.mlp.fc1.bias, &s.f16_act, self.intermediate_size, dim)?;
            b(); // barrier: GLU → fc2
            dispatch_gemv_bias_f16w(k, &s.f16_act, &layer.mlp.fc2.weight, &layer.mlp.fc2.bias, &s.f16_norm, dim, self.intermediate_size)?;
            // Fuse MLP residual-add with next layer's input layernorm (or final norm)
            // Saves 1 dispatch + 2 barriers per layer vs separate residual_add + layernorm
            b(); // barrier: fc2 → fused residual-add + layernorm
            let next_ln_weight = if layer_idx + 1 < self.num_layers {
                &self.layers[layer_idx + 1].input_layernorm
            } else {
                &self.final_norm_weight
            };
            dispatch_residual_add_layernorm(k, &s.f16_norm, cur_f32, next_f32,
                next_ln_weight, &s.f16_norm, 1, dim)?;
            cur_is_a = !cur_is_a;
        }

        // 4. LM head: [1, dim] @ proj_out [dim, vocab_size] → [1, vocab_size]
        b(); // barrier: norm → LM head
        dispatch_gemv_f16w(k, &s.f16_norm, &self.proj_out_weight, &s.f16_logits, self.vocab_size, dim)?;

        // Copy logits to readback buffer within the same batch (avoids 2nd fence wait)
        b(); // barrier: LM head → copy
        self.gpu.record_copy(&s.f16_logits, &s.logits_readback, (self.vocab_size * 2) as u64);

        self.gpu.end_batch()
            .map_err(|e| anyhow::anyhow!("end_batch token: {e}"))?;

        cache.self_len += 1;
        Ok(())
    }

    /// Read pre-copied f16 logits from readback buffer and return argmax token ID.
    fn argmax_logits(&self) -> Result<u32> {
        let bytes = self.gpu.map_readback_buffer(&self.scratch.logits_readback, (self.vocab_size * 2) as u64)
            .map_err(|e| anyhow::anyhow!("map readback: {e}"))?;
        let f16_data: Vec<half::f16> = bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
            .collect();

        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, v) in f16_data.iter().enumerate() {
            let val = v.to_f32();
            if val > best_val {
                best_val = val;
                best_idx = i as u32;
            }
        }
        Ok(best_idx)
    }

    /// Full greedy decode: encoder_hidden → token IDs.
    pub fn greedy_decode(
        &self,
        encoder_hidden: &Tensor,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        let mut cache = self.new_cache()?;

        // Prepare encoder projection (CPU, once)
        self.prepare_encoder_proj(encoder_hidden, &mut cache)?;

        // Compute cross-attention K/V for all layers (GPU, once)
        self.initialize_cross_attention(&mut cache)?;

        let mut generated = Vec::new();

        // First step with BOS
        self.forward_one_token(self.bos_id, &mut cache)?;
        let mut next_token = self.argmax_logits()?;
        generated.push(next_token);

        if next_token == self.eos_id {
            return Ok(generated);
        }

        // Continue generation
        for _step in 0..max_tokens - 1 {
            self.forward_one_token(next_token, &mut cache)?;
            next_token = self.argmax_logits()?;
            generated.push(next_token);
            if next_token == self.eos_id {
                break;
            }
        }

        Ok(generated)
    }
}

// ── Q8 packed quantization (dead code, kept for future use) ──

#[allow(dead_code)]
const HLSL_GEMV_Q8: &str =
    include_str!("../../kernels/out/hlsl/gemv_q8_v2.hlsl");
#[allow(dead_code)]
const HLSL_GEMV_Q8_BIAS: &str =
    include_str!("../../kernels/out/hlsl/gemv_q8_bias_v2.hlsl");
#[allow(dead_code)]
const HLSL_GEMV_Q8_BIAS_GLU: &str =
    include_str!("../../kernels/out/hlsl/gemv_q8_bias_glu_v2.hlsl");
#[allow(dead_code)]
const HLSL_GEMV_Q8_RESADD_LN: &str =
    include_str!("../../kernels/out/hlsl/gemv_q8_resadd_ln_v2.hlsl");

/// Q8 packed weights: int8 values packed 4-per-u32 + f16 per-block scales.
/// Layout: qs[K, cols/4] (i32), scales[K/32, cols] (f16).
#[allow(dead_code)]
struct Q8Weights {
    qs: GpuBuffer,
    scales: GpuBuffer,
    cols: usize,
    k: usize,
}

/// Load 2D weight from GGUF → dequantize → Q8 packed format → GPU buffers.
#[allow(dead_code)]
fn load_q8_weight(
    gpu: &Gpu, shape: (usize, usize), vb: &QVarBuilder,
) -> Result<Q8Weights> {
    let (n, k) = shape;
    assert!(k % 32 == 0, "K={k} must be multiple of 32 for Q8 blocks");
    assert!(n % 4 == 0, "N={n} must be multiple of 4 for Q8 packing");

    let qt = vb.get(shape, "weight")?;
    let t = qt.dequantize(&Device::Cpu)?;
    let data = t.to_vec2::<f32>()?;

    let n_k_blocks = k / 32;
    let n_div4 = n / 4;

    let mut int8_vals = vec![0i8; n * k];
    let mut scales_nk = vec![0.0f32; n * n_k_blocks];

    for ni in 0..n {
        for kb in 0..n_k_blocks {
            let mut max_abs = 0.0f32;
            for ki in 0..32 {
                max_abs = max_abs.max(data[ni][kb * 32 + ki].abs());
            }
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales_nk[ni * n_k_blocks + kb] = scale;
            let inv_scale = if max_abs > 0.0 { 127.0 / max_abs } else { 0.0 };
            for ki in 0..32 {
                let val = data[ni][kb * 32 + ki];
                int8_vals[ni * k + kb * 32 + ki] = (val * inv_scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }

    // Transpose [N,K] → [K,N] and pack 4 cols per u32
    let mut qs_packed = vec![0u32; k * n_div4];
    for ki in 0..k {
        for ni4 in 0..n_div4 {
            let b0 = int8_vals[(ni4 * 4) * k + ki] as u8 as u32;
            let b1 = int8_vals[(ni4 * 4 + 1) * k + ki] as u8 as u32;
            let b2 = int8_vals[(ni4 * 4 + 2) * k + ki] as u8 as u32;
            let b3 = int8_vals[(ni4 * 4 + 3) * k + ki] as u8 as u32;
            qs_packed[ki * n_div4 + ni4] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
    }

    // Transpose scales [N, K/32] → [K/32, N]
    let mut scales_transposed = vec![half::f16::ZERO; n_k_blocks * n];
    for ni in 0..n {
        for kb in 0..n_k_blocks {
            scales_transposed[kb * n + ni] = half::f16::from_f32(scales_nk[ni * n_k_blocks + kb]);
        }
    }

    let qs_bytes: Vec<u8> = qs_packed.iter().flat_map(|v| v.to_le_bytes()).collect();
    let qs_buf = create_f32_buffer(gpu, k * n_div4)?;
    upload_f32(gpu, &qs_bytes, &qs_buf)?;

    let scales_bytes: Vec<u8> = scales_transposed.iter().flat_map(|v| v.to_le_bytes()).collect();
    let scales_buf = create_f16_buffer(gpu, n_k_blocks * n)?;
    upload_f16(gpu, &scales_bytes, &scales_buf)?;

    Ok(Q8Weights { qs: qs_buf, scales: scales_buf, cols: n, k })
}
