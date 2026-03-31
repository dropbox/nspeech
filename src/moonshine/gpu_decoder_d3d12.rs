//! D3D12 backend for the shared GPU decoder.

use anyhow::Result;
use candle_d3d12_kernels::{BufferBinding, Gpu, GpuBuffer, ID3D12PipelineState};
use std::sync::Arc;

use crate::triton_d3d12_kernels::{create_f16_buffer, create_f32_buffer, upload_f16, upload_f32};
use super::gpu_decoder::{
    DecoderBackend, AttentionW,
    RopeCacheParams, SelfAttentionParams, CrossAttentionParams,
};

fn cdiv(a: usize, b: usize) -> usize { (a + b - 1) / b }
fn i32_as_u32(v: i32) -> u32 { v as u32 }

fn uav_f16<'a>(buf: &'a GpuBuffer, count: u32) -> BufferBinding<'a> {
    BufferBinding::structured_f16(buf, count)
}

fn uav_f32<'a>(buf: &'a GpuBuffer, count: u32) -> BufferBinding<'a> {
    BufferBinding::structured_f32(buf, count)
}

// ── Pre-compiled DXIL bytecode ──

const DXIL_MATMUL_F32W_64: &[u8] = include_bytes!("../../kernels/out/dxil/matmul_f16a_f32w_64x64x32.dxil");
const DXIL_MATMUL_F32W_32: &[u8] = include_bytes!("../../kernels/out/dxil/matmul_f16a_f32w_32x32x32.dxil");
const DXIL_LAYERNORM_STD_F32IN: &[u8] = include_bytes!("../../kernels/out/dxil/layernorm_standard_f32in_640.dxil");
const DXIL_GEMV_F16W: &[u8] = include_bytes!("../../kernels/out/dxil/gemv_f16w.dxil");
const DXIL_GEMV_BIAS_F16W: &[u8] = include_bytes!("../../kernels/out/dxil/gemv_bias_f16w.dxil");
const DXIL_ATTENTION_DECODE: &[u8] = include_bytes!("../../kernels/out/dxil/attention_decode_1d_d80.dxil");
const DXIL_ROPE_QK_CACHE_FUSED: &[u8] = include_bytes!("../../kernels/out/dxil/rope_qk_cache_fused.dxil");
const DXIL_KV_CACHE_APPEND: &[u8] = include_bytes!("../../kernels/out/dxil/kv_cache_append.dxil");
const DXIL_RESIDUAL_ADD_LAYERNORM: &[u8] = include_bytes!("../../kernels/out/dxil/residual_add_layernorm_fused.dxil");
const DXIL_GEMV_BIAS_GLU: &[u8] = include_bytes!("../../kernels/out/dxil/gemv_bias_glu_fused.dxil");
const DXIL_GEMV_RESADD_LN: &[u8] = include_bytes!("../../kernels/out/dxil/gemv_resadd_ln_fused.dxil");

// ── Kernel PSOs ──

struct D3D12Kernels {
    gpu: Arc<Gpu>,
    matmul_f32w_64: ID3D12PipelineState,
    matmul_f32w_32: ID3D12PipelineState,
    gemv_f16w: ID3D12PipelineState,
    gemv_bias_f16w: ID3D12PipelineState,
    layernorm_std_f32in: ID3D12PipelineState,
    attention_decode: ID3D12PipelineState,
    rope_qk_cache_fused: ID3D12PipelineState,
    kv_cache_append: ID3D12PipelineState,
    residual_add_layernorm: ID3D12PipelineState,
    gemv_bias_glu: ID3D12PipelineState,
    gemv_resadd_ln: ID3D12PipelineState,
}

impl D3D12Kernels {
    fn load(gpu: &Arc<Gpu>) -> Result<Self> {
        let load = |name: &str, dxil: &[u8]| -> Result<ID3D12PipelineState> {
            gpu.create_compute_pso(dxil)
                .map_err(|e| anyhow::anyhow!("PSO {name}: {e}"))
        };
        Ok(Self {
            gpu: gpu.clone(),
            matmul_f32w_64: load("matmul_f32w_64", DXIL_MATMUL_F32W_64)?,
            matmul_f32w_32: load("matmul_f32w_32", DXIL_MATMUL_F32W_32)?,
            gemv_f16w: load("gemv_f16w", DXIL_GEMV_F16W)?,
            gemv_bias_f16w: load("gemv_bias_f16w", DXIL_GEMV_BIAS_F16W)?,
            layernorm_std_f32in: load("layernorm_std_f32in", DXIL_LAYERNORM_STD_F32IN)?,
            attention_decode: load("attention_decode", DXIL_ATTENTION_DECODE)?,
            rope_qk_cache_fused: load("rope_qk_cache_fused", DXIL_ROPE_QK_CACHE_FUSED)?,
            kv_cache_append: load("kv_cache_append", DXIL_KV_CACHE_APPEND)?,
            residual_add_layernorm: load("residual_add_layernorm", DXIL_RESIDUAL_ADD_LAYERNORM)?,
            gemv_bias_glu: load("gemv_bias_glu", DXIL_GEMV_BIAS_GLU)?,
            gemv_resadd_ln: load("gemv_resadd_ln", DXIL_GEMV_RESADD_LN)?,
        })
    }
}

// ── Dispatch helpers ──

fn dispatch_gemv_f16w(k: &D3D12Kernels, x: &GpuBuffer, w: &GpuBuffer, out: &GpuBuffer,
                       n: usize, kk: usize) -> Result<()> {
    let grid_x = cdiv(n, 128) as u32;
    let rc: Vec<u32> = vec![
        i32_as_u32(n as i32), i32_as_u32(kk as i32),
        1, i32_as_u32(n as i32), grid_x, 1, 1,
    ];
    let uavs = [uav_f16(x, kk as u32), uav_f16(w, (kk * n) as u32), uav_f16(out, n as u32)];
    k.gpu.record_dispatch(&k.gemv_f16w, &rc, &uavs, [grid_x, 1, 1])
        .map_err(|e| anyhow::anyhow!("gemv_f16w: {e}"))
}

fn dispatch_gemv_resadd_ln(k: &D3D12Kernels, x: &GpuBuffer, w: &GpuBuffer,
                            f32_res: &GpuBuffer, f32_out: &GpuBuffer,
                            ln_weight: &GpuBuffer, f16_norm: &GpuBuffer,
                            dim: usize, gemv_k: usize) -> Result<()> {
    let rc: Vec<u32> = vec![
        i32_as_u32(dim as i32), i32_as_u32(gemv_k as i32),
        1, i32_as_u32(dim as i32), 1, 1, 1,
    ];
    let uavs = [
        uav_f16(x, gemv_k as u32), uav_f16(w, (gemv_k * dim) as u32),
        uav_f32(f32_res, dim as u32), uav_f32(f32_out, dim as u32),
        uav_f16(ln_weight, dim as u32), uav_f16(f16_norm, dim as u32),
    ];
    k.gpu.record_dispatch(&k.gemv_resadd_ln, &rc, &uavs, [1, 1, 1])
        .map_err(|e| anyhow::anyhow!("gemv_resadd_ln: {e}"))
}

fn dispatch_gemv_bias_f16w(k: &D3D12Kernels, x: &GpuBuffer, w: &GpuBuffer,
                            bias: &GpuBuffer, out: &GpuBuffer,
                            n: usize, kk: usize) -> Result<()> {
    let grid_x = cdiv(n, 128) as u32;
    let rc: Vec<u32> = vec![
        i32_as_u32(n as i32), i32_as_u32(kk as i32),
        1, i32_as_u32(n as i32), grid_x, 1, 1,
    ];
    let uavs = [
        uav_f16(x, kk as u32), uav_f16(w, (kk * n) as u32),
        uav_f32(bias, n as u32), uav_f16(out, n as u32),
    ];
    k.gpu.record_dispatch(&k.gemv_bias_f16w, &rc, &uavs, [grid_x, 1, 1])
        .map_err(|e| anyhow::anyhow!("gemv_bias_f16w: {e}"))
}

fn dispatch_matmul_f32w(k: &D3D12Kernels, a: &GpuBuffer, b: &GpuBuffer, out: &GpuBuffer,
                         m: usize, n: usize, kk: usize) -> Result<()> {
    let (pso, bm, bn) = if m <= 32 {
        (&k.matmul_f32w_32, 32, 32)
    } else {
        (&k.matmul_f32w_64, 64, 64)
    };
    let grid_x = cdiv(m, bm) as u32;
    let grid_y = cdiv(n, bn) as u32;
    let rc: Vec<u32> = vec![
        i32_as_u32(m as i32), i32_as_u32(n as i32), i32_as_u32(kk as i32),
        i32_as_u32(kk as i32), 1, i32_as_u32(n as i32), 1,
        i32_as_u32(n as i32), 1, grid_x, grid_y, 1,
    ];
    let uavs = [
        uav_f16(a, (m * kk) as u32), uav_f32(b, (kk * n) as u32), uav_f16(out, (m * n) as u32),
    ];
    k.gpu.record_dispatch(pso, &rc, &uavs, [grid_x, grid_y, 1])
        .map_err(|e| anyhow::anyhow!("matmul_f32w: {e}"))
}

fn dispatch_layernorm_std_f32in(k: &D3D12Kernels, x: &GpuBuffer, weight: &GpuBuffer,
                                 out: &GpuBuffer, n_rows: usize, n_cols: usize) -> Result<()> {
    let grid_x = n_rows as u32;
    let rc: Vec<u32> = vec![
        i32_as_u32(n_rows as i32), i32_as_u32(n_cols as i32),
        i32_as_u32(n_cols as i32), i32_as_u32(n_cols as i32),
        grid_x, 1, 1,
    ];
    let uavs = [
        uav_f32(x, (n_rows * n_cols) as u32), uav_f16(weight, n_cols as u32),
        uav_f16(out, (n_rows * n_cols) as u32),
    ];
    k.gpu.record_dispatch(&k.layernorm_std_f32in, &rc, &uavs, [grid_x, 1, 1])
        .map_err(|e| anyhow::anyhow!("layernorm_std_f32in: {e}"))
}

fn dispatch_gemv_bias_glu(k: &D3D12Kernels, x: &GpuBuffer, w: &GpuBuffer,
                           bias: &GpuBuffer, out: &GpuBuffer,
                           n_intermediate: usize, kk: usize) -> Result<()> {
    let grid_x = cdiv(n_intermediate, 128) as u32;
    let rc: Vec<u32> = vec![
        i32_as_u32(n_intermediate as i32), i32_as_u32(kk as i32),
        1, i32_as_u32((n_intermediate * 2) as i32), grid_x, 1, 1,
    ];
    let uavs = [
        uav_f16(x, kk as u32), uav_f16(w, (kk * n_intermediate * 2) as u32),
        uav_f32(bias, (n_intermediate * 2) as u32), uav_f16(out, n_intermediate as u32),
    ];
    k.gpu.record_dispatch(&k.gemv_bias_glu, &rc, &uavs, [grid_x, 1, 1])
        .map_err(|e| anyhow::anyhow!("gemv_bias_glu: {e}"))
}

fn dispatch_residual_add_layernorm(k: &D3D12Kernels, f16_proj: &GpuBuffer,
                                    f32_residual: &GpuBuffer, f32_out: &GpuBuffer,
                                    weight: &GpuBuffer, f16_norm: &GpuBuffer,
                                    n_rows: usize, dim: usize) -> Result<()> {
    let grid_x = n_rows as u32;
    let rc: Vec<u32> = vec![
        i32_as_u32(n_rows as i32), i32_as_u32(dim as i32),
        i32_as_u32(dim as i32), i32_as_u32(dim as i32),
        grid_x, 1, 1,
    ];
    let uavs = [
        uav_f16(f16_proj, (n_rows * dim) as u32),
        uav_f32(f32_residual, (n_rows * dim) as u32),
        uav_f32(f32_out, (n_rows * dim) as u32),
        uav_f16(weight, dim as u32),
        uav_f16(f16_norm, (n_rows * dim) as u32),
    ];
    k.gpu.record_dispatch(&k.residual_add_layernorm, &rc, &uavs, [grid_x, 1, 1])
        .map_err(|e| anyhow::anyhow!("residual_add_layernorm: {e}"))
}

fn dispatch_attention_decode(k: &D3D12Kernels, q: &GpuBuffer, kk: &GpuBuffer,
                              v: &GpuBuffer, out: &GpuBuffer,
                              kv_len: usize, head_dim: usize, n_kv_heads: usize,
                              n_q_heads: usize, sm_scale: f32,
                              stride_kv_head: usize, stride_kv_seq: usize,
                              kv_buf_elems: usize) -> Result<()> {
    let grid_x = n_q_heads as u32;
    let rc: Vec<u32> = vec![
        i32_as_u32(kv_len as i32), i32_as_u32(n_q_heads as i32),
        i32_as_u32(n_kv_heads as i32), sm_scale.to_bits(),
        i32_as_u32(stride_kv_head as i32), i32_as_u32(stride_kv_seq as i32),
        i32_as_u32(head_dim as i32), grid_x, 1, 1,
    ];
    let q_count = (n_q_heads * head_dim) as u32;
    let kv_count = kv_buf_elems as u32;
    let uavs = [
        uav_f16(q, q_count), uav_f16(kk, kv_count),
        uav_f16(v, kv_count), uav_f16(out, q_count),
    ];
    k.gpu.record_dispatch(&k.attention_decode, &rc, &uavs, [grid_x, 1, 1])
        .map_err(|e| anyhow::anyhow!("attention_decode: {e}"))
}

fn dispatch_rope_qk_cache_fused(k: &D3D12Kernels, q: &GpuBuffer, kk: &GpuBuffer,
                                  rope_table: &GpuBuffer, cache_k: &GpuBuffer,
                                  n_q_heads: usize, n_kv_heads: usize, head_dim: usize,
                                  half_rot: usize, pos: usize, max_kv_len: usize) -> Result<()> {
    let rc: Vec<u32> = vec![
        i32_as_u32(n_q_heads as i32), i32_as_u32(n_kv_heads as i32),
        i32_as_u32(head_dim as i32), i32_as_u32(half_rot as i32),
        i32_as_u32(pos as i32), i32_as_u32(max_kv_len as i32), 1, 1, 1,
    ];
    let kv_cache_total = (n_kv_heads * max_kv_len * head_dim) as u32;
    let uavs = [
        uav_f16(q, (n_q_heads * head_dim) as u32),
        uav_f16(kk, (n_kv_heads * head_dim) as u32),
        uav_f32(rope_table, (512 * half_rot * 2) as u32),
        uav_f16(cache_k, kv_cache_total),
    ];
    k.gpu.record_dispatch(&k.rope_qk_cache_fused, &rc, &uavs, [1, 1, 1])
        .map_err(|e| anyhow::anyhow!("rope_qk_cache_fused: {e}"))
}

fn dispatch_kv_cache_append(k: &D3D12Kernels, new_kv: &GpuBuffer, cache: &GpuBuffer,
                             n_kv_heads: usize, head_dim: usize, max_kv_len: usize,
                             pos: usize) -> Result<()> {
    let total_elems = n_kv_heads * head_dim;
    let grid_x = cdiv(total_elems, 256) as u32;
    let rc: Vec<u32> = vec![
        i32_as_u32(total_elems as i32), i32_as_u32(max_kv_len as i32),
        i32_as_u32(head_dim as i32), i32_as_u32(pos as i32), grid_x, 1, 1,
    ];
    let cache_total = (n_kv_heads * max_kv_len * head_dim) as u32;
    let uavs = [uav_f16(new_kv, total_elems as u32), uav_f16(cache, cache_total)];
    k.gpu.record_dispatch(&k.kv_cache_append, &rc, &uavs, [grid_x, 1, 1])
        .map_err(|e| anyhow::anyhow!("kv_cache_append: {e}"))
}

// ── D3D12 Backend ──

pub struct D3D12Backend {
    gpu: Arc<Gpu>,
    kernels: D3D12Kernels,
    logits_readback: GpuBuffer,
    embed_staging: GpuBuffer,
    vocab_size: usize,
    dim: usize,
}

impl D3D12Backend {
    pub fn new(gpu: &Arc<Gpu>, vocab_size: usize, dim: usize) -> Result<Self> {
        println!("  Loading decoder DXIL kernels...");
        let kernels = D3D12Kernels::load(gpu)?;
        let logits_readback = gpu.create_readback_buffer((vocab_size * 2) as u64)
            .map_err(|e| anyhow::anyhow!("create readback: {e}"))?;
        let embed_staging = gpu.create_upload_buffer((dim * 4) as u64)
            .map_err(|e| anyhow::anyhow!("create upload: {e}"))?;
        Ok(Self {
            gpu: gpu.clone(),
            kernels,
            logits_readback,
            embed_staging,
            vocab_size,
            dim,
        })
    }

    fn barrier(&self) {
        self.gpu.record_uav_barrier();
    }
}

impl DecoderBackend for D3D12Backend {
    type Buf = GpuBuffer;

    fn alloc_f16(&self, count: usize) -> Result<GpuBuffer> {
        create_f16_buffer(&self.gpu, count)
    }

    fn alloc_f32(&self, count: usize) -> Result<GpuBuffer> {
        create_f32_buffer(&self.gpu, count)
    }

    fn upload_f16_weight(&self, data_f32: &[f32]) -> Result<GpuBuffer> {
        let f16_data: Vec<half::f16> = data_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let bytes: Vec<u8> = f16_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buf = create_f16_buffer(&self.gpu, data_f32.len())?;
        upload_f16(&self.gpu, &bytes, &buf)?;
        Ok(buf)
    }

    fn upload_f32_data(&self, data: &[f32]) -> Result<GpuBuffer> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buf = create_f32_buffer(&self.gpu, data.len())?;
        upload_f32(&self.gpu, &bytes, &buf)?;
        Ok(buf)
    }

    fn upload_cross_kv_weight(&self, data_f32: &[f32]) -> Result<GpuBuffer> {
        // D3D12 keeps cross-attn K/V weights as f32 for mixed-precision matmul
        self.upload_f32_data(data_f32)
    }

    fn begin_pass(&self) -> Result<()> {
        self.gpu.begin_batch()
            .map_err(|e| anyhow::anyhow!("begin_batch: {e}"))
    }

    fn upload_embed(&self, dst: &GpuBuffer, data: &[f32]) -> Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        self.gpu.write_upload_buffer(&self.embed_staging, bytes)
            .map_err(|e| anyhow::anyhow!("write upload: {e}"))?;
        self.gpu.record_copy(&self.embed_staging, dst, (data.len() * 4) as u64);
        Ok(())
    }

    fn end_pass(&self) -> Result<()> {
        self.gpu.end_batch()
            .map_err(|e| anyhow::anyhow!("end_batch: {e}"))
    }

    fn argmax_logits(&self, _logits: &GpuBuffer, vocab_size: usize) -> Result<u32> {
        let bytes = self.gpu.map_readback_buffer(&self.logits_readback, (vocab_size * 2) as u64)
            .map_err(|e| anyhow::anyhow!("map readback: {e}"))?;
        let f16_data: Vec<half::f16> = bytes.chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]])).collect();
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

    fn layernorm_f32in(&self, x: &GpuBuffer, w: &GpuBuffer, out: &GpuBuffer, dim: usize) {
        self.barrier();
        dispatch_layernorm_std_f32in(&self.kernels, x, w, out, 1, dim).unwrap();
    }

    fn qkv_proj(&self, norm: &GpuBuffer, attn: &AttentionW<GpuBuffer>,
                 q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer,
                 q_dim: usize, kv_dim: usize, dim: usize) {
        self.barrier();
        dispatch_gemv_f16w(&self.kernels, norm, &attn.q_proj.weight, q, q_dim, dim).unwrap();
        dispatch_gemv_f16w(&self.kernels, norm, &attn.k_proj.weight, k, kv_dim, dim).unwrap();
        dispatch_gemv_f16w(&self.kernels, norm, &attn.v_proj.weight, v, kv_dim, dim).unwrap();
    }

    fn rope_kv_cache(&self, q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer,
                      rope: &GpuBuffer, cache_k: &GpuBuffer, cache_v: &GpuBuffer,
                      p: &RopeCacheParams) {
        self.barrier();
        dispatch_rope_qk_cache_fused(&self.kernels, q, k, rope, cache_k,
            p.n_q_heads, p.n_kv_heads, p.head_dim, p.half_rot,
            p.pos, p.max_kv_len).unwrap();
        dispatch_kv_cache_append(&self.kernels, v, cache_v,
            p.n_kv_heads, p.head_dim, p.max_kv_len, p.pos).unwrap();
    }

    fn self_attention(&self, q: &GpuBuffer, cache_k: &GpuBuffer, cache_v: &GpuBuffer,
                       out: &GpuBuffer, p: &SelfAttentionParams) {
        self.barrier();
        let kv_buf_elems = p.n_kv_heads * p.max_kv_len * p.head_dim;
        dispatch_attention_decode(&self.kernels,
            q, cache_k, cache_v, out,
            p.kv_len, p.head_dim, p.n_kv_heads, p.n_q_heads,
            p.sm_scale,
            p.max_kv_len * p.head_dim,  // stride_kv_head
            p.head_dim,                  // stride_kv_seq
            kv_buf_elems).unwrap();
    }

    fn cross_attention(&self, q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer,
                        out: &GpuBuffer, p: &CrossAttentionParams) {
        self.barrier();
        let kv_buf_elems = p.kv_len * p.n_kv_heads * p.head_dim;
        dispatch_attention_decode(&self.kernels,
            q, k, v, out,
            p.kv_len, p.head_dim, p.n_kv_heads, p.n_q_heads,
            p.sm_scale,
            p.head_dim,                    // stride_kv_head (cross: seq-major)
            p.n_kv_heads * p.head_dim,     // stride_kv_seq
            kv_buf_elems).unwrap();
    }

    fn gemv_resadd_ln(&self, x: &GpuBuffer, w: &GpuBuffer,
                       res_in: &GpuBuffer, res_out: &GpuBuffer,
                       ln_w: &GpuBuffer, norm_out: &GpuBuffer,
                       _temp: &GpuBuffer, dim: usize, in_dim: usize) {
        self.barrier();
        dispatch_gemv_resadd_ln(&self.kernels, x, w,
            res_in, res_out, ln_w, norm_out,
            dim, in_dim).unwrap();
    }

    fn cross_q_proj(&self, x: &GpuBuffer, w: &GpuBuffer, out: &GpuBuffer,
                     n: usize, k: usize) {
        self.barrier();
        dispatch_gemv_f16w(&self.kernels, x, w, out, n, k).unwrap();
    }

    fn mlp_fc1_glu(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer,
                    out: &GpuBuffer, intermediate: usize, dim: usize) {
        self.barrier();
        dispatch_gemv_bias_glu(&self.kernels, x, w, bias, out, intermediate, dim).unwrap();
    }

    fn mlp_fc2_bias(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer,
                     out: &GpuBuffer, dim: usize, intermediate: usize) {
        self.barrier();
        dispatch_gemv_bias_f16w(&self.kernels, x, w, bias, out, dim, intermediate).unwrap();
    }

    fn residual_add_ln(&self, proj: &GpuBuffer, res_in: &GpuBuffer, res_out: &GpuBuffer,
                        ln_w: &GpuBuffer, norm_out: &GpuBuffer, dim: usize) {
        self.barrier();
        dispatch_residual_add_layernorm(&self.kernels, proj, res_in, res_out,
            ln_w, norm_out, 1, dim).unwrap();
    }

    fn lm_head(&self, x: &GpuBuffer, w: &GpuBuffer, out: &GpuBuffer,
                vocab: usize, dim: usize) {
        self.barrier();
        dispatch_gemv_f16w(&self.kernels, x, w, out, vocab, dim).unwrap();
        // Copy logits to readback buffer for CPU argmax
        self.barrier();
        self.gpu.record_copy(out, &self.logits_readback, (vocab * 2) as u64);
    }

    fn matmul_cross_kv(&self, enc_proj: &GpuBuffer, w: &GpuBuffer, out: &GpuBuffer,
                        m: usize, n: usize, k: usize) {
        dispatch_matmul_f32w(&self.kernels, enc_proj, w, out, m, n, k).unwrap();
    }
}
