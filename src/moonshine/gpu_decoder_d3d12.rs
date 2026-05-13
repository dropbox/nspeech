//! D3D12 backend for the shared GPU decoder.
//!
//! Uses generated D3D12DecoderKernels (from gen_rust.py) for kernel loading
//! and dispatch. This file provides the DecoderBackend trait implementation.

use anyhow::Result;
use candle_d3d12_kernels::{Gpu, GpuBuffer};
use std::sync::Arc;

use crate::triton_d3d12_kernels::{D3D12DecoderKernels, create_f16_buffer, create_f32_buffer, upload_f16, upload_f32};
use super::gpu_decoder::{
    DecoderBackend, AttentionW,
    RopeCacheParams, SelfAttentionParams, CrossAttentionParams,
};

pub struct D3D12Backend {
    gpu: Arc<Gpu>,
    kernels: D3D12DecoderKernels,
    logits_readback: GpuBuffer,
    embed_staging: GpuBuffer,
}

impl D3D12Backend {
    pub fn new(gpu: &Arc<Gpu>, vocab_size: usize, dim: usize) -> Result<Self> {
        println!("  Loading decoder DXIL kernels...");
        let kernels = D3D12DecoderKernels::load(gpu)?;
        let logits_readback = gpu.create_readback_buffer((vocab_size * 2) as u64)
            .map_err(|e| anyhow::anyhow!("create readback: {e}"))?;
        let embed_staging = gpu.create_upload_buffer((dim * 4) as u64)
            .map_err(|e| anyhow::anyhow!("create upload: {e}"))?;
        Ok(Self {
            gpu: gpu.clone(),
            kernels,
            logits_readback,
            embed_staging,
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
        self.gpu.record_copy(&self.embed_staging, dst, (data.len() * 4) as u64)
            .map_err(|e| anyhow::anyhow!("record_copy: {e}"))?;
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
        self.kernels.dispatch_layernorm_std_f32in(
            x, dim as u32, w, dim as u32, out, dim as u32,
            1, dim as i32, dim as i32, dim as i32,
        ).unwrap();
    }

    fn qkv_proj(&self, norm: &GpuBuffer, attn: &AttentionW<GpuBuffer>,
                 q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer,
                 q_dim: usize, kv_dim: usize, dim: usize) {
        self.barrier();
        self.kernels.dispatch_gemv_f16w(
            norm, dim as u32, &attn.q_proj.weight, (dim * q_dim) as u32,
            q, q_dim as u32,
            q_dim as i32, dim as i32, 1, q_dim as i32,
        ).unwrap();
        self.kernels.dispatch_gemv_f16w(
            norm, dim as u32, &attn.k_proj.weight, (dim * kv_dim) as u32,
            k, kv_dim as u32,
            kv_dim as i32, dim as i32, 1, kv_dim as i32,
        ).unwrap();
        self.kernels.dispatch_gemv_f16w(
            norm, dim as u32, &attn.v_proj.weight, (dim * kv_dim) as u32,
            v, kv_dim as u32,
            kv_dim as i32, dim as i32, 1, kv_dim as i32,
        ).unwrap();
    }

    fn rope_kv_cache(&self, q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer,
                      rope: &GpuBuffer, cache_k: &GpuBuffer, cache_v: &GpuBuffer,
                      p: &RopeCacheParams) {
        self.barrier();
        let kv_cache_total = (p.n_kv_heads * p.max_kv_len * p.head_dim) as u32;
        self.kernels.dispatch_rope_qk_cache_fused(
            q, (p.n_q_heads * p.head_dim) as u32,
            k, (p.n_kv_heads * p.head_dim) as u32,
            rope, (512 * p.half_rot * 2) as u32,
            cache_k, kv_cache_total,
            p.pos as i32, p.max_kv_len as i32,
        ).unwrap();
        let total_elems = p.n_kv_heads * p.head_dim;
        self.kernels.dispatch_kv_cache_append(
            v, total_elems as u32, cache_v, kv_cache_total,
            total_elems as i32, p.max_kv_len as i32,
            p.head_dim as i32, p.pos as i32,
        ).unwrap();
    }

    fn self_attention(&self, q: &GpuBuffer, cache_k: &GpuBuffer, cache_v: &GpuBuffer,
                       out: &GpuBuffer, p: &SelfAttentionParams) {
        self.barrier();
        let kv_buf_elems = (p.n_kv_heads * p.max_kv_len * p.head_dim) as u32;
        let q_count = (p.n_q_heads * p.head_dim) as u32;
        self.kernels.dispatch_attention_decode(
            q, q_count, cache_k, kv_buf_elems, cache_v, kv_buf_elems, out, q_count,
            p.kv_len as i32, p.n_q_heads as i32, p.n_kv_heads as i32,
            p.sm_scale,
            (p.max_kv_len * p.head_dim) as i32,  // stride_kv_head
            p.head_dim as i32,                    // stride_kv_seq
            p.head_dim as i32,
        ).unwrap();
    }

    fn cross_attention(&self, q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer,
                        out: &GpuBuffer, p: &CrossAttentionParams) {
        self.barrier();
        let kv_buf_elems = (p.kv_len * p.n_kv_heads * p.head_dim) as u32;
        let q_count = (p.n_q_heads * p.head_dim) as u32;
        self.kernels.dispatch_attention_decode(
            q, q_count, k, kv_buf_elems, v, kv_buf_elems, out, q_count,
            p.kv_len as i32, p.n_q_heads as i32, p.n_kv_heads as i32,
            p.sm_scale,
            p.head_dim as i32,                    // stride_kv_head (cross: seq-major)
            (p.n_kv_heads * p.head_dim) as i32,   // stride_kv_seq
            p.head_dim as i32,
        ).unwrap();
    }

    fn gemv_resadd_ln(&self, x: &GpuBuffer, w: &GpuBuffer,
                       res_in: &GpuBuffer, res_out: &GpuBuffer,
                       ln_w: &GpuBuffer, norm_out: &GpuBuffer,
                       _temp: &GpuBuffer, dim: usize, in_dim: usize) {
        self.barrier();
        self.kernels.dispatch_gemv_resadd_ln(
            x, in_dim as u32, w, (in_dim * dim) as u32,
            res_in, dim as u32, res_out, dim as u32,
            ln_w, dim as u32, norm_out, dim as u32,
            dim as i32, in_dim as i32, 1, dim as i32,
        ).unwrap();
    }

    fn cross_q_proj(&self, x: &GpuBuffer, w: &GpuBuffer, out: &GpuBuffer,
                     n: usize, k: usize) {
        self.barrier();
        self.kernels.dispatch_gemv_f16w(
            x, k as u32, w, (k * n) as u32, out, n as u32,
            n as i32, k as i32, 1, n as i32,
        ).unwrap();
    }

    fn mlp_fc1_glu(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer,
                    out: &GpuBuffer, intermediate: usize, dim: usize) {
        self.barrier();
        self.kernels.dispatch_gemv_bias_glu(
            x, dim as u32, w, (dim * intermediate * 2) as u32,
            bias, (intermediate * 2) as u32, out, intermediate as u32,
            intermediate as i32, dim as i32, 1, (intermediate * 2) as i32,
        ).unwrap();
    }

    fn mlp_fc2_bias(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer,
                     out: &GpuBuffer, dim: usize, intermediate: usize) {
        self.barrier();
        self.kernels.dispatch_gemv_bias_f16w(
            x, intermediate as u32, w, (intermediate * dim) as u32,
            bias, dim as u32, out, dim as u32,
            dim as i32, intermediate as i32, 1, dim as i32,
        ).unwrap();
    }

    fn residual_add_ln(&self, proj: &GpuBuffer, res_in: &GpuBuffer, res_out: &GpuBuffer,
                        ln_w: &GpuBuffer, norm_out: &GpuBuffer, dim: usize) {
        self.barrier();
        self.kernels.dispatch_residual_add_layernorm(
            proj, dim as u32, res_in, dim as u32, res_out, dim as u32,
            ln_w, dim as u32, norm_out, dim as u32,
            1, dim as i32, dim as i32, dim as i32,
        ).unwrap();
    }

    fn lm_head(&self, x: &GpuBuffer, w: &GpuBuffer, out: &GpuBuffer,
                vocab: usize, dim: usize) {
        self.barrier();
        self.kernels.dispatch_gemv_f16w(
            x, dim as u32, w, (dim * vocab) as u32, out, vocab as u32,
            vocab as i32, dim as i32, 1, vocab as i32,
        ).unwrap();
        self.barrier();
        self.gpu.record_copy(out, &self.logits_readback, (vocab * 2) as u64).unwrap();
    }

    fn matmul_cross_kv(&self, enc_proj: &GpuBuffer, w: &GpuBuffer, out: &GpuBuffer,
                        m: usize, n: usize, k: usize) {
        self.kernels.dispatch_matmul_f32w(
            enc_proj, (m * k) as u32, w, (k * n) as u32, out, (m * n) as u32,
            m as i32, n as i32, k as i32,
        ).unwrap();
    }
}
