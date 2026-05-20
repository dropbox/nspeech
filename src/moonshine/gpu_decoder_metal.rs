//! Metal backend for the shared GPU decoder.

use std::cell::RefCell;
use anyhow::Result;
use candle_core::MetalDevice;
use candle_metal_kernels::metal::ComputeCommandEncoder;

use crate::triton_kernels::{
    DecoderKernels, GpuBuffer, TritonKernels,
    enc_gemv_f16w, enc_matmul,
    enc_layernorm_std_f32in, enc_attention_decode, enc_attention_splitkv,
    enc_rope_qk_cache_fused, enc_kv_cache_append,
    enc_residual_add_layernorm,
    enc_gemv_splitk,
    enc_gemv_qkv_splitk, enc_gemv_splitk_bias,
    enc_gemv_glu_splitk,
};
use super::gpu_decoder::{
    DecoderBackend, AttentionW,
    RopeCacheParams, SelfAttentionParams, CrossAttentionParams,
};

pub struct MetalBackend {
    device: MetalDevice,
    kernels: DecoderKernels,
    encoder_kernels: TritonKernels,
    /// Active compute command encoder (set during begin_pass..end_pass).
    encoder: RefCell<Option<ComputeCommandEncoder>>,
    /// Split-KV partial buffer for cross-attention.
    f32_splitkv_partial: GpuBuffer,
    /// Split-K partial buffer for GEMV operations.
    f16_partial: GpuBuffer,
}

impl MetalBackend {
    pub fn new(device: &MetalDevice, n_q_heads: usize, intermediate_size: usize,
               _vocab_size: usize, _decoder_dim: usize) -> Result<Self> {
        let kernels = DecoderKernels::load(device)?;
        let encoder_kernels = TritonKernels::load(device)?;

        // n_splits=32, BLOCK_D=128, 3 arrays (m, l, acc) per partial
        let f32_splitkv_partial = GpuBuffer::alloc_shared_f32(device, n_q_heads * 32 * 3 * 128)?;
        // F16 partial buffer for split-K GEMV
        let f16_partial = GpuBuffer::alloc_shared_f16(device, 3 * 16 * 2 * intermediate_size)?;

        Ok(Self {
            device: device.clone(),
            kernels,
            encoder_kernels,
            encoder: RefCell::new(None),
            f32_splitkv_partial,
            f16_partial,
        })
    }

    pub fn device(&self) -> &MetalDevice { &self.device }

    fn enc(&self) -> std::cell::Ref<'_, ComputeCommandEncoder> {
        std::cell::Ref::map(self.encoder.borrow(), |e| e.as_ref().unwrap())
    }
}

impl DecoderBackend for MetalBackend {
    type Buf = GpuBuffer;

    fn alloc_f16(&self, count: usize) -> Result<GpuBuffer> {
        Ok(GpuBuffer::alloc_shared_f16(&self.device, count)?)
    }

    fn alloc_f32(&self, count: usize) -> Result<GpuBuffer> {
        Ok(GpuBuffer::alloc_shared_f32(&self.device, count)?)
    }

    fn upload_f16_weight(&self, data_f32: &[f32]) -> Result<GpuBuffer> {
        let f16_data: Vec<half::f16> = data_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        Ok(GpuBuffer::from_f16_data(&self.device, &f16_data)?)
    }

    fn upload_f32_data(&self, data: &[f32]) -> Result<GpuBuffer> {
        Ok(GpuBuffer::from_f32_data(&self.device, data)?)
    }

    fn upload_cross_kv_weight(&self, data_f32: &[f32]) -> Result<GpuBuffer> {
        // Metal uses f16 for all weights including cross-attention K/V
        self.upload_f16_weight(data_f32)
    }

    fn begin_pass(&self) -> Result<()> {
        *self.encoder.borrow_mut() = Some(self.device.command_encoder()?);
        Ok(())
    }

    fn upload_embed(&self, dst: &GpuBuffer, data: &[f32]) -> Result<()> {
        unsafe {
            let ptr = dst.contents_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        Ok(())
    }

    fn end_pass(&self) -> Result<()> {
        let enc = self.encoder.borrow_mut().take().unwrap();
        drop(enc);
        self.device.wait_until_completed()?;
        Ok(())
    }

    fn argmax_logits(&self, logits: &GpuBuffer, vocab_size: usize) -> Result<u32> {
        let ptr = logits.contents_ptr() as *const half::f16;
        let data = unsafe { std::slice::from_raw_parts(ptr, vocab_size) };
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, v) in data.iter().enumerate() {
            let f = v.to_f32();
            if f > best_val {
                best_val = f;
                best_idx = i;
            }
        }
        Ok(best_idx as u32)
    }

    fn layernorm_f32in(&self, x: &GpuBuffer, w: &GpuBuffer, out: &GpuBuffer, dim: usize) {
        enc_layernorm_std_f32in(&self.enc(), &self.kernels.layernorm_std_f32in,
            x, w, out, 1, dim);
    }

    fn qkv_proj(&self, norm: &GpuBuffer, attn: &AttentionW<GpuBuffer>,
                 q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer,
                 q_dim: usize, _kv_dim: usize, dim: usize) {
        enc_gemv_qkv_splitk(&self.enc(),
            &self.kernels.gemv_qkv_splitk_partial, &self.kernels.gemv_qkv_splitk_reduce,
            norm,
            &attn.q_proj.weight, &attn.k_proj.weight, &attn.v_proj.weight,
            q, k, v,
            &self.f16_partial,
            q_dim, dim, 16);
    }

    fn rope_kv_cache(&self, q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer,
                      rope: &GpuBuffer, cache_k: &GpuBuffer, cache_v: &GpuBuffer,
                      p: &RopeCacheParams) {
        let enc = self.enc();
        enc_rope_qk_cache_fused(&enc, &self.kernels.rope_qk_cache_fused,
            q, k, rope, cache_k,
            p.pos, p.max_kv_len);
        enc_kv_cache_append(&enc, &self.kernels.kv_cache_append,
            v, cache_v,
            p.n_kv_heads, p.head_dim, p.max_kv_len, p.pos);
    }

    fn self_attention(&self, q: &GpuBuffer, cache_k: &GpuBuffer, cache_v: &GpuBuffer,
                       out: &GpuBuffer, p: &SelfAttentionParams) {
        enc_attention_decode(&self.enc(), &self.kernels.attention_decode,
            q, cache_k, cache_v, out,
            p.kv_len, p.head_dim, p.n_kv_heads, p.n_q_heads,
            p.sm_scale,
            p.max_kv_len * p.head_dim,  // stride_kv_head
            p.head_dim);                 // stride_kv_seq
    }

    fn cross_attention(&self, q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer,
                        out: &GpuBuffer, p: &CrossAttentionParams) {
        let n_splits = 32usize.min(p.kv_len);
        enc_attention_splitkv(&self.enc(),
            &self.kernels.attention_splitkv_partial, &self.kernels.attention_splitkv_reduce,
            q, k, v, out, &self.f32_splitkv_partial,
            p.kv_len, p.head_dim, p.n_kv_heads, p.n_q_heads,
            p.sm_scale,
            p.head_dim,                     // stride_kv_head (cross: seq-major)
            p.n_kv_heads * p.head_dim,      // stride_kv_seq
            n_splits);
    }

    fn gemv_resadd_ln(&self, x: &GpuBuffer, w: &GpuBuffer,
                       res_in: &GpuBuffer, res_out: &GpuBuffer,
                       ln_w: &GpuBuffer, norm_out: &GpuBuffer,
                       temp: &GpuBuffer, dim: usize, in_dim: usize) {
        let enc = self.enc();
        // Phase 1: split-K GEMV (O proj) → temp
        enc_gemv_splitk(&enc,
            &self.kernels.gemv_splitk_partial, &self.kernels.gemv_splitk_reduce,
            x, w, temp, &self.f16_partial,
            dim, in_dim, 16);
        // Phase 2: residual add + layernorm: temp + res_in → res_out + norm_out
        enc_residual_add_layernorm(&enc, &self.kernels.residual_add_layernorm,
            temp, res_in, res_out,
            ln_w, norm_out,
            1, dim);
    }

    fn cross_q_proj(&self, x: &GpuBuffer, w: &GpuBuffer, out: &GpuBuffer,
                     n: usize, k: usize) {
        enc_gemv_splitk(&self.enc(),
            &self.kernels.gemv_splitk_partial, &self.kernels.gemv_splitk_reduce,
            x, w, out, &self.f16_partial,
            n, k, 16);
    }

    fn mlp_fc1_glu(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer,
                    out: &GpuBuffer, intermediate: usize, dim: usize) {
        enc_gemv_glu_splitk(&self.enc(),
            &self.kernels.gemv_glu_splitk_partial, &self.kernels.gemv_glu_splitk_reduce,
            x, w, bias, out, &self.f16_partial,
            intermediate, dim, 16);
    }

    fn mlp_fc2_bias(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer,
                     out: &GpuBuffer, dim: usize, intermediate: usize) {
        enc_gemv_splitk_bias(&self.enc(),
            &self.kernels.gemv_splitk_partial, &self.kernels.gemv_splitk_bias_reduce,
            x, w, bias, out, &self.f16_partial,
            dim, intermediate, 32);
    }

    fn residual_add_ln(&self, proj: &GpuBuffer, res_in: &GpuBuffer, res_out: &GpuBuffer,
                        ln_w: &GpuBuffer, norm_out: &GpuBuffer, dim: usize) {
        enc_residual_add_layernorm(&self.enc(), &self.kernels.residual_add_layernorm,
            proj, res_in, res_out,
            ln_w, norm_out,
            1, dim);
    }

    fn lm_head(&self, x: &GpuBuffer, w: &GpuBuffer, out: &GpuBuffer,
                vocab: usize, dim: usize) {
        enc_gemv_f16w(&self.enc(), &self.kernels.gemv_f16w,
            x, w, out, vocab, dim);
    }

    fn matmul_cross_kv(&self, enc_proj: &GpuBuffer, w: &GpuBuffer, out: &GpuBuffer,
                        m: usize, n: usize, k: usize) {
        let (block_m, pipeline) = if let Some(ref p128) = self.encoder_kernels.matmul_128x128 {
            (128, p128)
        } else {
            (64, &self.encoder_kernels.matmul_64x64)
        };
        enc_matmul(&self.enc(), pipeline,
            enc_proj, w, out,
            m, n, k, block_m, block_m);
    }
}
