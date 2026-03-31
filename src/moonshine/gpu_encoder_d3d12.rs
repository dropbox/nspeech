//! D3D12 backend for the shared GPU encoder.

use anyhow::Result;
use candle_d3d12_kernels::{Gpu, GpuBuffer};
use std::sync::Arc;

use crate::triton_d3d12_kernels::{
    TritonD3D12Kernels,
    create_f16_buffer, create_f32_buffer,
    upload_f16, upload_f32, download_f16,
    triton_d3d12_matmul_f32w, triton_d3d12_matmul_bias_f32w,
    triton_d3d12_layernorm_f32in,
    triton_d3d12_gelu, triton_d3d12_residual_add_f32,
    triton_d3d12_flash_attention,
};
use super::gpu_encoder::{EncoderBackend, FlashAttentionParams};

pub struct D3D12EncoderBackend {
    gpu: Arc<Gpu>,
    kernels: TritonD3D12Kernels,
    /// All-zeros f16 gamma for bare layernorm: (1 + 0) * LN(x) = LN(x).
    zero_gamma: GpuBuffer,
}

impl D3D12EncoderBackend {
    pub fn new(gpu: &Arc<Gpu>, use_fp16_acc: bool, encoder_dim: usize) -> Result<Self> {
        let kernels = TritonD3D12Kernels::load(gpu, use_fp16_acc)?;
        // Upload all-zeros f16 buffer for bare layernorm
        let zero_gamma = create_f16_buffer(gpu, encoder_dim)?;
        let zeros: Vec<u8> = vec![0u8; encoder_dim * 2]; // f16 zeros = 0x0000
        upload_f16(gpu, &zeros, &zero_gamma)?;
        Ok(Self {
            gpu: gpu.clone(),
            kernels,
            zero_gamma,
        })
    }
}

impl EncoderBackend for D3D12EncoderBackend {
    type Buf = GpuBuffer;

    fn alloc_activation(&self, count: usize) -> Result<GpuBuffer> {
        create_f16_buffer(&self.gpu, count)
    }

    fn alloc_residual(&self, count: usize) -> Result<GpuBuffer> {
        // D3D12 encoder uses f32 residual stream
        create_f32_buffer(&self.gpu, count)
    }

    fn upload_matmul_weight(&self, data_f32: &[f32]) -> Result<GpuBuffer> {
        // D3D12 keeps weights in f32 for Q8 precision
        let buf = create_f32_buffer(&self.gpu, data_f32.len())?;
        let bytes: Vec<u8> = data_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        upload_f32(&self.gpu, &bytes, &buf)?;
        Ok(buf)
    }

    fn upload_f16_1d(&self, data_f32: &[f32]) -> Result<GpuBuffer> {
        let f16_data: Vec<half::f16> = data_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let buf = create_f16_buffer(&self.gpu, f16_data.len())?;
        let bytes: Vec<u8> = f16_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        upload_f16(&self.gpu, &bytes, &buf)?;
        Ok(buf)
    }

    fn upload_bias_1d(&self, data_f32: &[f32]) -> Result<GpuBuffer> {
        // D3D12 matmul_bias kernel expects f32 bias
        let buf = create_f32_buffer(&self.gpu, data_f32.len())?;
        let bytes: Vec<u8> = data_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        upload_f32(&self.gpu, &bytes, &buf)?;
        Ok(buf)
    }

    fn upload_input_f16(&self, _dst: &GpuBuffer, _data: &[half::f16]) -> Result<()> {
        anyhow::bail!("D3D12 encoder uses f32 residual; f16 upload not supported")
    }

    fn upload_input_f32(&self, dst: &GpuBuffer, data: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        upload_f32(&self.gpu, &bytes, dst)
    }

    fn download_f16(&self, buf: &GpuBuffer, count: usize) -> Result<Vec<half::f16>> {
        let bytes = download_f16(&self.gpu, buf, count)?;
        let f16_data: Vec<half::f16> = bytes.chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
            .collect();
        Ok(f16_data)
    }

    fn layernorm_bare(&self, hidden: &GpuBuffer, out: &GpuBuffer,
                       n_rows: usize, n_cols: usize) {
        // D3D12 has no bare LN kernel; use layernorm_f32in with zero gamma
        // (1 + 0) * LN(x) = LN(x)
        triton_d3d12_layernorm_f32in(&self.kernels, hidden, &self.zero_gamma, out, n_rows, n_cols).unwrap();
    }

    fn layernorm_unit_offset(&self, hidden: &GpuBuffer, gamma: &GpuBuffer,
                              out: &GpuBuffer, n_rows: usize, n_cols: usize) {
        triton_d3d12_layernorm_f32in(&self.kernels, hidden, gamma, out, n_rows, n_cols).unwrap();
    }

    fn matmul(&self, a: &GpuBuffer, b: &GpuBuffer, out: &GpuBuffer,
              m: usize, n: usize, k: usize) {
        triton_d3d12_matmul_f32w(&self.kernels, a, b, out, m, n, k).unwrap();
    }

    fn matmul_bias(&self, a: &GpuBuffer, b: &GpuBuffer, bias: &GpuBuffer,
                    out: &GpuBuffer, m: usize, n: usize, k: usize) {
        triton_d3d12_matmul_bias_f32w(&self.kernels, a, b, bias, out, m, n, k).unwrap();
    }

    fn gelu(&self, x: &GpuBuffer, out: &GpuBuffer, n_elem: usize) {
        triton_d3d12_gelu(&self.kernels, x, out, n_elem).unwrap();
    }

    fn bias_add(&self, _x: &GpuBuffer, _bias: &GpuBuffer, _out: &GpuBuffer,
                _n_elem: usize, _n_cols: usize) {
        // Not used on D3D12 — matmul_bias handles bias directly
        panic!("D3D12 encoder does not use separate bias_add");
    }

    fn residual_add(&self, proj: &GpuBuffer, res_in: &GpuBuffer,
                     res_out: &GpuBuffer, n_elem: usize) {
        // f16 proj + f32 residual → f32 output
        triton_d3d12_residual_add_f32(&self.kernels, proj, res_in, res_out, n_elem).unwrap();
    }

    fn flash_attention(&self, q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer,
                        out: &GpuBuffer, p: &FlashAttentionParams) {
        triton_d3d12_flash_attention(
            &self.kernels, q, k, v, out,
            p.n_heads, p.seq_len, p.seq_len, // padded_seq == seq_len for D3D12
            p.head_dim, p.sm_scale,
            p.window_left, p.window_right,
        ).unwrap();
    }
}
