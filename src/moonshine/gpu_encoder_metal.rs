//! Metal backend for the shared GPU encoder.

use anyhow::Result;
use candle_core::{MetalDevice, Storage, Tensor};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};

use crate::triton_kernels::{
    GpuBuffer, TritonKernels,
    enc_matmul, enc_matmul_bias, enc_matmul_bias_gelu,
    enc_layernorm_bare, enc_layernorm_unit_offset,
    enc_gelu, enc_residual_add, enc_bias_add, enc_flash_attention,
    enc_convert_f32_to_f16, load_kernel_pipeline,
};
use super::gpu_encoder::{EncoderBackend, FlashAttentionParams};

pub struct MetalEncoderBackend {
    device: MetalDevice,
    kernels: TritonKernels,
    convert_f32_to_f16: ComputePipeline,
}

impl MetalEncoderBackend {
    pub fn new(device: &MetalDevice) -> Result<Self> {
        let kernels = TritonKernels::load(device)?;
        let convert_f32_to_f16 = load_kernel_pipeline(device, "convert_f32_to_f16", "convert_f32_to_f16")?;
        Ok(Self {
            device: device.clone(),
            kernels,
            convert_f32_to_f16,
        })
    }

    pub fn device(&self) -> &MetalDevice { &self.device }

    /// Flush pending command buffers without waiting (for GPU pipeline overlap).
    fn flush(&self) -> Result<()> {
        self.device.flush().map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Get a command encoder from Candle's command buffer pool.
    /// Each call may return an encoder on the same or a different command buffer,
    /// matching the per-dispatch pattern that gives the GPU scheduler maximum freedom.
    fn enc(&self) -> ComputeCommandEncoder {
        self.device.command_encoder().expect("Failed to get command encoder")
    }

    /// Select the best matmul pipeline and tile size.
    fn matmul_config(&self) -> (usize, &candle_metal_kernels::metal::ComputePipeline) {
        if let Some(ref p128) = self.kernels.matmul_128x128 {
            (128, p128)
        } else {
            (64, &self.kernels.matmul_64x64)
        }
    }

    /// Select the best matmul_bias pipeline and tile size.
    fn matmul_bias_config(&self) -> (usize, &candle_metal_kernels::metal::ComputePipeline) {
        if let Some(ref p) = self.kernels.matmul_bias_128x128 { (128, p) }
        else if let Some(ref p) = self.kernels.matmul_bias_64x64 { (64, p) }
        else { (32, &self.kernels.matmul_bias_32x32) }
    }

    /// Select the best matmul_bias_gelu pipeline and tile size.
    fn matmul_bias_gelu_config(&self) -> (usize, &candle_metal_kernels::metal::ComputePipeline) {
        if let Some(ref p) = self.kernels.matmul_bias_gelu_128x128 { (128, p) }
        else if let Some(ref p) = self.kernels.matmul_bias_gelu_64x64 { (64, p) }
        else { (32, &self.kernels.matmul_bias_gelu_32x32) }
    }
}

impl EncoderBackend for MetalEncoderBackend {
    type Buf = GpuBuffer;

    fn alloc_activation(&self, count: usize) -> Result<GpuBuffer> {
        Ok(GpuBuffer::alloc_f16(&self.device, count)?)
    }

    fn alloc_residual(&self, count: usize) -> Result<GpuBuffer> {
        Ok(GpuBuffer::alloc_f16(&self.device, count)?)
    }

    fn upload_matmul_weight(&self, data_f32: &[f32]) -> Result<GpuBuffer> {
        let f16_data: Vec<half::f16> = data_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        Ok(GpuBuffer::from_f16_data(&self.device, &f16_data)?)
    }

    fn upload_f16_1d(&self, data_f32: &[f32]) -> Result<GpuBuffer> {
        let f16_data: Vec<half::f16> = data_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        Ok(GpuBuffer::from_f16_data(&self.device, &f16_data)?)
    }

    fn upload_input_f16(&self, dst: &GpuBuffer, data: &[half::f16]) -> Result<()> {
        unsafe {
            let ptr = dst.contents_ptr() as *mut half::f16;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        Ok(())
    }

    fn upload_input_f32(&self, _dst: &GpuBuffer, _data: &[f32]) -> Result<()> {
        anyhow::bail!("Metal encoder uses f16 residual; f32 upload not supported")
    }

    fn download_f16(&self, buf: &GpuBuffer, count: usize) -> Result<Vec<half::f16>> {
        let ptr = buf.contents_ptr() as *const half::f16;
        let data = unsafe { std::slice::from_raw_parts(ptr, count) };
        Ok(data.to_vec())
    }

    fn begin_pass(&self) -> Result<()> {
        // Metal uses Candle's command buffer pool — no explicit pass needed.
        // Each enc() call gets an encoder from the pool automatically.
        Ok(())
    }

    fn end_pass(&self) -> Result<()> {
        // Flush and wait for all pending command buffers.
        self.device.wait_until_completed()?;
        Ok(())
    }

    fn layernorm_bare(&self, hidden: &GpuBuffer, out: &GpuBuffer,
                       n_rows: usize, n_cols: usize) {
        enc_layernorm_bare(&self.enc(), self.kernels.layernorm_bare.as_ref().unwrap(),
            hidden, out, n_rows, n_cols);
    }

    fn layernorm_unit_offset(&self, hidden: &GpuBuffer, gamma: &GpuBuffer,
                              out: &GpuBuffer, n_rows: usize, n_cols: usize) {
        enc_layernorm_unit_offset(&self.enc(), &self.kernels.layernorm_unit_offset,
            hidden, gamma, out, n_rows, n_cols);
    }

    fn matmul(&self, a: &GpuBuffer, b: &GpuBuffer, out: &GpuBuffer,
              m: usize, n: usize, k: usize) {
        let (block, pipeline) = self.matmul_config();
        enc_matmul(&self.enc(), pipeline, a, b, out, m, n, k, block, block);
    }

    fn matmul_bias(&self, a: &GpuBuffer, b: &GpuBuffer, bias: &GpuBuffer,
                    out: &GpuBuffer, m: usize, n: usize, k: usize) {
        let (block, pipeline) = self.matmul_bias_config();
        if block < 64 {
            let (mm_block, mm_pipeline) = self.matmul_config();
            if mm_block >= 64 {
                enc_matmul(&self.enc(), mm_pipeline, a, b, out, m, n, k, mm_block, mm_block);
                enc_bias_add(&self.enc(), &self.kernels.bias_add, out, bias, out, m * n, n);
                return;
            }
        }
        enc_matmul_bias(&self.enc(), pipeline, a, b, bias, out, m, n, k, block, block);
    }

    fn matmul_bias_gelu(&self, a: &GpuBuffer, b: &GpuBuffer, bias: &GpuBuffer,
                         out: &GpuBuffer, m: usize, n: usize, k: usize) {
        let (block, pipeline) = self.matmul_bias_gelu_config();
        if block < 64 {
            let (mm_block, mm_pipeline) = self.matmul_config();
            if mm_block >= 64 {
                enc_matmul(&self.enc(), mm_pipeline, a, b, out, m, n, k, mm_block, mm_block);
                enc_bias_add(&self.enc(), &self.kernels.bias_add, out, bias, out, m * n, n);
                enc_gelu(&self.enc(), &self.kernels.gelu, out, out, m * n);
                return;
            }
        }
        enc_matmul_bias_gelu(&self.enc(), pipeline, a, b, bias, out, m, n, k, block, block);
    }

    fn gelu(&self, x: &GpuBuffer, out: &GpuBuffer, n_elem: usize) {
        enc_gelu(&self.enc(), &self.kernels.gelu, x, out, n_elem);
    }

    fn bias_add(&self, x: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                n_elem: usize, n_cols: usize) {
        enc_bias_add(&self.enc(), &self.kernels.bias_add, x, bias, out, n_elem, n_cols);
    }

    fn residual_add(&self, proj: &GpuBuffer, res_in: &GpuBuffer,
                     res_out: &GpuBuffer, n_elem: usize) {
        enc_residual_add(&self.enc(), &self.kernels.residual_add, proj, res_in, res_out, n_elem);
    }

    fn flash_attention(&self, q: &GpuBuffer, k: &GpuBuffer, v: &GpuBuffer,
                        out: &GpuBuffer, p: &FlashAttentionParams) {
        enc_flash_attention(&self.enc(), &self.kernels.flash_attention,
            q, k, v, out,
            p.n_heads, p.padded_seq, p.head_dim,
            p.stride_h, p.stride_m, p.stride_o,
            p.sm_scale, p.window_left, p.window_right);
    }

    fn supports_buf_slice(&self) -> bool { true }

    fn buf_slice(&self, buf: &GpuBuffer, byte_offset: usize) -> GpuBuffer {
        buf.with_offset(byte_offset)
    }

    fn sync(&self) -> Result<()> {
        self.device.wait_until_completed()?;
        Ok(())
    }

    fn upload_input_gpu(&self, x: &Tensor, dst: &GpuBuffer,
                         n: usize, padded_n: usize) -> Result<bool> {
        // Only works for F32 tensors on Metal device
        if x.dtype() != candle_core::DType::F32 {
            return Ok(false);
        }
        if !matches!(x.device(), candle_core::Device::Metal(_)) {
            return Ok(false);
        }

        // Ensure tensor is contiguous (may dispatch a Candle copy kernel)
        let x = x.contiguous().map_err(|e| anyhow::anyhow!("{e}"))?;

        let (storage, layout) = x.storage_and_layout();
        let metal_storage = match &*storage {
            Storage::Metal(ms) => ms,
            _ => return Ok(false),
        };
        let src_buf = metal_storage.buffer();
        let src_offset = layout.start_offset() * 4; // f32 = 4 bytes

        // Flush Candle's pending command buffers (frontend + contiguous copy).
        // Metal's FIFO queue ordering guarantees these ops complete before
        // our new command encoder starts reading the buffer.
        self.flush()?;

        // Zero padding region via CPU memset (shared memory buffer).
        // FA reads all padded_seq rows, so padding must be zero.
        if padded_n > n {
            unsafe {
                let ptr = dst.contents_ptr() as *mut u8;
                std::ptr::write_bytes(ptr.add(n * 2), 0, (padded_n - n) * 2);
            }
        }

        // Dispatch f32→f16 convert kernel.
        // Candle's command pool handles encoder management; hazard tracking
        // ensures the convert completes before subsequent reads from dst.
        let enc = self.enc();
        enc_convert_f32_to_f16(&enc, &self.convert_f32_to_f16,
            src_buf, src_offset, dst, n);

        Ok(true)
    }
}
