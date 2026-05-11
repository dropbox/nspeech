//! D3D12 backend for Kokoro TTS decoder.
//!
//! Implements KokoroGpuBackend using Triton-compiled DXIL kernels.
//! Weights are cached on first upload. Each high-level op does a single
//! begin_batch/record_dispatch/end_batch round trip.

use anyhow::Result;
use candle_d3d12_kernels::{BufferBinding, Gpu, GpuBuffer, ID3D12PipelineState};
use std::collections::HashMap;
use std::cell::RefCell;
use std::sync::Arc;

use super::gpu_backend::KokoroGpuBackend;

include!("../../kernels/out/generated/kokoro_d3d12_gen.rs");

fn cdiv(a: usize, b: usize) -> usize { (a + b - 1) / b }

fn uav_f16<'a>(buf: &'a GpuBuffer, count: u32) -> BufferBinding<'a> {
    BufferBinding::structured_f16(buf, count)
}

pub struct KokoroGpuDecoderD3D12 {
    kernels: KokoroD3D12Kernels,
    gpu: Arc<Gpu>,
    weight_cache: RefCell<HashMap<usize, GpuBuffer>>,
}

impl KokoroGpuDecoderD3D12 {
    pub fn try_new() -> Result<Option<Self>> {
        let gpu = match Gpu::new(0) {
            Ok(g) => Arc::new(g),
            Err(_) => return Ok(None),
        };
        match KokoroD3D12Kernels::load(&gpu) {
            Ok(kernels) => Ok(Some(Self {
                kernels,
                gpu,
                weight_cache: RefCell::new(HashMap::new()),
            })),
            Err(e) => {
                eprintln!("    Kokoro D3D12 GPU unavailable: {e}");
                Ok(None)
            }
        }
    }
}

impl KokoroGpuBackend for KokoroGpuDecoderD3D12 {
    type Buf = GpuBuffer;

    fn alloc(&self, count: usize) -> Result<GpuBuffer> {
        self.gpu.create_buffer((count * 2) as u64)
            .map_err(|e| anyhow::anyhow!("create_buffer: {e}"))
    }

    fn upload_f16(&self, data: &[half::f16]) -> Result<GpuBuffer> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)
        };
        let buf = self.gpu.create_buffer((data.len() * 2) as u64)
            .map_err(|e| anyhow::anyhow!("create_buffer: {e}"))?;
        self.gpu.upload_to_buffer(bytes, &buf)
            .map_err(|e| anyhow::anyhow!("upload: {e}"))?;
        Ok(buf)
    }

    fn upload_weight(&self, id: usize, data: &[half::f16]) -> Result<GpuBuffer> {
        {
            let cache = self.weight_cache.borrow();
            if let Some(buf) = cache.get(&id) {
                return Ok(buf.clone());
            }
        }
        let buf = self.upload_f16(data)?;
        self.weight_cache.borrow_mut().insert(id, buf.clone());
        Ok(buf)
    }

    fn download_f16(&self, buf: &GpuBuffer, count: usize) -> Result<Vec<half::f16>> {
        let bytes = self.gpu.download_buffer(buf, (count * 2) as u64)
            .map_err(|e| anyhow::anyhow!("download: {e}"))?;
        Ok(bytes.chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
            .collect())
    }

    fn add(&self, a: &GpuBuffer, b: &GpuBuffer, n: usize) -> Result<GpuBuffer> {
        let out = self.alloc(n)?;
        let grid_x = cdiv(n, 1024) as u32;
        let rc: Vec<u32> = vec![n as u32, grid_x, 1, 1];
        let uavs = [uav_f16(a, n as u32), uav_f16(b, n as u32), uav_f16(&out, n as u32)];
        self.gpu.record_dispatch(&self.kernels.add, &rc, &uavs, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("add: {e}"))?;
        Ok(out)
    }

    fn scale(&self, x: &GpuBuffer, n: usize, _s: f32) -> Result<GpuBuffer> {
        let out = self.alloc(n)?;
        let grid_x = cdiv(n, 1024) as u32;
        let rc: Vec<u32> = vec![n as u32, grid_x, 1, 1];
        let uavs = [uav_f16(x, n as u32), uav_f16(&out, n as u32)];
        self.gpu.record_dispatch(&self.kernels.scale_third, &rc, &uavs, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("scale_third: {e}"))?;
        Ok(out)
    }

    fn leaky_relu(&self, x: &GpuBuffer, out: &GpuBuffer, n_elements: usize, slope: f32) -> Result<()> {
        let pso = if slope < 0.05 {
            &self.kernels.leaky_relu_001
        } else if slope < 0.15 {
            &self.kernels.leaky_relu_01
        } else {
            &self.kernels.leaky_relu_02
        };
        let grid_x = cdiv(n_elements, 1024) as u32;
        let rc: Vec<u32> = vec![n_elements as u32, grid_x, 1, 1];
        let uavs = [uav_f16(x, n_elements as u32), uav_f16(out, n_elements as u32)];
        self.gpu.record_dispatch(pso, &rc, &uavs, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("leaky_relu: {e}"))
    }

    fn snake(&self, x: &GpuBuffer, alpha: &GpuBuffer, out: &GpuBuffer,
             n_elements: usize, channels: usize, seq_len: usize) -> Result<()> {
        let grid_x = cdiv(n_elements, 1024) as u32;
        let rc: Vec<u32> = vec![
            n_elements as u32, channels as u32, seq_len as u32,
            grid_x, 1, 1,
        ];
        let uavs = [
            uav_f16(x, n_elements as u32),
            uav_f16(alpha, channels as u32),
            uav_f16(out, n_elements as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.snake, &rc, &uavs, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("snake: {e}"))
    }

    fn adain_snake(&self, x: &GpuBuffer, gamma: &GpuBuffer, beta: &GpuBuffer,
                   alpha: &GpuBuffer, out: &GpuBuffer,
                   channels: usize, seq_len: usize) -> Result<()> {
        if seq_len <= 1024 {
            let grid_x = channels as u32;
            let n_elements = channels * seq_len;
            let rc: Vec<u32> = vec![channels as u32, seq_len as u32, grid_x, 1, 1];
            let uavs = [
                uav_f16(x, n_elements as u32),
                uav_f16(gamma, channels as u32),
                uav_f16(beta, channels as u32),
                uav_f16(alpha, channels as u32),
                uav_f16(out, n_elements as u32),
            ];
            return self.gpu.record_dispatch(&self.kernels.adain_snake_1k, &rc, &uavs, [grid_x, 1, 1])
                .map_err(|e| anyhow::anyhow!("adain_snake: {e}"));
        }

        // Two-pass approach for seq_len > 1024:
        // Pass 1: compute per-channel mean+rstd into f32 stats buffer
        let stats_pso = if seq_len <= 2048 {
            &self.kernels.instance_norm_stats_2k
        } else if seq_len <= 8192 {
            &self.kernels.instance_norm_stats_8k
        } else {
            &self.kernels.instance_norm_stats_32k
        };
        let stats_buf = self.gpu.create_buffer((channels * 2 * 4) as u64)
            .map_err(|e| anyhow::anyhow!("create stats buf: {e}"))?;
        let grid_x = channels as u32;
        let n_elements = channels * seq_len;
        let rc1: Vec<u32> = vec![channels as u32, seq_len as u32, grid_x, 1, 1];
        let uavs1 = [
            uav_f16(x, n_elements as u32),
            BufferBinding::structured_f32(&stats_buf, (channels * 2) as u32),
        ];
        self.gpu.record_dispatch(stats_pso, &rc1, &uavs1, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("instance_norm_stats: {e}"))?;

        // Pass 2: element-wise normalize + style + snake
        let grid2 = cdiv(n_elements, 1024) as u32;
        let rc2: Vec<u32> = vec![n_elements as u32, channels as u32, seq_len as u32, grid2, 1, 1];
        let uavs2 = [
            uav_f16(x, n_elements as u32),
            BufferBinding::structured_f32(&stats_buf, (channels * 2) as u32),
            uav_f16(gamma, channels as u32),
            uav_f16(beta, channels as u32),
            uav_f16(alpha, channels as u32),
            uav_f16(out, n_elements as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.norm_style_snake, &rc2, &uavs2, [grid2, 1, 1])
            .map_err(|e| anyhow::anyhow!("norm_style_snake: {e}"))
    }

    fn conv1d(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
              c_in: usize, c_out: usize, t_in: usize, t_out: usize,
              k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let grid_x = c_out as u32;
        let grid_y = cdiv(t_out, 256) as u32;
        let rc: Vec<u32> = vec![
            c_in as u32, c_out as u32, t_in as u32, t_out as u32,
            k as u32, stride as u32, padding as u32, dilation as u32,
            grid_x, grid_y, 1,
        ];
        let uavs = [
            uav_f16(x, (c_in * t_in) as u32),
            uav_f16(w, (c_out * c_in * k) as u32),
            uav_f16(bias, c_out as u32),
            uav_f16(out, (c_out * t_out) as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.conv1d, &rc, &uavs, [grid_x, grid_y, 1])
            .map_err(|e| anyhow::anyhow!("conv1d: {e}"))
    }

    fn conv1d_k(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let kk = c_in * k;
        if kk % 32 == 0 {
            self.conv1d_matmul(x, w, bias, out, c_in, c_out, t_in, t_out, k, stride, padding, dilation)
        } else {
            self.conv1d(x, w, bias, out, c_in, c_out, t_in, t_out, k, stride, padding, dilation)
        }
    }

    fn conv_transpose1d(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                        c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                        k: usize, stride: usize, padding: usize) -> Result<()> {
        let grid_x = c_out as u32;
        let grid_y = cdiv(t_out, 256) as u32;
        let rc: Vec<u32> = vec![
            c_in as u32, c_out as u32, t_in as u32, t_out as u32,
            k as u32, stride as u32, padding as u32,
            grid_x, grid_y, 1,
        ];
        let uavs = [
            uav_f16(x, (c_in * t_in) as u32),
            uav_f16(w, (c_in * c_out * k) as u32),
            uav_f16(bias, c_out as u32),
            uav_f16(out, (c_out * t_out) as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.conv_transpose1d, &rc, &uavs, [grid_x, grid_y, 1])
            .map_err(|e| anyhow::anyhow!("conv_transpose1d: {e}"))
    }

    fn conv_transpose1d_lrelu(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                              c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                              k: usize, stride: usize, padding: usize) -> Result<()> {
        let n = c_in * t_in;
        let tmp = self.alloc(n)?;
        self.leaky_relu(x, &tmp, n, 0.1)?;
        self.conv_transpose1d(&tmp, w, bias, out, c_in, c_out, t_in, t_out, k, stride, padding)
    }

    fn conv1d_lrelu001(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                       c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                       k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let grid_x = c_out as u32;
        let grid_y = cdiv(t_out, 256) as u32;
        let rc: Vec<u32> = vec![
            c_in as u32, c_out as u32, t_in as u32, t_out as u32,
            k as u32, stride as u32, padding as u32, dilation as u32,
            grid_x, grid_y, 1,
        ];
        let uavs = [
            uav_f16(x, (c_in * t_in) as u32),
            uav_f16(w, (c_out * c_in * k) as u32),
            uav_f16(bias, c_out as u32),
            uav_f16(out, (c_out * t_out) as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.conv1d_lrelu001, &rc, &uavs, [grid_x, grid_y, 1])
            .map_err(|e| anyhow::anyhow!("conv1d_lrelu001: {e}"))
    }

    fn reflection_pad1d(&self, x: &GpuBuffer, out: &GpuBuffer, channels: usize, seq_len: usize) -> Result<()> {
        let n_out = channels * (seq_len + 1);
        let grid_x = cdiv(n_out, 1024) as u32;
        let rc: Vec<u32> = vec![channels as u32, seq_len as u32, grid_x, 1, 1];
        let uavs = [
            uav_f16(x, (channels * seq_len) as u32),
            uav_f16(out, n_out as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.reflection_pad1d, &rc, &uavs, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("reflection_pad1d: {e}"))
    }

    fn im2col(&self, x: &GpuBuffer, out: &GpuBuffer,
              c_in: usize, t_in: usize, t_out: usize, k: usize,
              stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let n_elements = c_in * k * t_out;
        let grid_x = cdiv(n_elements, 1024) as u32;
        let rc: Vec<u32> = vec![
            c_in as u32, t_in as u32, t_out as u32, k as u32,
            stride as u32, padding as u32, dilation as u32,
            grid_x, 1, 1,
        ];
        let uavs = [
            uav_f16(x, (c_in * t_in) as u32),
            uav_f16(out, n_elements as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.im2col, &rc, &uavs, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("im2col: {e}"))
    }

    fn im2col_lrelu(&self, x: &GpuBuffer, out: &GpuBuffer,
                    c_in: usize, t_in: usize, t_out: usize, k: usize,
                    stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let n_elements = c_in * k * t_out;
        let grid_x = cdiv(n_elements, 1024) as u32;
        let rc: Vec<u32> = vec![
            c_in as u32, t_in as u32, t_out as u32, k as u32,
            stride as u32, padding as u32, dilation as u32,
            grid_x, 1, 1,
        ];
        let uavs = [
            uav_f16(x, (c_in * t_in) as u32),
            uav_f16(out, n_elements as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.im2col_lrelu, &rc, &uavs, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("im2col_lrelu: {e}"))
    }

    fn matmul_bias(&self, a: &GpuBuffer, b: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                   m: usize, n: usize, k: usize) -> Result<()> {
        let total = m * n;
        let tmp = self.alloc(total)?;

        let grid_x = cdiv(m, 64) as u32;
        let grid_y = cdiv(n, 64) as u32;
        let rc: Vec<u32> = vec![
            m as u32, n as u32, k as u32,
            k as u32, 1,    // stride_am, stride_ak
            n as u32, 1,    // stride_bk, stride_bn
            n as u32, 1,    // stride_cm, stride_cn
            grid_x, grid_y, 1,
        ];
        let uavs = [
            uav_f16(a, (m * k) as u32),
            uav_f16(b, (k * n) as u32),
            uav_f16(&tmp, total as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.matmul, &rc, &uavs, [grid_x, grid_y, 1])
            .map_err(|e| anyhow::anyhow!("matmul: {e}"))?;

        // Row-broadcast bias add: out[i] = tmp[i] + bias[i / n]
        let bias_grid = cdiv(total, 1024) as u32;
        let bias_rc: Vec<u32> = vec![total as u32, n as u32, bias_grid, 1, 1];
        let bias_uavs = [
            uav_f16(&tmp, total as u32),
            uav_f16(bias, m as u32),
            uav_f16(out, total as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.row_bias_add, &bias_rc, &bias_uavs, [bias_grid, 1, 1])
            .map_err(|e| anyhow::anyhow!("row_bias_add: {e}"))
    }

    fn conv1d_matmul(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                     c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                     k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let kk = c_in * k;
        let kk_padded = cdiv(kk, 32) * 32;
        if kk_padded == kk {
            let col_buf = self.alloc(kk * t_out)?;
            self.im2col(x, &col_buf, c_in, t_in, t_out, k, stride, padding, dilation)?;
            return self.matmul_bias(w, &col_buf, bias, out, c_out, t_out, kk);
        }
        // K not aligned to 32: pad weight rows and im2col to avoid matmul reading across rows.
        let col_buf = self.alloc(kk_padded * t_out)?;
        self.im2col(x, &col_buf, c_in, t_in, t_out, k, stride, padding, dilation)?;
        // im2col writes kk*t_out elements contiguously; the extra (kk_padded-kk)*t_out stay zero.
        // But matmul reads B as [K_padded, N] row-major → need B reshaped with stride N per K row.
        // Actually im2col writes [kk, t_out] contiguously. With stride_bk=N, B[row,col] = b[row*N+col].
        // The first kk rows are from im2col, rows kk..kk_padded are zeros (from alloc).
        // This works as-is IF alloc zeroes memory. But it might not.
        // Instead: allocate exact kk*t_out for im2col, then create kk_padded*t_out buffer with zeros
        // and copy im2col data into it. But that requires a copy kernel.
        //
        // Simpler: just pad the WEIGHT matrix (which is already on GPU, cached).
        let w_data = self.gpu.download_buffer(w, (c_out * kk * 2) as u64)
            .map_err(|e| anyhow::anyhow!("download w: {e}"))?;
        let mut w_padded = vec![0u8; c_out * kk_padded * 2];
        for row in 0..c_out {
            let src = &w_data[row * kk * 2..(row + 1) * kk * 2];
            let dst = &mut w_padded[row * kk_padded * 2..row * kk_padded * 2 + kk * 2];
            dst.copy_from_slice(src);
        }
        let w_pad_buf = self.gpu.create_buffer((c_out * kk_padded * 2) as u64)
            .map_err(|e| anyhow::anyhow!("create padded w: {e}"))?;
        self.gpu.upload_to_buffer(&w_padded, &w_pad_buf)
            .map_err(|e| anyhow::anyhow!("upload padded w: {e}"))?;

        // For B (im2col): allocate kk_padded * t_out (extra rows zero from alloc... hopefully).
        // Actually, alloc creates uninitialized memory. Need to write zeros.
        // The im2col kernel writes exactly kk * t_out elements. If we allocate kk_padded * t_out,
        // the layout is contiguous but B is read as stride_bk=N row-major.
        // im2col writes: for each (c, k_pos, t) → col_buf[(c * k + k_pos) * t_out + t]
        // So col_buf is [kk, t_out] contiguous. Extra rows (kk..kk_padded) need to be 0.
        // Simple: allocate larger, im2col fills first kk*t_out, rest must be zero.
        let col_full = self.gpu.create_buffer((kk_padded * t_out * 2) as u64)
            .map_err(|e| anyhow::anyhow!("create padded col: {e}"))?;
        // Zero it
        let zeros = vec![0u8; kk_padded * t_out * 2];
        self.gpu.upload_to_buffer(&zeros, &col_full)
            .map_err(|e| anyhow::anyhow!("zero col: {e}"))?;
        // Run im2col into first kk*t_out elements
        let n_elements = kk * t_out;
        let grid_x = cdiv(n_elements, 1024) as u32;
        let rc: Vec<u32> = vec![
            c_in as u32, t_in as u32, t_out as u32, k as u32,
            stride as u32, padding as u32, dilation as u32,
            grid_x, 1, 1,
        ];
        let uavs = [
            uav_f16(x, (c_in * t_in) as u32),
            uav_f16(&col_full, (kk_padded * t_out) as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.im2col, &rc, &uavs, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("im2col: {e}"))?;

        self.matmul_bias(&w_pad_buf, &col_full, bias, out, c_out, t_out, kk_padded)
    }

    // ── F32-intermediate operations ──

    fn has_f32_intermediates(&self) -> bool { true }

    fn download_f32(&self, buf: &GpuBuffer, count: usize) -> Result<Vec<f32>> {
        let bytes = self.gpu.download_buffer(buf, (count * 4) as u64)
            .map_err(|e| anyhow::anyhow!("download f32: {e}"))?;
        Ok(bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    fn alloc_f32(&self, count: usize) -> Result<GpuBuffer> {
        self.gpu.create_buffer((count * 4) as u64)
            .map_err(|e| anyhow::anyhow!("create_buffer f32: {e}"))
    }

    fn f16_to_f32(&self, x: &GpuBuffer, out: &GpuBuffer, n: usize) -> Result<()> {
        let grid_x = cdiv(n, 1024) as u32;
        let rc: Vec<u32> = vec![n as u32, grid_x, 1, 1];
        let uavs = [
            uav_f16(x, n as u32),
            BufferBinding::structured_f32(out, n as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.f16_to_f32, &rc, &uavs, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("f16_to_f32: {e}"))
    }

    fn f32_to_f16(&self, x: &GpuBuffer, out: &GpuBuffer, n: usize) -> Result<()> {
        let grid_x = cdiv(n, 1024) as u32;
        let rc: Vec<u32> = vec![n as u32, grid_x, 1, 1];
        let uavs = [
            BufferBinding::structured_f32(x, n as u32),
            uav_f16(out, n as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.f32_to_f16, &rc, &uavs, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("f32_to_f16: {e}"))
    }

    fn im2col_f32_to_f16(&self, x: &GpuBuffer, out: &GpuBuffer,
                         c_in: usize, t_in: usize, t_out: usize, k: usize,
                         stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let n_elements = c_in * k * t_out;
        let grid_x = cdiv(n_elements, 1024) as u32;
        let rc: Vec<u32> = vec![
            c_in as u32, t_in as u32, t_out as u32, k as u32,
            stride as u32, padding as u32, dilation as u32,
            grid_x, 1, 1,
        ];
        let uavs = [
            BufferBinding::structured_f32(x, (c_in * t_in) as u32),
            uav_f16(out, n_elements as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.im2col_f32_to_f16, &rc, &uavs, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("im2col_f32_to_f16: {e}"))
    }

    fn conv1d_f32(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                  c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                  k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let kk = c_in * k;
        if kk % 32 == 0 {
            let col_buf = self.alloc(kk * t_out)?;
            self.im2col_f32_to_f16(x, &col_buf, c_in, t_in, t_out, k, stride, padding, dilation)?;
            let matmul_out = self.alloc(c_out * t_out)?;
            self.matmul_bias(w, &col_buf, bias, &matmul_out, c_out, t_out, kk)?;
            self.f16_to_f32(&matmul_out, out, c_out * t_out)?;
            return Ok(());
        }
        // Fallback to naive kernel
        let grid_x = c_out as u32;
        let grid_y = cdiv(t_out, 256) as u32;
        let rc: Vec<u32> = vec![
            c_in as u32, c_out as u32, t_in as u32, t_out as u32,
            k as u32, stride as u32, padding as u32, dilation as u32,
            grid_x, grid_y, 1,
        ];
        let uavs = [
            BufferBinding::structured_f32(x, (c_in * t_in) as u32),
            uav_f16(w, (c_out * c_in * k) as u32),
            uav_f16(bias, c_out as u32),
            BufferBinding::structured_f32(out, (c_out * t_out) as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.conv1d_f32io, &rc, &uavs, [grid_x, grid_y, 1])
            .map_err(|e| anyhow::anyhow!("conv1d_f32io: {e}"))
    }

    fn adain_snake_f32(&self, x: &GpuBuffer, gamma: &GpuBuffer, beta: &GpuBuffer,
                       alpha: &GpuBuffer, out: &GpuBuffer,
                       channels: usize, seq_len: usize) -> Result<()> {
        let n_elements = channels * seq_len;
        // Two-pass: compute stats from f32 input, then normalize+style+snake in f32
        let stats_pso = if seq_len <= 2048 {
            &self.kernels.instance_norm_stats_f32in_2k
        } else if seq_len <= 8192 {
            &self.kernels.instance_norm_stats_f32in_8k
        } else if seq_len <= 32768 {
            &self.kernels.instance_norm_stats_f32in_32k
        } else if seq_len <= 65536 {
            &self.kernels.instance_norm_stats_f32in_64k
        } else {
            &self.kernels.instance_norm_stats_f32in_128k
        };
        let stats_buf = self.gpu.create_buffer((channels * 2 * 4) as u64)
            .map_err(|e| anyhow::anyhow!("create stats buf: {e}"))?;
        let grid_x = channels as u32;
        let rc1: Vec<u32> = vec![channels as u32, seq_len as u32, grid_x, 1, 1];
        let uavs1 = [
            BufferBinding::structured_f32(x, n_elements as u32),
            BufferBinding::structured_f32(&stats_buf, (channels * 2) as u32),
        ];
        self.gpu.record_dispatch(stats_pso, &rc1, &uavs1, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("instance_norm_stats_f32in: {e}"))?;

        // Pass 2: normalize + style + snake (f32 in, f32 out)
        let grid2 = cdiv(n_elements, 1024) as u32;
        let rc2: Vec<u32> = vec![n_elements as u32, channels as u32, seq_len as u32, grid2, 1, 1];
        let uavs2 = [
            BufferBinding::structured_f32(x, n_elements as u32),
            BufferBinding::structured_f32(&stats_buf, (channels * 2) as u32),
            uav_f16(gamma, channels as u32),
            uav_f16(beta, channels as u32),
            uav_f16(alpha, channels as u32),
            BufferBinding::structured_f32(out, n_elements as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.norm_style_snake_f32io, &rc2, &uavs2, [grid2, 1, 1])
            .map_err(|e| anyhow::anyhow!("norm_style_snake_f32io: {e}"))
    }

    fn add_f32(&self, a: &GpuBuffer, b: &GpuBuffer, out: &GpuBuffer, n: usize) -> Result<()> {
        let grid_x = cdiv(n, 1024) as u32;
        let rc: Vec<u32> = vec![n as u32, grid_x, 1, 1];
        let uavs = [
            BufferBinding::structured_f32(a, n as u32),
            BufferBinding::structured_f32(b, n as u32),
            BufferBinding::structured_f32(out, n as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.add_f32, &rc, &uavs, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("add_f32: {e}"))
    }

    fn scale_third_f32(&self, x: &GpuBuffer, out: &GpuBuffer, n: usize) -> Result<()> {
        let grid_x = cdiv(n, 1024) as u32;
        let rc: Vec<u32> = vec![n as u32, grid_x, 1, 1];
        let uavs = [
            BufferBinding::structured_f32(x, n as u32),
            BufferBinding::structured_f32(out, n as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.scale_third_f32, &rc, &uavs, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("scale_third_f32: {e}"))
    }

    fn leaky_relu_f32(&self, x: &GpuBuffer, out: &GpuBuffer, n: usize, slope: f32) -> Result<()> {
        let pso = if slope == 0.01 {
            &self.kernels.leaky_relu_f32_001
        } else {
            &self.kernels.leaky_relu_f32_01
        };
        let grid_x = cdiv(n, 1024) as u32;
        let rc: Vec<u32> = vec![n as u32, grid_x, 1, 1];
        let uavs = [
            BufferBinding::structured_f32(x, n as u32),
            BufferBinding::structured_f32(out, n as u32),
        ];
        self.gpu.record_dispatch(pso, &rc, &uavs, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("leaky_relu_f32: {e}"))
    }

    fn conv_transpose1d_f32io_lrelu(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                                    c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                                    k: usize, stride: usize, padding: usize) -> Result<()> {
        let grid_x = c_out as u32;
        let total_y = cdiv(t_out, 256);
        let max_y = if c_in * k > 5000 { 4 } else if c_in * k > 2000 { 8 } else { total_y };
        let uavs = [
            BufferBinding::structured_f32(x, (c_in * t_in) as u32),
            uav_f16(w, (c_in * c_out * k) as u32),
            uav_f16(bias, c_out as u32),
            BufferBinding::structured_f32(out, (c_out * t_out) as u32),
        ];
        let mut y_off = 0;
        while y_off < total_y {
            let chunk_y = max_y.min(total_y - y_off);
            let rc: Vec<u32> = vec![
                c_in as u32, c_out as u32, t_in as u32, t_out as u32,
                k as u32, stride as u32, padding as u32, y_off as u32,
                grid_x, chunk_y as u32, 1,
            ];
            self.gpu.record_dispatch(&self.kernels.conv_transpose1d_f32io_lrelu, &rc, &uavs, [grid_x, chunk_y as u32, 1])
                .map_err(|e| anyhow::anyhow!("conv_transpose1d_f32io_lrelu: {e}"))?;
            y_off += chunk_y;
        }
        Ok(())
    }

    fn conv_transpose1d_f32io(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                              c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                              k: usize, stride: usize, padding: usize) -> Result<()> {
        let grid_x = c_out as u32;
        let total_y = cdiv(t_out, 256);
        // Chunk Y to avoid TDR: limit work per dispatch based on c_in*k cost per thread.
        let max_y = if c_in * k > 5000 { 4 } else if c_in * k > 2000 { 8 } else { total_y };
        let uavs = [
            BufferBinding::structured_f32(x, (c_in * t_in) as u32),
            uav_f16(w, (c_in * c_out * k) as u32),
            uav_f16(bias, c_out as u32),
            BufferBinding::structured_f32(out, (c_out * t_out) as u32),
        ];
        let mut y_off = 0;
        while y_off < total_y {
            let chunk_y = max_y.min(total_y - y_off);
            let rc: Vec<u32> = vec![
                c_in as u32, c_out as u32, t_in as u32, t_out as u32,
                k as u32, stride as u32, padding as u32, y_off as u32,
                grid_x, chunk_y as u32, 1,
            ];
            self.gpu.record_dispatch(&self.kernels.conv_transpose1d_f32io, &rc, &uavs, [grid_x, chunk_y as u32, 1])
                .map_err(|e| anyhow::anyhow!("conv_transpose1d_f32io: {e}"))?;
            y_off += chunk_y;
        }
        Ok(())
    }

    fn reflection_pad1d_f32(&self, x: &GpuBuffer, out: &GpuBuffer, channels: usize, seq_len: usize) -> Result<()> {
        let n_out = channels * (seq_len + 1);
        let grid_x = cdiv(n_out, 1024) as u32;
        let rc: Vec<u32> = vec![channels as u32, seq_len as u32, grid_x, 1, 1];
        let uavs = [
            BufferBinding::structured_f32(x, (channels * seq_len) as u32),
            BufferBinding::structured_f32(out, n_out as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.reflection_pad1d_f32, &rc, &uavs, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("reflection_pad1d_f32: {e}"))
    }
}

