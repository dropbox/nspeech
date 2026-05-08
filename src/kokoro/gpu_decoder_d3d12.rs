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
        self.gpu.record_dispatch(&self.kernels.adain_snake_1k, &rc, &uavs, [grid_x, 1, 1])
            .map_err(|e| anyhow::anyhow!("adain_snake: {e}"))
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
        self.conv1d_matmul(x, w, bias, out, c_in, c_out, t_in, t_out, k, stride, padding, dilation)
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
        self.gpu.record_dispatch(&self.kernels.conv_transpose1d_lrelu, &rc, &uavs, [grid_x, grid_y, 1])
            .map_err(|e| anyhow::anyhow!("conv_transpose1d_lrelu: {e}"))
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
            uav_f16(out, (m * n) as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.matmul, &rc, &uavs, [grid_x, grid_y, 1])
            .map_err(|e| anyhow::anyhow!("matmul: {e}"))?;

        // Row-broadcast bias add: out[i] += bias[i / n]
        let total = m * n;
        let bias_grid = cdiv(total, 1024) as u32;
        let bias_rc: Vec<u32> = vec![total as u32, n as u32, bias_grid, 1, 1];
        let bias_uavs = [
            uav_f16(out, total as u32),
            uav_f16(bias, m as u32),
            uav_f16(out, total as u32),
        ];
        self.gpu.record_dispatch(&self.kernels.row_bias_add, &bias_rc, &bias_uavs, [bias_grid, 1, 1])
            .map_err(|e| anyhow::anyhow!("row_bias_add: {e}"))
    }
}
