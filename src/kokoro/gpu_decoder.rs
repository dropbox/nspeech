//! Metal backend for Kokoro TTS decoder.
//!
//! Implements KokoroGpuBackend using Triton-compiled Metal kernels.
//! Weights are cached on first upload; activations upload each call.

use anyhow::Result;
use candle_core::{Device, GpuBuffer, MetalDevice};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;
use std::collections::HashMap;
use std::cell::RefCell;

use super::gpu_backend::KokoroGpuBackend;

include!("../../kernels/out/generated/kokoro_metal_gen.rs");

fn cdiv(a: usize, b: usize) -> usize { (a + b - 1) / b }

pub struct KokoroGpuDecoder {
    kernels: KokoroKernels,
    pub(super) device: MetalDevice,
    weight_cache: RefCell<HashMap<usize, GpuBuffer>>,
}

impl KokoroGpuDecoder {
    pub fn try_new(model_device: &Device) -> Result<Option<Self>> {
        let md = match model_device {
            Device::Metal(m) => m.clone(),
            _ => {
                let d = match Device::new_metal(0) {
                    Ok(d) => d,
                    Err(_) => return Ok(None),
                };
                match &d {
                    Device::Metal(m) => m.clone(),
                    _ => return Ok(None),
                }
            }
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            KokoroKernels::load(&md)
        }));
        match result {
            Ok(Ok(kernels)) => Ok(Some(Self {
                kernels,
                device: md,
                weight_cache: RefCell::new(HashMap::new()),
            })),
            Ok(Err(e)) => {
                eprintln!("    GPU decoder unavailable: {e}");
                Ok(None)
            }
            Err(_) => {
                eprintln!("    GPU decoder unavailable: kernel exceeds device limits");
                Ok(None)
            }
        }
    }
}

impl KokoroGpuDecoder {
    fn adain_snake_cpu_fallback(&self, x: &GpuBuffer, gamma: &GpuBuffer, beta: &GpuBuffer,
                                alpha: &GpuBuffer, out: &GpuBuffer,
                                channels: usize, seq_len: usize) -> Result<()> {
        self.device.wait_until_completed()?;
        let n = channels * seq_len;
        let x_data = self.download_f16(x, n)?;
        let gamma_data = self.download_f16(gamma, channels)?;
        let beta_data = self.download_f16(beta, channels)?;
        let alpha_data = self.download_f16(alpha, channels)?;

        let mut result = vec![half::f16::ZERO; n];
        for ch in 0..channels {
            let base = ch * seq_len;
            let slice = &x_data[base..base + seq_len];
            let sum: f32 = slice.iter().map(|v| v.to_f32()).sum();
            let mean = sum / seq_len as f32;
            let var: f32 = slice.iter().map(|v| { let d = v.to_f32() - mean; d * d }).sum::<f32>() / seq_len as f32;
            let rstd = 1.0 / (var + 1e-5_f32).sqrt();

            let g = gamma_data[ch].to_f32();
            let b = beta_data[ch].to_f32();
            let a = alpha_data[ch].to_f32();
            let scale = (g + 1.0) * rstd;
            let inv_a = 1.0 / (a + 1e-9_f32);

            for t in 0..seq_len {
                let val = x_data[base + t].to_f32();
                let styled = scale * (val - mean) + b;
                let ax = a * styled;
                let s = ax.sin();
                let out_val = styled + s * s * inv_a;
                result[base + t] = half::f16::from_f32(out_val);
            }
        }

        let out_ptr = out.contents_ptr() as *mut half::f16;
        unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), out_ptr, n); }
        Ok(())
    }
}

impl KokoroGpuBackend for KokoroGpuDecoder {
    type Buf = GpuBuffer;

    fn alloc(&self, count: usize) -> Result<GpuBuffer> {
        let buffer = self.device.allocate_zeros(count * 2)?;
        Ok(GpuBuffer::from_arc(buffer))
    }

    fn upload_f16(&self, data: &[half::f16]) -> Result<GpuBuffer> {
        GpuBuffer::from_f16_data(&self.device, data).map_err(Into::into)
    }

    fn upload_weight(&self, id: usize, data: &[half::f16]) -> Result<GpuBuffer> {
        {
            let cache = self.weight_cache.borrow();
            if let Some(buf) = cache.get(&id) {
                return Ok(buf.clone());
            }
        }
        let buf = GpuBuffer::from_f16_data(&self.device, data)?;
        self.weight_cache.borrow_mut().insert(id, buf.clone());
        Ok(buf)
    }

    fn download_f16(&self, buf: &GpuBuffer, count: usize) -> Result<Vec<half::f16>> {
        let ptr = buf.contents_ptr() as *const half::f16;
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        Ok(slice.to_vec())
    }

    fn add(&self, a: &GpuBuffer, b: &GpuBuffer, n: usize) -> Result<GpuBuffer> {
        let out = self.alloc(n)?;
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.add);
        encoder.set_buffer(0, Some(a.buf()), a.offset);
        encoder.set_buffer(1, Some(b.buf()), b.offset);
        encoder.set_buffer(2, Some(out.buf()), out.offset);
        encoder.set_bytes(3, &(n as i32));
        let max_tg = self.kernels.add.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(out)
    }

    fn scale(&self, x: &GpuBuffer, n: usize, _s: f32) -> Result<GpuBuffer> {
        let out = self.alloc(n)?;
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.scale_third);
        encoder.set_buffer(0, Some(x.buf()), x.offset);
        encoder.set_buffer(1, Some(out.buf()), out.offset);
        encoder.set_bytes(2, &(n as i32));
        let max_tg = self.kernels.scale_third.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(out)
    }

    fn leaky_relu(&self, x: &GpuBuffer, out: &GpuBuffer, n_elements: usize, slope: f32) -> Result<()> {
        let pipeline = if slope < 0.05 {
            &self.kernels.leaky_relu_001
        } else if slope < 0.15 {
            &self.kernels.leaky_relu_01
        } else {
            &self.kernels.leaky_relu_02
        };
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(x.buf()), x.offset);
        encoder.set_buffer(1, Some(out.buf()), out.offset);
        encoder.set_bytes(2, &(n_elements as i32));
        let max_tg = pipeline.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n_elements, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn snake(&self, x: &GpuBuffer, alpha: &GpuBuffer, out: &GpuBuffer,
             n_elements: usize, channels: usize, seq_len: usize) -> Result<()> {
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.snake);
        encoder.set_buffer(0, Some(x.buf()), x.offset);
        encoder.set_buffer(1, Some(alpha.buf()), alpha.offset);
        encoder.set_buffer(2, Some(out.buf()), out.offset);
        encoder.set_bytes(3, &(n_elements as i32));
        encoder.set_bytes(4, &(channels as i32));
        encoder.set_bytes(5, &(seq_len as i32));
        let max_tg = self.kernels.snake.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n_elements, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn adain_snake(&self, x: &GpuBuffer, gamma: &GpuBuffer, beta: &GpuBuffer,
                   alpha: &GpuBuffer, out: &GpuBuffer,
                   channels: usize, seq_len: usize) -> Result<()> {
        if seq_len > 1024 {
            return self.adain_snake_cpu_fallback(x, gamma, beta, alpha, out, channels, seq_len);
        }
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.adain_snake_1k);
        encoder.set_buffer(0, Some(x.buf()), x.offset);
        encoder.set_buffer(1, Some(gamma.buf()), gamma.offset);
        encoder.set_buffer(2, Some(beta.buf()), beta.offset);
        encoder.set_buffer(3, Some(alpha.buf()), alpha.offset);
        encoder.set_buffer(4, Some(out.buf()), out.offset);
        encoder.set_bytes(5, &(channels as i32));
        encoder.set_bytes(6, &(seq_len as i32));
        let grid = MTLSize { width: channels, height: 1, depth: 1 };
        let max_tg = self.kernels.adain_snake_1k.max_total_threads_per_threadgroup() as usize;
        let tg_width = seq_len.next_power_of_two().min(max_tg);
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn conv1d(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
              c_in: usize, c_out: usize, t_in: usize, t_out: usize,
              k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.conv1d);
        encoder.set_buffer(0, Some(x.buf()), x.offset);
        encoder.set_buffer(1, Some(w.buf()), w.offset);
        encoder.set_buffer(2, Some(bias.buf()), bias.offset);
        encoder.set_buffer(3, Some(out.buf()), out.offset);
        encoder.set_bytes(4, &(c_in as i32));
        encoder.set_bytes(5, &(c_out as i32));
        encoder.set_bytes(6, &(t_in as i32));
        encoder.set_bytes(7, &(t_out as i32));
        encoder.set_bytes(8, &(k as i32));
        encoder.set_bytes(9, &(stride as i32));
        encoder.set_bytes(10, &(padding as i32));
        encoder.set_bytes(11, &(dilation as i32));
        let grid = MTLSize { width: c_out, height: cdiv(t_out, 256), depth: 1 };
        let tg = MTLSize { width: 256, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn conv1d_k(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let kk = c_in * k;
        let matmul_ok = self.kernels.matmul.max_total_threads_per_threadgroup() >= 256;
        if matmul_ok && kk % 32 == 0 && c_out % 64 == 0 && t_out % 64 == 0 {
            self.conv1d_matmul(x, w, bias, out, c_in, c_out, t_in, t_out, k, stride, padding, dilation)
        } else if matmul_ok && kk % 32 == 0 && c_out % 64 == 0 {
            self.conv1d_matmul_npad(x, w, bias, out, c_in, c_out, t_in, t_out, k, stride, padding, dilation)
        } else {
            self.conv1d(x, w, bias, out, c_in, c_out, t_in, t_out, k, stride, padding, dilation)
        }
    }

    fn conv_transpose1d(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                        c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                        k: usize, stride: usize, padding: usize) -> Result<()> {
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.conv_transpose1d);
        encoder.set_buffer(0, Some(x.buf()), x.offset);
        encoder.set_buffer(1, Some(w.buf()), w.offset);
        encoder.set_buffer(2, Some(bias.buf()), bias.offset);
        encoder.set_buffer(3, Some(out.buf()), out.offset);
        encoder.set_bytes(4, &(c_in as i32));
        encoder.set_bytes(5, &(c_out as i32));
        encoder.set_bytes(6, &(t_in as i32));
        encoder.set_bytes(7, &(t_out as i32));
        encoder.set_bytes(8, &(k as i32));
        encoder.set_bytes(9, &(stride as i32));
        encoder.set_bytes(10, &(padding as i32));
        let grid = MTLSize { width: c_out, height: cdiv(t_out, 256), depth: 1 };
        let tg = MTLSize { width: 256, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
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
        let n = c_in * t_in;
        let lrelu_tmp = self.alloc(n)?;
        self.leaky_relu(x, &lrelu_tmp, n, 0.01)?;
        let kk = c_in * k;
        let matmul_ok = self.kernels.matmul.max_total_threads_per_threadgroup() >= 256;
        if matmul_ok && kk % 32 == 0 && c_out % 64 == 0 && t_out % 64 == 0 {
            self.conv1d_matmul(&lrelu_tmp, w, bias, out, c_in, c_out, t_in, t_out, k, stride, padding, dilation)
        } else if matmul_ok && kk % 32 == 0 && c_out % 64 == 0 {
            self.conv1d_matmul_npad(&lrelu_tmp, w, bias, out, c_in, c_out, t_in, t_out, k, stride, padding, dilation)
        } else {
            self.conv1d(&lrelu_tmp, w, bias, out, c_in, c_out, t_in, t_out, k, stride, padding, dilation)
        }
    }

    fn reflection_pad1d(&self, x: &GpuBuffer, out: &GpuBuffer, channels: usize, seq_len: usize) -> Result<()> {
        let n_out = channels * (seq_len + 1);
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.reflection_pad1d);
        encoder.set_buffer(0, Some(x.buf()), x.offset);
        encoder.set_buffer(1, Some(out.buf()), out.offset);
        encoder.set_bytes(2, &(channels as i32));
        encoder.set_bytes(3, &(seq_len as i32));
        let max_tg = self.kernels.reflection_pad1d.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n_out, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn im2col(&self, x: &GpuBuffer, out: &GpuBuffer,
              c_in: usize, t_in: usize, t_out: usize, k: usize,
              stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let n_elements = c_in * k * t_out;
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.im2col);
        encoder.set_buffer(0, Some(x.buf()), x.offset);
        encoder.set_buffer(1, Some(out.buf()), out.offset);
        encoder.set_bytes(2, &(c_in as i32));
        encoder.set_bytes(3, &(t_in as i32));
        encoder.set_bytes(4, &(t_out as i32));
        encoder.set_bytes(5, &(k as i32));
        encoder.set_bytes(6, &(stride as i32));
        encoder.set_bytes(7, &(padding as i32));
        encoder.set_bytes(8, &(dilation as i32));
        let max_tg = self.kernels.im2col.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n_elements, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn im2col_lrelu(&self, x: &GpuBuffer, out: &GpuBuffer,
                    c_in: usize, t_in: usize, t_out: usize, k: usize,
                    stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let n_elements = c_in * k * t_out;
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.im2col_lrelu);
        encoder.set_buffer(0, Some(x.buf()), x.offset);
        encoder.set_buffer(1, Some(out.buf()), out.offset);
        encoder.set_bytes(2, &(c_in as i32));
        encoder.set_bytes(3, &(t_in as i32));
        encoder.set_bytes(4, &(t_out as i32));
        encoder.set_bytes(5, &(k as i32));
        encoder.set_bytes(6, &(stride as i32));
        encoder.set_bytes(7, &(padding as i32));
        encoder.set_bytes(8, &(dilation as i32));
        let max_tg = self.kernels.im2col_lrelu.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n_elements, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn matmul_bias(&self, w: &GpuBuffer, col: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                   c_out: usize, t_out: usize, kk: usize) -> Result<()> {
        let pipeline = &self.kernels.matmul;

        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(w.buf()), w.offset);       // A
        encoder.set_buffer(1, Some(col.buf()), col.offset);   // B
        encoder.set_buffer(2, Some(out.buf()), out.offset);   // C
        encoder.set_bytes(3, &(c_out as i32));    // M
        encoder.set_bytes(4, &(t_out as i32));    // N
        encoder.set_bytes(5, &(kk as i32));       // K
        encoder.set_bytes(6, &(kk as i32));       // stride_am = K
        encoder.set_bytes(7, &1i32);              // stride_ak = 1
        encoder.set_bytes(8, &(t_out as i32));    // stride_bk = N
        encoder.set_bytes(9, &1i32);              // stride_bn = 1
        encoder.set_bytes(10, &(t_out as i32));   // stride_cm = N
        encoder.set_bytes(11, &1i32);             // stride_cn = 1
        let grid = MTLSize { width: cdiv(c_out, 64), height: cdiv(t_out, 64), depth: 1 };
        let max_tg = pipeline.max_total_threads_per_threadgroup() as usize;
        let tg = MTLSize { width: 1024.min(max_tg), height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);

        // Row-broadcast bias add on GPU: out[i] += bias[i / t_out]
        let n = c_out * t_out;
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.row_bias_add);
        encoder.set_buffer(0, Some(out.buf()), out.offset);
        encoder.set_buffer(1, Some(bias.buf()), bias.offset);
        encoder.set_buffer(2, Some(out.buf()), out.offset);
        encoder.set_bytes(3, &(n as i32));
        encoder.set_bytes(4, &(t_out as i32));
        let max_tg = self.kernels.row_bias_add.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }
}

impl KokoroGpuDecoder {
    /// Conv1d via im2col + matmul with N-padding for alignment.
    /// Used when K is 32-aligned but t_out is not.
    fn conv1d_matmul_npad(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                          c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                          k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let kk = c_in * k;
        let m_pad = cdiv(t_out, 64) * 64;

        let col_buf = self.alloc(kk * m_pad)?;
        self.im2col(x, &col_buf, c_in, t_in, m_pad, k, stride, padding, dilation)?;

        let tmp = self.alloc(c_out * m_pad)?;
        self.matmul_bias(w, &col_buf, bias, &tmp, c_out, m_pad, kk)?;

        // Copy valid columns: tmp[c, 0..t_out] → out[c, 0..t_out]
        self.im2col(&tmp, out, c_out, m_pad, t_out, 1, 1, 0, 1)?;
        Ok(())
    }
}
