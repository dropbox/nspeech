//! Metal backend for Kokoro TTS decoder.
//!
//! Implements KokoroGpuBackend using Triton-compiled Metal kernels.
//! Weights are cached on first upload; activations upload each call.

use anyhow::Result;
use candle_core::{Device, MetalDevice};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;
use std::collections::HashMap;
use std::cell::RefCell;
use std::sync::Arc;

use super::gpu_backend::KokoroGpuBackend;

type Buffer = candle_metal_kernels::metal::Buffer;

include!("../../kernels/out/generated/kokoro_metal_gen.rs");

fn cdiv(a: usize, b: usize) -> usize { (a + b - 1) / b }

pub struct KokoroGpuDecoder {
    kernels: KokoroKernels,
    pub(super) device: MetalDevice,
    weight_cache: RefCell<HashMap<usize, Arc<Buffer>>>,
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

impl KokoroGpuBackend for KokoroGpuDecoder {
    type Buf = Arc<Buffer>;

    fn alloc(&self, count: usize) -> Result<Arc<Buffer>> {
        self.device.allocate_zeros(count * 2).map_err(Into::into)
    }

    fn upload_f16(&self, data: &[half::f16]) -> Result<Arc<Buffer>> {
        self.device.new_buffer_with_data(data).map_err(Into::into)
    }

    fn upload_weight(&self, id: usize, data: &[half::f16]) -> Result<Arc<Buffer>> {
        {
            let cache = self.weight_cache.borrow();
            if let Some(buf) = cache.get(&id) {
                return Ok(buf.clone());
            }
        }
        let buf = self.device.new_buffer_with_data(data)?;
        self.weight_cache.borrow_mut().insert(id, buf.clone());
        Ok(buf)
    }

    fn download_f16(&self, buf: &Arc<Buffer>, count: usize) -> Result<Vec<half::f16>> {
        let ptr = buf.contents() as *const half::f16;
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        Ok(slice.to_vec())
    }

    fn add(&self, a: &Arc<Buffer>, b: &Arc<Buffer>, n: usize) -> Result<Arc<Buffer>> {
        let out = self.alloc(n)?;
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.add);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_bytes(3, &(n as i32));
        let max_tg = self.kernels.add.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(out)
    }

    fn scale(&self, x: &Arc<Buffer>, n: usize, _s: f32) -> Result<Arc<Buffer>> {
        let out = self.alloc(n)?;
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.scale_third);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(&out), 0);
        encoder.set_bytes(2, &(n as i32));
        let max_tg = self.kernels.scale_third.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(out)
    }

    fn leaky_relu(&self, x: &Arc<Buffer>, out: &Arc<Buffer>, n_elements: usize, slope: f32) -> Result<()> {
        let pipeline = if slope < 0.05 {
            &self.kernels.leaky_relu_001
        } else if slope < 0.15 {
            &self.kernels.leaky_relu_01
        } else {
            &self.kernels.leaky_relu_02
        };
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(out), 0);
        encoder.set_bytes(2, &(n_elements as i32));
        let max_tg = pipeline.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n_elements, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn snake(&self, x: &Arc<Buffer>, alpha: &Arc<Buffer>, out: &Arc<Buffer>,
             n_elements: usize, channels: usize, seq_len: usize) -> Result<()> {
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.snake);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(alpha), 0);
        encoder.set_buffer(2, Some(out), 0);
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

    fn adain_snake(&self, x: &Arc<Buffer>, gamma: &Arc<Buffer>, beta: &Arc<Buffer>,
                   alpha: &Arc<Buffer>, out: &Arc<Buffer>,
                   channels: usize, seq_len: usize) -> Result<()> {
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.adain_snake_1k);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(gamma), 0);
        encoder.set_buffer(2, Some(beta), 0);
        encoder.set_buffer(3, Some(alpha), 0);
        encoder.set_buffer(4, Some(out), 0);
        encoder.set_bytes(5, &(channels as i32));
        encoder.set_bytes(6, &(seq_len as i32));
        let grid = MTLSize { width: channels, height: 1, depth: 1 };
        let max_tg = self.kernels.adain_snake_1k.max_total_threads_per_threadgroup() as usize;
        let tg_width = seq_len.next_power_of_two().min(max_tg);
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn conv1d(&self, x: &Arc<Buffer>, w: &Arc<Buffer>, bias: &Arc<Buffer>, out: &Arc<Buffer>,
              c_in: usize, c_out: usize, t_in: usize, t_out: usize,
              k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.conv1d);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(w), 0);
        encoder.set_buffer(2, Some(bias), 0);
        encoder.set_buffer(3, Some(out), 0);
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

    fn conv1d_k(&self, x: &Arc<Buffer>, w: &Arc<Buffer>, bias: &Arc<Buffer>, out: &Arc<Buffer>,
                c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let pipeline = match k {
            3 => &self.kernels.conv1d_k3,
            7 => &self.kernels.conv1d_k7,
            11 => &self.kernels.conv1d_k11,
            _ => return self.conv1d(x, w, bias, out, c_in, c_out, t_in, t_out, k, stride, padding, dilation),
        };
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(w), 0);
        encoder.set_buffer(2, Some(bias), 0);
        encoder.set_buffer(3, Some(out), 0);
        encoder.set_bytes(4, &(c_in as i32));
        encoder.set_bytes(5, &(c_out as i32));
        encoder.set_bytes(6, &(t_in as i32));
        encoder.set_bytes(7, &(t_out as i32));
        encoder.set_bytes(8, &(stride as i32));
        encoder.set_bytes(9, &(padding as i32));
        encoder.set_bytes(10, &(dilation as i32));
        let grid = MTLSize { width: c_out, height: cdiv(t_out, 256), depth: 1 };
        let tg = MTLSize { width: 256, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn conv_transpose1d(&self, x: &Arc<Buffer>, w: &Arc<Buffer>, bias: &Arc<Buffer>, out: &Arc<Buffer>,
                        c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                        k: usize, stride: usize, padding: usize) -> Result<()> {
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.conv_transpose1d);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(w), 0);
        encoder.set_buffer(2, Some(bias), 0);
        encoder.set_buffer(3, Some(out), 0);
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

    fn conv_transpose1d_lrelu(&self, x: &Arc<Buffer>, w: &Arc<Buffer>, bias: &Arc<Buffer>, out: &Arc<Buffer>,
                              c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                              k: usize, stride: usize, padding: usize) -> Result<()> {
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.conv_transpose1d_lrelu);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(w), 0);
        encoder.set_buffer(2, Some(bias), 0);
        encoder.set_buffer(3, Some(out), 0);
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

    fn conv1d_lrelu001(&self, x: &Arc<Buffer>, w: &Arc<Buffer>, bias: &Arc<Buffer>, out: &Arc<Buffer>,
                       c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                       k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.conv1d_lrelu001);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(w), 0);
        encoder.set_buffer(2, Some(bias), 0);
        encoder.set_buffer(3, Some(out), 0);
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

    fn reflection_pad1d(&self, x: &Arc<Buffer>, out: &Arc<Buffer>, channels: usize, seq_len: usize) -> Result<()> {
        let n_out = channels * (seq_len + 1);
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.reflection_pad1d);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(out), 0);
        encoder.set_bytes(2, &(channels as i32));
        encoder.set_bytes(3, &(seq_len as i32));
        let max_tg = self.kernels.reflection_pad1d.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n_out, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }
}
