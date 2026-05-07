//! GPU-accelerated Kokoro TTS decoder using Triton-compiled Metal kernels.
//!
//! Element-wise kernels: snake activation, fused AdaIN+snake, leaky_relu.
//! Conv1d stays on Candle's native Metal path (already GPU-dispatched).

use anyhow::Result;
use candle_core::{DType, Device, MetalDevice, Shape, Storage, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;

include!("../../kernels/out/generated/kokoro_metal_gen.rs");

fn cdiv(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// AdaIN via Candle ops (for sequences too long for the fused kernel).
fn adain_cpu(x: &Tensor, gamma: &Tensor, beta: &Tensor, _channels: usize) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let gamma = gamma.to_dtype(DType::F32)?;
    let beta = beta.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(2)?;
    let diff = x.broadcast_sub(&mean)?;
    let var = diff.sqr()?.mean_keepdim(2)?;
    let norm = diff.broadcast_div(&(var + 1e-5)?.sqrt()?)?;
    let gamma_unsq = gamma.unsqueeze(2)?;
    let beta_unsq = beta.unsqueeze(2)?;
    let scale = (gamma_unsq + 1.0)?;
    let result = norm.broadcast_mul(&scale)?.broadcast_add(&beta_unsq)?;
    result.to_dtype(DType::F16).map_err(Into::into)
}

pub struct KokoroGpuDecoder {
    kernels: KokoroKernels,
    device: MetalDevice,
}

impl KokoroGpuDecoder {
    pub fn new(device: &Device) -> Result<Option<Self>> {
        match device {
            Device::Metal(metal_device) => {
                let md = metal_device.clone();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    KokoroKernels::load(&md)
                }));
                match result {
                    Ok(Ok(kernels)) => Ok(Some(Self {
                        kernels,
                        device: metal_device.clone(),
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
            _ => Ok(None),
        }
    }

    /// Fused AdaIN + snake activation on GPU.
    /// x: [1, C, T], gamma/beta: [1, C] from Linear, alpha: [1, C, 1]
    pub fn adain_snake(
        &self, x: &Tensor, gamma: &Tensor, beta: &Tensor, alpha: &Tensor,
    ) -> Result<Tensor> {
        let (_, channels, seq_len) = x.dims3()?;

        if seq_len <= 1024 {
            let out = self.empty_f16((1, channels, seq_len))?;
            let x = x.contiguous()?.to_dtype(DType::F16)?;
            let gamma = gamma.reshape(channels)?.contiguous()?.to_dtype(DType::F16)?;
            let beta = beta.reshape(channels)?.contiguous()?.to_dtype(DType::F16)?;
            let alpha = alpha.reshape(channels)?.contiguous()?.to_dtype(DType::F16)?;
            self.dispatch_adain_snake(&self.kernels.adain_snake_1k, &x, &gamma, &beta, &alpha, &out, channels, seq_len)?;
            Ok(out)
        } else {
            let adain_result = adain_cpu(x, gamma, beta, channels)?;
            self.snake(&adain_result, alpha)
        }
    }

    /// Snake activation: x + sin²(αx)/α. x: [1, C, T], alpha: [1, C, 1]
    pub fn snake(&self, x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
        let (_, channels, seq_len) = x.dims3()?;
        let n_elements = channels * seq_len;
        let out = self.empty_f16((1, channels, seq_len))?;

        let x = x.contiguous()?.to_dtype(DType::F16)?;
        let alpha = alpha.reshape(channels)?.to_dtype(DType::F16)?;

        self.dispatch_snake(&x, &alpha, &out, n_elements, channels, seq_len)?;
        Ok(out)
    }

    /// LeakyReLU with specified slope.
    pub fn leaky_relu(&self, x: &Tensor, slope: f32) -> Result<Tensor> {
        let shape = x.shape().clone();
        let n_elements = shape.elem_count();
        let out = self.empty_f16(shape)?;

        let x = x.contiguous()?.to_dtype(DType::F16)?;

        let pipeline = if slope < 0.05 {
            &self.kernels.leaky_relu_001
        } else if slope < 0.15 {
            &self.kernels.leaky_relu_01
        } else {
            &self.kernels.leaky_relu_02
        };

        self.dispatch_elementwise(pipeline, &x, &out, n_elements)?;
        Ok(out)
    }


    fn empty_f16(&self, shape: impl Into<Shape>) -> Result<Tensor> {
        Ok(self.device.empty_tensor(shape, DType::F16)?)
    }

    // ── Dispatch helpers ──

    fn dispatch_adain_snake(
        &self, pipeline: &ComputePipeline,
        x: &Tensor, gamma: &Tensor, beta: &Tensor, alpha: &Tensor, out: &Tensor,
        n_channels: usize, seq_len: usize,
    ) -> Result<()> {
        let (sx, _) = x.storage_and_layout();
        let (sg, _) = gamma.storage_and_layout();
        let (sb, _) = beta.storage_and_layout();
        let (sa, _) = alpha.storage_and_layout();
        let (so, _) = out.storage_and_layout();
        match (&*sx, &*sg, &*sb, &*sa, &*so) {
            (Storage::Metal(mx), Storage::Metal(mg), Storage::Metal(mb), Storage::Metal(ma), Storage::Metal(mo)) => {
                let encoder = self.device.command_encoder()?;
                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_buffer(0, Some(mx.buffer()), 0);
                encoder.set_buffer(1, Some(mg.buffer()), 0);
                encoder.set_buffer(2, Some(mb.buffer()), 0);
                encoder.set_buffer(3, Some(ma.buffer()), 0);
                encoder.set_buffer(4, Some(mo.buffer()), 0);
                encoder.set_bytes(5, &(n_channels as i32));
                encoder.set_bytes(6, &(seq_len as i32));
                let grid = MTLSize { width: n_channels, height: 1, depth: 1 };
                let max_tg = pipeline.max_total_threads_per_threadgroup() as usize;
                let tg_width = seq_len.next_power_of_two().min(max_tg);
                let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
                encoder.dispatch_thread_groups(grid, tg);
                Ok(())
            }
            _ => anyhow::bail!("All tensors must be on Metal"),
        }
    }

    fn dispatch_snake(
        &self, x: &Tensor, alpha: &Tensor, out: &Tensor,
        n_elements: usize, n_channels: usize, seq_len: usize,
    ) -> Result<()> {
        let (sx, _) = x.storage_and_layout();
        let (sa, _) = alpha.storage_and_layout();
        let (so, _) = out.storage_and_layout();
        match (&*sx, &*sa, &*so) {
            (Storage::Metal(mx), Storage::Metal(ma), Storage::Metal(mo)) => {
                let encoder = self.device.command_encoder()?;
                encoder.set_compute_pipeline_state(&self.kernels.snake);
                encoder.set_buffer(0, Some(mx.buffer()), 0);
                encoder.set_buffer(1, Some(ma.buffer()), 0);
                encoder.set_buffer(2, Some(mo.buffer()), 0);
                encoder.set_bytes(3, &(n_elements as i32));
                encoder.set_bytes(4, &(n_channels as i32));
                encoder.set_bytes(5, &(seq_len as i32));
                let max_tg = self.kernels.snake.max_total_threads_per_threadgroup() as usize;
                let tg_width = 1024.min(max_tg);
                let grid = MTLSize { width: cdiv(n_elements, tg_width), height: 1, depth: 1 };
                let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
                encoder.dispatch_thread_groups(grid, tg);
                Ok(())
            }
            _ => anyhow::bail!("All tensors must be on Metal"),
        }
    }

    fn dispatch_elementwise(
        &self, pipeline: &ComputePipeline,
        x: &Tensor, out: &Tensor, n_elements: usize,
    ) -> Result<()> {
        let (sx, _) = x.storage_and_layout();
        let (so, _) = out.storage_and_layout();
        match (&*sx, &*so) {
            (Storage::Metal(mx), Storage::Metal(mo)) => {
                let encoder = self.device.command_encoder()?;
                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_buffer(0, Some(mx.buffer()), 0);
                encoder.set_buffer(1, Some(mo.buffer()), 0);
                encoder.set_bytes(2, &(n_elements as i32));
                let max_tg = pipeline.max_total_threads_per_threadgroup() as usize;
                let tg_width = 1024.min(max_tg);
                let grid = MTLSize { width: cdiv(n_elements, tg_width), height: 1, depth: 1 };
                let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
                encoder.dispatch_thread_groups(grid, tg);
                Ok(())
            }
            _ => anyhow::bail!("All tensors must be on Metal"),
        }
    }

}
