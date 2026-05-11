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

#[allow(dead_code)]
impl KokoroGpuDecoder {
    pub fn test_conv1d_f32io(&self) {
        self.test_adain_snake_f32();
        self.test_adain_snake_f32_large();
        self.test_f16_to_f32_roundtrip();
        // Stage 0 dimensions
        for &(k, dilation, padding) in &[(3usize, 1usize, 1usize), (7, 3, 9), (11, 5, 25)] {
            self.test_conv1d_f32io_params(256, 256, 1880, k, 1, padding, dilation);
        }
        // Stage 1 dimensions (128 channels, seq_len=11281)
        for &(k, dilation, padding) in &[(3usize, 1usize, 1usize), (3, 3, 3), (3, 5, 5), (7, 1, 3), (7, 3, 9), (7, 5, 15), (11, 1, 5), (11, 3, 15), (11, 5, 25)] {
            self.test_conv1d_f32io_params(128, 128, 11281, k, 1, padding, dilation);
        }
    }

    fn test_adain_snake_f32_large(&self) {
        // Test with stage 1 dimensions: 128 channels, seq_len=11281 (uses 32k stats kernel)
        let channels = 128usize;
        let seq_len = 11281;
        let n = channels * seq_len;

        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin() * 3.0).collect();
        let gamma_data: Vec<half::f16> = (0..channels).map(|i| half::f16::from_f32((i as f32 * 0.01).sin() * 0.5)).collect();
        let beta_data: Vec<half::f16> = (0..channels).map(|i| half::f16::from_f32((i as f32 * 0.02).cos() * 0.3)).collect();
        let alpha_data: Vec<half::f16> = (0..channels).map(|i| half::f16::from_f32(1.0 + (i as f32 * 0.005).sin())).collect();

        let x_buf = GpuBuffer::from_f32_data(&self.device, &x_data).unwrap();
        let gamma_buf = self.upload_f16(&gamma_data).unwrap();
        let beta_buf = self.upload_f16(&beta_data).unwrap();
        let alpha_buf = self.upload_f16(&alpha_data).unwrap();
        let out_buf = self.alloc_f32(n).unwrap();

        self.adain_snake_f32(&x_buf, &gamma_buf, &beta_buf, &alpha_buf, &out_buf, channels, seq_len).unwrap();
        self.device.wait_until_completed().unwrap();

        let ptr = out_buf.contents_ptr() as *const f32;
        let gpu_out: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec();

        let mut max_err: f32 = 0.0;
        let mut max_ch = 0;
        let mut max_t = 0;
        for ch in 0..channels {
            let base = ch * seq_len;
            let slice = &x_data[base..base + seq_len];
            let sum: f64 = slice.iter().map(|v| *v as f64).sum();
            let mean = (sum / seq_len as f64) as f32;
            let var: f64 = slice.iter().map(|v| { let d = *v as f64 - mean as f64; d * d }).sum::<f64>() / seq_len as f64;
            let rstd = 1.0 / (var as f32 + 1e-5f32).sqrt();
            let g = gamma_data[ch].to_f32();
            let b = beta_data[ch].to_f32();
            let a = alpha_data[ch].to_f32();
            let scale = (g + 1.0) * rstd;
            let inv_a = 1.0 / (a + 1e-9);
            for t in 0..seq_len {
                let val = x_data[base + t];
                let styled = scale * (val - mean) + b;
                let ax = a * styled;
                let s = ax.sin();
                let expected = styled + s * s * inv_a;
                let gpu_val = gpu_out[base + t];
                let d = (gpu_val - expected).abs();
                if d > max_err { max_err = d; max_ch = ch; max_t = t; }
            }
        }
        eprintln!("[adain_snake_f32 LARGE test] channels={channels} seq_len={seq_len} max_err={max_err:.6} (ch={max_ch} t={max_t} gpu={:.4} expected)",
            gpu_out[max_ch * seq_len + max_t]);
    }

    fn test_instance_norm_stats_f32in_v2(&self) {
        let channels = 4usize;
        let seq_len = 100;
        let n = channels * seq_len;

        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() * 3.0).collect();
        let x_f32_buf = GpuBuffer::from_f32_data(&self.device, &x_data).unwrap();

        let stats_buf = self.alloc_f32(channels * 2).unwrap();
        let encoder = self.device.command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&self.kernels.instance_norm_stats_f32in_2k);
        encoder.set_buffer(0, Some(x_f32_buf.buf()), x_f32_buf.offset);
        encoder.set_buffer(1, Some(stats_buf.buf()), stats_buf.offset);
        encoder.set_bytes(2, &(channels as i32));
        encoder.set_bytes(3, &(seq_len as i32));
        let max_tg = self.kernels.instance_norm_stats_f32in_2k.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: channels, height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);

        self.device.wait_until_completed().unwrap();
        let ptr = stats_buf.contents_ptr() as *const f32;
        let stats: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, channels * 2) }.to_vec();

        let mut max_err: f32 = 0.0;
        for ch in 0..channels {
            let base = ch * seq_len;
            let slice = &x_data[base..base + seq_len];
            let sum: f32 = slice.iter().sum();
            let mean = sum / seq_len as f32;
            let var: f32 = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / seq_len as f32;
            let rstd = 1.0 / (var + 1e-5f32).sqrt();
            let d_mean = (stats[ch * 2] - mean).abs();
            let d_rstd = (stats[ch * 2 + 1] - rstd).abs();
            if d_mean > max_err { max_err = d_mean; }
            if d_rstd > max_err { max_err = d_rstd; }
        }
        eprintln!("[instance_norm_stats_f32in test] channels={channels} seq_len={seq_len} max_err={max_err:.6}");
    }

    fn test_adain_snake_f16_direct(&self) {
        let channels = 4usize;
        let seq_len = 100;
        let n = channels * seq_len;

        let x_f16: Vec<half::f16> = (0..n).map(|i| half::f16::from_f32((i as f32 * 0.001).sin() * 3.0)).collect();
        let gamma_data: Vec<half::f16> = (0..channels).map(|i| half::f16::from_f32((i as f32 * 0.01).sin() * 0.5)).collect();
        let beta_data: Vec<half::f16> = (0..channels).map(|i| half::f16::from_f32((i as f32 * 0.02).cos() * 0.3)).collect();
        let alpha_data: Vec<half::f16> = (0..channels).map(|i| half::f16::from_f32(1.0 + (i as f32 * 0.005).sin())).collect();

        let x_buf = self.upload_f16(&x_f16).unwrap();
        let gamma_buf = self.upload_f16(&gamma_data).unwrap();
        let beta_buf = self.upload_f16(&beta_data).unwrap();
        let alpha_buf = self.upload_f16(&alpha_data).unwrap();
        let out_buf = self.alloc(n).unwrap();

        self.adain_snake(&x_buf, &gamma_buf, &beta_buf, &alpha_buf, &out_buf, channels, seq_len).unwrap();
        self.device.wait_until_completed().unwrap();

        let gpu_out = self.download_f16(&out_buf, n).unwrap();

        // CPU reference from f16 input
        let x_f32: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();
        let mut max_err: f32 = 0.0;
        for ch in 0..channels {
            let base = ch * seq_len;
            let slice = &x_f32[base..base + seq_len];
            let sum: f32 = slice.iter().sum();
            let mean = sum / seq_len as f32;
            let var: f32 = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / seq_len as f32;
            let rstd = 1.0 / (var + 1e-5f32).sqrt();
            let g = gamma_data[ch].to_f32();
            let b = beta_data[ch].to_f32();
            let a = alpha_data[ch].to_f32();
            let scale = (g + 1.0) * rstd;
            let inv_a = 1.0 / (a + 1e-9);
            for t in 0..seq_len {
                let val = x_f32[base + t];
                let styled = scale * (val - mean) + b;
                let ax = a * styled;
                let s = ax.sin();
                let expected = styled + s * s * inv_a;
                let d = (gpu_out[base + t].to_f32() - expected).abs();
                if d > max_err { max_err = d; }
            }
        }
        eprintln!("[adain_snake_f16 direct test] channels={channels} seq_len={seq_len} max_err={max_err:.6} gpu[0]={:.4}",
            gpu_out[0].to_f32());
    }

    fn test_f16_to_f32_roundtrip(&self) {
        let n = 1024;
        let data: Vec<half::f16> = (0..n).map(|i| half::f16::from_f32(i as f32 * 0.1 - 50.0)).collect();
        let f16_buf = self.upload_f16(&data).unwrap();
        let f32_buf = self.alloc_f32(n).unwrap();
        self.f16_to_f32(&f16_buf, &f32_buf, n).unwrap();
        self.device.wait_until_completed().unwrap();
        let ptr = f32_buf.contents_ptr() as *const f32;
        let out: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec();
        let mut max_err: f32 = 0.0;
        for i in 0..n {
            let expected = data[i].to_f32();
            let d = (out[i] - expected).abs();
            if d > max_err { max_err = d; }
        }
        eprintln!("[f16_to_f32 test] n={n} max_err={max_err:.6} out[100]={:.4} expected={:.4}",
            out[100], data[100].to_f32());

        // Test add_f32: a + b where a=f32_buf (from f16_to_f32), b=zeros
        let zeros = self.alloc_f32(n).unwrap();
        let sum_buf = self.alloc_f32(n).unwrap();
        self.add_f32(&f32_buf, &zeros, &sum_buf, n).unwrap();
        self.device.wait_until_completed().unwrap();
        let ptr = sum_buf.contents_ptr() as *const f32;
        let sum_out: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec();
        let mut max_err2: f32 = 0.0;
        for i in 0..n {
            let d = (sum_out[i] - out[i]).abs();
            if d > max_err2 { max_err2 = d; }
        }
        eprintln!("[add_f32 test] n={n} max_err={max_err2:.6} sum[100]={:.4} expected={:.4}",
            sum_out[100], out[100]);
    }

    fn test_instance_norm_stats_f32in(&self) {
        let channels = 4usize;
        let seq_len = 100;
        let n = channels * seq_len;

        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() * 3.0).collect();
        let x_buf = {
            let buffer = self.device.allocate_zeros(n * 4).unwrap();
            let ptr = buffer.contents() as *mut f32;
            unsafe { std::ptr::copy_nonoverlapping(x_data.as_ptr(), ptr, n); }
            GpuBuffer::from_arc(buffer)
        };
        let stats_buf = self.alloc_f32(channels * 2).unwrap();

        // Use 2k variant (seq_len <= 2048)
        let encoder = self.device.command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&self.kernels.instance_norm_stats_f32in_2k);
        encoder.set_buffer(0, Some(x_buf.buf()), x_buf.offset);
        encoder.set_buffer(1, Some(stats_buf.buf()), stats_buf.offset);
        encoder.set_bytes(2, &(channels as i32));
        encoder.set_bytes(3, &(seq_len as i32));
        let max_tg = self.kernels.instance_norm_stats_f32in_2k.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: channels, height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);

        self.device.wait_until_completed().unwrap();
        let ptr = stats_buf.contents_ptr() as *const f32;
        let stats: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, channels * 2) }.to_vec();

        for ch in 0..channels {
            let base = ch * seq_len;
            let slice = &x_data[base..base + seq_len];
            let sum: f32 = slice.iter().sum();
            let mean = sum / seq_len as f32;
            let var: f32 = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / seq_len as f32;
            let rstd = 1.0 / (var + 1e-5f32).sqrt();
            let gpu_mean = stats[ch * 2];
            let gpu_rstd = stats[ch * 2 + 1];
            if ch < 2 {
                eprintln!("  [stats ch{ch}] mean: gpu={gpu_mean:.6} cpu={mean:.6} | rstd: gpu={gpu_rstd:.6} cpu={rstd:.6}");
            }
        }
    }

    fn test_adain_snake_f32(&self) {
        let channels = 4usize;
        let seq_len = 100;
        let n = channels * seq_len;

        // Create test f32 input
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin() * 3.0).collect();
        let gamma_data: Vec<half::f16> = (0..channels).map(|i| half::f16::from_f32((i as f32 * 0.01).sin() * 0.5)).collect();
        let beta_data: Vec<half::f16> = (0..channels).map(|i| half::f16::from_f32((i as f32 * 0.02).cos() * 0.3)).collect();
        let alpha_data: Vec<half::f16> = (0..channels).map(|i| half::f16::from_f32(1.0 + (i as f32 * 0.005).sin())).collect();

        let x_buf = GpuBuffer::from_f32_data(&self.device, &x_data).unwrap();
        let gamma_buf = self.upload_f16(&gamma_data).unwrap();
        let beta_buf = self.upload_f16(&beta_data).unwrap();
        let alpha_buf = self.upload_f16(&alpha_data).unwrap();
        let out_buf = self.alloc_f32(n).unwrap();

        self.adain_snake_f32(&x_buf, &gamma_buf, &beta_buf, &alpha_buf, &out_buf, channels, seq_len).unwrap();
        self.device.wait_until_completed().unwrap();

        let ptr = out_buf.contents_ptr() as *const f32;
        let gpu_out: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec();

        // CPU reference — use f16-quantized input for stats (matches GPU path)
        let x_f16: Vec<f32> = x_data.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
        let mut max_err: f32 = 0.0;
        for ch in 0..channels {
            let base = ch * seq_len;
            let slice = &x_f16[base..base + seq_len];
            let sum: f32 = slice.iter().sum();
            let mean = sum / seq_len as f32;
            let var: f32 = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / seq_len as f32;
            let rstd = 1.0 / (var + 1e-5f32).sqrt();
            let g = gamma_data[ch].to_f32();
            let b = beta_data[ch].to_f32();
            let a = alpha_data[ch].to_f32();
            let scale = (g + 1.0) * rstd;
            let inv_a = 1.0 / (a + 1e-9);
            for t in 0..seq_len {
                let val = x_f16[base + t];
                let styled = scale * (val - mean) + b;
                let ax = a * styled;
                let s = ax.sin();
                let expected = styled + s * s * inv_a;
                let gpu_val = gpu_out[base + t];
                let d = (gpu_val - expected).abs();
                if d > max_err { max_err = d; }
            }
        }
        eprintln!("[adain_snake_f32 test] channels={channels} seq_len={seq_len} max_err={max_err:.6} gpu[0]={:.4}",
            gpu_out[0]);
    }

    fn test_conv1d_f16_params(&self, c_in: usize, c_out: usize, t_in: usize, k: usize, stride: usize, padding: usize, dilation: usize) {
        let t_out = (t_in + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
        let n_in = c_in * t_in;
        let n_out = c_out * t_out;
        let n_w = c_out * c_in * k;

        let x_data: Vec<half::f16> = (0..n_in).map(|i| half::f16::from_f32((i as f32 * 0.001).sin() * 2.0)).collect();
        let w_data: Vec<half::f16> = (0..n_w).map(|i| half::f16::from_f32((i as f32 * 0.0001).cos() * 0.1)).collect();
        let bias_data: Vec<half::f16> = (0..c_out).map(|i| half::f16::from_f32(i as f32 * 0.001)).collect();

        let x_buf = self.upload_f16(&x_data).unwrap();
        let w_buf = self.upload_f16(&w_data).unwrap();
        let bias_buf = self.upload_f16(&bias_data).unwrap();
        let out_buf = self.alloc(n_out).unwrap();

        self.conv1d(&x_buf, &w_buf, &bias_buf, &out_buf, c_in, c_out, t_in, t_out, k, stride, padding, dilation).unwrap();
        self.device.wait_until_completed().unwrap();

        let gpu_out = self.download_f16(&out_buf, n_out).unwrap();

        // CPU reference (from f16 inputs, same as GPU)
        let mut max_err: f32 = 0.0;
        let mut max_idx = 0;
        for co in 0..1.min(c_out) {  // just check channel 0
            for t in 0..t_out {
                let mut acc = 0.0f32;
                for c in 0..c_in {
                    for ki in 0..k {
                        let t_in_pos = t as i32 * stride as i32 - padding as i32 + ki as i32 * dilation as i32;
                        if t_in_pos >= 0 && (t_in_pos as usize) < t_in {
                            let x_val = x_data[c * t_in + t_in_pos as usize].to_f32();
                            let w_val = w_data[co * c_in * k + c * k + ki].to_f32();
                            acc += x_val * w_val;
                        }
                    }
                }
                acc += bias_data[co].to_f32();
                let idx = co * t_out + t;
                let d = (gpu_out[idx].to_f32() - acc).abs();
                if d > max_err { max_err = d; max_idx = idx; }
            }
        }
        eprintln!("[conv1d_f16 test] K={k} C_in={c_in} dil={dilation} max_err={max_err:.6} at idx={max_idx} gpu={:.4} cpu_ref",
            gpu_out[max_idx].to_f32());
    }

    fn test_conv1d_f32io_params(&self, c_in: usize, c_out: usize, t_in: usize, k: usize, stride: usize, padding: usize, dilation: usize) {
        let t_out = (t_in + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

        // Create test data: simple pattern
        let n_in = c_in * t_in;
        let n_out = c_out * t_out;
        let n_w = c_out * c_in * k;

        // Random-ish f32 input
        let x_data: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.001).sin() * 2.0).collect();
        let w_data: Vec<half::f16> = (0..n_w).map(|i| half::f16::from_f32((i as f32 * 0.0001).cos() * 0.1)).collect();
        let bias_data: Vec<half::f16> = (0..c_out).map(|i| half::f16::from_f32(i as f32 * 0.001)).collect();

        let x_buf = GpuBuffer::from_f32_data(&self.device, &x_data).unwrap();
        let w_buf = self.upload_f16(&w_data).unwrap();
        let bias_buf = self.upload_f16(&bias_data).unwrap();
        let out_buf = self.alloc_f32(n_out).unwrap();

        // Run GPU kernel
        self.conv1d_f32(&x_buf, &w_buf, &bias_buf, &out_buf, c_in, c_out, t_in, t_out, k, stride, padding, dilation).unwrap();
        self.device.wait_until_completed().unwrap();

        // Read back
        let ptr = out_buf.contents_ptr() as *const f32;
        let gpu_out: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, n_out) }.to_vec();

        // CPU reference
        let mut cpu_out = vec![0.0f32; n_out];
        for co in 0..c_out {
            for t in 0..t_out {
                let mut acc = 0.0f32;
                for c in 0..c_in {
                    for ki in 0..k {
                        let t_in_pos = t as i32 * stride as i32 - padding as i32 + ki as i32 * dilation as i32;
                        if t_in_pos >= 0 && (t_in_pos as usize) < t_in {
                            let x_val = x_data[c * t_in + t_in_pos as usize];
                            let w_val = w_data[co * c_in * k + c * k + ki].to_f32();
                            acc += x_val * w_val;
                        }
                    }
                }
                acc += bias_data[co].to_f32();
                cpu_out[co * t_out + t] = acc;
            }
        }

        // Compare
        let mut max_err: f32 = 0.0;
        let mut max_idx = 0;
        for i in 0..n_out {
            let d = (gpu_out[i] - cpu_out[i]).abs();
            if d > max_err { max_err = d; max_idx = i; }
        }
        eprintln!("[conv1d_f32io test] K={k} C_in={c_in} max_err={max_err:.6} at idx={max_idx} gpu={:.4} cpu={:.4}",
            gpu_out[max_idx], cpu_out[max_idx]);
    }
}

impl KokoroGpuBackend for KokoroGpuDecoder {
    type Buf = GpuBuffer;

    fn alloc(&self, count: usize) -> Result<GpuBuffer> {
        GpuBuffer::alloc_shared_f16(&self.device, count).map_err(Into::into)
    }

    fn upload_f16(&self, data: &[half::f16]) -> Result<GpuBuffer> {
        GpuBuffer::from_f16_data(&self.device, data).map_err(Into::into)
    }

    fn upload_f32(&self, data: &[f32]) -> Result<GpuBuffer> {
        GpuBuffer::from_f32_data(&self.device, data).map_err(Into::into)
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
        self.device.wait_until_completed()?;
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
        let pipeline = if seq_len <= 1024 {
            &self.kernels.adain_snake_1k
        } else if seq_len <= 2048 {
            &self.kernels.adain_snake_2k
        } else {
            &self.kernels.adain_snake_8k
        };
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(x.buf()), x.offset);
        encoder.set_buffer(1, Some(gamma.buf()), gamma.offset);
        encoder.set_buffer(2, Some(beta.buf()), beta.offset);
        encoder.set_buffer(3, Some(alpha.buf()), alpha.offset);
        encoder.set_buffer(4, Some(out.buf()), out.offset);
        encoder.set_bytes(5, &(channels as i32));
        encoder.set_bytes(6, &(seq_len as i32));
        let grid = MTLSize { width: channels, height: 1, depth: 1 };
        let max_tg = pipeline.max_total_threads_per_threadgroup() as usize;
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
        // Single encoder for matmul + bias add → sequential execution guaranteed
        let encoder = self.device.command_encoder()?;

        encoder.set_compute_pipeline_state(&self.kernels.matmul);
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
        let max_tg = self.kernels.matmul.max_total_threads_per_threadgroup() as usize;
        let tg = MTLSize { width: 1024.min(max_tg), height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);

        // Row-broadcast bias add (reads/writes out written by matmul above)
        let n = c_out * t_out;
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

    fn has_f32_intermediates(&self) -> bool { true }

    fn download_f32(&self, buf: &GpuBuffer, count: usize) -> Result<Vec<f32>> {
        self.device.wait_until_completed()?;
        let ptr = buf.contents_ptr() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        Ok(slice.to_vec())
    }

    fn alloc_f32(&self, count: usize) -> Result<GpuBuffer> {
        GpuBuffer::alloc_shared_f32(&self.device, count).map_err(Into::into)
    }

    fn f16_to_f32(&self, x: &GpuBuffer, out: &GpuBuffer, n: usize) -> Result<()> {
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.f16_to_f32);
        encoder.set_buffer(0, Some(x.buf()), x.offset);
        encoder.set_buffer(1, Some(out.buf()), out.offset);
        encoder.set_bytes(2, &(n as i32));
        let max_tg = self.kernels.f16_to_f32.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn f32_to_f16(&self, x: &GpuBuffer, out: &GpuBuffer, n: usize) -> Result<()> {
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.f32_to_f16);
        encoder.set_buffer(0, Some(x.buf()), x.offset);
        encoder.set_buffer(1, Some(out.buf()), out.offset);
        encoder.set_bytes(2, &(n as i32));
        let max_tg = self.kernels.f32_to_f16.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn conv1d_f32(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                  c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                  k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.conv1d_f32io);
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

    fn adain_snake_f32(&self, x: &GpuBuffer, gamma: &GpuBuffer, beta: &GpuBuffer,
                       alpha: &GpuBuffer, out: &GpuBuffer,
                       channels: usize, seq_len: usize) -> Result<()> {
        let n_elements = channels * seq_len;
        let stats_buf = self.alloc_f32(channels * 2)?;

        // 1) Compute stats from f32 input
        let stats_pipeline = if seq_len <= 2048 {
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
        let max_tg = stats_pipeline.max_total_threads_per_threadgroup() as usize;
        if max_tg >= 1024 {
            let encoder = self.device.command_encoder()?;
            encoder.set_compute_pipeline_state(stats_pipeline);
            encoder.set_buffer(0, Some(x.buf()), x.offset);
            encoder.set_buffer(1, Some(stats_buf.buf()), stats_buf.offset);
            encoder.set_bytes(2, &(channels as i32));
            encoder.set_bytes(3, &(seq_len as i32));
            let grid = MTLSize { width: channels, height: 1, depth: 1 };
            let tg = MTLSize { width: 1024, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid, tg);
        } else {
            // Twopass stats kernels require 1024 threads; compute on CPU when unavailable
            let x_data = self.download_f32(x, n_elements)?;
            let stats_ptr = stats_buf.contents_ptr() as *mut f32;
            let stats_slice = unsafe { std::slice::from_raw_parts_mut(stats_ptr, channels * 2) };
            for ch in 0..channels {
                let base = ch * seq_len;
                let slice = &x_data[base..base + seq_len];
                let sum: f64 = slice.iter().map(|v| *v as f64).sum();
                let mean = (sum / seq_len as f64) as f32;
                let var: f64 = slice.iter().map(|v| { let d = *v as f64 - mean as f64; d * d }).sum::<f64>() / seq_len as f64;
                let rstd = 1.0 / (var as f32 + 1e-5f32).sqrt();
                stats_slice[ch * 2] = mean;
                stats_slice[ch * 2 + 1] = rstd;
            }
        }

        // 2) Normalize + style + snake (reads stats_buf written above)
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.norm_style_snake_f32io);
        encoder.set_buffer(0, Some(x.buf()), x.offset);
        encoder.set_buffer(1, Some(stats_buf.buf()), stats_buf.offset);
        encoder.set_buffer(2, Some(gamma.buf()), gamma.offset);
        encoder.set_buffer(3, Some(beta.buf()), beta.offset);
        encoder.set_buffer(4, Some(alpha.buf()), alpha.offset);
        encoder.set_buffer(5, Some(out.buf()), out.offset);
        encoder.set_bytes(6, &(n_elements as i32));
        encoder.set_bytes(7, &(channels as i32));
        encoder.set_bytes(8, &(seq_len as i32));
        let max_tg = self.kernels.norm_style_snake_f32io.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n_elements, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn add_f32(&self, a: &GpuBuffer, b: &GpuBuffer, out: &GpuBuffer, n: usize) -> Result<()> {
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.add_f32);
        encoder.set_buffer(0, Some(a.buf()), a.offset);
        encoder.set_buffer(1, Some(b.buf()), b.offset);
        encoder.set_buffer(2, Some(out.buf()), out.offset);
        encoder.set_bytes(3, &(n as i32));
        let max_tg = self.kernels.add_f32.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn scale_third_f32(&self, x: &GpuBuffer, out: &GpuBuffer, n: usize) -> Result<()> {
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.scale_third_f32);
        encoder.set_buffer(0, Some(x.buf()), x.offset);
        encoder.set_buffer(1, Some(out.buf()), out.offset);
        encoder.set_bytes(2, &(n as i32));
        let max_tg = self.kernels.scale_third_f32.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn leaky_relu_f32(&self, x: &GpuBuffer, out: &GpuBuffer, n: usize, slope: f32) -> Result<()> {
        let pipeline = if slope == 0.01 {
            &self.kernels.leaky_relu_f32_001
        } else {
            &self.kernels.leaky_relu_f32_01
        };
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(x.buf()), x.offset);
        encoder.set_buffer(1, Some(out.buf()), out.offset);
        encoder.set_bytes(2, &(n as i32));
        let max_tg = pipeline.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n, tg_width), height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn conv_transpose1d_f32io_lrelu(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                                    c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                                    k: usize, stride: usize, padding: usize) -> Result<()> {
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.conv_transpose1d_f32io_lrelu);
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
        encoder.set_bytes(11, &0i32);
        let grid = MTLSize { width: c_out, height: cdiv(t_out, 256), depth: 1 };
        let tg = MTLSize { width: 256, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn conv_transpose1d_f32io(&self, x: &GpuBuffer, w: &GpuBuffer, bias: &GpuBuffer, out: &GpuBuffer,
                              c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                              k: usize, stride: usize, padding: usize) -> Result<()> {
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.conv_transpose1d_f32io);
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
        encoder.set_bytes(11, &0i32);
        let grid = MTLSize { width: c_out, height: cdiv(t_out, 256), depth: 1 };
        let tg = MTLSize { width: 256, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    fn reflection_pad1d_f32(&self, x: &GpuBuffer, out: &GpuBuffer, channels: usize, seq_len: usize) -> Result<()> {
        let n_out = channels * (seq_len + 1);
        let encoder = self.device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.kernels.reflection_pad1d_f32);
        encoder.set_buffer(0, Some(x.buf()), x.offset);
        encoder.set_buffer(1, Some(out.buf()), out.offset);
        encoder.set_bytes(2, &(channels as i32));
        encoder.set_bytes(3, &(seq_len as i32));
        let max_tg = self.kernels.reflection_pad1d_f32.max_total_threads_per_threadgroup() as usize;
        let tg_width = 1024.min(max_tg);
        let grid = MTLSize { width: cdiv(n_out, tg_width), height: 1, depth: 1 };
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
