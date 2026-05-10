//! ISTFTNet decoder — direct waveform synthesis via inverse STFT.
//!
//! Structure (from safetensors keys):
//!   decoder.encode — single AdaINResBlock (in=514, out=1024)
//!   decoder.decode.{0,1,2,3} — AdaINResBlocks (in=1090, out varies)
//!   decoder.asr_res.0 — Conv1d [64, 512, 1] (ASR feature projection)
//!   decoder.F0_conv — Conv1d [1, 1, 3]
//!   decoder.N_conv — Conv1d [1, 1, 3]
//!   decoder.generator.ups.{0,1} — ConvTranspose1d
//!   decoder.generator.resblocks.{0..5} — Snake ResBlocks with AdaIN
//!   decoder.generator.noise_res.{0,1} — Noise ResBlocks
//!   decoder.generator.noise_convs.{0,1} — Noise injection convs
//!   decoder.generator.conv_post — Final conv [22, 128, 7]
//!   decoder.generator.m_source — Sine source generator

use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::config::KokoroConfig;

#[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
use super::gpu_backend::KokoroGpuBackend;
#[cfg(feature = "triton-metal")]
use super::gpu_decoder::KokoroGpuDecoder;
#[cfg(feature = "triton-d3d12")]
use super::gpu_decoder_d3d12::KokoroGpuDecoderD3D12;


pub struct ISTFTNetDecoder {
    // Encoder/decoder blocks
    encode: DecoderBlock,
    decode_blocks: Vec<DecoderBlock>,
    // Feature projections
    asr_res_weight: Tensor,
    asr_res_bias: Tensor,
    f0_conv_weight: Tensor,
    f0_conv_bias: Tensor,
    n_conv_weight: Tensor,
    n_conv_bias: Tensor,
    // Generator
    generator: Generator,
}

#[allow(dead_code)]
struct DecoderBlock {
    conv1_weight: Tensor,
    conv1_bias: Tensor,
    conv2_weight: Tensor,
    conv2_bias: Tensor,
    conv1x1_weight: Option<Tensor>,
    norm1_fc: Linear,
    norm2_fc: Linear,
    pool_weight: Option<Tensor>,
    pool_bias: Option<Tensor>,
}

struct Generator {
    ups: Vec<(Tensor, Tensor)>,     // (weight, bias) for ConvTranspose1d
    resblocks: Vec<ResBlock>,
    noise_res: Vec<ResBlock>,
    noise_convs: Vec<(Tensor, Tensor)>,
    noise_conv_strides: Vec<usize>,
    conv_post_weight: Tensor,
    conv_post_bias: Tensor,
    m_source_weight: Tensor,
    m_source_bias: Tensor,
    istft_n_fft: usize,
    istft_hop: usize,
    upsample_scale: usize,
}

struct ResBlock {
    convs1: Vec<(Tensor, Tensor)>,
    convs2: Vec<(Tensor, Tensor)>,
    alpha1: Vec<Tensor>,
    alpha2: Vec<Tensor>,
    adain1: Vec<Linear>,
    kernel_size: usize,
    adain2: Vec<Linear>,
    channels: usize,
}

impl ISTFTNetDecoder {
    pub fn load(vb: VarBuilder, cfg: &KokoroConfig) -> Result<Self> {
        // Single encode block: in=514 (512+1+1), out=1024
        let encode = DecoderBlock::load(vb.pp("encode"), 514, 1024, cfg.style_dim, false)?;

        // 4 decode blocks: first 3 output 1024, last outputs 512
        let decode_out_channels = [1024, 1024, 1024, 512];
        let mut decode_blocks = Vec::new();
        for i in 0..4 {
            let is_last = i == 3;
            decode_blocks.push(DecoderBlock::load(
                vb.pp(format!("decode.{}", i)), 1090, decode_out_channels[i], cfg.style_dim, is_last,
            )?);
        }

        // Feature projections
        let asr_vb = vb.pp("asr_res.0");
        let asr_res_weight = asr_vb.get((64, 512, 1), "weight")?;
        let asr_res_bias = asr_vb.get(64, "bias")?;

        let f0_conv_weight = vb.get((1, 1, 3), "F0_conv.weight")?;
        let f0_conv_bias = vb.get(1, "F0_conv.bias")?;
        let n_conv_weight = vb.get((1, 1, 3), "N_conv.weight")?;
        let n_conv_bias = vb.get(1, "N_conv.bias")?;

        let generator = Generator::load(vb.pp("generator"), cfg)?;

        Ok(Self {
            encode,
            decode_blocks,
            asr_res_weight,
            asr_res_bias,
            f0_conv_weight,
            f0_conv_bias,
            n_conv_weight,
            n_conv_bias,
            generator,
        })
    }

    pub fn forward(&self, asr: &Tensor, f0: &Tensor, n: &Tensor, style: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "triton-metal")]
        {
            if let Ok(Some(gpu)) = KokoroGpuDecoder::try_new(asr.device()) {
                return self.forward_gpu(asr, f0, n, style, &gpu);
            }
        }
        #[cfg(feature = "triton-d3d12")]
        {
            if let Ok(Some(gpu)) = KokoroGpuDecoderD3D12::try_new() {
                return self.forward_gpu(asr, f0, n, style, &gpu);
            }
        }
        self.forward_inner(asr, f0, n, style)
    }

    pub fn forward_cpu(&self, asr: &Tensor, f0: &Tensor, n: &Tensor, style: &Tensor) -> Result<Tensor> {
        self.forward_inner(asr, f0, n, style)
    }

    #[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
    fn forward_gpu<G: KokoroGpuBackend>(&self, asr: &Tensor, f0: &Tensor, n: &Tensor, style: &Tensor, gpu: &G) -> Result<Tensor> {
        let f0_feat = f0.unsqueeze(1)?.conv1d(&self.f0_conv_weight, 1, 2, 1, 1)?;
        let f0_feat = f0_feat.broadcast_add(&self.f0_conv_bias.unsqueeze(0)?.unsqueeze(2)?)?;
        let n_feat = n.unsqueeze(1)?.conv1d(&self.n_conv_weight, 1, 2, 1, 1)?;
        let n_feat = n_feat.broadcast_add(&self.n_conv_bias.unsqueeze(0)?.unsqueeze(2)?)?;

        let x = Tensor::cat(&[asr, &f0_feat, &n_feat], 1)?;
        let encoded = self.encode.forward(&x, style)?;

        let asr_res = asr.conv1d(&self.asr_res_weight, 0, 1, 1, 1)?;
        let asr_res = asr_res.broadcast_add(&self.asr_res_bias.unsqueeze(0)?.unsqueeze(2)?)?;

        let mut h = encoded;
        let mut concat_residuals = true;
        for block in &self.decode_blocks {
            if concat_residuals {
                h = Tensor::cat(&[&h, &asr_res, &f0_feat, &n_feat], 1)?;
            }
            h = block.forward(&h, style)?;
            if block.pool_weight.is_some() {
                concat_residuals = false;
            }
        }

        self.generator.forward_gpu(&h, f0, style, gpu)
    }

    fn forward_inner(&self, asr: &Tensor, f0: &Tensor, n: &Tensor, style: &Tensor) -> Result<Tensor> {
        let f0_feat = f0.unsqueeze(1)?.conv1d(&self.f0_conv_weight, 1, 2, 1, 1)?;
        let f0_feat = f0_feat.broadcast_add(&self.f0_conv_bias.unsqueeze(0)?.unsqueeze(2)?)?;
        let n_feat = n.unsqueeze(1)?.conv1d(&self.n_conv_weight, 1, 2, 1, 1)?;
        let n_feat = n_feat.broadcast_add(&self.n_conv_bias.unsqueeze(0)?.unsqueeze(2)?)?;

        let x = Tensor::cat(&[asr, &f0_feat, &n_feat], 1)?;

        let encoded = self.encode.forward(&x, style)?;

        let asr_res = asr.conv1d(&self.asr_res_weight, 0, 1, 1, 1)?;
        let asr_res = asr_res.broadcast_add(&self.asr_res_bias.unsqueeze(0)?.unsqueeze(2)?)?;

        let mut h = encoded;
        let mut concat_residuals = true;
        for block in &self.decode_blocks {
            if concat_residuals {
                h = Tensor::cat(&[&h, &asr_res, &f0_feat, &n_feat], 1)?;
            }
            h = block.forward(&h, style)?;
            if block.pool_weight.is_some() {
                concat_residuals = false;
            }
        }

        self.generator.forward(&h, f0, style)
    }
}

impl DecoderBlock {
    fn load(vb: VarBuilder, in_ch: usize, out_ch: usize, style_dim: usize, has_pool: bool) -> Result<Self> {
        let conv1_weight = vb.get((out_ch, in_ch, 3), "conv1.weight")?;
        let conv1_bias = vb.get(out_ch, "conv1.bias")?;
        let conv2_weight = vb.get((out_ch, out_ch, 3), "conv2.weight")?;
        let conv2_bias = vb.get(out_ch, "conv2.bias")?;

        let conv1x1_weight = if in_ch != out_ch {
            Some(vb.get((out_ch, in_ch, 1), "conv1x1.weight")?)
        } else {
            None
        };

        let norm1_fc = candle_nn::linear(style_dim, in_ch * 2, vb.pp("norm1").pp("fc"))?;
        let norm2_fc = candle_nn::linear(style_dim, out_ch * 2, vb.pp("norm2").pp("fc"))?;

        let (pool_weight, pool_bias) = if has_pool {
            (
                Some(vb.get((1090, 1, 3), "pool.weight")?),
                Some(vb.get(1090, "pool.bias")?),
            )
        } else {
            (None, None)
        };

        Ok(Self { conv1_weight, conv1_bias, conv2_weight, conv2_bias, conv1x1_weight, norm1_fc, norm2_fc, pool_weight, pool_bias })
    }

    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let (_, in_ch, _) = x.dims3()?;

        // Residual path: AdaIN1 → LeakyReLU → pool (upsample if present) → conv1
        let h = adain(x, style, &self.norm1_fc, in_ch)?;
        let h = leaky_relu(&h, 0.2)?;
        // Pool: ConvTranspose1d for upsample, identity otherwise
        let h = match (&self.pool_weight, &self.pool_bias) {
            (Some(pw), Some(pb)) => {
                // Depthwise ConvTranspose1d: stride=2, padding=1, output_padding=1, groups=in_ch
                let h = h.conv_transpose1d(pw, 1, 1, 2, 1, in_ch)?;
                h.broadcast_add(&pb.unsqueeze(0)?.unsqueeze(2)?)?
            }
            _ => h,
        };
        let h = h.conv1d(&self.conv1_weight, 1, 1, 1, 1)?;
        let h = h.broadcast_add(&self.conv1_bias.unsqueeze(0)?.unsqueeze(2)?)?;

        let (_, out_ch, _) = h.dims3()?;

        // AdaIN2 → LeakyReLU → conv2
        let h = adain(&h, style, &self.norm2_fc, out_ch)?;
        let h = leaky_relu(&h, 0.2)?;
        let h = h.conv1d(&self.conv2_weight, 1, 1, 1, 1)?;
        let h = h.broadcast_add(&self.conv2_bias.unsqueeze(0)?.unsqueeze(2)?)?;

        // Shortcut: upsample (nearest 2x if pool present) → conv1x1
        let shortcut = if self.pool_weight.is_some() {
            upsample_nearest_2x(x)?
        } else {
            x.clone()
        };
        let shortcut = match &self.conv1x1_weight {
            Some(w) => shortcut.conv1d(w, 0, 1, 1, 1)?,
            None => shortcut,
        };

        // Combine with rsqrt(2) normalization
        let rsqrt2 = 1.0 / 2.0f64.sqrt();
        ((h + shortcut)? * rsqrt2).map_err(Into::into)
    }
}

impl Generator {
    fn load(vb: VarBuilder, cfg: &KokoroConfig) -> Result<Self> {
        // Upsample layers
        let mut ups = Vec::new();
        let up_channels = [(512, 256, 20), (256, 128, 12)];
        for (i, &(in_ch, out_ch, kernel)) in up_channels.iter().enumerate() {
            let w = vb.get((in_ch, out_ch, kernel), &format!("ups.{}.weight", i))?;
            let b = vb.get(out_ch, &format!("ups.{}.bias", i))?;
            ups.push((w, b));
        }

        // ResBlocks: 3 per upsample stage (kernel sizes [3,7,11]) = 6 total
        let rb_channels = [256, 256, 256, 128, 128, 128];
        let rb_kernels = [3, 7, 11, 3, 7, 11];
        let mut resblocks = Vec::new();
        for i in 0..6 {
            resblocks.push(ResBlock::load(
                vb.pp(format!("resblocks.{}", i)), rb_channels[i], rb_kernels[i], cfg.style_dim,
            )?);
        }

        // Noise ResBlocks (kernel sizes 7, 11)
        let noise_channels = [256, 128];
        let noise_kernels = [7, 11];
        let mut noise_res = Vec::new();
        for i in 0..2 {
            noise_res.push(ResBlock::load(
                vb.pp(format!("noise_res.{}", i)), noise_channels[i], noise_kernels[i], cfg.style_dim,
            )?);
        }

        // Noise convs — stride for each is prod(upsample_rates[i+1:]) (without hop)
        let upsample_rates = &cfg.istftnet.upsample_rates;
        let mut noise_convs = Vec::new();
        let mut noise_conv_strides = Vec::new();
        let nc_shapes = [(256, 22, 12), (128, 22, 1)];
        for (i, &(out_ch, in_ch, k)) in nc_shapes.iter().enumerate() {
            let w = vb.get((out_ch, in_ch, k), &format!("noise_convs.{}.weight", i))?;
            let b = vb.get(out_ch, &format!("noise_convs.{}.bias", i))?;
            noise_convs.push((w, b));
            // stride = prod(upsample_rates[i+1:]) for intermediate, 1 for last
            let stride: usize = upsample_rates[i + 1..].iter().product::<usize>().max(1);
            noise_conv_strides.push(stride);
        }

        let conv_post_weight = vb.get((22, 128, 7), "conv_post.weight")?;
        let conv_post_bias = vb.get(22, "conv_post.bias")?;

        let m_source_weight = vb.get((1, 9), "m_source.l_linear.weight")?;
        let m_source_bias = vb.get(1, "m_source.l_linear.bias")?;

        let upsample_scale: usize = upsample_rates.iter().product::<usize>() * cfg.istftnet.gen_istft_hop_size;

        Ok(Self {
            ups,
            resblocks,
            noise_res,
            noise_convs,
            noise_conv_strides,
            conv_post_weight,
            conv_post_bias,
            m_source_weight,
            m_source_bias,
            istft_n_fft: cfg.istftnet.gen_istft_n_fft,
            istft_hop: cfg.istftnet.gen_istft_hop_size,
            upsample_scale,
        })
    }

    #[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
    fn forward_gpu<G: KokoroGpuBackend>(&self, x: &Tensor, f0: &Tensor, style: &Tensor, gpu: &G) -> Result<Tensor> {
        let compare = std::env::var("KOKORO_COMPARE").is_ok();
        let dtype = x.dtype();
        let (_batch, c_in, t_frames) = x.dims3()?;
        let device = x.device().clone();

        let har_source = self.generate_harmonic_source(f0, 1, t_frames, self.m_source_weight.device(), dtype)?;
        let har_buf = upload_activation(gpu, &har_source.to_dtype(DType::F16)?)?;
        let har_c = har_source.dim(1)?;

        // Precompute all adain gamma/beta for resblocks + noise_res and upload to GPU
        let all_resblock_params: Vec<Vec<(G::Buf, G::Buf)>> = self.resblocks.iter()
            .map(|rb| rb.precompute_adain_params(style, gpu))
            .collect::<Result<_>>()?;
        let all_noise_params: Vec<Vec<(G::Buf, G::Buf)>> = self.noise_res.iter()
            .map(|rb| rb.precompute_adain_params(style, gpu))
            .collect::<Result<_>>()?;

        // CPU reference (only when comparing)
        let mut cpu_h = if compare { Some(x.clone()) } else { None };

        let mut h_buf = upload_activation(gpu, &x.to_dtype(DType::F16)?)?;
        let mut h_c = c_in;
        let mut h_t = t_frames;
        let num_ups = self.ups.len();

        // f32 carry buffer: when available, used as input to next stage's upsample
        // to avoid f16 quantization between stages.
        let mut h_f32_carry: Option<G::Buf> = None;

        for (stage, (up_w, up_b)) in self.ups.iter().enumerate() {
            // Noise source: conv1d(har_source) → noise_res
            let (nc_w, nc_b) = &self.noise_convs[stage];
            let nc_stride = self.noise_conv_strides[stage];
            let nc_k = nc_w.dim(2)?;
            let nc_padding = if nc_k > 1 { (nc_stride + 1) / 2 } else { 0 };
            let har_t = har_source.dim(2)?;
            let mut _noise_keep: Vec<G::Buf> = Vec::new();
            let (src_buf, _src_c, _src_t) = if gpu.has_f32_intermediates() {
                let (c_out, _, k) = nc_w.dims3()?;
                let t_out = (har_t + 2 * nc_padding - 1 * (k - 1) - 1) / nc_stride + 1;
                let har_f32 = gpu.alloc_f32(har_c * har_t)?;
                gpu.f16_to_f32(&har_buf, &har_f32, har_c * har_t)?;
                let w_buf = upload_weight(gpu, nc_w)?;
                let b_buf = upload_weight(gpu, nc_b)?;
                let out_f32 = gpu.alloc_f32(c_out * t_out)?;
                gpu.conv1d_f32(&har_f32, &w_buf, &b_buf, &out_f32, har_c, c_out, har_t, t_out, k, nc_stride, nc_padding, 1)?;
                _noise_keep.push(har_f32);
                // noise_res takes f32 input directly (skip f16 round-trip)
                let src_buf = self.noise_res[stage].forward_gpu_f32_direct(&out_f32, &all_noise_params[stage], gpu, c_out, t_out, &mut _noise_keep)?;
                (src_buf, c_out, t_out)
            } else {
                let (src_buf, src_c, src_t) = buf_conv1d(gpu, &har_buf, nc_w, nc_b, har_c, har_t, nc_padding, nc_stride, 1)?;
                let src_buf = self.noise_res[stage].forward_gpu_precomputed(&src_buf, &all_noise_params[stage], gpu, src_c, src_t)?;
                (src_buf, src_c, src_t)
            };

            // Leaky_relu(0.1) + ConvTranspose1d upsample
            let up_stride = if stage == 0 { 10 } else { 6 };
            let up_padding = (up_w.dim(2)? - up_stride) / 2;
            // Track f32 version of h for precision when f32 intermediates available
            let mut h_f32_buf: Option<G::Buf> = None;
            let (h_new, new_c, new_t) = if gpu.has_f32_intermediates() {
                let (_, c_out, k) = up_w.dims3()?;
                let t_out = (h_t - 1) * up_stride - 2 * up_padding + k;
                let w_buf = upload_weight(gpu, up_w)?;
                let b_buf = upload_weight(gpu, up_b)?;
                // Apply leaky_relu(0.1) in f32 on GPU, then conv_transpose1d_f32io
                let f32_in = if let Some(carry) = h_f32_carry.take() {
                    carry
                } else {
                    let tmp = gpu.alloc_f32(h_c * h_t)?;
                    gpu.f16_to_f32(&h_buf, &tmp, h_c * h_t)?;
                    tmp
                };
                let lrelu_f32 = gpu.alloc_f32(h_c * h_t)?;
                gpu.leaky_relu_f32(&f32_in, &lrelu_f32, h_c * h_t, 0.1)?;
                let out_f32 = gpu.alloc_f32(c_out * t_out)?;
                gpu.conv_transpose1d_f32io(&lrelu_f32, &w_buf, &b_buf, &out_f32, h_c, c_out, h_t, t_out, k, up_stride, up_padding)?;
                // Also produce f16 version for h_buf
                let out_f16 = gpu.alloc(c_out * t_out)?;
                gpu.f32_to_f16(&out_f32, &out_f16, c_out * t_out)?;
                h_f32_buf = Some(out_f32);
                (out_f16, c_out, t_out)
            } else {
                buf_conv_transpose1d_lrelu(gpu, &h_buf, up_w, up_b, h_c, h_t, up_padding, up_stride)?
            };
            h_f32_carry = None;
            h_buf = h_new;
            h_c = new_c;
            h_t = new_t;

            if compare {
                let ch = cpu_h.as_ref().unwrap();
                let cpu_up = leaky_relu(ch, 0.1)?;
                let cpu_up = cpu_up.conv_transpose1d(up_w, up_padding, 0, up_stride, 1, 1)?;
                let cpu_up = cpu_up.broadcast_add(&up_b.unsqueeze(0)?.unsqueeze(2)?)?;
                compare_gpu_cpu(gpu, &h_buf, &cpu_up, &format!("stage{stage} upsample"));
                cpu_h = Some(cpu_up);
            }

            // Reflection pad on last stage
            if stage == num_ups - 1 {
                let (padded, new_t) = buf_reflection_pad1d(gpu, &h_buf, h_c, h_t, 1, 0)?;
                h_buf = padded;
                // Also pad the f32 version
                if let Some(ref f32_in) = h_f32_buf {
                    let padded_f32 = gpu.alloc_f32(h_c * (h_t + 1))?;
                    gpu.reflection_pad1d_f32(f32_in, &padded_f32, h_c, h_t)?;
                    h_f32_buf = Some(padded_f32);
                }
                h_t = new_t;
                if compare {
                    let ch = cpu_h.as_ref().unwrap();
                    let cpu_pad = reflection_pad1d(ch, 1, 0)?;
                    compare_gpu_cpu(gpu, &h_buf, &cpu_pad, &format!("stage{stage} reflpad"));
                    cpu_h = Some(cpu_pad);
                }
            }

            // h = h + x_source
            let n = h_c * h_t;
            h_buf = gpu.add(&h_buf, &src_buf, n)?;
            // Also add in f32 if we have the f32 upsample result
            if let Some(ref h_f32) = h_f32_buf {
                let src_f32 = gpu.alloc_f32(n)?;
                gpu.f16_to_f32(&src_buf, &src_f32, n)?;
                let sum_f32 = gpu.alloc_f32(n)?;
                gpu.add_f32(h_f32, &src_f32, &sum_f32, n)?;
                h_f32_buf = Some(sum_f32);
            }
            if compare {
                let ch = cpu_h.as_ref().unwrap();
                let nc_src = har_source.conv1d(nc_w, nc_padding, nc_stride, 1, 1)?;
                let nc_src = nc_src.broadcast_add(&nc_b.unsqueeze(0)?.unsqueeze(2)?)?;
                let nc_src_final = self.noise_res[stage].forward(&nc_src, style)?;
                let cpu_added = (ch + nc_src_final)?;
                compare_gpu_cpu(gpu, &h_buf, &cpu_added, &format!("stage{stage} h+src"));
                cpu_h = Some(cpu_added);
            }

            // 3 resblocks averaged (using precomputed gamma/beta)
            let base = stage * 3;
            let dbg_rb = if compare && stage == 0 {
                Some((cpu_h.as_ref().unwrap(), style))
            } else { None };
            // Hold f32 intermediate buffers alive until all resblocks complete
            // to prevent the allocator from reusing them while GPU is still reading.
            let mut _rb_keep: Vec<G::Buf> = Vec::new();
            if gpu.has_f32_intermediates() {
                // Resblocks in f32; use f32 input when available (avoids f16 quantization)
                let r0 = if let Some(ref f32_in) = h_f32_buf {
                    self.resblocks[base].forward_gpu_f32_direct_noconv(f32_in, &all_resblock_params[base], gpu, h_c, h_t, &mut _rb_keep)?
                } else {
                    self.resblocks[base].forward_gpu_f32_noconv(&h_buf, &all_resblock_params[base], gpu, h_c, h_t, &mut _rb_keep)?
                };
                let r1 = if let Some(ref f32_in) = h_f32_buf {
                    self.resblocks[base + 1].forward_gpu_f32_direct_noconv(f32_in, &all_resblock_params[base + 1], gpu, h_c, h_t, &mut _rb_keep)?
                } else {
                    self.resblocks[base + 1].forward_gpu_f32_noconv(&h_buf, &all_resblock_params[base + 1], gpu, h_c, h_t, &mut _rb_keep)?
                };
                let r2 = if let Some(ref f32_in) = h_f32_buf {
                    self.resblocks[base + 2].forward_gpu_f32_direct_noconv(f32_in, &all_resblock_params[base + 2], gpu, h_c, h_t, &mut _rb_keep)?
                } else {
                    self.resblocks[base + 2].forward_gpu_f32_noconv(&h_buf, &all_resblock_params[base + 2], gpu, h_c, h_t, &mut _rb_keep)?
                };
                let sum = gpu.alloc_f32(n)?;
                gpu.add_f32(&r0, &r1, &sum, n)?;
                let sum2 = gpu.alloc_f32(n)?;
                gpu.add_f32(&sum, &r2, &sum2, n)?;
                let avg = gpu.alloc_f32(n)?;
                gpu.scale_third_f32(&sum2, &avg, n)?;
                if stage == num_ups - 1 {
                    // Last stage: keep f32 for conv_post (avoid f16 quantization before exp())
                    _rb_keep.push(r0); _rb_keep.push(r1); _rb_keep.push(r2);
                    _rb_keep.push(sum); _rb_keep.push(sum2);
                    drop(_rb_keep);
                    if compare {
                        let avg_cmp = gpu.download_f32(&avg, n)?;
                        let ch = cpu_h.as_ref().unwrap();
                        let cpu_r0 = self.resblocks[base].forward(ch, style)?;
                        let cpu_r1 = self.resblocks[base + 1].forward(ch, style)?;
                        let cpu_r2 = self.resblocks[base + 2].forward(ch, style)?;
                        let cpu_avg = ((cpu_r0 + cpu_r1)? + cpu_r2)? / 3.0;
                        let cpu_avg = cpu_avg?;
                        let cpu_data: Vec<f32> = cpu_avg.flatten_all()?.to_vec1()?;
                        let mut max_err: f32 = 0.0;
                        let mut rms_sum: f64 = 0.0;
                        for i in 0..avg_cmp.len().min(cpu_data.len()) {
                            let d = (avg_cmp[i] - cpu_data[i]).abs();
                            rms_sum += (d as f64) * (d as f64);
                            if d > max_err { max_err = d; }
                        }
                        let rms = (rms_sum / avg_cmp.len() as f64).sqrt();
                        eprintln!("  [stage{stage} resblocks f32] n={} max_err={max_err:.4} rms={rms:.6}", avg_cmp.len());
                    }
                    // conv_post: leaky_relu(0.01) on f32 → conv1d_f32
                    let lrelu_buf = gpu.alloc_f32(n)?;
                    gpu.leaky_relu_f32(&avg, &lrelu_buf, n, 0.01)?;
                    let (c_out, _, k) = self.conv_post_weight.dims3()?;
                    let cp_padding = (k - 1) / 2;
                    let t_out = h_t;
                    let w_buf = upload_weight(gpu, &self.conv_post_weight)?;
                    let b_buf = upload_weight(gpu, &self.conv_post_bias)?;
                    let out_f32 = gpu.alloc_f32(c_out * t_out)?;
                    gpu.conv1d_f32(&lrelu_buf, &w_buf, &b_buf, &out_f32, h_c, c_out, h_t, t_out, k, 1, cp_padding, 1)?;
                    let out_data = gpu.download_f32(&out_f32, c_out * t_out)?;
                    if compare {
                        let ch = cpu_h.as_ref().unwrap();
                        let cpu_r0 = self.resblocks[base].forward(ch, style)?;
                        let cpu_r1 = self.resblocks[base + 1].forward(ch, style)?;
                        let cpu_r2 = self.resblocks[base + 2].forward(ch, style)?;
                        let cpu_avg = ((cpu_r0 + cpu_r1)? + cpu_r2)? / 3.0;
                        let cpu_avg = cpu_avg?;
                        let cpu_lrelu = leaky_relu(&cpu_avg, 0.01)?;
                        let cpu_conv = cpu_lrelu.conv1d(&self.conv_post_weight, cp_padding, 1, 1, 1)?;
                        let cpu_conv = cpu_conv.broadcast_add(&self.conv_post_bias.unsqueeze(0)?.unsqueeze(2)?)?;
                        let cpu_data: Vec<f32> = cpu_conv.flatten_all()?.to_vec1()?;
                        let mut max_err: f32 = 0.0;
                        let mut rms_sum: f64 = 0.0;
                        for i in 0..out_data.len().min(cpu_data.len()) {
                            let d = (out_data[i] - cpu_data[i]).abs();
                            rms_sum += (d as f64) * (d as f64);
                            if d > max_err { max_err = d; }
                        }
                        let rms = (rms_sum / out_data.len() as f64).sqrt();
                        eprintln!("  [conv_post f32] n={} max_err={max_err:.4} rms={rms:.6}", out_data.len());
                    }
                    let h = Tensor::from_vec(out_data, &[1, c_out, t_out][..], &device)?;
                    let n_fft = self.istft_n_fft;
                    let n_bins = n_fft / 2 + 1;
                    let mag = h.narrow(1, 0, n_bins)?.exp()?;
                    let phase = h.narrow(1, n_bins, n_bins)?.sin()?;
                    return istft(&mag, &phase, n_fft, self.istft_hop);
                }
                // Non-last stage: save f32 avg as carry for next stage's upsample
                h_f32_carry = Some(avg);
                h_buf = gpu.alloc(n)?;
                gpu.f32_to_f16(h_f32_carry.as_ref().unwrap(), &h_buf, n)?;
                _rb_keep.push(r0);
                _rb_keep.push(r1);
                _rb_keep.push(r2);
                _rb_keep.push(sum);
                _rb_keep.push(sum2);
            } else {
                let r0 = self.resblocks[base].forward_gpu_precomputed_dbg(&h_buf, &all_resblock_params[base], gpu, h_c, h_t, dbg_rb)?;
                let r1 = self.resblocks[base + 1].forward_gpu_precomputed(&h_buf, &all_resblock_params[base + 1], gpu, h_c, h_t)?;
                let r2 = self.resblocks[base + 2].forward_gpu_precomputed(&h_buf, &all_resblock_params[base + 2], gpu, h_c, h_t)?;
                let sum = gpu.add(&r0, &r1, n)?;
                let sum = gpu.add(&sum, &r2, n)?;
                h_buf = gpu.scale(&sum, n, 1.0 / 3.0)?;
            }
            drop(_rb_keep);

            if compare {
                let ch = cpu_h.as_ref().unwrap();
                let cpu_r0 = self.resblocks[base].forward(ch, style)?;
                let cpu_r1 = self.resblocks[base + 1].forward(ch, style)?;
                let cpu_r2 = self.resblocks[base + 2].forward(ch, style)?;
                let cpu_avg = ((cpu_r0 + cpu_r1)? + cpu_r2)? / 3.0;
                let cpu_avg = cpu_avg?;
                compare_gpu_cpu(gpu, &h_buf, &cpu_avg, &format!("stage{stage} resblocks"));
                cpu_h = Some(cpu_avg);
            }
        }

        // Leaky_relu(0.01) + conv_post
        // When f32 intermediates available, output conv_post as f32 to avoid
        // exp() amplifying f16 quantization error in iSTFT magnitude
        let h = if gpu.has_f32_intermediates() {
            let n = h_c * h_t;
            let lrelu_buf = gpu.alloc(n)?;
            gpu.leaky_relu(&h_buf, &lrelu_buf, n, 0.01)?;
            let h_f32 = gpu.alloc_f32(n)?;
            gpu.f16_to_f32(&lrelu_buf, &h_f32, n)?;
            let (c_out, _, k) = self.conv_post_weight.dims3()?;
            let padding = (k - 1) / 2;
            let t_out = h_t; // padding=(k-1)/2, stride=1 → t_out = t_in
            let w_buf = upload_weight(gpu, &self.conv_post_weight)?;
            let b_buf = upload_weight(gpu, &self.conv_post_bias)?;
            let out_f32 = gpu.alloc_f32(c_out * t_out)?;
            gpu.conv1d_f32(&h_f32, &w_buf, &b_buf, &out_f32, h_c, c_out, h_t, t_out, k, 1, padding, 1)?;
            let out_data = gpu.download_f32(&out_f32, c_out * t_out)?;
            if compare {
                let ch = cpu_h.as_ref().unwrap();
                let cpu_lrelu = leaky_relu(ch, 0.01)?;
                let cpu_conv = cpu_lrelu.conv1d(&self.conv_post_weight, 3, 1, 1, 1)?;
                let cpu_conv = cpu_conv.broadcast_add(&self.conv_post_bias.unsqueeze(0)?.unsqueeze(2)?)?;
                let cpu_data: Vec<f32> = cpu_conv.flatten_all()?.to_vec1()?;
                let mut max_err: f32 = 0.0;
                let mut max_idx = 0;
                for i in 0..out_data.len().min(cpu_data.len()) {
                    let d = (out_data[i] - cpu_data[i]).abs();
                    if d > max_err { max_err = d; max_idx = i; }
                }
                eprintln!("  [conv_post f32] n={} max_err={max_err:.4} (gpu={:.4} cpu={:.4} at {max_idx})",
                    out_data.len(), out_data[max_idx], cpu_data[max_idx]);
            }
            Tensor::from_vec(out_data, &[1, c_out, t_out][..], &device)?
        } else {
            let (h_buf, h_c, h_t) = buf_conv1d_lrelu001(gpu, &h_buf, &self.conv_post_weight, &self.conv_post_bias, h_c, h_t, 3, 1, 1)?;

            if compare {
                let ch = cpu_h.as_ref().unwrap();
                let cpu_lrelu = leaky_relu(ch, 0.01)?;
                let cpu_conv = cpu_lrelu.conv1d(&self.conv_post_weight, 3, 1, 1, 1)?;
                let cpu_conv = cpu_conv.broadcast_add(&self.conv_post_bias.unsqueeze(0)?.unsqueeze(2)?)?;
                compare_gpu_cpu(gpu, &h_buf, &cpu_conv, "conv_post");
            }

            // Download and do iSTFT on CPU
            #[cfg(feature = "triton-metal")]
            let _ = self.conv_post_weight.device().as_metal_device().map(|md| md.wait_until_completed());
            let out_data = gpu.download_f16(&h_buf, h_c * h_t)?;
            Tensor::from_vec(out_data, &[1, h_c, h_t][..], &device)?.to_dtype(DType::F32)?
        };

        let n_fft = self.istft_n_fft;
        let n_bins = n_fft / 2 + 1;
        let mag = h.narrow(1, 0, n_bins)?.exp()?;
        let phase = h.narrow(1, n_bins, n_bins)?.sin()?;

        istft(&mag, &phase, n_fft, self.istft_hop)
    }

    fn forward(&self, x: &Tensor, f0: &Tensor, style: &Tensor) -> Result<Tensor> {
        let device = x.device();
        let dtype = x.dtype();
        let (_batch, _, t_frames) = x.dims3()?;

        // Generate harmonic source
        let har_source = self.generate_harmonic_source(f0, 1, t_frames, device, dtype)?;

        let mut h = x.clone();
        let num_ups = self.ups.len();

        for (stage, (up_w, up_b)) in self.ups.iter().enumerate() {
            h = leaky_relu(&h, 0.1)?;

            // Noise source injection
            let (nc_w, nc_b) = &self.noise_convs[stage];
            let nc_stride = self.noise_conv_strides[stage];
            let nc_padding = if nc_w.dim(2)? > 1 { (nc_stride + 1) / 2 } else { 0 };
            let x_source = har_source.conv1d(nc_w, nc_padding, nc_stride, 1, 1)?;
            let x_source = x_source.broadcast_add(&nc_b.unsqueeze(0)?.unsqueeze(2)?)?;
            let x_source = self.noise_res[stage].forward(&x_source, style)?;

            // Upsample
            let up_stride = if stage == 0 { 10 } else { 6 };
            let up_padding = (up_w.dim(2)? - up_stride) / 2;
            h = h.conv_transpose1d(up_w, up_padding, 0, up_stride, 1, 1)?;
            h = h.broadcast_add(&up_b.unsqueeze(0)?.unsqueeze(2)?)?.contiguous()?;

            // Reflection pad on last upsample stage
            if stage == num_ups - 1 {
                h = reflection_pad1d(&h, 1, 0)?;
            }

            // Add source
            h = (h + x_source)?;

            // Apply 3 resblocks per stage (sum and average)
            let base = stage * 3;
            let mut sum = self.resblocks[base].forward(&h, style)?;
            sum = (sum + self.resblocks[base + 1].forward(&h, style)?)?;
            sum = (sum + self.resblocks[base + 2].forward(&h, style)?)?;
            h = (sum / 3.0)?;
        }

        // Final LeakyReLU (default slope 0.01) + conv_post
        h = leaky_relu(&h, 0.01)?;
        let h = h.conv1d(&self.conv_post_weight, 3, 1, 1, 1)?;
        let h = h.broadcast_add(&self.conv_post_bias.unsqueeze(0)?.unsqueeze(2)?)?;

        // Split into magnitude + phase
        let n_fft = self.istft_n_fft;
        let n_bins = n_fft / 2 + 1;
        let mag = h.narrow(1, 0, n_bins)?.exp()?;
        let phase = h.narrow(1, n_bins, n_bins)?.sin()?;

        istft(&mag, &phase, n_fft, self.istft_hop)
    }

    /// Generate harmonic source: upsample F0, create 9 harmonics, combine, STFT.
    /// Returns [B, 22, T_stft] (magnitude + phase concatenated).
    fn generate_harmonic_source(
        &self, f0: &Tensor, _batch: usize, t_frames: usize,
        device: &candle_core::Device, dtype: candle_core::DType,
    ) -> Result<Tensor> {
        let audio_len = t_frames * self.upsample_scale;
        let sr = 24000.0f64;

        // Upsample F0 from frame rate to audio rate via nearest-neighbor (matches nn.Upsample)
        // f0: [B, T_frames]
        let f0_vec: Vec<f32> = f0.squeeze(0)?.to_vec1()?;
        let f0_len = f0_vec.len();
        let mut f0_audio = vec![0.0f32; audio_len];
        for i in 0..audio_len {
            let idx = (i * f0_len / audio_len).min(f0_len - 1);
            f0_audio[i] = f0_vec[idx];
        }

        // Generate 9 harmonics: sin(2π * cumsum(k * f0 / sr)) for k=1..9
        let num_harmonics = 9usize;
        let sine_amp = 0.1f32;
        let noise_std = 0.003f32;
        let voiced_threshold = 10.0f32;

        let mut harmonics = vec![0.0f32; audio_len * num_harmonics];
        for k in 0..num_harmonics {
            let harmonic_num = (k + 1) as f32;
            let mut phase_acc = 0.0f64;
            for t in 0..audio_len {
                let freq = f0_audio[t] as f64 * harmonic_num as f64;
                phase_acc += freq / sr;
                let voiced = f0_audio[t] > voiced_threshold;
                let sine_val = (2.0 * std::f64::consts::PI * phase_acc).sin() as f32 * sine_amp;
                harmonics[t * num_harmonics + k] = if voiced {
                    sine_val + noise_std * rand_normal()
                } else {
                    sine_amp / 3.0 * rand_normal()
                };
            }
        }

        // Combine via l_linear + tanh: [audio_len, 9] @ [9, 1] + bias -> [audio_len, 1]
        let harmonics_t = Tensor::from_vec(harmonics, (audio_len, num_harmonics), device)?.to_dtype(dtype)?;
        let combined = harmonics_t.matmul(&self.m_source_weight.t()?)?
            .broadcast_add(&self.m_source_bias)?;
        let combined = combined.tanh()?; // [audio_len, 1]
        let har_source = combined.squeeze(1)?; // [audio_len]

        // STFT of harmonic source: n_fft=20, hop=5, periodic Hann window
        let har_stft = stft_source(&har_source, self.istft_n_fft, self.istft_hop, device, dtype)?;
        // har_stft: [22, T_stft] — add batch dim
        har_stft.unsqueeze(0).map_err(Into::into)
    }
}

// ── Tensor↔Buffer bridge: all GPU work stays in buffer space ──

#[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
fn compare_gpu_cpu<G: KokoroGpuBackend>(gpu: &G, gpu_buf: &G::Buf, cpu_tensor: &Tensor, label: &str) {
    let cpu_f16: Vec<half::f16> = cpu_tensor.to_dtype(DType::F16).unwrap()
        .flatten_all().unwrap().to_vec1().unwrap();
    let n = cpu_f16.len();
    let gpu_data = gpu.download_f16(gpu_buf, n).unwrap();
    let mut max_err: f32 = 0.0;
    let mut sum_sq: f64 = 0.0;
    let mut max_idx = 0;
    for i in 0..n {
        let diff = (gpu_data[i].to_f32() - cpu_f16[i].to_f32()).abs();
        sum_sq += (diff as f64) * (diff as f64);
        if diff > max_err {
            max_err = diff;
            max_idx = i;
        }
    }
    let rms = (sum_sq / n as f64).sqrt();
    let gpu_val = gpu_data[max_idx].to_f32();
    let cpu_val = cpu_f16[max_idx].to_f32();
    eprintln!("  [{label}] n={n} max_err={max_err:.4} rms={rms:.6} (at {max_idx}: gpu={gpu_val:.4} cpu={cpu_val:.4})");
}

#[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
fn tensor_id_usize(t: &Tensor) -> usize {
    let id = t.id();
    unsafe { std::mem::transmute::<candle_core::TensorId, usize>(id) }
}

#[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
fn tensor_to_f16_vec(t: &Tensor) -> Result<Vec<half::f16>> {
    t.to_dtype(DType::F16)?.flatten_all()?.to_vec1::<half::f16>().map_err(Into::into)
}

/// Upload a weight tensor (cached by TensorId).
#[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
fn upload_weight<G: KokoroGpuBackend>(gpu: &G, t: &Tensor) -> Result<G::Buf> {
    let data = tensor_to_f16_vec(t)?;
    gpu.upload_weight(tensor_id_usize(t), &data)
}

/// Upload an activation tensor (not cached).
#[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
fn upload_activation<G: KokoroGpuBackend>(gpu: &G, t: &Tensor) -> Result<G::Buf> {
    let data = tensor_to_f16_vec(t)?;
    gpu.upload_f16(&data)
}

/// GPU conv1d operating entirely in buffer space.
/// Uses specialized unrolled kernel when K is 3, 7, or 11.
#[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
fn buf_conv1d<G: KokoroGpuBackend>(
    gpu: &G, x: &G::Buf, w: &Tensor, b: &Tensor,
    c_in: usize, t_in: usize, padding: usize, stride: usize, dilation: usize,
) -> Result<(G::Buf, usize, usize)> {
    let (c_out, _, k) = w.dims3()?;
    let t_out = (t_in + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    let w_buf = upload_weight(gpu, w)?;
    let b_buf = upload_weight(gpu, b)?;
    let out = gpu.alloc(c_out * t_out)?;
    gpu.conv1d_k(x, &w_buf, &b_buf, &out, c_in, c_out, t_in, t_out, k, stride, padding, dilation)?;
    Ok((out, c_out, t_out))
}

/// Fused leaky_relu(0.1) + conv_transpose1d (activation applied to input on load).
#[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
fn buf_conv_transpose1d_lrelu<G: KokoroGpuBackend>(
    gpu: &G, x: &G::Buf, w: &Tensor, b: &Tensor,
    c_in: usize, t_in: usize, padding: usize, stride: usize,
) -> Result<(G::Buf, usize, usize)> {
    let (_, c_out, k) = w.dims3()?;
    let t_out = (t_in - 1) * stride - 2 * padding + k;
    let w_buf = upload_weight(gpu, w)?;
    let b_buf = upload_weight(gpu, b)?;
    let out = gpu.alloc(c_out * t_out)?;
    gpu.conv_transpose1d_lrelu(x, &w_buf, &b_buf, &out, c_in, c_out, t_in, t_out, k, stride, padding)?;
    Ok((out, c_out, t_out))
}

/// Fused leaky_relu(0.01) + conv1d (activation applied to input on load).
#[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
fn buf_conv1d_lrelu001<G: KokoroGpuBackend>(
    gpu: &G, x: &G::Buf, w: &Tensor, b: &Tensor,
    c_in: usize, t_in: usize, padding: usize, stride: usize, dilation: usize,
) -> Result<(G::Buf, usize, usize)> {
    let (c_out, _, k) = w.dims3()?;
    let t_out = (t_in + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    let w_buf = upload_weight(gpu, w)?;
    let b_buf = upload_weight(gpu, b)?;
    let out = gpu.alloc(c_out * t_out)?;
    gpu.conv1d_lrelu001(x, &w_buf, &b_buf, &out, c_in, c_out, t_in, t_out, k, stride, padding, dilation)?;
    Ok((out, c_out, t_out))
}

/// Reflection pad1d (pad_left=1, pad_right=0) entirely on GPU.
#[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
fn buf_reflection_pad1d<G: KokoroGpuBackend>(
    gpu: &G, x: &G::Buf, channels: usize, seq_len: usize, _pad_left: usize, _pad_right: usize,
) -> Result<(G::Buf, usize)> {
    let new_len = seq_len + 1;
    let out = gpu.alloc(channels * new_len)?;
    gpu.reflection_pad1d(x, &out, channels, seq_len)?;
    Ok((out, new_len))
}

#[cfg(any(feature = "triton-metal", feature = "triton-d3d12"))]
impl ResBlock {
    /// Precompute all adain gamma/beta pairs on CPU and upload to GPU.
    /// Returns 6 pairs: (gamma_buf, beta_buf) for adain1[0..3] then adain2[0..3].
    fn precompute_adain_params<G: KokoroGpuBackend>(&self, style: &Tensor, gpu: &G) -> Result<Vec<(G::Buf, G::Buf)>> {
        let mut params = Vec::with_capacity(6);
        for i in 0..3 {
            let p = self.adain1[i].forward(style)?;
            let gamma = tensor_to_f16_vec(&p.narrow(1, 0, self.channels)?.reshape(self.channels)?.to_dtype(DType::F16)?)?;
            let beta = tensor_to_f16_vec(&p.narrow(1, self.channels, self.channels)?.reshape(self.channels)?.to_dtype(DType::F16)?)?;
            params.push((gpu.upload_f16(&gamma)?, gpu.upload_f16(&beta)?));
        }
        for i in 0..3 {
            let p = self.adain2[i].forward(style)?;
            let gamma = tensor_to_f16_vec(&p.narrow(1, 0, self.channels)?.reshape(self.channels)?.to_dtype(DType::F16)?)?;
            let beta = tensor_to_f16_vec(&p.narrow(1, self.channels, self.channels)?.reshape(self.channels)?.to_dtype(DType::F16)?)?;
            params.push((gpu.upload_f16(&gamma)?, gpu.upload_f16(&beta)?));
        }
        Ok(params)
    }

    /// Forward with precomputed adain params (no CPU work in the hot loop).
    fn forward_gpu_precomputed<G: KokoroGpuBackend>(
        &self, x: &G::Buf, adain_params: &[(G::Buf, G::Buf)], gpu: &G,
        channels: usize, seq_len: usize,
    ) -> Result<G::Buf> {
        self.forward_gpu_precomputed_dbg(x, adain_params, gpu, channels, seq_len, None)
    }

    fn forward_gpu_precomputed_dbg<G: KokoroGpuBackend>(
        &self, x: &G::Buf, adain_params: &[(G::Buf, G::Buf)], gpu: &G,
        channels: usize, seq_len: usize, dbg_input: Option<(&Tensor, &Tensor)>,
    ) -> Result<G::Buf> {
        if gpu.has_f32_intermediates() {
            return self.forward_gpu_f32(x, adain_params, gpu, channels, seq_len);
        }

        let dilations = [1usize, 3, 5];
        let n = channels * seq_len;
        let mut h = x.clone();
        let mut cpu_h_opt: Option<Tensor> = dbg_input.map(|(t, _)| t.clone());

        for i in 0..3 {
            let residual = h.clone();

            let alpha1_buf = upload_weight(gpu, &self.alpha1[i].reshape(channels)?)?;
            let out = gpu.alloc(n)?;
            gpu.adain_snake(&h, &adain_params[i].0, &adain_params[i].1, &alpha1_buf, &out, channels, seq_len)?;
            h = out;

            if let Some(ref ch) = cpu_h_opt {
                let style = dbg_input.unwrap().1;
                let cpu_val = adain(ch, style, &self.adain1[i], channels)?;
                let cpu_val = snake_activation_tensor(&cpu_val, &self.alpha1[i])?;
                compare_gpu_cpu(gpu, &h, &cpu_val.to_dtype(DType::F16)?.to_dtype(DType::F32)?, &format!("rb_iter{i} adain_snake1"));
                cpu_h_opt = Some(cpu_val);
            }

            let dilation = dilations[i];
            let padding1 = (self.kernel_size - 1) * dilation / 2;
            let (w1, b1) = &self.convs1[i];
            let (h_new, _, _) = buf_conv1d(gpu, &h, w1, b1, channels, seq_len, padding1, 1, dilation)?;
            h = h_new;

            if let Some(ref ch) = cpu_h_opt {
                let cpu_val = ch.conv1d(w1, padding1, 1, dilation, 1)?;
                let cpu_val = cpu_val.broadcast_add(&b1.unsqueeze(0)?.unsqueeze(2)?)?;
                compare_gpu_cpu(gpu, &h, &cpu_val.to_dtype(DType::F16)?.to_dtype(DType::F32)?, &format!("rb_iter{i} conv1"));
                cpu_h_opt = Some(cpu_val);
            }

            let alpha2_buf = upload_weight(gpu, &self.alpha2[i].reshape(channels)?)?;
            let out = gpu.alloc(n)?;
            gpu.adain_snake(&h, &adain_params[3 + i].0, &adain_params[3 + i].1, &alpha2_buf, &out, channels, seq_len)?;
            h = out;

            if let Some(ref ch) = cpu_h_opt {
                let style = dbg_input.unwrap().1;
                let cpu_val = adain(ch, style, &self.adain2[i], channels)?;
                let cpu_val = snake_activation_tensor(&cpu_val, &self.alpha2[i])?;
                compare_gpu_cpu(gpu, &h, &cpu_val.to_dtype(DType::F16)?.to_dtype(DType::F32)?, &format!("rb_iter{i} adain_snake2"));
                cpu_h_opt = Some(cpu_val);
            }

            let padding2 = (self.kernel_size - 1) / 2;
            let (w2, b2) = &self.convs2[i];
            let (h_new, _, _) = buf_conv1d(gpu, &h, w2, b2, channels, seq_len, padding2, 1, 1)?;
            h = h_new;

            if let Some(ref ch) = cpu_h_opt {
                let cpu_val = ch.conv1d(w2, padding2, 1, 1, 1)?;
                let cpu_val = cpu_val.broadcast_add(&b2.unsqueeze(0)?.unsqueeze(2)?)?;
                compare_gpu_cpu(gpu, &h, &cpu_val.to_dtype(DType::F16)?.to_dtype(DType::F32)?, &format!("rb_iter{i} conv2"));
                cpu_h_opt = Some(cpu_val);
            }

            h = gpu.add(&h, &residual, n)?;
        }

        Ok(h)
    }

    /// F32-intermediate resblock: keeps activations in f32 within the resblock
    /// to prevent instance normalization from amplifying f16 quantization errors.
    fn forward_gpu_f32<G: KokoroGpuBackend>(
        &self, x: &G::Buf, adain_params: &[(G::Buf, G::Buf)], gpu: &G,
        channels: usize, seq_len: usize,
    ) -> Result<G::Buf> {
        let mut keep = Vec::new();
        self.forward_gpu_f32_into(x, adain_params, gpu, channels, seq_len, &mut keep)
    }

    /// Like forward_gpu_f32_noconv but takes an already-f32 input buffer (no f16→f32 step).
    /// Returns f32 buffer.
    fn forward_gpu_f32_direct_noconv<G: KokoroGpuBackend>(
        &self, x_f32: &G::Buf, adain_params: &[(G::Buf, G::Buf)], gpu: &G,
        channels: usize, seq_len: usize, keep: &mut Vec<G::Buf>,
    ) -> Result<G::Buf> {
        let dilations = [1usize, 3, 5];
        let n = channels * seq_len;
        let mut h = x_f32.clone();
        for i in 0..3 {
            let residual = h.clone();
            let alpha1_buf = upload_weight(gpu, &self.alpha1[i].reshape(channels)?)?;
            let out = gpu.alloc_f32(n)?;
            gpu.adain_snake_f32(&h, &adain_params[i].0, &adain_params[i].1, &alpha1_buf, &out, channels, seq_len)?;
            keep.push(h); h = out;
            let dilation = dilations[i];
            let padding1 = (self.kernel_size - 1) * dilation / 2;
            let (w1, b1) = &self.convs1[i];
            let w1_buf = upload_weight(gpu, w1)?;
            let b1_buf = upload_weight(gpu, b1)?;
            let out = gpu.alloc_f32(n)?;
            gpu.conv1d_f32(&h, &w1_buf, &b1_buf, &out, channels, channels, seq_len, seq_len, self.kernel_size, 1, padding1, dilation)?;
            keep.push(h); h = out;
            let alpha2_buf = upload_weight(gpu, &self.alpha2[i].reshape(channels)?)?;
            let out = gpu.alloc_f32(n)?;
            gpu.adain_snake_f32(&h, &adain_params[3 + i].0, &adain_params[3 + i].1, &alpha2_buf, &out, channels, seq_len)?;
            keep.push(h); h = out;
            let padding2 = (self.kernel_size - 1) / 2;
            let (w2, b2) = &self.convs2[i];
            let w2_buf = upload_weight(gpu, w2)?;
            let b2_buf = upload_weight(gpu, b2)?;
            let out = gpu.alloc_f32(n)?;
            gpu.conv1d_f32(&h, &w2_buf, &b2_buf, &out, channels, channels, seq_len, seq_len, self.kernel_size, 1, padding2, 1)?;
            keep.push(h); h = out;
            let out = gpu.alloc_f32(n)?;
            gpu.add_f32(&h, &residual, &out, n)?;
            keep.push(h); h = out;
        }
        Ok(h)
    }

    /// Like forward_gpu_f32_into but takes an already-f32 input buffer (no f16→f32 step).
    /// Returns f16 buffer (for noise path).
    fn forward_gpu_f32_direct<G: KokoroGpuBackend>(
        &self, x_f32: &G::Buf, adain_params: &[(G::Buf, G::Buf)], gpu: &G,
        channels: usize, seq_len: usize, keep: &mut Vec<G::Buf>,
    ) -> Result<G::Buf> {
        let dilations = [1usize, 3, 5];
        let n = channels * seq_len;
        let mut h = x_f32.clone();

        for i in 0..3 {
            let residual = h.clone();

            let alpha1_buf = upload_weight(gpu, &self.alpha1[i].reshape(channels)?)?;
            let out = gpu.alloc_f32(n)?;
            gpu.adain_snake_f32(&h, &adain_params[i].0, &adain_params[i].1, &alpha1_buf, &out, channels, seq_len)?;
            keep.push(h);
            h = out;

            let dilation = dilations[i];
            let padding1 = (self.kernel_size - 1) * dilation / 2;
            let (w1, b1) = &self.convs1[i];
            let w1_buf = upload_weight(gpu, w1)?;
            let b1_buf = upload_weight(gpu, b1)?;
            let out = gpu.alloc_f32(n)?;
            gpu.conv1d_f32(&h, &w1_buf, &b1_buf, &out, channels, channels, seq_len, seq_len, self.kernel_size, 1, padding1, dilation)?;
            keep.push(h);
            h = out;

            let alpha2_buf = upload_weight(gpu, &self.alpha2[i].reshape(channels)?)?;
            let out = gpu.alloc_f32(n)?;
            gpu.adain_snake_f32(&h, &adain_params[3 + i].0, &adain_params[3 + i].1, &alpha2_buf, &out, channels, seq_len)?;
            keep.push(h);
            h = out;

            let padding2 = (self.kernel_size - 1) / 2;
            let (w2, b2) = &self.convs2[i];
            let w2_buf = upload_weight(gpu, w2)?;
            let b2_buf = upload_weight(gpu, b2)?;
            let out = gpu.alloc_f32(n)?;
            gpu.conv1d_f32(&h, &w2_buf, &b2_buf, &out, channels, channels, seq_len, seq_len, self.kernel_size, 1, padding2, 1)?;
            keep.push(h);
            h = out;

            let out = gpu.alloc_f32(n)?;
            gpu.add_f32(&h, &residual, &out, n)?;
            keep.push(h);
            h = out;
        }

        let out_f16 = gpu.alloc(n)?;
        gpu.f32_to_f16(&h, &out_f16, n)?;

        Ok(out_f16)
    }

    fn forward_gpu_f32_into<G: KokoroGpuBackend>(
        &self, x: &G::Buf, adain_params: &[(G::Buf, G::Buf)], gpu: &G,
        channels: usize, seq_len: usize, keep: &mut Vec<G::Buf>,
    ) -> Result<G::Buf> {
        let dbg = std::env::var("KOKORO_DBG_RB").is_ok() && channels == 128;
        let dilations = [1usize, 3, 5];
        let n = channels * seq_len;

        let mut h = gpu.alloc_f32(n)?;
        gpu.f16_to_f32(x, &h, n)?;

        if dbg {
            let data = gpu.download_f32(&h, n)?;
            let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            eprintln!("    [rb_dbg] after f16_to_f32: max_abs={max_abs:.4} first10={:.4?}", &data[..10]);
        }

        for i in 0..3 {
            let residual = h.clone();

            let alpha1_buf = upload_weight(gpu, &self.alpha1[i].reshape(channels)?)?;
            let out = gpu.alloc_f32(n)?;
            gpu.adain_snake_f32(&h, &adain_params[i].0, &adain_params[i].1, &alpha1_buf, &out, channels, seq_len)?;
            keep.push(h);
            h = out;

            if dbg && i == 0 {
                let data = gpu.download_f32(&h, n)?;
                let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                eprintln!("    [rb_dbg] i={i} after adain1: max_abs={max_abs:.4} first10={:.4?}", &data[..10]);
            }

            let dilation = dilations[i];
            let padding1 = (self.kernel_size - 1) * dilation / 2;
            let (w1, b1) = &self.convs1[i];
            let w1_buf = upload_weight(gpu, w1)?;
            let b1_buf = upload_weight(gpu, b1)?;
            let out = gpu.alloc_f32(n)?;
            gpu.conv1d_f32(&h, &w1_buf, &b1_buf, &out, channels, channels, seq_len, seq_len, self.kernel_size, 1, padding1, dilation)?;
            keep.push(h);
            h = out;

            if dbg && i == 0 {
                let data = gpu.download_f32(&h, n)?;
                let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                eprintln!("    [rb_dbg] i={i} after conv1: max_abs={max_abs:.4} first10={:.4?}", &data[..10]);
            }

            let alpha2_buf = upload_weight(gpu, &self.alpha2[i].reshape(channels)?)?;
            let out = gpu.alloc_f32(n)?;
            gpu.adain_snake_f32(&h, &adain_params[3 + i].0, &adain_params[3 + i].1, &alpha2_buf, &out, channels, seq_len)?;
            keep.push(h);
            h = out;

            if dbg && i == 0 {
                let data = gpu.download_f32(&h, n)?;
                let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                eprintln!("    [rb_dbg] i={i} after adain2: max_abs={max_abs:.4} first10={:.4?}", &data[..10]);
            }

            let padding2 = (self.kernel_size - 1) / 2;
            let (w2, b2) = &self.convs2[i];
            let w2_buf = upload_weight(gpu, w2)?;
            let b2_buf = upload_weight(gpu, b2)?;
            let out = gpu.alloc_f32(n)?;
            gpu.conv1d_f32(&h, &w2_buf, &b2_buf, &out, channels, channels, seq_len, seq_len, self.kernel_size, 1, padding2, 1)?;
            keep.push(h);
            h = out;

            if dbg && i == 0 {
                let data = gpu.download_f32(&h, n)?;
                let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                eprintln!("    [rb_dbg] i={i} after conv2: max_abs={max_abs:.4} first10={:.4?}", &data[..10]);
            }

            let out = gpu.alloc_f32(n)?;
            gpu.add_f32(&h, &residual, &out, n)?;
            keep.push(h);
            h = out;

            if dbg && i == 0 {
                let data = gpu.download_f32(&h, n)?;
                let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                eprintln!("    [rb_dbg] i={i} after add: max_abs={max_abs:.4} first10={:.4?}", &data[..10]);
            }
        }

        let out_f16 = gpu.alloc(n)?;
        gpu.f32_to_f16(&h, &out_f16, n)?;
        Ok(out_f16)
    }

    /// Like forward_gpu_f32_into but returns the f32 buffer directly (no f32→f16 conversion).
    fn forward_gpu_f32_noconv<G: KokoroGpuBackend>(
        &self, x: &G::Buf, adain_params: &[(G::Buf, G::Buf)], gpu: &G,
        channels: usize, seq_len: usize, keep: &mut Vec<G::Buf>,
    ) -> Result<G::Buf> {
        let dilations = [1usize, 3, 5];
        let n = channels * seq_len;

        let mut h = gpu.alloc_f32(n)?;
        gpu.f16_to_f32(x, &h, n)?;

        for i in 0..3 {
            let residual = h.clone();

            let alpha1_buf = upload_weight(gpu, &self.alpha1[i].reshape(channels)?)?;
            let out = gpu.alloc_f32(n)?;
            gpu.adain_snake_f32(&h, &adain_params[i].0, &adain_params[i].1, &alpha1_buf, &out, channels, seq_len)?;
            keep.push(h);
            h = out;

            let dilation = dilations[i];
            let padding1 = (self.kernel_size - 1) * dilation / 2;
            let (w1, b1) = &self.convs1[i];
            let w1_buf = upload_weight(gpu, w1)?;
            let b1_buf = upload_weight(gpu, b1)?;
            let out = gpu.alloc_f32(n)?;
            gpu.conv1d_f32(&h, &w1_buf, &b1_buf, &out, channels, channels, seq_len, seq_len, self.kernel_size, 1, padding1, dilation)?;
            keep.push(h);
            h = out;

            let alpha2_buf = upload_weight(gpu, &self.alpha2[i].reshape(channels)?)?;
            let out = gpu.alloc_f32(n)?;
            gpu.adain_snake_f32(&h, &adain_params[3 + i].0, &adain_params[3 + i].1, &alpha2_buf, &out, channels, seq_len)?;
            keep.push(h);
            h = out;

            let padding2 = (self.kernel_size - 1) / 2;
            let (w2, b2) = &self.convs2[i];
            let w2_buf = upload_weight(gpu, w2)?;
            let b2_buf = upload_weight(gpu, b2)?;
            let out = gpu.alloc_f32(n)?;
            gpu.conv1d_f32(&h, &w2_buf, &b2_buf, &out, channels, channels, seq_len, seq_len, self.kernel_size, 1, padding2, 1)?;
            keep.push(h);
            h = out;

            let out = gpu.alloc_f32(n)?;
            gpu.add_f32(&h, &residual, &out, n)?;
            keep.push(h);
            h = out;
        }

        Ok(h)
    }
}

impl ResBlock {
    fn load(vb: VarBuilder, channels: usize, kernel_size: usize, style_dim: usize) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut alpha1 = Vec::new();
        let mut alpha2 = Vec::new();
        let mut adain1 = Vec::new();
        let mut adain2 = Vec::new();

        for i in 0..3 {
            let w1 = vb.get((channels, channels, kernel_size), &format!("convs1.{}.weight", i))?;
            let b1 = vb.get(channels, &format!("convs1.{}.bias", i))?;
            convs1.push((w1, b1));

            let w2 = vb.get((channels, channels, kernel_size), &format!("convs2.{}.weight", i))?;
            let b2 = vb.get(channels, &format!("convs2.{}.bias", i))?;
            convs2.push((w2, b2));

            alpha1.push(vb.get((1, channels, 1), &format!("alpha1.{}", i))?);
            alpha2.push(vb.get((1, channels, 1), &format!("alpha2.{}", i))?);

            adain1.push(candle_nn::linear(
                style_dim, channels * 2,
                vb.pp(format!("adain1.{}.fc", i)),
            )?);
            adain2.push(candle_nn::linear(
                style_dim, channels * 2,
                vb.pp(format!("adain2.{}.fc", i)),
            )?);
        }

        Ok(Self { convs1, convs2, alpha1, alpha2, adain1, adain2, kernel_size, channels })
    }

    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let dilations = [1usize, 3, 5];
        let mut h = x.clone();

        for i in 0..3 {
            let residual = h.clone();
            let dilation = dilations[i];
            let padding1 = (self.kernel_size - 1) * dilation / 2;
            let padding2 = (self.kernel_size - 1) / 2;

            // AdaIN → Snake → dilated Conv1
            h = adain(&h, style, &self.adain1[i], self.channels)?;
            h = snake_activation_tensor(&h, &self.alpha1[i])?;
            let (w1, b1) = &self.convs1[i];
            h = h.conv1d(w1, padding1, 1, dilation, 1)?;
            h = h.broadcast_add(&b1.unsqueeze(0)?.unsqueeze(2)?)?;

            // AdaIN → Snake → Conv2 (dilation=1)
            h = adain(&h, style, &self.adain2[i], self.channels)?;
            h = snake_activation_tensor(&h, &self.alpha2[i])?;
            let (w2, b2) = &self.convs2[i];
            h = h.conv1d(w2, padding2, 1, 1, 1)?;
            h = h.broadcast_add(&b2.unsqueeze(0)?.unsqueeze(2)?)?;

            h = (h + residual)?;
        }

        Ok(h)
    }

    #[allow(dead_code)]
    fn forward_f16_weights(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let dilations = [1usize, 3, 5];
        let mut h = x.clone();

        for i in 0..3 {
            let residual = h.clone();
            let dilation = dilations[i];
            let padding1 = (self.kernel_size - 1) * dilation / 2;
            let padding2 = (self.kernel_size - 1) / 2;

            h = adain(&h, style, &self.adain1[i], self.channels)?;
            h = snake_activation_tensor(&h, &self.alpha1[i])?;
            let (w1, b1) = &self.convs1[i];
            let w1_f16 = w1.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
            let b1_f16 = b1.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
            h = h.conv1d(&w1_f16, padding1, 1, dilation, 1)?;
            h = h.broadcast_add(&b1_f16.unsqueeze(0)?.unsqueeze(2)?)?;

            h = adain(&h, style, &self.adain2[i], self.channels)?;
            h = snake_activation_tensor(&h, &self.alpha2[i])?;
            let (w2, b2) = &self.convs2[i];
            let w2_f16 = w2.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
            let b2_f16 = b2.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
            h = h.conv1d(&w2_f16, padding2, 1, 1, 1)?;
            h = h.broadcast_add(&b2_f16.unsqueeze(0)?.unsqueeze(2)?)?;

            h = (h + residual)?;
        }

        Ok(h)
    }
}

fn adain(x: &Tensor, style: &Tensor, fc: &Linear, channels: usize) -> Result<Tensor> {
    let params = fc.forward(style)?;
    let gamma = params.narrow(1, 0, channels)?.unsqueeze(2)?;
    let beta = params.narrow(1, channels, channels)?.unsqueeze(2)?;

    let mean = x.mean_keepdim(2)?;
    let diff = x.broadcast_sub(&mean)?;
    let var = diff.sqr()?.mean_keepdim(2)?;
    let norm = diff.broadcast_div(&(var + 1e-5)?.sqrt()?)?;

    let scale = (gamma + 1.0)?;
    norm.broadcast_mul(&scale)?.broadcast_add(&beta).map_err(Into::into)
}

fn upsample_nearest_2x(x: &Tensor) -> Result<Tensor> {
    let (batch, channels, len) = x.dims3()?;
    let expanded = x.unsqueeze(3)?; // [B, C, T, 1]
    let expanded = expanded.expand((batch, channels, len, 2))?.contiguous()?; // [B, C, T, 2]
    expanded.reshape((batch, channels, len * 2)).map_err(Into::into)
}

fn leaky_relu(x: &Tensor, negative_slope: f64) -> Result<Tensor> {
    let zeros = x.zeros_like()?;
    let pos = x.maximum(&zeros)?;
    let neg = x.minimum(&zeros)?;
    (pos + neg * negative_slope).map_err(Into::into)
}

fn reflection_pad1d(x: &Tensor, pad_left: usize, pad_right: usize) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let orig_device = x.device().clone();
    let x = x.to_dtype(candle_core::DType::F32)?;
    let (batch, channels, len) = x.dims3()?;
    let new_len = len + pad_left + pad_right;
    let data: Vec<f32> = x.flatten_all()?.to_vec1()?;
    let mut out = vec![0.0f32; batch * channels * new_len];
    for b in 0..batch {
        for c in 0..channels {
            let src_off = (b * channels + c) * len;
            let dst_off = (b * channels + c) * new_len;
            for i in 0..pad_left {
                out[dst_off + i] = data[src_off + pad_left - i];
            }
            for i in 0..len {
                out[dst_off + pad_left + i] = data[src_off + i];
            }
            for i in 0..pad_right {
                out[dst_off + pad_left + len + i] = data[src_off + len - 2 - i];
            }
        }
    }
    let result = Tensor::from_vec(out, (batch, channels, new_len), &orig_device)?;
    result.to_dtype(orig_dtype).map_err(Into::into)
}

fn snake_activation_tensor(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let ax = x.broadcast_mul(alpha)?;
    let sin_sq = ax.sin()?.sqr()?;
    let inv_alpha = (alpha + 1e-9)?.recip()?;
    (x + sin_sq.broadcast_mul(&inv_alpha)?).map_err(Into::into)
}

thread_local! {
    static RAND_STATE: std::cell::Cell<u64> = std::cell::Cell::new(0x12345678_9abcdef0);
}

pub fn reset_rng() {
    RAND_STATE.with(|s| s.set(0x12345678_9abcdef0));
}

fn rand_normal() -> f32 {
    RAND_STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        let u1 = (x & 0xFFFFFFFF) as f32 / u32::MAX as f32;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        let u2 = (x & 0xFFFFFFFF) as f32 / u32::MAX as f32;
        (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    })
}

/// Forward STFT of a 1D signal with center=True padding, returns [n_fft+2, T_stft].
fn stft_source(signal: &Tensor, n_fft: usize, hop_size: usize, device: &candle_core::Device, dtype: candle_core::DType) -> Result<Tensor> {
    let sig: Vec<f32> = signal.to_vec1()?;
    let freq_bins = n_fft / 2 + 1;

    // Pad signal by n_fft/2 on each side (center=True mode)
    let pad = n_fft / 2;
    let padded_len = sig.len() + 2 * pad;
    let mut padded = vec![0.0f32; padded_len];
    padded[pad..pad + sig.len()].copy_from_slice(&sig);

    // Periodic Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / n_fft as f32).cos())
        .collect();

    let n_frames = (padded_len - n_fft) / hop_size + 1;

    let mut mag_data = vec![0.0f32; freq_bins * n_frames];
    let mut phase_data = vec![0.0f32; freq_bins * n_frames];

    for frame in 0..n_frames {
        let offset = frame * hop_size;
        for k in 0..freq_bins {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            for n in 0..n_fft {
                let val = padded[offset + n] * window[n];
                let angle = 2.0 * std::f32::consts::PI * k as f32 * n as f32 / n_fft as f32;
                real += val * angle.cos();
                imag -= val * angle.sin();
            }
            mag_data[k * n_frames + frame] = (real * real + imag * imag).sqrt();
            phase_data[k * n_frames + frame] = imag.atan2(real);
        }
    }

    let mag = Tensor::from_vec(mag_data, (freq_bins, n_frames), device)?.to_dtype(dtype)?;
    let phase = Tensor::from_vec(phase_data, (freq_bins, n_frames), device)?.to_dtype(dtype)?;
    Tensor::cat(&[&mag, &phase], 0).map_err(Into::into)
}

/// Inverse STFT matching torch.istft with periodic Hann window.
fn istft(magnitude: &Tensor, phase: &Tensor, n_fft: usize, hop_size: usize) -> Result<Tensor> {
    let (batch, freq_bins, n_frames) = magnitude.dims3()?;
    let device = magnitude.device();
    let dtype = magnitude.dtype();

    // Pull data to CPU for the small n_fft=20 iSTFT (faster than tensor ops per-frame)
    let mag_data: Vec<f32> = magnitude.flatten_all()?.to_vec1()?;
    let phase_data: Vec<f32> = phase.flatten_all()?.to_vec1()?;

    let output_len = (n_frames - 1) * hop_size + n_fft;

    // Periodic Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / n_fft as f32).cos())
        .collect();

    // Build iDFT basis (cos/sin tables)
    let mut cos_basis = vec![0.0f32; n_fft * freq_bins];
    let mut sin_basis = vec![0.0f32; n_fft * freq_bins];
    for n in 0..n_fft {
        for k in 0..freq_bins {
            let angle = 2.0 * std::f32::consts::PI * k as f32 * n as f32 / n_fft as f32;
            cos_basis[n * freq_bins + k] = angle.cos();
            sin_basis[n * freq_bins + k] = angle.sin();
        }
    }

    let mut all_output = vec![0.0f32; batch * output_len];
    let mut window_sum = vec![0.0f32; output_len];

    // Pre-compute window squared sum (same for all batches)
    for frame in 0..n_frames {
        let offset = frame * hop_size;
        for n in 0..n_fft {
            if offset + n < output_len {
                window_sum[offset + n] += window[n] * window[n];
            }
        }
    }

    for b in 0..batch {
        for frame in 0..n_frames {
            let mut frame_signal = vec![0.0f32; n_fft];
            for n in 0..n_fft {
                let mut val = 0.0f32;
                for k in 0..freq_bins {
                    let idx = b * freq_bins * n_frames + k * n_frames + frame;
                    let m = mag_data[idx];
                    let p = phase_data[idx];
                    let real = m * p.cos();
                    let imag = m * p.sin();
                    // iDFT: x[n] = (1/N) * sum Re(X[k])*cos(angle) - Im(X[k])*sin(angle)
                    // One-sided: scale *2 for non-DC/Nyquist bins
                    let scale = if k == 0 || k == freq_bins - 1 { 1.0 } else { 2.0 };
                    val += scale * (real * cos_basis[n * freq_bins + k]
                                  - imag * sin_basis[n * freq_bins + k]);
                }
                frame_signal[n] = val / n_fft as f32;
            }

            // Apply synthesis window and overlap-add
            let offset = frame * hop_size;
            for n in 0..n_fft {
                if offset + n < output_len {
                    all_output[b * output_len + offset + n] += frame_signal[n] * window[n];
                }
            }
        }

        // COLA normalization: divide by sum of squared windows
        for i in 0..output_len {
            if window_sum[i] > 1e-8 {
                all_output[b * output_len + i] /= window_sum[i];
            }
        }
    }

    // Trim center padding (n_fft/2 from each side, matching torch.istft center=True)
    let trim = n_fft / 2;
    let trimmed_len = output_len - 2 * trim;
    let mut trimmed = vec![0.0f32; batch * trimmed_len];
    for b in 0..batch {
        trimmed[b * trimmed_len..(b + 1) * trimmed_len]
            .copy_from_slice(&all_output[b * output_len + trim..b * output_len + trim + trimmed_len]);
    }
    Tensor::from_vec(trimmed, (batch, trimmed_len), device)?.to_dtype(dtype).map_err(Into::into)
}
