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
use candle_core::Tensor;
use candle_nn::{Linear, Module, VarBuilder};

use super::config::KokoroConfig;

#[cfg(feature = "triton-metal")]
use super::gpu_decoder::KokoroGpuDecoder;


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
            if let Ok(Some(gpu)) = KokoroGpuDecoder::new(asr.device()) {
                return self.forward_gpu(asr, f0, n, style, &gpu);
            }
        }
        self.forward_inner(asr, f0, n, style)
    }

    #[cfg(feature = "triton-metal")]
    fn forward_gpu(&self, asr: &Tensor, f0: &Tensor, n: &Tensor, style: &Tensor, gpu: &KokoroGpuDecoder) -> Result<Tensor> {
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

    #[cfg(feature = "triton-metal")]
    fn forward_gpu(&self, x: &Tensor, f0: &Tensor, style: &Tensor, gpu: &KokoroGpuDecoder) -> Result<Tensor> {
        let device = x.device();
        let dtype = x.dtype();
        let (_batch, _, t_frames) = x.dims3()?;

        let har_source = self.generate_harmonic_source(f0, 1, t_frames, device, dtype)?;

        let mut h = x.clone();
        let num_ups = self.ups.len();

        for (stage, (up_w, up_b)) in self.ups.iter().enumerate() {
            h = gpu.leaky_relu(&h, 0.1)?;
            h = h.to_dtype(candle_core::DType::F32)?;

            // Noise source injection
            let (nc_w, nc_b) = &self.noise_convs[stage];
            let nc_stride = self.noise_conv_strides[stage];
            let nc_padding = if nc_w.dim(2)? > 1 { (nc_stride + 1) / 2 } else { 0 };
            let x_source = har_source.conv1d(nc_w, nc_padding, nc_stride, 1, 1)?;
            let x_source = x_source.broadcast_add(&nc_b.unsqueeze(0)?.unsqueeze(2)?)?;
            let x_source = self.noise_res[stage].forward_gpu(&x_source, style, gpu)?;

            // Upsample via conv_transpose1d
            let up_stride = if stage == 0 { 10 } else { 6 };
            let up_padding = (up_w.dim(2)? - up_stride) / 2;
            h = h.conv_transpose1d(up_w, up_padding, 0, up_stride, 1, 1)?;
            h = h.broadcast_add(&up_b.unsqueeze(0)?.unsqueeze(2)?)?.contiguous()?;

            if stage == num_ups - 1 {
                h = reflection_pad1d(&h, 1, 0)?;
            }

            // x_source is F32 (from resblock forward_gpu which outputs F32)
            h = (h + x_source)?;

            // Resblocks with GPU-fused AdaIN+Snake
            let base = stage * 3;
            let mut sum = self.resblocks[base].forward_gpu(&h, style, gpu)?;
            sum = (sum + self.resblocks[base + 1].forward_gpu(&h, style, gpu)?)?;
            sum = (sum + self.resblocks[base + 2].forward_gpu(&h, style, gpu)?)?;
            h = (sum / 3.0)?;
        }

        // Final: leaky_relu + conv_post
        h = gpu.leaky_relu(&h, 0.01)?;
        h = h.to_dtype(candle_core::DType::F32)?;
        let h = h.conv1d(&self.conv_post_weight, 3, 1, 1, 1)?;
        let h = h.broadcast_add(&self.conv_post_bias.unsqueeze(0)?.unsqueeze(2)?)?;

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
        let n_bins = n_fft / 2 + 1; // 11
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

#[cfg(feature = "triton-metal")]
impl ResBlock {
    fn forward_gpu(&self, x: &Tensor, style: &Tensor, gpu: &KokoroGpuDecoder) -> Result<Tensor> {
        let dilations = [1usize, 3, 5];
        let mut h = x.clone();

        for i in 0..3 {
            let residual = h.clone();
            let dilation = dilations[i];
            let padding1 = (self.kernel_size - 1) * dilation / 2;
            let padding2 = (self.kernel_size - 1) / 2;

            let params = self.adain1[i].forward(style)?;
            let gamma = params.narrow(1, 0, self.channels)?.contiguous()?;
            let beta = params.narrow(1, self.channels, self.channels)?.contiguous()?;
            h = gpu.adain_snake(&h, &gamma, &beta, &self.alpha1[i])?;
            h = h.to_dtype(candle_core::DType::F32)?;

            let (w1, b1) = &self.convs1[i];
            h = h.conv1d(w1, padding1, 1, dilation, 1)?;
            h = h.broadcast_add(&b1.unsqueeze(0)?.unsqueeze(2)?)?;

            let params = self.adain2[i].forward(style)?;
            let gamma = params.narrow(1, 0, self.channels)?.contiguous()?;
            let beta = params.narrow(1, self.channels, self.channels)?.contiguous()?;
            h = gpu.adain_snake(&h, &gamma, &beta, &self.alpha2[i])?;
            h = h.to_dtype(candle_core::DType::F32)?;

            let (w2, b2) = &self.convs2[i];
            h = h.conv1d(w2, padding2, 1, 1, 1)?;
            h = h.broadcast_add(&b2.unsqueeze(0)?.unsqueeze(2)?)?;

            h = (h + residual)?;
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
    Tensor::from_vec(out, (batch, channels, new_len), x.device()).map_err(Into::into)
}

fn snake_activation_tensor(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let ax = x.broadcast_mul(alpha)?;
    let sin_sq = ax.sin()?.sqr()?;
    let inv_alpha = (alpha + 1e-9)?.recip()?;
    (x + sin_sq.broadcast_mul(&inv_alpha)?).map_err(Into::into)
}

/// Simple xorshift64 PRNG for reproducible noise generation.
fn rand_normal() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = Cell::new(0x12345678_9abcdef0);
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        // Box-Muller approximation using two uniform samples
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
