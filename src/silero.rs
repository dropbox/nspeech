// silero_vad.rs
use anyhow::{bail, Result};
use candle_core::{Device, Module, Tensor};
use candle_nn::{self as nn, RNN};

#[derive(Clone, Debug)]
pub struct StftConfig {
    pub n_fft: usize,
    pub hop: usize,
    pub win: usize,
}

pub struct StftMag {
    cfg: StftConfig,
    // [out_channels = n_fft/2+1, in_channels=1, kernel_size=win]
    // These are fixed DFT kernels (cos/sin), multiplied by window.
    k_re: Tensor,
    k_im: Tensor,
}

impl StftMag {
    pub fn new(cfg: StftConfig, device: &Device) -> Result<Self> {
        let n_bins = cfg.n_fft / 2 + 1;

        // Hann window
        let mut window = vec![0f32; cfg.win];
        for i in 0..cfg.win {
            window[i] = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / cfg.win as f32).cos();
        }

        // Build DFT kernels for real/imag: bin k uses cos/sin(2π k n / n_fft)
        // kernel_size = win, but uses n_fft in phase
        let mut k_re = vec![0f32; n_bins * cfg.win];
        let mut k_im = vec![0f32; n_bins * cfg.win];

        for k in 0..n_bins {
            for n in 0..cfg.win {
                let phase = 2.0 * std::f32::consts::PI * (k as f32) * (n as f32) / (cfg.n_fft as f32);
                let w = window[n];
                k_re[k * cfg.win + n] = w * phase.cos();
                k_im[k * cfg.win + n] = -w * phase.sin(); // negative to match common STFT convention
            }
        }

        // Shape for Conv1d weight: [out_ch, in_ch, k]
        let k_re = Tensor::from_vec(k_re, (n_bins, 1, cfg.win), device)?;
        let k_im = Tensor::from_vec(k_im, (n_bins, 1, cfg.win), device)?;
        Ok(Self { cfg, k_re, k_im })
    }

    /// x: [B, 1, T] float32 PCM in [-1, 1]
    /// returns magnitude: [B, n_bins, frames]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // NOTE: Silero often uses reflection padding.
        // Here we do a simple symmetric padding with zeros (works, but you may want to match reflect).
        // pad = n_fft/2 is common for "center=True" STFT behavior.
        let pad = self.cfg.n_fft / 2;
        let (b, c, _t) = x.dims3()?;
        if c != 1 {
            bail!("expected mono with shape [B,1,T], got c={c}");
        }

        let left = Tensor::zeros((b, 1, pad), x.dtype(), x.device())?;
        let right = Tensor::zeros((b, 1, pad), x.dtype(), x.device())?;
        let x = Tensor::cat(&[&left, x, &right], 2)?;

        // Conv1d parameters: stride = hop, padding = 0 (we already padded)
        // Use Tensor's conv1d method
        let y_re = x.conv1d(&self.k_re, 0, self.cfg.hop, 1, 1)?;
        let y_im = x.conv1d(&self.k_im, 0, self.cfg.hop, 1, 1)?;

        // magnitude = sqrt(re^2 + im^2)
        let mag = ((&y_re * &y_re)? + (&y_im * &y_im)?)?.sqrt()?;
        Ok(mag)
    }
}

pub struct SileroVad {
    stft: StftMag,
    conv1: nn::Conv1d,
    conv2: nn::Conv1d,
    conv3: nn::Conv1d,
    conv4: nn::Conv1d,
    lstm: nn::rnn::LSTM,
    head: nn::Conv1d,
}

impl SileroVad {
    pub fn load_from_safetensors(
        vb: nn::VarBuilder,
        device: &Device,
        stft_cfg: StftConfig,
    ) -> Result<Self> {
        // STFT front-end computed in Rust, not loaded.
        let stft = StftMag::new(stft_cfg, device)?;

        // IMPORTANT: These channel sizes match the common Silero layout:
        // (n_fft/2+1) -> 128 -> 64 -> 64 -> 128
        // If your exported state_dict differs, adjust these sizes and/or key names.
        let n_bins = stft.cfg.n_fft / 2 + 1;

        let conv1 = nn::conv1d(
            n_bins,
            128,
            3,
            nn::Conv1dConfig { padding: 0, stride: 1, ..Default::default() },
            vb.pp("conv1"),
        )?;
        let conv2 = nn::conv1d(
            128,
            64,
            3,
            nn::Conv1dConfig { padding: 0, stride: 2, ..Default::default() },
            vb.pp("conv2"),
        )?;
        let conv3 = nn::conv1d(
            64,
            64,
            3,
            nn::Conv1dConfig { padding: 0, stride: 2, ..Default::default() },
            vb.pp("conv3"),
        )?;
        let conv4 = nn::conv1d(
            64,
            128,
            3,
            nn::Conv1dConfig { padding: 0, stride: 1, ..Default::default() },
            vb.pp("conv4"),
        )?;

        let lstm_cfg = nn::rnn::LSTMConfig::default();
        let lstm = nn::rnn::lstm(128, 128, lstm_cfg, vb.pp("lstm"))?;

        let head = nn::conv1d(
            128,
            1,
            1,
            nn::Conv1dConfig { padding: 0, stride: 1, ..Default::default() },
            vb.pp("head"),
        )?;

        Ok(Self { stft, conv1, conv2, conv3, conv4, lstm, head })
    }

    /// x: [B, T] f32 PCM, returns speech prob [B, frames] (or [B,1,frames] depending on how you squeeze)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.unsqueeze(1)?;              // [B,1,T]
        let mut z = self.stft.forward(&x)?;   // [B, n_bins, frames]

        z = self.conv1.forward(&z)?.relu()?;
        z = self.conv2.forward(&z)?.relu()?;
        z = self.conv3.forward(&z)?.relu()?;
        z = self.conv4.forward(&z)?.relu()?;  // [B,128,frames2]

        // LSTM expects sequence. We'll convert [B, C, T] -> [T, B, C]
        let z = z.transpose(1, 2)?; // [B, T, C]
        let z = z.transpose(0, 1)?; // [T, B, C]

        // Run LSTM (no explicit state passed here; for streaming, keep (h,c) and use step-by-step)
        let states = self.lstm.seq(&z)?;

        // Extract hidden states and stack them into a tensor [T, B, C]
        let y: Vec<Tensor> = states.iter().map(|s| s.h().clone()).collect();
        let y = Tensor::stack(&y, 0)?;

        // Back to [B, C, T]
        let y = y.transpose(0, 1)?; // [B, T, C]
        let y = y.transpose(1, 2)?; // [B, C, T]

        let p = nn::ops::sigmoid(&self.head.forward(&y)?)?; // [B,1,T]
        Ok(p.squeeze(1)?) // [B,T]
    }
}

