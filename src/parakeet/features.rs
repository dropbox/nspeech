use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;

// ----------------- Audio / log-mel frontend -----------------
const SAMPLE_RATE: u32 = 16_000;

//const LOG_ZERO_GUARD_VALUE: f32 = 2.0_f32.powi(-24);
const LOG_ZERO_GUARD_VALUE: f32 = 5.9604645e-08; // 2^-24

pub struct ParakeetFeatureExtractor {
    pub feature_size: usize,  // 80
    pub sampling_rate: usize, // 16000
    pub hop_length: usize,    // 160
    pub n_fft: usize,         // 512
    pub win_length: usize,    // 400
    pub preemphasis: f32,     // 0.97
    pub padding_value: f32,   // 0.0 (only for later padding if batching)
    pub normalize: bool,      // true = per_feature normalization, false = no normalization

    window: Vec<f32>,
    mel_filters: Vec<Vec<f32>>, // [feature_size][n_fft/2+1]
    fft: Arc<dyn Fft<f32>>,
}

impl ParakeetFeatureExtractor {
    /// Create feature extractor with per-feature normalization (default)
    pub fn new(feature_size: usize) -> Self {
        Self::new_with_config(feature_size, true)
    }

    /// Create feature extractor with configurable normalization
    /// normalize=true: per-feature normalization (for standard TDT model)
    /// normalize=false: no normalization (for streaming TDT model)
    pub fn new_with_config(feature_size: usize, normalize: bool) -> Self {
        let sampling_rate = 16_000usize;
        let hop_length = 160usize;
        let n_fft = 512usize;
        let win_length = 400usize;
        let preemphasis = 0.97f32;  // Re-enabled - NeMo uses preemphasis
        let padding_value = 0.0f32;

        let window = hann_window(win_length);
        let mel_filters =
            mel_filterbank_slaney_norm(feature_size, sampling_rate, n_fft, 0.0, sampling_rate as f32 / 2.0);

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        Self {
            feature_size,
            sampling_rate,
            hop_length,
            n_fft,
            win_length,
            preemphasis,
            padding_value,
            normalize,
            window,
            mel_filters,
            fft,
        }
    }

    /// Input: 16kHz mono f32 PCM
    /// Output: Candle tensor [1, T, feature_size]
    pub fn extract_to_tensor(&self, pcm16k: &[f32], device: &Device) -> Result<Tensor> {
        let (frames, feats) = self.extract_flat(pcm16k);
        Ok(Tensor::from_vec(feats, (1, frames, self.feature_size), device)?)
    }

    /// Output flattened row-major [T, F]
    pub fn extract_flat(&self, x: &[f32]) -> (usize, Vec<f32>) {
        // 1) preemphasis
        let x = if self.preemphasis != 0.0 {
            preemphasis(x, self.preemphasis)
        } else {
            x.to_vec()
        };

        // 2) torch.stft center padding: reflection padding, pad = n_fft/2 on both sides
        let pad = self.n_fft / 2;
        let mut padded = Vec::with_capacity(x.len() + 2 * pad);

        // Reflect left side
        for i in 0..pad {
            let idx = (i + 1).min(x.len() - 1);
            padded.push(x[idx]);
        }
        padded.extend_from_slice(&x);
        // Reflect right side
        for i in 0..pad {
            let idx = x.len().saturating_sub(2 + i);
            padded.push(x[idx]);
        }

        // 3) number of frames
        let frames = if padded.len() >= self.n_fft {
            1 + (padded.len() - self.n_fft) / self.hop_length
        } else {
            0
        };

        let n_freq = self.n_fft / 2 + 1;
        let mut feats = Vec::with_capacity(frames * self.feature_size);

        let mut fft_in = vec![Complex32::new(0.0, 0.0); self.n_fft];

        for t in 0..frames {
            let start = t * self.hop_length;

            // zero buffer
            for v in fft_in.iter_mut() {
                *v = Complex32::new(0.0, 0.0);
            }

            // windowed frame in first win_length, then zero pad to n_fft
            for i in 0..self.win_length {
                fft_in[i].re = padded[start + i] * self.window[i];
            }

            // FFT
            self.fft.process(&mut fft_in);

            // power spectrum
            let mut power = vec![0.0f32; n_freq];
            for k in 0..n_freq {
                let c = fft_in[k];
                power[k] = c.re * c.re + c.im * c.im;
            }

            // mel filterbank + log10
            for m in 0..self.feature_size {
                let filt = &self.mel_filters[m];
                let mut acc = 0.0f32;
                for k in 0..n_freq {
                    acc += filt[k] * power[k];
                }
                feats.push((acc + LOG_ZERO_GUARD_VALUE).log10());
            }
        }

        // Apply per-feature normalization only if enabled
        // NeMo's normalize='per_feature': Normalize each mel bin independently to mean=0, std=1
        // NeMo's normalize='NA': Skip normalization entirely
        if self.normalize {
            let num_features = self.feature_size;

            if frames > 0 {
                // Calculate mean and std for each feature dimension
                let mut means = vec![0.0f32; num_features];
                let mut stds = vec![0.0f32; num_features];

                // Calculate means
                for t in 0..frames {
                    for f in 0..num_features {
                        means[f] += feats[t * num_features + f];
                    }
                }
                for mean in means.iter_mut() {
                    *mean /= frames as f32;
                }

                // Calculate standard deviations
                for t in 0..frames {
                    for f in 0..num_features {
                        let diff = feats[t * num_features + f] - means[f];
                        stds[f] += diff * diff;
                    }
                }
                for std in stds.iter_mut() {
                    *std = (*std / frames as f32).sqrt();
                    // Avoid division by zero
                    if *std < 1e-10 {
                        *std = 1.0;
                    }
                }

                // Normalize: (x - mean) / std for each feature
                for t in 0..frames {
                    for f in 0..num_features {
                        let idx = t * num_features + f;
                        feats[idx] = (feats[idx] - means[f]) / stds[f];
                    }
                }
            }
        }

        (frames, feats)
    }
}

/* ------------------------- helpers ------------------------- */

fn preemphasis(x: &[f32], coef: f32) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }
    let mut y = Vec::with_capacity(x.len());
    y.push(x[0]);
    for i in 1..x.len() {
        y.push(x[i] - coef * x[i - 1]);
    }
    y
}

/// Hann window, periodic: w[n]=0.5-0.5*cos(2*pi*n/N)
/// This matches PyTorch/NeMo's default (periodic, NOT symmetric)
fn hann_window(n: usize) -> Vec<f32> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = n as f32;  // Periodic window uses N, not N-1
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * (i as f32) / denom).cos())
        .collect()
}

/// Slaney mel scale
fn hz_to_mel_slaney(f: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp; // 15
    let logstep = (6.4_f32).ln() / 27.0;
    if f < min_log_hz {
        f / f_sp
    } else {
        min_log_mel + (f / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz_slaney(m: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp; // 15
    let logstep = (6.4_f32).ln() / 27.0;
    if m < min_log_mel {
        f_sp * m
    } else {
        min_log_hz * ((m - min_log_mel) * logstep).exp()
    }
}

/// librosa mel with norm="slaney" (area norm), fmin=0, fmax=sr/2
fn mel_filterbank_slaney_norm(
    n_mels: usize,
    sr: usize,
    n_fft: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<Vec<f32>> {
    let n_freq = n_fft / 2 + 1;

    let fft_freqs: Vec<f32> = (0..n_freq)
        .map(|k| (k as f32) * (sr as f32) / (n_fft as f32))
        .collect();

    let mel_min = hz_to_mel_slaney(fmin);
    let mel_max = hz_to_mel_slaney(fmax);

    // n_mels + 2 points
    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..(n_mels + 2) {
        let t = i as f32 / (n_mels + 1) as f32;
        mel_points.push(mel_min + t * (mel_max - mel_min));
    }
    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz_slaney).collect();

    let mut filters = vec![vec![0.0f32; n_freq]; n_mels];

    for m in 0..n_mels {
        let f_left = hz_points[m];
        let f_center = hz_points[m + 1];
        let f_right = hz_points[m + 2];

        // Slaney area normalization
        let denom = (f_right - f_left).max(1e-12);
        let enorm = 2.0 / denom;

        for (k, &f) in fft_freqs.iter().enumerate() {
            let w = if f < f_left || f > f_right {
                0.0
            } else if f <= f_center {
                (f - f_left) / (f_center - f_left).max(1e-12)
            } else {
                (f_right - f) / (f_right - f_center).max(1e-12)
            };
            filters[m][k] = w * enorm;
        }
    }

    filters
}

/// Load pre-computed encoder input from Python (bypasses mel+subsampling)
pub fn load_python_encoder_input<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<Tensor> {
    let path_str = path.as_ref().to_str().unwrap();
    let subsamp_file = if path_str.contains("dots.wav") {
        "python_subsamp_dots.bin"
    } else {
        return Err(anyhow!("Pre-computed subsampling not available for this file. Use dots.wav"));
    };
    let data = std::fs::read(subsamp_file)?;
    let n_floats = data.len() / 4;
    let mut feats = Vec::with_capacity(n_floats);
    for chunk in data.chunks_exact(4) {
        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
        feats.push(f32::from_le_bytes(bytes));
    }
    let n_frames = feats.len() / 1024;
    let tensor = Tensor::from_slice(&feats, (1, n_frames, 1024), device)?;
    Ok(tensor)
}

/// Extract features from raw PCM samples (16kHz mono, normalized [-1, 1])
pub fn extract_features_from_samples(
    samples: &[f32],
    feat_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    if samples.is_empty() {
        return Err(anyhow!("empty audio samples"));
    }

    let fe = ParakeetFeatureExtractor::new(feat_dim);
    let tensor: Tensor = fe.extract_to_tensor(samples, device)?;
    Ok(tensor)
}

pub fn load_wav_as_features<P: AsRef<Path>>(
    path: P,
    feat_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut reader = hound::WavReader::open(&path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(anyhow!("expected mono wav, got {} channels", spec.channels));
    }
    if spec.sample_rate != SAMPLE_RATE {
        return Err(anyhow!(
            "expected {} Hz audio, got {} Hz",
            SAMPLE_RATE,
            spec.sample_rate
        ));
    }
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Int, 24) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / 8_388_608.0))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()?,
        _ => return Err(anyhow!("unsupported WAV format")),
    };

    extract_features_from_samples(&samples, feat_dim, device)
}
