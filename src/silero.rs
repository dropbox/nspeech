use anyhow::{bail, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn as nn;
use serde::Deserialize;
use std::path::Path;

// Import VAD assets
use crate::parakeet::{VAD_CONFIG, VAD_MODEL};

#[derive(Debug, Clone, Deserialize)]
pub struct VadConfig {
    pub sample_rate: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_fft: usize,
    pub stft_right_padding: usize,
    pub encoder_padding: usize,
    pub context_size: usize,
    pub chunk_size: usize,
}

pub struct StftMag {
    /// forward_basis_buffer exported from silero:
    /// shape is typically [2 * n_bins, 1, win_length] (real then imag filters)
    basis: Tensor,
    hop: usize,
    n_bins: usize,
    n_fft: usize,
    /// Reflection padding on the right (official model uses 64 samples)
    right_pad: usize,
}

impl StftMag {
    pub fn new(basis: Tensor, cfg: &VadConfig) -> Result<Self> {
        let (oc, ic, k) = basis.dims3()?;
        if ic != 1 {
            bail!("stft basis expected in_ch=1, got {ic}");
        }
        if k != cfg.win_length {
            bail!("win_length mismatch: basis kernel={k} cfg.win_length={}", cfg.win_length);
        }
        // Commonly oc == 2*(n_fft/2+1)
        let n_bins = oc / 2;
        Ok(Self {
            basis,
            hop: cfg.hop_length,
            n_bins,
            n_fft: cfg.n_fft,
            right_pad: cfg.stft_right_padding,
        })
    }

    /// x: [B, 1, T] float
    /// returns magnitude: [B, n_bins, frames]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Most Silero STFTs behave like center=True (pad n_fft/2 both sides).
        let pad = self.n_fft / 2;
        let (b, c, _t) = x.dims3()?;
        if c != 1 { bail!("expected [B,1,T], got c={c}"); }

        let left = Tensor::zeros((b, 1, pad), x.dtype(), x.device())?;
        let right = Tensor::zeros((b, 1, pad), x.dtype(), x.device())?;
        let xpad = Tensor::cat(&[&left, x, &right], 2)?;

        // Conv1d: output [B, 2*n_bins, frames]
        let y = xpad.conv1d(&self.basis, 0, self.hop, 1, 1)?;

        // split real/imag along channel dim
        let y_re = y.narrow(1, 0, self.n_bins)?;
        let y_im = y.narrow(1, self.n_bins, self.n_bins)?;
        let mag = ((&y_re * &y_re)? + (&y_im * &y_im)?)?.sqrt()?;

        // log1p
        Ok((&mag + 1.0)?.log()?)


        //let mag = ((&y_re * &y_re)? + (&y_im * &y_im)?)?.sqrt()?;
        //Ok(mag)
    }

    /// Streaming STFT magnitude with reflection padding on the right.
    /// Expects x: [B, 1, T] where T is the input length (typically context + chunk).
    /// Applies reflection padding on the right: [a,b,c,d] + pad=2 => [a,b,c,d,c,b]
    /// Returns [B, n_bins, frames] - raw magnitude (no log transform)
    pub fn forward_streaming(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, c, t) = x.dims3()?;
        anyhow::ensure!(c == 1, "expected mono [B,1,T], got c={c}");

        // Apply reflection padding on the right
        // For reflection: mirror the last right_pad samples
        if self.right_pad > 0 {
            if t < self.right_pad {
                bail!("input too short for reflection padding: t={t} < right_pad={}", self.right_pad);
            }
            // Extract the last right_pad samples and reverse them
            // x[:, :, t-right_pad:t] becomes the padding
            let pad_region = x.narrow(2, t - self.right_pad, self.right_pad)?;

            // Reverse along the time dimension by manually building reversed tensor
            // This is a workaround since Candle may not have a flip operation
            // We'll just extract each sample in reverse order
            let mut pad_samples = Vec::new();
            for i in (0..self.right_pad).rev() {
                pad_samples.push(pad_region.narrow(2, i, 1)?);
            }
            let pad_reversed = Tensor::cat(&pad_samples, 2)?;

            // Concatenate original + reflected padding
            let x = Tensor::cat(&[x, &pad_reversed], 2)?;

            // Conv1d with no padding
            let y = x.conv1d(&self.basis, 0, self.hop, 1, 1)?;

            let y_re = y.narrow(1, 0, self.n_bins)?;
            let y_im = y.narrow(1, self.n_bins, self.n_bins)?;

            // Return raw magnitude (no log transform - that was the issue!)
            Ok(((&y_re * &y_re)? + (&y_im * &y_im)?)?.sqrt()?)
        } else {
            // No padding case
            let y = x.conv1d(&self.basis, 0, self.hop, 1, 1)?;
            let y_re = y.narrow(1, 0, self.n_bins)?;
            let y_im = y.narrow(1, self.n_bins, self.n_bins)?;
            Ok(((&y_re * &y_re)? + (&y_im * &y_im)?)?.sqrt()?)
        }
    }
}

/// Minimal single-layer LSTM cell matching weights:
/// weight_ih: [4H, I], weight_hh: [4H, H], biases: [4H]
pub struct LstmCell {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
    hidden: usize,
}

impl LstmCell {
    pub fn new(w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> Result<Self> {
        let (g1, _i) = w_ih.dims2()?;
        let (g2, h) = w_hh.dims2()?;
        if g1 != g2 { bail!("w_ih gates {g1} != w_hh gates {g2}"); }
        if g1 % 4 != 0 { bail!("gate dim {g1} not divisible by 4"); }
        if g1 / 4 != h { bail!("hidden mismatch: gates/4={} != h={}", g1/4, h); }
        Ok(Self { w_ih, w_hh, b_ih, b_hh, hidden: h })
    }

    /// x: [T, B, I]
    /// h0,c0: [B, H]
    /// returns y: [T, B, H], and final (h,c)
    pub fn forward(&self, x: &Tensor, mut h: Tensor, mut c: Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (t, _b, _i) = x.dims3()?;

        let mut outs: Vec<Tensor> = Vec::with_capacity(t);

        for ti in 0..t {
            // xt: [B, I]
            let xt = x.get(ti)?;

            // gates = xt @ w_ih^T + h @ w_hh^T + b_ih + b_hh
            // w_ih: [4H, I] so w_ih^T: [I, 4H]
            let g1 = xt.matmul(&self.w_ih.transpose(0, 1)?)?;
            let g2 = h.matmul(&self.w_hh.transpose(0, 1)?)?;
            let mut gates = (g1 + g2)?;
            let gate_shape = gates.shape().clone();
            gates = (gates + &self.b_ih.broadcast_as(&gate_shape)?)?;
            gates = (gates + &self.b_hh.broadcast_as(&gate_shape)?)?;

            // split gates: i,f,g,o each [B,H]
            let hidden = self.hidden;
            let i_gate = nn::ops::sigmoid(&gates.narrow(1, 0*hidden, hidden)?)?;
            let f_gate = nn::ops::sigmoid(&gates.narrow(1, 1*hidden, hidden)?)?;
            let g_gate = gates.narrow(1, 2*hidden, hidden)?.tanh()?;
            let o_gate = nn::ops::sigmoid(&gates.narrow(1, 3*hidden, hidden)?)?;

            c = ((&f_gate * &c)? + (&i_gate * &g_gate)?)?;
            let c_tanh = c.tanh()?;
            h = (&o_gate * c_tanh)?;

            outs.push(h.clone());
        }

        let y = Tensor::stack(&outs, 0)?; // [T,B,H]
        Ok((y, h, c))
    }
}

pub struct SileroVad {
    cfg: VadConfig,
    stft: StftMag,
    enc: [nn::Conv1d; 4],
    rnn: LstmCell,
    head: nn::Conv1d,
}

impl SileroVad {
    pub fn load<P: AsRef<Path>>(assets: P, device: &Device) -> Result<Self> {
        let assets = assets.as_ref().to_path_buf();

        // Load config from embedded asset (decompresses automatically)
        let cfg_bytes = VAD_CONFIG.bytes(&assets).map_err(|_| {
            anyhow::anyhow!("failed to load VAD config from assets")
        })?;
        let cfg: VadConfig = serde_json::from_slice(cfg_bytes)?;

        // Load model from embedded asset (decompresses automatically)
        let model_bytes = VAD_MODEL.bytes(&assets).map_err(|_| {
            anyhow::anyhow!("failed to load VAD model from assets")
        })?;
        let st = safetensors::SafeTensors::deserialize(model_bytes)?;

        let (basis, enc, rnn, head) = Self::load_tensors(device, &st, &cfg)?;

        let stft = StftMag::new(basis, &cfg)?;

        Ok(Self { cfg, stft, enc, rnn, head })
    }

    fn load_tensors(
        device: &Device,
        st: &safetensors::SafeTensors,
        cfg: &VadConfig,
    ) -> Result<(Tensor, [nn::Conv1d; 4], LstmCell, nn::Conv1d)> {
        // Helper to load a tensor from safetensors
        fn load_tensor(st: &safetensors::SafeTensors, name: &str, device: &Device) -> Result<Tensor> {
            let view = st.tensor(name)?;
            let shape = view.shape();
            let data = view.data();
            let dtype = match view.dtype() {
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::F16 => DType::F16,
                _ => bail!("unsupported dtype for tensor {}", name),
            };
            Ok(Tensor::from_raw_buffer(data, dtype, shape, device)?)
        }

        // --- STFT basis ---
        let basis_info = st.tensor("stft.forward_basis_buffer")?;
        let shape = basis_info.shape();
        if shape.len() != 3 { bail!("basis shape expected 3D, got {:?}", shape); }
        let basis = load_tensor(st, "stft.forward_basis_buffer", device)?;

        // --- Encoder convs (infer in/out/kernel from weight shapes) ---
        fn conv_from(
            st: &safetensors::SafeTensors,
            device: &Device,
            wkey: &str,
            bkey: &str,
            stride: usize,
            padding: usize,
        ) -> Result<nn::Conv1d> {
            // Load weight and bias tensors
            let weight = load_tensor(st, wkey, device)?;
            let bias = load_tensor(st, bkey, device)?;

            // weight: [out_ch, in_ch, k]
            let (out_ch, in_ch, k) = weight.dims3()?;

            // Use padding from config file
            let cfg = nn::Conv1dConfig { stride, padding, ..Default::default() };

            // Create conv1d with dummy var builder (zeros), then replace weights manually
            let mut tensors = std::collections::HashMap::new();
            tensors.insert("weight".to_string(), weight);
            tensors.insert("bias".to_string(), bias);
            let vb = nn::VarBuilder::from_tensors(tensors, DType::F32, device);

            Ok(nn::conv1d(in_ch, out_ch, k, cfg, vb)?)
        }

        // Strides: [1, 2, 2, 1] as used in the original Silero VAD model
        let padding = cfg.encoder_padding;
        let enc0 = conv_from(st, device, "enc.0.weight", "enc.0.bias", 1, padding)?;
        let enc1 = conv_from(st, device, "enc.1.weight", "enc.1.bias", 2, padding)?;
        let enc2 = conv_from(st, device, "enc.2.weight", "enc.2.bias", 2, padding)?;
        let enc3 = conv_from(st, device, "enc.3.weight", "enc.3.bias", 1, padding)?;

        // --- RNN tensors ---
        let w_ih = load_tensor(st, "rnn.weight_ih", device)?;
        let w_hh = load_tensor(st, "rnn.weight_hh", device)?;
        let b_ih = load_tensor(st, "rnn.bias_ih", device)?;
        let b_hh = load_tensor(st, "rnn.bias_hh", device)?;

        let rnn = LstmCell::new(w_ih, w_hh, b_ih, b_hh)?;

        // --- Head conv (usually 1x1, no padding needed) ---
        let head = conv_from(st, device, "head.weight", "head.bias", 1, 0)?;

        Ok((basis, [enc0, enc1, enc2, enc3], rnn, head))
    }

    /// x: [B, T] float PCM in [-1,1]
    /// h, c: optional RNN state from previous chunk
    /// returns (p, h_new, c_new): probabilities and new RNN state
    pub fn forward_stateful(&self, x: &Tensor, h: Option<Tensor>, c: Option<Tensor>) -> Result<(Tensor, Tensor, Tensor)> {
        let x = x.unsqueeze(1)?; // [B,1,T]

        // STFT magnitude with reflection padding: [B, n_bins, frames]
        let mut z = self.stft.forward_streaming(&x)?;

        // Conv stack with relu
        z = self.enc[0].forward(&z)?.relu()?;
        z = self.enc[1].forward(&z)?.relu()?;
        z = self.enc[2].forward(&z)?.relu()?;
        z = self.enc[3].forward(&z)?.relu()?; // [B, C, T']

        // Prepare for RNN: [T', B, C]
        let z = z.transpose(1, 2)?; // [B,T',C]
        let z = z.transpose(0, 1)?; // [T',B,C]

        let (_t, b, _ch) = z.dims3()?;

        // Use provided state or zeros
        let h0 = match h {
            Some(h) => h,
            None => Tensor::zeros((b, self.rnn.hidden), DType::F32, z.device())?,
        };
        let c0 = match c {
            Some(c) => c,
            None => Tensor::zeros((b, self.rnn.hidden), DType::F32, z.device())?,
        };

        let (y, h_new, c_new) = self.rnn.forward(&z, h0, c0)?; // [T',B,H]

        // back to [B,H,T']
        let y = y.transpose(0, 1)?; // [B,T',H]
        let y = y.transpose(1, 2)?; // [B,H,T']

        // Apply ReLU before head (official model has Dropout->ReLU->Conv->Sigmoid)
        let y = y.relu()?;

        let p = nn::ops::sigmoid(&self.head.forward(&y)?)?; // [B,1,T']
        Ok((p.squeeze(1)?, h_new, c_new))
    }

    /// Stateless version for compatibility
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (p, _h, _c) = self.forward_stateful(x, None, None)?;
        Ok(p)
    }
}

pub struct VadStream {
    vad: SileroVad,
    device: Device,
    pcm_buf: Vec<f32>,
    chunk_size: usize,
    context_size: usize,
    /// Context buffer: last context_size samples from previous chunk
    context: Vec<f32>,
    /// RNN state maintained across chunks
    h: Option<Tensor>,
    c: Option<Tensor>,
}

impl VadStream {
    pub fn new(vad: SileroVad, device: &Device) -> Result<Self> {
        let chunk_size = vad.cfg.chunk_size;
        let context_size = vad.cfg.context_size;

        Ok(Self {
            vad,
            device: device.clone(),
            pcm_buf: Vec::new(),
            chunk_size,
            context_size,
            context: vec![0.0; context_size],
            h: None,
            c: None,
        })
    }

    pub fn push(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        self.pcm_buf.extend_from_slice(pcm);

        let mut out = Vec::new();

        // Process complete chunks
        while self.pcm_buf.len() >= self.chunk_size {
            // Extract one chunk
            let chunk: Vec<f32> = self.pcm_buf.drain(..self.chunk_size).collect();

            // Prepend context to chunk (like official model does)
            let mut with_context = self.context.clone();
            with_context.extend_from_slice(&chunk);

            // Create tensor [1, context_size + chunk_size]
            let x = Tensor::from_slice(&with_context, (1, with_context.len()), &self.device)?;

            // Call stateful forward, passing and receiving RNN state
            let (p, h_new, c_new) = self.vad.forward_stateful(&x, self.h.take(), self.c.take())?;

            // Store new state for next chunk
            self.h = Some(h_new);
            self.c = Some(c_new);

            // Update context buffer with last context_size samples from this chunk
            self.context.clear();
            self.context.extend_from_slice(&chunk[chunk.len() - self.context_size..]);

            // Output each frame separately for better temporal resolution
            let probs = p.to_vec2::<f32>()?;
            if !probs.is_empty() && !probs[0].is_empty() {
                for &prob in &probs[0] {
                    out.push(prob);
                }
            }
        }

        Ok(out)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.pcm_buf.clear();
        self.context.clear();
        self.context.resize(self.context_size, 0.0);
        self.h = None;
        self.c = None;
        Ok(())
    }
}

