use anyhow::{bail, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn as nn;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct VadConfig {
    pub sample_rate: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_fft: usize,
}

pub struct StftMag {
    /// forward_basis_buffer exported from silero:
    /// shape is typically [2 * n_bins, 1, win_length] (real then imag filters)
    basis: Tensor,
    hop: usize,
    n_bins: usize,
    n_fft: usize,
}

impl StftMag {
    pub fn new(basis: Tensor, cfg: &VadConfig) -> Result<Self> {
        let (oc, ic, k) = basis.dims3()?;
        if ic != 1 {
            bail!("stft basis expected in_ch=1, got {ic}");
        }
        if k != cfg.win_length {
            // Some versions use win_length == n_fft; if cfg is wrong you’ll notice here.
            // Still allow it but warn by erroring loudly.
            bail!("win_length mismatch: basis kernel={k} cfg.win_length={}", cfg.win_length);
        }
        // Commonly oc == 2*(n_fft/2+1)
        let n_bins = oc / 2;
        Ok(Self { basis, hop: cfg.hop_length, n_bins, n_fft: cfg.n_fft })
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

    /// Streaming STFT magnitude with NO center padding.
    /// Expects x: [B, 1, T] where T is whatever you currently have buffered.
    /// Returns [B, n_bins, frames] where frames = floor((T - win)/hop) + 1 (if T>=win)
    pub fn forward_streaming(&self, x: &Tensor) -> Result<Tensor> {
    // x: [B, 1, T]
        let (_b, c, _t) = x.dims3()?;
        println!("c={c} t={_t}");
        anyhow::ensure!(c == 1, "expected mono [B,1,T], got c={c}");

        // output: [B, 2*n_bins, frames]
        let y = x.conv1d(&self.basis, 0, self.hop, 1, 1)?;

        let y_re = y.narrow(1, 0, self.n_bins)?;
        let y_im = y.narrow(1, self.n_bins, self.n_bins)?;
        let mag = ((&y_re * &y_re)? + (&y_im * &y_im)?)?.sqrt()?;


        // ✅ log1p compression: ln(1 + mag)
        let mag1p = (&mag + 1.0)?;
        let feat = mag1p.log()?; // if .log() exists; otherwise nn::ops::log(&mag1p)?
        Ok(feat)


        //Ok(((&y_re * &y_re)? + (&y_im * &y_im)?)?.sqrt()?)

       // 🔥 add log-compression (Silero-style)
        // mag_log = log(mag + 1e-6)

        //let mag = (&mag + 1e-6)?;    // broadcast scalar add
        //let mag = mag.log()?;          // if your Candle has .log()
        //Ok(mag)
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
    pub fn load(device: &Device, st_path: &str, cfg_path: &str) -> Result<Self> {
        let cfg: VadConfig = serde_json::from_slice(&std::fs::read(cfg_path)?)?;

        let buffer = std::fs::read(st_path)?;
        let st = safetensors::SafeTensors::deserialize(&buffer)?;

        let (basis, enc, rnn, head) = Self::load_tensors(device, &st)?;

        let stft = StftMag::new(basis, &cfg)?;

        Ok(Self { cfg, stft, enc, rnn, head })
    }

    fn load_tensors(
        device: &Device,
        st: &safetensors::SafeTensors,
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
        ) -> Result<nn::Conv1d> {
            // Load weight and bias tensors
            let weight = load_tensor(st, wkey, device)?;
            let bias = load_tensor(st, bkey, device)?;

            // weight: [out_ch, in_ch, k]
            let (out_ch, in_ch, k) = weight.dims3()?;

            // Use padding=0 to match the original model (no padding in encoder)
            let cfg = nn::Conv1dConfig { stride, padding: 0, ..Default::default() };

            // Create conv1d with dummy var builder (zeros), then replace weights manually
            let mut tensors = std::collections::HashMap::new();
            tensors.insert("weight".to_string(), weight);
            tensors.insert("bias".to_string(), bias);
            let vb = nn::VarBuilder::from_tensors(tensors, DType::F32, device);

            Ok(nn::conv1d(in_ch, out_ch, k, cfg, vb)?)
        }

        // Strides: [1, 2, 2, 1] as used in the original Silero VAD model
        let enc0 = conv_from(st, device, "enc.0.weight", "enc.0.bias", 1)?;
        let enc1 = conv_from(st, device, "enc.1.weight", "enc.1.bias", 2)?;
        let enc2 = conv_from(st, device, "enc.2.weight", "enc.2.bias", 2)?;
        let enc3 = conv_from(st, device, "enc.3.weight", "enc.3.bias", 1)?;

        // --- RNN tensors ---
        let w_ih = load_tensor(st, "rnn.weight_ih", device)?;
        let w_hh = load_tensor(st, "rnn.weight_hh", device)?;
        let b_ih = load_tensor(st, "rnn.bias_ih", device)?;
        let b_hh = load_tensor(st, "rnn.bias_hh", device)?;

        let rnn = LstmCell::new(w_ih, w_hh, b_ih, b_hh)?;

        // --- Head conv (usually 1x1) ---
        let head = conv_from(st, device, "head.weight", "head.bias", 1)?;

        Ok((basis, [enc0, enc1, enc2, enc3], rnn, head))
    }

    /// x: [B, T] float PCM in [-1,1]
    /// returns p: [B, frames] speech probability
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.unsqueeze(1)?; // [B,1,T]

        // STFT magnitude: [B, n_bins, frames]
        let mut z = self.stft.forward(&x)?;

        // Conv stack with relu
        z = self.enc[0].forward(&z)?.relu()?;
        z = self.enc[1].forward(&z)?.relu()?;
        z = self.enc[2].forward(&z)?.relu()?;
        z = self.enc[3].forward(&z)?.relu()?; // [B, C, T']

        // Prepare for RNN: [T', B, C]
        let z = z.transpose(1, 2)?; // [B,T',C]
        let z = z.transpose(0, 1)?; // [T',B,C]

        // init state zeros
        let (_t, b, _c) = z.dims3()?;
        let h0 = Tensor::zeros((b, self.rnn.hidden), DType::F32, z.device())?;
        let c0 = Tensor::zeros((b, self.rnn.hidden), DType::F32, z.device())?;

        let (y, _hf, _cf) = self.rnn.forward(&z, h0, c0)?; // [T',B,H]

        // back to [B,H,T']
        let y = y.transpose(0, 1)?; // [B,T',H]
        let y = y.transpose(1, 2)?; // [B,H,T']

        let p = nn::ops::sigmoid(&self.head.forward(&y)?)?; // [B,1,T']
        Ok(p.squeeze(1)?) // [B,T']
    }
}

pub struct VadStream {
    vad: SileroVad,
    device: Device,
    pcm_buf: Vec<f32>,
    chunk_size: usize, // 512 samples @ 16kHz = 32ms
}

impl VadStream {
    pub fn new(vad: SileroVad, device: &Device) -> Result<Self> {
        // Use 2048-sample chunks (128ms at 16kHz)
        // Note: The official Silero model uses 512 samples, but our exported weights
        // with strides [1,2,2,1] require more input to avoid zero frames after encoder
        let chunk_size = 2048;

        Ok(Self {
            vad,
            device: device.clone(),
            pcm_buf: Vec::new(),
            chunk_size,
        })
    }

    pub fn push(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        self.pcm_buf.extend_from_slice(pcm);

        let mut out = Vec::new();

        // Process complete chunks
        while self.pcm_buf.len() >= self.chunk_size {
            // Extract one chunk
            let chunk: Vec<f32> = self.pcm_buf.drain(..self.chunk_size).collect();

            // Create tensor [1, chunk_size]
            let x = Tensor::from_slice(&chunk, (1, self.chunk_size), &self.device)?;

            // Call the standard forward method (processes chunk independently)
            let p = self.vad.forward(&x)?; // Returns [1, frames]

            // Get the mean probability across all frames (more robust than last frame)
            let probs = p.to_vec2::<f32>()?;
            if !probs.is_empty() && !probs[0].is_empty() {
                // Take the mean of all probability values
                let mean_prob: f32 = probs[0].iter().sum::<f32>() / probs[0].len() as f32;
                out.push(mean_prob);
            }
        }

        Ok(out)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.pcm_buf.clear();
        Ok(())
    }
}

