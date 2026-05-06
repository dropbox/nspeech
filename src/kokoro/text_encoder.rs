//! TextEncoder — phoneme embedding + Conv1D stack + BiLSTM.
//!
//! Tensor names in safetensors:
//!   text_encoder.embedding.weight: [178, 512]
//!   text_encoder.cnn.{0,1,2}.0.weight: [512, 512, 5] (conv, weight-norm folded)
//!   text_encoder.cnn.{0,1,2}.0.bias: [512]
//!   text_encoder.cnn.{0,1,2}.1.gamma: [512]
//!   text_encoder.cnn.{0,1,2}.1.beta: [512]
//!   text_encoder.lstm.weight_ih_l0: [1024, 512]
//!   text_encoder.lstm.weight_hh_l0: [1024, 256]
//!   (+ reverse variants)

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{self as nn, Embedding, Module, VarBuilder};

pub struct TextEncoder {
    embedding: Embedding,
    convs: Vec<ConvBlock>,
    lstm: BiLSTM,
}

struct ConvBlock {
    weight: Tensor,
    bias: Tensor,
    gamma: Tensor,
    beta: Tensor,
    padding: usize,
}

struct BiLSTM {
    forward_ih: Tensor,
    forward_hh: Tensor,
    forward_bias_ih: Tensor,
    forward_bias_hh: Tensor,
    reverse_ih: Tensor,
    reverse_hh: Tensor,
    reverse_bias_ih: Tensor,
    reverse_bias_hh: Tensor,
    hidden_size: usize,
}

impl TextEncoder {
    pub fn load(vb: VarBuilder, n_token: usize, hidden_dim: usize) -> Result<Self> {
        let embedding = Embedding::new(
            vb.get((n_token, hidden_dim), "embedding.weight")?,
            hidden_dim,
        );

        let mut convs = Vec::new();
        for i in 0..3 {
            let cvb = vb.pp(format!("cnn.{}", i));
            convs.push(ConvBlock::load(cvb, hidden_dim, 5)?);
        }

        let lstm = BiLSTM::load(vb.pp("lstm"), hidden_dim, hidden_dim / 2)?;

        Ok(Self { embedding, convs, lstm })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // [B, T] -> [B, T, 512] -> [B, 512, T]
        let x = self.embedding.forward(input_ids)?;
        let mut x = x.transpose(1, 2)?;

        for conv in &self.convs {
            x = conv.forward(&x)?;
        }

        // [B, 512, T] -> [B, T, 512] for LSTM
        let x = x.transpose(1, 2)?.contiguous()?;
        let x = self.lstm.forward(&x)?;
        // BiLSTM output: [B, T, 512] (256*2 concatenated)
        // Transpose back: [B, 512, T]
        x.transpose(1, 2).map_err(Into::into)
    }
}

impl ConvBlock {
    fn load(vb: VarBuilder, hidden_dim: usize, kernel_size: usize) -> Result<Self> {
        let conv_vb = vb.pp("0");
        let weight = conv_vb.get((hidden_dim, hidden_dim, kernel_size), "weight")?;
        let bias = conv_vb.get(hidden_dim, "bias")?;

        let norm_vb = vb.pp("1");
        let gamma = norm_vb.get(hidden_dim, "gamma")?;
        let beta = norm_vb.get(hidden_dim, "beta")?;

        let padding = (kernel_size - 1) / 2;
        Ok(Self { weight, bias, gamma, beta, padding })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Conv1d
        let x = x.conv1d(&self.weight, self.padding, 1, 1, 1)?;
        let x = x.broadcast_add(&self.bias.unsqueeze(0)?.unsqueeze(2)?)?;

        // LayerNorm over channel dim: transpose to [B, T, C], normalize, transpose back
        let xt = x.transpose(1, 2)?; // [B, T, C]
        let mean = xt.mean_keepdim(2)?;
        let diff = xt.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(2)?;
        let norm = diff.broadcast_div(&(var + 1e-5)?.sqrt()?)?;
        let xt = norm.broadcast_mul(&self.gamma.unsqueeze(0)?.unsqueeze(0)?)?
            .broadcast_add(&self.beta.unsqueeze(0)?.unsqueeze(0)?)?;
        let x = xt.transpose(1, 2)?; // [B, C, T]

        // LeakyReLU(0.2)
        let zeros = x.zeros_like()?;
        let pos = x.maximum(&zeros)?;
        let neg = x.minimum(&zeros)?;
        (pos + neg * 0.2).map_err(Into::into)
    }
}

impl BiLSTM {
    fn load(vb: VarBuilder, input_size: usize, hidden_size: usize) -> Result<Self> {
        let gate_size = 4 * hidden_size;
        Ok(Self {
            forward_ih: vb.get((gate_size, input_size), "weight_ih_l0")?,
            forward_hh: vb.get((gate_size, hidden_size), "weight_hh_l0")?,
            forward_bias_ih: vb.get(gate_size, "bias_ih_l0")?,
            forward_bias_hh: vb.get(gate_size, "bias_hh_l0")?,
            reverse_ih: vb.get((gate_size, input_size), "weight_ih_l0_reverse")?,
            reverse_hh: vb.get((gate_size, hidden_size), "weight_hh_l0_reverse")?,
            reverse_bias_ih: vb.get(gate_size, "bias_ih_l0_reverse")?,
            reverse_bias_hh: vb.get(gate_size, "bias_hh_l0_reverse")?,
            hidden_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fwd = self.run_direction(
            x, &self.forward_ih, &self.forward_hh,
            &self.forward_bias_ih, &self.forward_bias_hh, false,
        )?;
        let rev = self.run_direction(
            x, &self.reverse_ih, &self.reverse_hh,
            &self.reverse_bias_ih, &self.reverse_bias_hh, true,
        )?;

        // Concatenate forward + reverse: [B, T, 2*hidden]
        Tensor::cat(&[&fwd, &rev], 2).map_err(Into::into)
    }

    fn run_direction(
        &self, x: &Tensor,
        w_ih: &Tensor, w_hh: &Tensor,
        b_ih: &Tensor, b_hh: &Tensor,
        reverse: bool,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();
        let hs = self.hidden_size;

        let mut h = Tensor::zeros((batch, hs), dtype, device)?;
        let mut c = Tensor::zeros((batch, hs), dtype, device)?;
        let mut outputs = Vec::with_capacity(seq_len);

        let indices: Vec<usize> = if reverse {
            (0..seq_len).rev().collect()
        } else {
            (0..seq_len).collect()
        };

        for &t in &indices {
            let xt = x.narrow(1, t, 1)?.squeeze(1)?.contiguous()?;
            let gates = (xt.matmul(&w_ih.t()?)?.broadcast_add(b_ih)?
                + h.matmul(&w_hh.t()?)?.broadcast_add(b_hh)?)?;

            let i = nn::ops::sigmoid(&gates.narrow(1, 0, hs)?)?;
            let f = nn::ops::sigmoid(&gates.narrow(1, hs, hs)?)?;
            let g = gates.narrow(1, 2 * hs, hs)?.tanh()?;
            let o = nn::ops::sigmoid(&gates.narrow(1, 3 * hs, hs)?)?;

            c = ((&f * &c)? + (&i * &g)?)?;
            h = (&o * &c.tanh()?)?;
            outputs.push(h.unsqueeze(1)?);
        }

        if reverse {
            outputs.reverse();
        }
        Tensor::cat(&outputs, 1).map_err(Into::into)
    }
}
