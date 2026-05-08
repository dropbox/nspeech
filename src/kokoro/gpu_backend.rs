//! Cross-platform GPU backend trait for Kokoro TTS decoder.
//!
//! Both Metal and D3D12 implement this trait, enabling a single `forward_gpu`
//! implementation that works on all platforms.

use anyhow::Result;

pub trait KokoroGpuBackend {
    type Buf: Clone;

    /// Allocate a buffer for `count` f16 elements.
    fn alloc(&self, count: usize) -> Result<Self::Buf>;

    /// Upload f16 data (activation — not cached).
    fn upload_f16(&self, data: &[half::f16]) -> Result<Self::Buf>;

    /// Upload f16 weight data (cached by `id`).
    fn upload_weight(&self, id: usize, data: &[half::f16]) -> Result<Self::Buf>;

    /// Download f16 buffer to CPU.
    fn download_f16(&self, buf: &Self::Buf, count: usize) -> Result<Vec<half::f16>>;

    /// Element-wise add: returns new buffer where out[i] = a[i] + b[i].
    fn add(&self, a: &Self::Buf, b: &Self::Buf, n: usize) -> Result<Self::Buf> {
        let a_data = self.download_f16(a, n)?;
        let b_data = self.download_f16(b, n)?;
        let sum: Vec<half::f16> = a_data.iter().zip(b_data.iter())
            .map(|(x, y)| half::f16::from_f32(x.to_f32() + y.to_f32()))
            .collect();
        self.upload_f16(&sum)
    }

    /// Scale: returns new buffer where out[i] = x[i] * scalar.
    fn scale(&self, x: &Self::Buf, n: usize, s: f32) -> Result<Self::Buf> {
        let data = self.download_f16(x, n)?;
        let scaled: Vec<half::f16> = data.iter()
            .map(|v| half::f16::from_f32(v.to_f32() * s))
            .collect();
        self.upload_f16(&scaled)
    }

    /// Dispatch leaky_relu: out = x < 0 ? x*slope : x
    fn leaky_relu(&self, x: &Self::Buf, out: &Self::Buf, n_elements: usize, slope: f32) -> Result<()>;

    /// Dispatch snake: out = x + sin²(αx)/α
    fn snake(&self, x: &Self::Buf, alpha: &Self::Buf, out: &Self::Buf,
             n_elements: usize, channels: usize, seq_len: usize) -> Result<()>;

    /// Dispatch fused AdaIN + snake (seq_len <= 1024).
    fn adain_snake(&self, x: &Self::Buf, gamma: &Self::Buf, beta: &Self::Buf,
                   alpha: &Self::Buf, out: &Self::Buf,
                   channels: usize, seq_len: usize) -> Result<()>;

    /// Dispatch conv1d.
    fn conv1d(&self, x: &Self::Buf, w: &Self::Buf, bias: &Self::Buf, out: &Self::Buf,
              c_in: usize, c_out: usize, t_in: usize, t_out: usize,
              k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()>;

    /// Dispatch conv1d with compile-time K (uses specialized unrolled kernel).
    /// Default impl falls back to generic conv1d.
    fn conv1d_k(&self, x: &Self::Buf, w: &Self::Buf, bias: &Self::Buf, out: &Self::Buf,
                c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        self.conv1d(x, w, bias, out, c_in, c_out, t_in, t_out, k, stride, padding, dilation)
    }

    /// Dispatch conv_transpose1d.
    fn conv_transpose1d(&self, x: &Self::Buf, w: &Self::Buf, bias: &Self::Buf, out: &Self::Buf,
                        c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                        k: usize, stride: usize, padding: usize) -> Result<()>;

    /// Fused leaky_relu(0.1) + conv_transpose1d (activation applied to input on load).
    fn conv_transpose1d_lrelu(&self, x: &Self::Buf, w: &Self::Buf, bias: &Self::Buf, out: &Self::Buf,
                              c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                              k: usize, stride: usize, padding: usize) -> Result<()> {
        self.conv_transpose1d(x, w, bias, out, c_in, c_out, t_in, t_out, k, stride, padding)
    }

    /// Fused leaky_relu(0.01) + conv1d (activation applied to input on load).
    fn conv1d_lrelu001(&self, x: &Self::Buf, w: &Self::Buf, bias: &Self::Buf, out: &Self::Buf,
                       c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                       k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        self.conv1d(x, w, bias, out, c_in, c_out, t_in, t_out, k, stride, padding, dilation)
    }

    /// Reflection pad1d (pad_left=1, pad_right=0): out is [C, T+1].
    fn reflection_pad1d(&self, x: &Self::Buf, out: &Self::Buf, channels: usize, seq_len: usize) -> Result<()>;
}
