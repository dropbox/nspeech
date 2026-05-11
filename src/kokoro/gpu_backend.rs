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

    /// Upload f32 data (activation — not cached).
    fn upload_f32(&self, _data: &[f32]) -> Result<Self::Buf> {
        Err(anyhow::anyhow!("upload_f32 not supported"))
    }

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

    /// Im2col: rearrange [C_in, T_in] → [C_in*K, T_out] for matmul-based conv1d.
    fn im2col(&self, x: &Self::Buf, out: &Self::Buf,
              c_in: usize, t_in: usize, t_out: usize, k: usize,
              stride: usize, padding: usize, dilation: usize) -> Result<()>;

    /// Im2col with fused leaky_relu(0.1) on input values.
    fn im2col_lrelu(&self, x: &Self::Buf, out: &Self::Buf,
                    c_in: usize, t_in: usize, t_out: usize, k: usize,
                    stride: usize, padding: usize, dilation: usize) -> Result<()> {
        self.im2col(x, out, c_in, t_in, t_out, k, stride, padding, dilation)
    }

    /// Matmul with bias: out[M,N] = A[M,K] @ B[K,N] + bias[M].
    /// Used for im2col-based conv1d: W[C_out, C_in*K] @ im2col[C_in*K, T_out] + bias[C_out].
    fn matmul_bias(&self, a: &Self::Buf, b: &Self::Buf, bias: &Self::Buf, out: &Self::Buf,
                   m: usize, n: usize, k: usize) -> Result<()>;

    /// Conv1d via im2col + matmul. Default impl chains im2col → matmul_bias.
    /// Faster than naive conv1d on GPUs with weak scalar ALUs (Intel UHD).
    fn conv1d_matmul(&self, x: &Self::Buf, w: &Self::Buf, bias: &Self::Buf, out: &Self::Buf,
                     c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                     k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let col_buf = self.alloc(c_in * k * t_out)?;
        self.im2col(x, &col_buf, c_in, t_in, t_out, k, stride, padding, dilation)?;
        self.matmul_bias(w, &col_buf, bias, out, c_out, t_out, c_in * k)
    }

    /// Conv1d via im2col(with fused lrelu 0.1) + matmul.
    fn conv1d_matmul_lrelu(&self, x: &Self::Buf, w: &Self::Buf, bias: &Self::Buf, out: &Self::Buf,
                           c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                           k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let col_buf = self.alloc(c_in * k * t_out)?;
        self.im2col_lrelu(x, &col_buf, c_in, t_in, t_out, k, stride, padding, dilation)?;
        self.matmul_bias(w, &col_buf, bias, out, c_out, t_out, c_in * k)
    }

    // ── F32-intermediate operations (prevent precision loss through normalization) ──

    /// Whether this backend supports f32 intermediate buffers.
    /// If true, resblocks use f32 activations to avoid f16 error amplification.
    fn has_f32_intermediates(&self) -> bool { false }

    /// Download f32 buffer to CPU. Only available when has_f32_intermediates() is true.
    fn download_f32(&self, _buf: &Self::Buf, _count: usize) -> Result<Vec<f32>> {
        Err(anyhow::anyhow!("download_f32 not supported"))
    }

    /// Allocate an f32 buffer for `count` elements (4 bytes each).
    fn alloc_f32(&self, _count: usize) -> Result<Self::Buf> {
        Err(anyhow::anyhow!("f32 alloc not supported"))
    }

    /// Convert f16 buffer to f32.
    fn f16_to_f32(&self, _x: &Self::Buf, _out: &Self::Buf, _n: usize) -> Result<()> {
        Err(anyhow::anyhow!("f16_to_f32 not supported"))
    }

    /// Convert f32 buffer to f16.
    fn f32_to_f16(&self, _x: &Self::Buf, _out: &Self::Buf, _n: usize) -> Result<()> {
        Err(anyhow::anyhow!("f32_to_f16 not supported"))
    }

    /// Im2col from f32 input to f16 output: [C_in, T_in] f32 → [C_in*K, T_out] f16.
    fn im2col_f32_to_f16(&self, _x: &Self::Buf, _out: &Self::Buf,
                         _c_in: usize, _t_in: usize, _t_out: usize, _k: usize,
                         _stride: usize, _padding: usize, _dilation: usize) -> Result<()> {
        Err(anyhow::anyhow!("im2col_f32_to_f16 not supported"))
    }

    /// Conv1d with f32 input, f16 weights, f32 output.
    /// Default impl uses im2col_f32→f16 + matmul when dimensions align.
    fn conv1d_f32(&self, x: &Self::Buf, w: &Self::Buf, bias: &Self::Buf, out: &Self::Buf,
                  c_in: usize, c_out: usize, t_in: usize, t_out: usize,
                  k: usize, stride: usize, padding: usize, dilation: usize) -> Result<()> {
        let kk = c_in * k;
        if kk % 32 == 0 && c_out % 64 == 0 {
            let col_buf = self.alloc(kk * t_out)?;
            self.im2col_f32_to_f16(x, &col_buf, c_in, t_in, t_out, k, stride, padding, dilation)?;
            let matmul_out = self.alloc(c_out * t_out)?;
            self.matmul_bias(w, &col_buf, bias, &matmul_out, c_out, t_out, kk)?;
            self.f16_to_f32(&matmul_out, out, c_out * t_out)?;
            return Ok(());
        }
        Err(anyhow::anyhow!("conv1d_f32 not supported for these dimensions"))
    }

    /// AdaIN + snake with f32 input and f32 output.
    fn adain_snake_f32(&self, _x: &Self::Buf, _gamma: &Self::Buf, _beta: &Self::Buf,
                       _alpha: &Self::Buf, _out: &Self::Buf,
                       _channels: usize, _seq_len: usize) -> Result<()> {
        Err(anyhow::anyhow!("adain_snake_f32 not supported"))
    }

    /// Element-wise add of f32 buffers.
    fn add_f32(&self, _a: &Self::Buf, _b: &Self::Buf, _out: &Self::Buf, _n: usize) -> Result<()> {
        Err(anyhow::anyhow!("add_f32 not supported"))
    }

    /// Scale f32 buffer by 1/3.
    fn scale_third_f32(&self, _x: &Self::Buf, _out: &Self::Buf, _n: usize) -> Result<()> {
        Err(anyhow::anyhow!("scale_third_f32 not supported"))
    }

    /// LeakyReLU on f32 buffers: out = x >= 0 ? x : x * slope.
    fn leaky_relu_f32(&self, _x: &Self::Buf, _out: &Self::Buf, _n: usize, _slope: f32) -> Result<()> {
        Err(anyhow::anyhow!("leaky_relu_f32 not supported"))
    }

    /// ConvTranspose1d with f32 I/O, f16 weights, fused leaky_relu(0.1) on input.
    fn conv_transpose1d_f32io_lrelu(&self, _x: &Self::Buf, _w: &Self::Buf, _bias: &Self::Buf, _out: &Self::Buf,
                                    _c_in: usize, _c_out: usize, _t_in: usize, _t_out: usize,
                                    _k: usize, _stride: usize, _padding: usize) -> Result<()> {
        Err(anyhow::anyhow!("conv_transpose1d_f32io_lrelu not supported"))
    }

    /// ConvTranspose1d with f32 I/O, f16 weights (no activation).
    fn conv_transpose1d_f32io(&self, _x: &Self::Buf, _w: &Self::Buf, _bias: &Self::Buf, _out: &Self::Buf,
                              _c_in: usize, _c_out: usize, _t_in: usize, _t_out: usize,
                              _k: usize, _stride: usize, _padding: usize) -> Result<()> {
        Err(anyhow::anyhow!("conv_transpose1d_f32io not supported"))
    }

    /// Reflection pad1d (pad_left=1, pad_right=0) for f32 buffers.
    fn reflection_pad1d_f32(&self, _x: &Self::Buf, _out: &Self::Buf, _channels: usize, _seq_len: usize) -> Result<()> {
        Err(anyhow::anyhow!("reflection_pad1d_f32 not supported"))
    }

    /// Fused iSTFT on GPU: conv_post output [22, n_frames] f32 → audio [out_len] f32.
    /// Fuses exp(mag) + sin(phase) + iDFT + overlap-add + COLA normalization.
    fn istft_gpu(&self, _x: &Self::Buf, _out: &Self::Buf, _n_frames: usize, _out_len: usize) -> Result<()> {
        Err(anyhow::anyhow!("istft_gpu not supported"))
    }

    /// Begin batching dispatches (avoids per-dispatch submit+wait overhead).
    /// Default is no-op (Metal doesn't need this — its command encoder already batches).
    fn begin_batch(&self) -> Result<()> { Ok(()) }

    /// End batch and execute all recorded dispatches.
    fn end_batch(&self) -> Result<()> { Ok(()) }
}
