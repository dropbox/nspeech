//! Shared GPU encoder logic for Metal and D3D12 backends.
//!
//! The `EncoderBackend` trait abstracts per-platform dispatch.
//! `GpuEncoder<B>` implements the full Moonshine encoder layer loop, weight loading
//! (with gamma baking), scratch pre-allocation, and forward pass — once, shared
//! across both backends.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::config::MoonshineConfig;

type QVarBuilder = candle_transformers::quantized_var_builder::VarBuilder;

fn cdiv(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

// ── Trait ──

/// Backend-specific GPU operations for the encoder.
/// Backends decide kernel selection, tile sizes, barriers, etc.
/// All methods take `&self` — backends use interior mutability if needed.
pub trait EncoderBackend: Sized {
    type Buf;

    // ── Buffer management ──
    fn alloc_activation(&self, count: usize) -> Result<Self::Buf>;  // f16
    fn alloc_residual(&self, count: usize) -> Result<Self::Buf>;    // f16 on Metal, f32 on D3D12

    // ── Weight upload (CPU f32 → GPU) ──
    /// Upload 2D matmul weight. Metal: f16, D3D12: f32 (preserves Q8 precision).
    fn upload_matmul_weight(&self, data_f32: &[f32]) -> Result<Self::Buf>;
    /// Upload 1D data as f16 (gamma).
    fn upload_f16_1d(&self, data_f32: &[f32]) -> Result<Self::Buf>;
    /// Upload 1D bias data. Default: f16 (Metal). D3D12 overrides to f32.
    fn upload_bias_1d(&self, data_f32: &[f32]) -> Result<Self::Buf> {
        self.upload_f16_1d(data_f32)
    }

    // ── Input/output ──
    /// Upload f16 input data into a residual buffer.
    fn upload_input_f16(&self, dst: &Self::Buf, data: &[half::f16]) -> Result<()>;
    /// Upload f32 input data into a residual buffer (D3D12 f32 residual path).
    fn upload_input_f32(&self, dst: &Self::Buf, data: &[f32]) -> Result<()>;
    /// Download f16 activation data from GPU.
    fn download_f16(&self, buf: &Self::Buf, count: usize) -> Result<Vec<half::f16>>;

    // ── Command batching ──
    /// Begin a GPU command pass (Metal: create encoder, D3D12: begin_batch).
    fn begin_pass(&self) -> Result<()> { Ok(()) }
    /// End GPU command pass and wait for completion.
    fn end_pass(&self) -> Result<()> { Ok(()) }

    // ── Forward operations ──
    /// Bare layernorm: out_f16 = LN(hidden). No gamma scaling.
    /// Used when gamma is baked into downstream weights.
    /// `hidden` is residual-dtype (f16 on Metal, f32 on D3D12).
    fn layernorm_bare(&self, hidden: &Self::Buf, out: &Self::Buf,
                       n_rows: usize, n_cols: usize);
    /// LayerNorm with unit-offset gamma: out_f16 = LN(hidden) * (1 + gamma).
    fn layernorm_unit_offset(&self, hidden: &Self::Buf, gamma: &Self::Buf,
                              out: &Self::Buf, n_rows: usize, n_cols: usize);
    /// Matmul: out_f16 = a_f16 @ b_weight, where b_weight dtype is backend-specific.
    fn matmul(&self, a: &Self::Buf, b: &Self::Buf, out: &Self::Buf,
              m: usize, n: usize, k: usize);
    /// Matmul + bias: out_f16 = a_f16 @ b_weight + bias_f16.
    fn matmul_bias(&self, a: &Self::Buf, b: &Self::Buf, bias: &Self::Buf,
                    out: &Self::Buf, m: usize, n: usize, k: usize);
    /// Matmul + bias + GELU: out = GELU(a @ b + bias). Default: matmul_bias + gelu.
    fn matmul_bias_gelu(&self, a: &Self::Buf, b: &Self::Buf, bias: &Self::Buf,
                         out: &Self::Buf, m: usize, n: usize, k: usize) {
        self.matmul_bias(a, b, bias, out, m, n, k);
        self.gelu(out, out, m * n);
    }
    /// In-place GELU: x[i] = gelu(x[i]).
    fn gelu(&self, x: &Self::Buf, out: &Self::Buf, n_elem: usize);
    /// Bias add: out[i] = x[i] + bias[i % n_cols].
    fn bias_add(&self, x: &Self::Buf, bias: &Self::Buf, out: &Self::Buf,
                n_elem: usize, n_cols: usize);
    /// Residual add: res_out = proj_f16 + res_in (residual dtype).
    fn residual_add(&self, proj: &Self::Buf, res_in: &Self::Buf,
                     res_out: &Self::Buf, n_elem: usize);
    /// Flash Attention 2 with sliding window.
    fn flash_attention(&self, q: &Self::Buf, k: &Self::Buf, v: &Self::Buf,
                        out: &Self::Buf, p: &FlashAttentionParams);

    /// Whether this backend supports fused QKV via buf_slice.
    fn supports_buf_slice(&self) -> bool { false }
    /// Create an offset view into a buffer (for fused QKV slicing).
    /// `byte_offset` is the offset from the start of `buf`.
    /// Only called if `supports_buf_slice()` returns true.
    fn buf_slice(&self, _buf: &Self::Buf, _byte_offset: usize) -> Self::Buf {
        unimplemented!("buf_slice not supported on this backend")
    }

    /// Sync GPU for profiling (Metal: wait_until_completed, D3D12: no-op).
    fn sync(&self) -> Result<()> { Ok(()) }
}

// ── Parameter structs ──

pub struct FlashAttentionParams {
    pub n_heads: usize,
    pub padded_seq: usize,  // grid dimension
    pub seq_len: usize,     // actual seq len (for masking, D3D12 only)
    pub head_dim: usize,
    pub stride_h: i32,
    pub stride_m: i32,
    pub stride_o: i32,
    pub sm_scale: f32,
    pub window_left: i32,
    pub window_right: i32,
}

// ── Weight structs ──

pub struct EncoderLinearBiasW<B> {
    pub weight: B,
    pub bias: B,
}

pub struct EncoderAttentionW<B> {
    pub q_proj: B,  // baked: (1+gamma) * W
    pub k_proj: B,
    pub v_proj: B,
    pub qkv_proj: Option<B>,  // fused [dim, 3*kv_dim] weight (if backend supports buf_slice)
    pub o_proj: B,
    pub num_heads: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    pub scale: f32,
}

pub struct EncoderMlpW<B> {
    pub fc1: EncoderLinearBiasW<B>,  // fc1 weight is baked with post-attn gamma
    pub fc2: EncoderLinearBiasW<B>,
}

pub struct EncoderLayerW<B> {
    pub self_attn: EncoderAttentionW<B>,
    pub mlp: EncoderMlpW<B>,
}

// ── Scratch buffers ──

struct EncoderScratch<B> {
    normed: B,      // [padded_seq, encoder_dim] f16
    q: B,           // [padded_seq, kv_dim] f16
    k: B,           // [padded_seq, kv_dim] f16
    v: B,           // [padded_seq, kv_dim] f16
    qkv: Option<B>, // [padded_seq, 3*kv_dim] f16 — fused QKV output
    attn_out: B,    // [padded_seq, kv_dim] f16
    attn_proj: B,   // [padded_seq, encoder_dim] f16
    fc1: B,         // [padded_seq, intermediate_size] f16
    fc2: B,         // [padded_seq, encoder_dim] f16
    residual_a: B,  // [padded_seq, encoder_dim] residual-dtype
    residual_b: B,  // [padded_seq, encoder_dim] residual-dtype
}

// ── CPU-side weight dequantization helpers ──

/// Dequantize 2D weight: GGUF → f32, transpose, flatten row-major.
fn dequant_2d(shape: (usize, usize), vb: &QVarBuilder) -> Result<Vec<f32>> {
    let qt = vb.get(shape, "weight")?;
    let t = qt.dequantize(&Device::Cpu)?;
    let t = t.t()?.contiguous()?.flatten_all()?;
    Ok(t.to_vec1::<f32>()?)
}

/// Dequantize 1D weight/bias: GGUF → f32.
fn dequant_1d(dim: usize, name: &str, vb: &QVarBuilder) -> Result<Vec<f32>> {
    let qt = vb.get(dim, name)?;
    let t = qt.dequantize(&Device::Cpu)?;
    Ok(t.to_vec1::<f32>()?)
}

/// Bake (1 + gamma) into a 2D weight matrix on CPU.
/// weight is [in_dim, out_dim] row-major (already transposed).
/// gamma is [in_dim]. Result: weight[i,j] *= (1 + gamma[i]).
fn bake_gamma(weight: &mut [f32], gamma: &[f32], in_dim: usize, out_dim: usize) {
    for i in 0..in_dim {
        let scale = 1.0 + gamma[i];
        let row = &mut weight[i * out_dim..(i + 1) * out_dim];
        for v in row.iter_mut() {
            *v *= scale;
        }
    }
}

// ── Encoder data ──

struct EncoderData<B: EncoderBackend> {
    layers: Vec<EncoderLayerW<B::Buf>>,
    scratch: EncoderScratch<B::Buf>,
    final_norm_gamma: B::Buf,
    sliding_windows: Vec<[usize; 2]>,
    encoder_dim: usize,
    intermediate_size: usize,
    padded_seq: usize,         // pre-allocated size
}

// ── Main encoder struct ──

pub struct GpuEncoder<B: EncoderBackend> {
    pub(crate) backend: B,
    d: EncoderData<B>,
}

impl<B: EncoderBackend> GpuEncoder<B> {
    pub fn new(
        backend: B,
        cfg: &MoonshineConfig,
        vb: QVarBuilder,
        max_seq_len: usize,
    ) -> Result<Self> {
        let dim = cfg.encoder_dim;
        let kv_dim = cfg.encoder_num_kv_heads * cfg.encoder_head_dim;
        let intermediate = cfg.encoder_intermediate_size;
        let block_m = 128; // max tile size for padding
        let padded_seq = cdiv(max_seq_len, block_m) * block_m;

        let fuse_qkv = backend.supports_buf_slice();

        // Pre-allocate scratch buffers
        let scratch = EncoderScratch {
            normed: backend.alloc_activation(padded_seq * dim)?,
            q: backend.alloc_activation(padded_seq * kv_dim)?,
            k: backend.alloc_activation(padded_seq * kv_dim)?,
            v: backend.alloc_activation(padded_seq * kv_dim)?,
            qkv: if fuse_qkv {
                Some(backend.alloc_activation(padded_seq * 3 * kv_dim)?)
            } else { None },
            attn_out: backend.alloc_activation(padded_seq * kv_dim)?,
            attn_proj: backend.alloc_activation(padded_seq * dim)?,
            fc1: backend.alloc_activation(padded_seq * intermediate)?,
            fc2: backend.alloc_activation(padded_seq * dim)?,
            residual_a: backend.alloc_residual(padded_seq * dim)?,
            residual_b: backend.alloc_residual(padded_seq * dim)?,
        };

        // Load layers with gamma baking
        let mut layers = Vec::with_capacity(cfg.encoder_num_layers);
        for i in 0..cfg.encoder_num_layers {
            let lvb = vb.pp(&format!("layers.{i}"));
            let avb = lvb.pp("self_attn");

            // Load layernorm gammas for baking
            let input_ln_gamma = dequant_1d(dim, "gamma", &lvb.pp("input_layernorm"))?;
            let post_ln_gamma = dequant_1d(dim, "gamma", &lvb.pp("post_attention_layernorm"))?;

            // Bake (1 + input_gamma) into Q/K/V weights
            let mut w_q = dequant_2d((kv_dim, dim), &avb.pp("q_proj"))?;
            let mut w_k = dequant_2d((kv_dim, dim), &avb.pp("k_proj"))?;
            let mut w_v = dequant_2d((kv_dim, dim), &avb.pp("v_proj"))?;
            bake_gamma(&mut w_q, &input_ln_gamma, dim, kv_dim);
            bake_gamma(&mut w_k, &input_ln_gamma, dim, kv_dim);
            bake_gamma(&mut w_v, &input_ln_gamma, dim, kv_dim);

            // Fused QKV weight: [dim, 3*kv_dim] row-major (q|k|v interleaved per row)
            let qkv_proj = if fuse_qkv {
                let mut w_qkv = Vec::with_capacity(dim * 3 * kv_dim);
                for row in 0..dim {
                    w_qkv.extend_from_slice(&w_q[row * kv_dim..(row + 1) * kv_dim]);
                    w_qkv.extend_from_slice(&w_k[row * kv_dim..(row + 1) * kv_dim]);
                    w_qkv.extend_from_slice(&w_v[row * kv_dim..(row + 1) * kv_dim]);
                }
                Some(backend.upload_matmul_weight(&w_qkv)?)
            } else { None };

            let self_attn = EncoderAttentionW {
                q_proj: backend.upload_matmul_weight(&w_q)?,
                k_proj: backend.upload_matmul_weight(&w_k)?,
                v_proj: backend.upload_matmul_weight(&w_v)?,
                qkv_proj,
                o_proj: backend.upload_matmul_weight(&dequant_2d((dim, kv_dim), &avb.pp("o_proj"))?)?,
                num_heads: cfg.encoder_num_heads,
                head_dim: cfg.encoder_head_dim,
                kv_dim,
                scale: (cfg.encoder_head_dim as f32).powf(-0.5),
            };

            // Bake (1 + post_gamma) into fc1 weight
            let mut w_fc1 = dequant_2d((intermediate, dim), &lvb.pp("mlp").pp("fc1"))?;
            bake_gamma(&mut w_fc1, &post_ln_gamma, dim, intermediate);

            let mlp = EncoderMlpW {
                fc1: EncoderLinearBiasW {
                    weight: backend.upload_matmul_weight(&w_fc1)?,
                    bias: backend.upload_bias_1d(&dequant_1d(intermediate, "bias", &lvb.pp("mlp").pp("fc1"))?)?,
                },
                fc2: EncoderLinearBiasW {
                    weight: backend.upload_matmul_weight(&dequant_2d((dim, intermediate), &lvb.pp("mlp").pp("fc2"))?)?,
                    bias: backend.upload_bias_1d(&dequant_1d(dim, "bias", &lvb.pp("mlp").pp("fc2"))?)?,
                },
            };

            layers.push(EncoderLayerW { self_attn, mlp });
        }

        let final_norm_gamma = backend.upload_f16_1d(&dequant_1d(dim, "gamma", &vb.pp("final_norm"))?)?;

        Ok(Self {
            backend,
            d: EncoderData {
                layers,
                scratch,
                final_norm_gamma,
                sliding_windows: cfg.sliding_windows.clone(),
                encoder_dim: dim,
                intermediate_size: intermediate,
                padded_seq,
            },
        })
    }

    /// Forward pass. Input: [1, seq_len, dim] F32 on CPU.
    /// Output: [1, seq_len, dim] F32 on CPU.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, dim) = x.dims3()?;
        assert_eq!(batch, 1, "GpuEncoder only supports batch=1");
        assert_eq!(dim, self.d.encoder_dim);

        let block_m = 128; // padding granularity
        let padded_seq = cdiv(seq_len, block_m) * block_m;
        assert!(padded_seq <= self.d.padded_seq,
            "seq_len {} (padded {}) exceeds pre-allocated {}", seq_len, padded_seq, self.d.padded_seq);

        let s = &self.d.scratch;
        let b = &self.backend;

        // Upload input to residual_a
        self.upload_input(x, seq_len, padded_seq, dim, &s.residual_a)?;

        // Profiling mode: per-op timing (before begin_pass — profile manages its own passes)
        if std::env::var("TRITON_ENCODER_PROFILE").is_ok() {
            return self.forward_profile(x, seq_len, padded_seq);
        }

        // Batch all GPU dispatches into one command pass
        b.begin_pass()?;

        let (mut res_in, mut res_out) = (&s.residual_a, &s.residual_b);

        for (i, layer) in self.d.layers.iter().enumerate() {
            let kv_dim = layer.self_attn.kv_dim;
            let n_elem = padded_seq * dim;

            // Pre-norm (bare LN — gamma baked into QKV weights)
            b.layernorm_bare(res_in, &s.normed, padded_seq, dim);

            // Q/K/V projections
            let [win_left, win_right] = self.d.sliding_windows[i];
            if let (Some(qkv_w), Some(qkv_buf)) = (&layer.self_attn.qkv_proj, &s.qkv) {
                // Fused path: single matmul → [T, 3*kv_dim], slice into Q/K/V
                b.matmul(&s.normed, qkv_w, qkv_buf, padded_seq, 3 * kv_dim, dim);
                let q = b.buf_slice(qkv_buf, 0);
                let k = b.buf_slice(qkv_buf, kv_dim * 2); // f16 = 2 bytes
                let v = b.buf_slice(qkv_buf, 2 * kv_dim * 2);
                b.flash_attention(&q, &k, &v, &s.attn_out, &FlashAttentionParams {
                    n_heads: layer.self_attn.num_heads,
                    padded_seq,
                    seq_len,
                    head_dim: layer.self_attn.head_dim,
                    stride_h: layer.self_attn.head_dim as i32,
                    stride_m: (3 * kv_dim) as i32,
                    stride_o: kv_dim as i32,
                    sm_scale: layer.self_attn.scale,
                    window_left: win_left as i32,
                    window_right: win_right as i32,
                });
            } else {
                // Separate path: 3 matmuls
                b.matmul(&s.normed, &layer.self_attn.q_proj, &s.q, padded_seq, kv_dim, dim);
                b.matmul(&s.normed, &layer.self_attn.k_proj, &s.k, padded_seq, kv_dim, dim);
                b.matmul(&s.normed, &layer.self_attn.v_proj, &s.v, padded_seq, kv_dim, dim);
                b.flash_attention(&s.q, &s.k, &s.v, &s.attn_out, &FlashAttentionParams {
                    n_heads: layer.self_attn.num_heads,
                    padded_seq,
                    seq_len,
                    head_dim: layer.self_attn.head_dim,
                    stride_h: layer.self_attn.head_dim as i32,
                    stride_m: kv_dim as i32,
                    stride_o: kv_dim as i32,
                    sm_scale: layer.self_attn.scale,
                    window_left: win_left as i32,
                    window_right: win_right as i32,
                });
            }

            // O projection
            b.matmul(&s.attn_out, &layer.self_attn.o_proj, &s.attn_proj, padded_seq, dim, kv_dim);

            // Residual add
            b.residual_add(&s.attn_proj, res_in, res_out, n_elem);
            std::mem::swap(&mut res_in, &mut res_out);

            // Post-norm (bare LN — gamma baked into fc1 weights)
            b.layernorm_bare(res_in, &s.normed, padded_seq, dim);

            // FFN: fused matmul_bias_gelu + matmul_bias
            let intermediate = self.d.intermediate_size;
            b.matmul_bias_gelu(&s.normed, &layer.mlp.fc1.weight, &layer.mlp.fc1.bias,
                                &s.fc1, padded_seq, intermediate, dim);
            b.matmul_bias(&s.fc1, &layer.mlp.fc2.weight, &layer.mlp.fc2.bias,
                           &s.fc2, padded_seq, dim, intermediate);

            // Residual add
            b.residual_add(&s.fc2, res_in, res_out, n_elem);
            std::mem::swap(&mut res_in, &mut res_out);
        }

        // Final layernorm (with gamma — not baked)
        b.layernorm_unit_offset(res_in, &self.d.final_norm_gamma, &s.normed, padded_seq, dim);

        // End pass (commits command buffer, waits for GPU)
        b.end_pass()?;

        // Download and trim padding
        let f16_data = b.download_f16(&s.normed, padded_seq * dim)?;
        let f32_data: Vec<f32> = f16_data[..seq_len * dim]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        Ok(Tensor::from_vec(f32_data, (1, seq_len, dim), &Device::Cpu)?)
    }

    /// Upload input tensor to residual buffer (handles padding).
    fn upload_input(&self, x: &Tensor, seq_len: usize, padded_seq: usize,
                     dim: usize, dst: &B::Buf) -> Result<()> {
        // Flatten [1, seq_len, dim] → [seq_len * dim]
        let x_flat = x.reshape((seq_len, dim))?;
        let x_f32 = x_flat.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let data = x_f32.to_vec2::<f32>()?;

        // Convert to f16 for Metal (f16 residual) or f32 for D3D12 (f32 residual)
        // Try f16 first; if it fails (D3D12), use f32
        let mut f16_data: Vec<half::f16> = data.iter()
            .flat_map(|row| row.iter().map(|v| half::f16::from_f32(*v)))
            .collect();
        // Pad
        if padded_seq > seq_len {
            f16_data.resize(padded_seq * dim, half::f16::ZERO);
        }
        if self.backend.upload_input_f16(dst, &f16_data).is_ok() {
            return Ok(());
        }

        // F32 path (D3D12)
        let mut f32_data: Vec<f32> = data.into_iter().flatten().collect();
        if padded_seq > seq_len {
            f32_data.resize(padded_seq * dim, 0.0);
        }
        self.backend.upload_input_f32(dst, &f32_data)
    }

    /// Profiling forward pass — measures each operation individually.
    /// Each timed section gets its own begin_pass/end_pass for accurate timing.
    fn forward_profile(&self, x: &Tensor, seq_len: usize, padded_seq: usize) -> Result<Tensor> {
        use std::time::Instant;

        let dim = self.d.encoder_dim;
        let s = &self.d.scratch;
        let b = &self.backend;

        let (mut res_in, mut res_out) = (&s.residual_a, &s.residual_b);
        let mut totals = std::collections::HashMap::<&str, f64>::new();

        macro_rules! timed {
            ($name:expr, $body:expr) => {{
                b.begin_pass()?;
                let t = Instant::now();
                $body;
                b.end_pass()?;
                *totals.entry($name).or_default() += t.elapsed().as_secs_f64();
            }};
        }

        for (i, layer) in self.d.layers.iter().enumerate() {
            let kv_dim = layer.self_attn.kv_dim;
            let n_elem = padded_seq * dim;
            let [win_left, win_right] = self.d.sliding_windows[i];

            timed!("layernorm_bare", b.layernorm_bare(res_in, &s.normed, padded_seq, dim));

            if let (Some(qkv_w), Some(qkv_buf)) = (&layer.self_attn.qkv_proj, &s.qkv) {
                timed!("qkv_matmul", b.matmul(&s.normed, qkv_w, qkv_buf, padded_seq, 3 * kv_dim, dim));
                let q = b.buf_slice(qkv_buf, 0);
                let k = b.buf_slice(qkv_buf, kv_dim * 2);
                let v = b.buf_slice(qkv_buf, 2 * kv_dim * 2);
                timed!("flash_attn", b.flash_attention(&q, &k, &v, &s.attn_out, &FlashAttentionParams {
                    n_heads: layer.self_attn.num_heads,
                    padded_seq, seq_len,
                    head_dim: layer.self_attn.head_dim,
                    stride_h: layer.self_attn.head_dim as i32,
                    stride_m: (3 * kv_dim) as i32,
                    stride_o: kv_dim as i32,
                    sm_scale: layer.self_attn.scale,
                    window_left: win_left as i32, window_right: win_right as i32,
                }));
            } else {
                timed!("qkv_matmul", {
                    b.matmul(&s.normed, &layer.self_attn.q_proj, &s.q, padded_seq, kv_dim, dim);
                    b.matmul(&s.normed, &layer.self_attn.k_proj, &s.k, padded_seq, kv_dim, dim);
                    b.matmul(&s.normed, &layer.self_attn.v_proj, &s.v, padded_seq, kv_dim, dim);
                });
                timed!("flash_attn", b.flash_attention(&s.q, &s.k, &s.v, &s.attn_out, &FlashAttentionParams {
                    n_heads: layer.self_attn.num_heads,
                    padded_seq, seq_len,
                    head_dim: layer.self_attn.head_dim,
                    stride_h: layer.self_attn.head_dim as i32,
                    stride_m: kv_dim as i32,
                    stride_o: kv_dim as i32,
                    sm_scale: layer.self_attn.scale,
                    window_left: win_left as i32, window_right: win_right as i32,
                }));
            }

            timed!("o_matmul", b.matmul(&s.attn_out, &layer.self_attn.o_proj, &s.attn_proj, padded_seq, dim, kv_dim));
            timed!("residual", b.residual_add(&s.attn_proj, res_in, res_out, n_elem));
            std::mem::swap(&mut res_in, &mut res_out);

            timed!("layernorm_bare", b.layernorm_bare(res_in, &s.normed, padded_seq, dim));

            let intermediate = self.d.intermediate_size;
            timed!("fc1_matmul_bias_gelu",
                b.matmul_bias_gelu(&s.normed, &layer.mlp.fc1.weight, &layer.mlp.fc1.bias,
                                    &s.fc1, padded_seq, intermediate, dim));
            timed!("fc2_matmul_bias",
                b.matmul_bias(&s.fc1, &layer.mlp.fc2.weight, &layer.mlp.fc2.bias,
                               &s.fc2, padded_seq, dim, intermediate));
            timed!("residual", b.residual_add(&s.fc2, res_in, res_out, n_elem));
            std::mem::swap(&mut res_in, &mut res_out);
        }

        timed!("final_ln",
            b.layernorm_unit_offset(res_in, &self.d.final_norm_gamma, &s.normed, padded_seq, dim));

        // Print profile
        let total: f64 = totals.values().sum();
        eprintln!("\n  -- Encoder Profile (14 layers, T={padded_seq}) --");
        let mut items: Vec<_> = totals.iter().collect();
        items.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (name, ms) in &items {
            eprintln!("    {:<20} {:7.1}ms  ({:4.1}%)", name, *ms * 1000.0, *ms / total * 100.0);
        }
        eprintln!("    {:<20} {:7.1}ms", "TOTAL", total * 1000.0);

        // Download
        let f16_data = b.download_f16(&s.normed, padded_seq * dim)?;
        let f32_data: Vec<f32> = f16_data[..seq_len * dim]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        Ok(Tensor::from_vec(f32_data, (1, seq_len, dim), &Device::Cpu)?)
    }
}
