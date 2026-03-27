//! Triton-accelerated FastConformer encoder for Parakeet TDT.
//!
//! Replaces linear projections (FF1/FF2, Q/K/V/O, rel_k) with Triton F16 matmul.
//! Keeps ConvModule, LayerNorm, BatchNorm, and attention mechanics in Candle.

use anyhow::Result;
use candle_core::{DType, Device, Module, ModuleT, Tensor, D};
use candle_nn::{
    batch_norm, conv1d, layer_norm, BatchNorm, BatchNormConfig,
    Conv1d, Conv1dConfig, LayerNorm, LayerNormConfig, VarBuilder,
};

use crate::triton_kernels::{TritonKernels, triton_matmul};
use super::fast_conformer::{ConvSubsampling, FastConformerConfig, relative_positional_encoding};

fn cdiv(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Load a linear weight transposed to [in_dim, out_dim] F16 for Triton matmul.
fn load_f16_weight(vb: &VarBuilder, shape: (usize, usize)) -> Result<Tensor> {
    let w = vb.get(shape, "weight")?;
    Ok(w.to_dtype(DType::F16)?.t()?.contiguous()?)
}

/// Load bias in its original dtype.
fn load_bias(vb: &VarBuilder, dim: usize) -> Result<Tensor> {
    Ok(vb.get(dim, "bias")?)
}

// ── Per-layer weight structures ──────────────────────────────────────────────

struct FeedForwardWeights {
    w1: Tensor,    // [in_dim, hidden] F16 (transposed)
    b1: Tensor,    // [hidden] original dtype
    w2: Tensor,    // [hidden, in_dim] F16 (transposed)
    b2: Tensor,    // [in_dim] original dtype
}

impl FeedForwardWeights {
    fn new(d_model: usize, ff_mult: usize, vb: &VarBuilder) -> Result<Self> {
        let hidden = d_model * ff_mult;
        let w1_vb = vb.pp("linear1");
        let w2_vb = vb.pp("linear2");
        Ok(Self {
            w1: load_f16_weight(&w1_vb, (hidden, d_model))?,
            b1: load_bias(&w1_vb, hidden)?,
            w2: load_f16_weight(&w2_vb, (d_model, hidden))?,
            b2: load_bias(&w2_vb, d_model)?,
        })
    }
}

struct AttentionWeights {
    q_w: Tensor, q_b: Tensor,
    k_w: Tensor, k_b: Tensor,
    v_w: Tensor, v_b: Tensor,
    o_w: Tensor, o_b: Tensor,
    rel_k_w: Tensor,  // [d_model, d_model] F16 (transposed)
    bias_u: Tensor,    // [num_heads, head_dim] original dtype
    bias_v: Tensor,    // [num_heads, head_dim] original dtype
    num_heads: usize,
    head_dim: usize,
}

impl AttentionWeights {
    fn new(d_model: usize, num_heads: usize, vb: &VarBuilder) -> Result<Self> {
        let head_dim = d_model / num_heads;
        let q_vb = vb.pp("q_proj");
        let k_vb = vb.pp("k_proj");
        let v_vb = vb.pp("v_proj");
        let o_vb = vb.pp("o_proj");
        Ok(Self {
            q_w: load_f16_weight(&q_vb, (d_model, d_model))?,
            q_b: load_bias(&q_vb, d_model)?,
            k_w: load_f16_weight(&k_vb, (d_model, d_model))?,
            k_b: load_bias(&k_vb, d_model)?,
            v_w: load_f16_weight(&v_vb, (d_model, d_model))?,
            v_b: load_bias(&v_vb, d_model)?,
            o_w: load_f16_weight(&o_vb, (d_model, d_model))?,
            o_b: load_bias(&o_vb, d_model)?,
            rel_k_w: {
                let w = vb.get((d_model, d_model), "relative_k_proj.weight")?;
                // rel_k is used as: pos @ rel_k_weight^T, so store transposed
                w.to_dtype(DType::F16)?.contiguous()?
            },
            bias_u: vb.get((num_heads, head_dim), "bias_u")?,
            bias_v: vb.get((num_heads, head_dim), "bias_v")?,
            num_heads,
            head_dim,
        })
    }
}

struct ConvModuleWeights {
    pw1: Conv1d,
    dw: Conv1d,
    pw2: Conv1d,
    bn: BatchNorm,
    d_model: usize,
}

impl ConvModuleWeights {
    fn new(d_model: usize, kernel_size: usize, vb: &VarBuilder) -> Result<Self> {
        let mut cfg_pw = Conv1dConfig::default();
        cfg_pw.stride = 1;
        cfg_pw.padding = 0;
        let pw1 = conv1d(d_model, 2 * d_model, 1, cfg_pw, vb.pp("pointwise_conv1"))?;
        let mut cfg_dw = Conv1dConfig::default();
        cfg_dw.stride = 1;
        cfg_dw.padding = kernel_size / 2;
        cfg_dw.groups = d_model;
        let dw = conv1d(d_model, d_model, kernel_size, cfg_dw, vb.pp("depthwise_conv"))?;
        let eps = if vb.dtype() == DType::F16 || vb.dtype() == DType::BF16 { 1e-3 } else { 1e-5 };
        let bn_cfg = BatchNormConfig {
            eps,
            momentum: 0.1,
            affine: true,
            remove_mean: true,
        };
        let bn = batch_norm(d_model, bn_cfg, vb.pp("norm"))?;
        let pw2 = conv1d(d_model, d_model, 1, cfg_pw, vb.pp("pointwise_conv2"))?;
        Ok(Self { pw1, dw, pw2, bn, d_model })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let d = self.d_model;
        let xs = xs.transpose(1, 2)?;
        let xs = self.pw1.forward(&xs)?;
        let a = xs.narrow(1, 0, d)?;
        let b = xs.narrow(1, d, d)?;
        let gated = candle_nn::ops::sigmoid(&b)?;
        let xs = (a * gated)?;
        let xs = self.dw.forward(&xs)?;
        let xs = self.bn.forward_t(&xs, false)?;
        let xs = xs.silu()?;
        let xs = self.pw2.forward(&xs)?;
        Ok(xs.transpose(1, 2)?)
    }
}

struct TritonConformerBlock {
    ff1: FeedForwardWeights,
    ff2: FeedForwardWeights,
    attn: AttentionWeights,
    conv: ConvModuleWeights,
    ln_ff1: LayerNorm,
    ln_mha: LayerNorm,
    ln_conv: LayerNorm,
    ln_ff2: LayerNorm,
    ln_out: LayerNorm,
}

impl TritonConformerBlock {
    fn new(cfg: &FastConformerConfig, vb: &VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let eps = if vb.dtype() == DType::F16 || vb.dtype() == DType::BF16 { 1e-3 } else { 1e-5 };
        let ln_cfg = LayerNormConfig { eps, affine: true, remove_mean: true };
        Ok(Self {
            ff1: FeedForwardWeights::new(d, cfg.ff_mult, &vb.pp("feed_forward1"))?,
            ff2: FeedForwardWeights::new(d, cfg.ff_mult, &vb.pp("feed_forward2"))?,
            attn: AttentionWeights::new(d, cfg.num_heads, &vb.pp("self_attn"))?,
            conv: ConvModuleWeights::new(d, cfg.conv_kernel_size, &vb.pp("conv"))?,
            ln_ff1: layer_norm(d, ln_cfg, vb.pp("norm_feed_forward1"))?,
            ln_mha: layer_norm(d, ln_cfg, vb.pp("norm_self_att"))?,
            ln_conv: layer_norm(d, ln_cfg, vb.pp("norm_conv"))?,
            ln_ff2: layer_norm(d, ln_cfg, vb.pp("norm_feed_forward2"))?,
            ln_out: layer_norm(d, ln_cfg, vb.pp("norm_out"))?,
        })
    }
}

// ── Triton Encoder ───────────────────────────────────────────────────────────

pub struct TritonParakeetEncoder {
    subsampling: ConvSubsampling,
    blocks: Vec<TritonConformerBlock>,
    kernels: TritonKernels,
    metal_device: candle_core::MetalDevice,
    cfg: FastConformerConfig,
    model_dtype: DType,  // BF16 on GPU, F32 on CPU
}

impl TritonParakeetEncoder {
    pub fn new(
        cfg: FastConformerConfig,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let device = vb.device();
        let metal_device = match device {
            Device::Metal(md) => md.clone(),
            _ => anyhow::bail!("TritonParakeetEncoder requires Metal device"),
        };
        let model_dtype = vb.dtype();

        println!("  Loading Triton kernel pipelines...");
        let kernels = TritonKernels::load(&metal_device)?;

        let subsampling = ConvSubsampling::new(&cfg, vb.pp("subsampling"))?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(TritonConformerBlock::new(&cfg, &vb.pp(format!("layers.{i}")))?);
        }

        Ok(Self {
            subsampling,
            blocks,
            kernels,
            metal_device,
            cfg,
            model_dtype,
        })
    }

    /// Triton matmul: A[M,K] @ B[K,N] → C[M,N] in F16.
    fn matmul_f16(&self, a_f16: &Tensor, b_f16: &Tensor, m: usize, n: usize, k: usize) -> Result<Tensor> {
        triton_matmul(&self.metal_device, &self.kernels.matmul_64x64, a_f16, b_f16, m, n, k, 64, 64)
    }

    /// Linear projection via Triton: input[M,K] @ W[K,N] + bias[N].
    /// Converts input to F16 for matmul, output back to model_dtype for bias add.
    fn triton_linear(&self, input: &Tensor, w: &Tensor, bias: &Tensor, m: usize, n: usize, k: usize) -> Result<Tensor> {
        let input_f16 = input.to_dtype(DType::F16)?;
        let out = self.matmul_f16(&input_f16, w, m, n, k)?;
        let out = out.to_dtype(self.model_dtype)?;
        Ok(out.broadcast_add(bias)?)
    }

    /// Linear projection without bias (for rel_k).
    fn triton_linear_no_bias(&self, input: &Tensor, w: &Tensor, m: usize, n: usize, k: usize) -> Result<Tensor> {
        let input_f16 = input.to_dtype(DType::F16)?;
        let out = self.matmul_f16(&input_f16, w, m, n, k)?;
        Ok(out.to_dtype(self.model_dtype)?)
    }

    /// Transformer-XL relative position shift.
    fn rel_shift(x: &Tensor) -> Result<Tensor> {
        let (b, h, t, p) = x.dims4()?;
        let zeros = Tensor::zeros((b, h, t, 1), x.dtype(), x.device())?;
        let x = Tensor::cat(&[zeros, x.clone()], 3)?;
        let x = x.reshape((b, h, p + 1, t))?;
        let x = x.narrow(D::Minus2, 1, p)?;
        Ok(x.reshape((b, h, t, p))?)
    }

    fn forward_block(&self, block: &TritonConformerBlock, xs: &Tensor, pos: &Tensor, padded_t: usize) -> Result<Tensor> {
        let d = self.cfg.d_model;
        let hidden = d * self.cfg.ff_mult;
        let nh = block.attn.num_heads;
        let hd = block.attn.head_dim;

        // ── FF1 (0.5× scaled residual) ──
        let residual = xs.clone();
        let normed = block.ln_ff1.forward(xs)?;
        let normed_2d = normed.reshape((padded_t, d))?;
        let h = self.triton_linear(&normed_2d, &block.ff1.w1, &block.ff1.b1, padded_t, hidden, d)?;
        let h = h.silu()?;
        let h = self.triton_linear(&h, &block.ff1.w2, &block.ff1.b2, padded_t, d, hidden)?;
        let h = h.reshape((1, padded_t, d))?;
        let mut y = (residual + (h * 0.5)?)?;

        // ── Self-attention ──
        let residual = y.clone();
        let normed = block.ln_mha.forward(&y)?;
        let normed_2d = normed.reshape((padded_t, d))?;

        // Q/K/V projections via Triton
        let q = self.triton_linear(&normed_2d, &block.attn.q_w, &block.attn.q_b, padded_t, d, d)?;
        let k = self.triton_linear(&normed_2d, &block.attn.k_w, &block.attn.k_b, padded_t, d, d)?;
        let v = self.triton_linear(&normed_2d, &block.attn.v_w, &block.attn.v_b, padded_t, d, d)?;

        // Relative position K projection via Triton
        let pos_len = pos.dims()[1]; // 2*T-1
        let padded_pos = cdiv(pos_len, 64) * 64;
        let pos_2d = if padded_pos > pos_len {
            let pad = Tensor::zeros((padded_pos - pos_len, d), self.model_dtype, pos.device())?;
            let p = pos.reshape((pos_len, d))?;
            Tensor::cat(&[&p, &pad], 0)?
        } else {
            pos.reshape((pos_len, d))?
        };
        let k_rel = self.triton_linear_no_bias(&pos_2d, &block.attn.rel_k_w, padded_pos, d, d)?;
        let k_rel = if padded_pos > pos_len {
            k_rel.narrow(0, 0, pos_len)?
        } else {
            k_rel
        };

        let scale = 1.0 / (hd as f64).sqrt();

        let context = {
            let q = q.reshape((1, padded_t, nh, hd))?.transpose(1, 2)?.contiguous()?;
            let k = k.reshape((1, padded_t, nh, hd))?.transpose(1, 2)?.contiguous()?;
            let v = v.reshape((1, padded_t, nh, hd))?.transpose(1, 2)?.contiguous()?;
            let k_rel = k_rel.reshape((1, pos_len, nh, hd))?.transpose(1, 2)?.contiguous()?;

            let bu = block.attn.bias_u.unsqueeze(0)?.unsqueeze(2)?;
            let bv = block.attn.bias_v.unsqueeze(0)?.unsqueeze(2)?;
            let q_bias_u = q.broadcast_add(&bu)?;
            let q_bias_v = q.broadcast_add(&bv)?;

            let attn_scores_c = q_bias_u.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
            let mut attn_scores_r = q_bias_v.matmul(&k_rel.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
            attn_scores_r = Self::rel_shift(&attn_scores_r)?;
            let last = attn_scores_r.dims4()?.3;
            let take = last.min(padded_t);
            attn_scores_r = attn_scores_r.narrow(D::Minus1, 0, take)?;

            let mut attn_scores = (attn_scores_c + attn_scores_r)?;
            let scale_f = scale as f32;
            let scale_t = Tensor::from_slice(&[scale_f], (), xs.device())?.to_dtype(self.model_dtype)?;
            let scale_t = scale_t.broadcast_as(attn_scores.shape())?;
            attn_scores = (attn_scores / scale_t)?;

            let needs_upcast = self.model_dtype == DType::F16 || self.model_dtype == DType::BF16;
            let attn_f32 = if needs_upcast { attn_scores.to_dtype(DType::F32)? } else { attn_scores };
            let attn_weights = candle_nn::ops::softmax(&attn_f32, D::Minus1)?;
            let attn_weights = if needs_upcast { attn_weights.to_dtype(self.model_dtype)? } else { attn_weights };

            let context = attn_weights.matmul(&v)?;
            context.transpose(1, 2)?.reshape((padded_t, d))?
        };

        // O projection via Triton
        let attn_out = self.triton_linear(&context, &block.attn.o_w, &block.attn.o_b, padded_t, d, d)?;
        let attn_out = attn_out.reshape((1, padded_t, d))?;
        y = (residual + attn_out)?;

        // ── ConvModule (Candle) ──
        let residual = y.clone();
        let normed = block.ln_conv.forward(&y)?;
        let y_conv = block.conv.forward(&normed)?;
        y = (residual + y_conv)?;

        // ── FF2 (0.5× scaled residual) ──
        let residual = y.clone();
        let normed = block.ln_ff2.forward(&y)?;
        let normed_2d = normed.reshape((padded_t, d))?;
        let h = self.triton_linear(&normed_2d, &block.ff2.w1, &block.ff2.b1, padded_t, hidden, d)?;
        let h = h.silu()?;
        let h = self.triton_linear(&h, &block.ff2.w2, &block.ff2.b2, padded_t, d, hidden)?;
        let h = h.reshape((1, padded_t, d))?;
        y = (residual + (h * 0.5)?)?;

        // ── Output LayerNorm ──
        Ok(block.ln_out.forward(&y)?)
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let device = xs.device();
        let (_, _, input_dim) = xs.dims3()?;

        // Subsampling (Candle, only runs once)
        let xs = if input_dim == self.cfg.d_model {
            xs.clone()
        } else {
            self.subsampling.forward(xs)?
        };

        let (b, t, d) = xs.dims3()?;
        assert_eq!(b, 1, "TritonParakeetEncoder only supports batch=1");
        assert_eq!(d, self.cfg.d_model);

        // Scale input
        let xs = if self.cfg.scale_input {
            let scale = (self.cfg.d_model as f64).sqrt() as f32;
            let scale_t = Tensor::from_slice(&[scale], (), device)?.to_dtype(xs.dtype())?;
            let scale_t = scale_t.broadcast_as(xs.shape())?;
            (xs * scale_t)?
        } else {
            xs
        };

        // Pad T to multiple of 64 for Triton matmul tiles
        let block_m = 64;
        let padded_t = cdiv(t, block_m) * block_m;
        let xs = if padded_t > t {
            let pad = Tensor::zeros((1, padded_t - t, d), self.model_dtype, device)?;
            Tensor::cat(&[&xs, &pad], 1)?
        } else {
            xs
        };

        // Relative positional encoding
        let pos = relative_positional_encoding(b, padded_t, d, device, self.model_dtype)?;

        let mut h = xs;
        for block in &self.blocks {
            h = self.forward_block(block, &h, &pos, padded_t)?;
        }

        // Slice back to original T
        if padded_t > t {
            h = h.narrow(1, 0, t)?;
        }
        Ok(h)
    }
}
