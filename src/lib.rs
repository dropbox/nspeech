use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Module, ModuleT, Tensor, D};
use candle_nn::{
    batch_norm, conv1d, conv1d_no_bias, layer_norm, linear, BatchNorm, BatchNormConfig, Conv1d,
    Conv1dConfig, Dropout, LayerNorm, LayerNormConfig, Linear, VarBuilder,
};
use hf_hub::api::sync::Api;
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct FastConformerConfig {
    pub feat_in: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub ff_mult: usize,
    pub num_layers: usize,
    pub conv_kernel_size: usize,
    pub dropout: f64,
    pub subsampling_channels: usize,
    pub vocab_size: usize,
    pub blank_id: usize,
}

impl Default for FastConformerConfig {
    fn default() -> Self {
        Self {
            feat_in: 80,
            d_model: 512,
            num_heads: 8,
            ff_mult: 4,
            num_layers: 16,
            conv_kernel_size: 31,
            dropout: 0.1,
            subsampling_channels: 256,
            vocab_size: 1024,
            blank_id: 0,
        }
    }
}

// Simple sinusoidal positional encoding.
fn sinusoidal_positional_encoding(length: usize, dim: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; length * dim];
    for pos in 0..length {
        for i in 0..(dim / 2) {
            let idx = 2 * i;
            let div_term = (pos as f32) / (10000_f32.powf(2.0 * i as f32 / dim as f32));
            data[pos * dim + idx] = div_term.sin();
            if idx + 1 < dim {
                data[pos * dim + idx + 1] = div_term.cos();
            }
        }
    }

    Ok(Tensor::from_slice(&data, (1, length, dim), device)?)
}

/// 8x time-reduction conv front-end:
/// input:  [B, T, F]  (features)
/// output: [B, T/8, D_model]
pub struct ConvSubsampling {
    conv1: Conv1d,
    conv2: Conv1d,
    conv3: Conv1d,
    proj: Linear,
}

impl ConvSubsampling {
    pub fn new(cfg: &FastConformerConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let mut c1 = Conv1dConfig::default();
        c1.stride = 2;
        c1.padding = 1;
        let conv1 = conv1d(
            cfg.feat_in,
            cfg.subsampling_channels,
            3,
            c1,
            vb.pp("conv1"),
        )?;

        let mut c2 = Conv1dConfig::default();
        c2.stride = 2;
        c2.padding = 1;
        let conv2 = conv1d(
            cfg.subsampling_channels,
            cfg.subsampling_channels,
            3,
            c2,
            vb.pp("conv2"),
        )?;

        let mut c3 = Conv1dConfig::default();
        c3.stride = 2;
        c3.padding = 1;
        let conv3 = conv1d(
            cfg.subsampling_channels,
            cfg.subsampling_channels,
            3,
            c3,
            vb.pp("conv3"),
        )?;

        let proj = linear(cfg.subsampling_channels, cfg.d_model, vb.pp("proj"))?;

        Ok(Self {
            conv1,
            conv2,
            conv3,
            proj,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: [B, T, F] -> [B, F, T]
        let xs = xs.transpose(1, 2)?;
        let xs = self.conv1.forward(&xs)?.relu()?;
        let xs = self.conv2.forward(&xs)?.relu()?;
        let xs = self.conv3.forward(&xs)?.relu()?;
        let xs = xs.transpose(1, 2)?; // [B, T/8, C]
        let xs = self.proj.forward(&xs)?; // [B, T/8, D]
        Ok(xs)
    }
}

pub struct FeedForward {
    w1: Linear,
    w2: Linear,
    dropout: Dropout,
}

impl FeedForward {
    pub fn new(d_model: usize, ff_mult: usize, drop: f64, vb: VarBuilder<'_>) -> Result<Self> {
        let hidden = d_model * ff_mult;
        let w1 = linear(d_model, hidden, vb.pp("w1"))?;
        let w2 = linear(hidden, d_model, vb.pp("w2"))?;
        let dropout = Dropout::new(drop as f32);
        Ok(Self { w1, w2, dropout })
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let xs = self.w1.forward(xs)?.silu()?; // swish-ish non-linearity
        let xs = self.dropout.forward(&xs, train)?;
        let xs = self.w2.forward(&xs)?;
        Ok(xs)
    }
}

pub struct MultiHeadSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    dropout: Dropout,
}

impl MultiHeadSelfAttention {
    pub fn new(d_model: usize, num_heads: usize, drop: f64, vb: VarBuilder<'_>) -> Result<Self> {
        if d_model % num_heads != 0 {
            return Err(anyhow!(
                "d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            ));
        }
        let head_dim = d_model / num_heads;
        let q_proj = linear(d_model, d_model, vb.pp("q"))?;
        let k_proj = linear(d_model, d_model, vb.pp("k"))?;
        let v_proj = linear(d_model, d_model, vb.pp("v"))?;
        let o_proj = linear(d_model, d_model, vb.pp("o"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
            dropout: Dropout::new(drop as f32),
        })
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        // xs: [B, T, D]
        let (b, t, d) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // [B, T, D] -> [B, H, T, Dh]
        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let scale = (self.head_dim as f64).sqrt();
        let attn_scores =
            (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? / scale)?;
        let attn_weights = candle_nn::ops::softmax(&attn_scores, D::Minus1)?;

        let attn_weights = self.dropout.forward(&attn_weights, train)?;
        let context = attn_weights.matmul(&v)?; // [B, H, T, Dh]

        // -> [B, T, D]
        let context = context.transpose(1, 2)?.reshape((b, t, d))?;
        let out = self.o_proj.forward(&context)?;
        Ok(out)
    }
}

pub struct ConvModule {
    pw1: Conv1d,  // pointwise: D -> 2D
    dw: Conv1d,   // depthwise: 2D -> 2D
    pw2: Conv1d,  // pointwise: 2D -> D
    bn: BatchNorm,
    dropout: Dropout,
    d_model: usize,
}

impl ConvModule {
    pub fn new(d_model: usize, kernel_size: usize, drop: f64, vb: VarBuilder<'_>) -> Result<Self> {
        let mut cfg_pw = Conv1dConfig::default();
        cfg_pw.stride = 1;
        cfg_pw.padding = 0;
        let pw1 = conv1d(d_model, 2 * d_model, 1, cfg_pw, vb.pp("pw1"))?;

        let mut cfg_dw = Conv1dConfig::default();
        cfg_dw.stride = 1;
        cfg_dw.padding = kernel_size / 2;
        cfg_dw.groups = d_model;
        let dw = conv1d_no_bias(d_model, d_model, kernel_size, cfg_dw, vb.pp("dw"))?;

        let bn_cfg = BatchNormConfig {
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            remove_mean: false,
        };
        let bn = batch_norm(d_model, bn_cfg, vb.pp("bn"))?;

        let pw2 = conv1d(d_model, d_model, 1, cfg_pw, vb.pp("pw2"))?;

        Ok(Self {
            pw1,
            dw,
            pw2,
            bn,
            dropout: Dropout::new(drop as f32),
            d_model,
        })
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        // xs: [B, T, D] -> [B, D, T]
        let (_b, _t, d) = xs.dims3()?;
        if d != self.d_model {
            return Err(anyhow!(
                "conv module expected d_model {}, got {}",
                self.d_model,
                d
            ));
        }

        let xs = xs.transpose(1, 2)?; // [B, D, T]

        // GLU pointwise: D -> 2D
        let xs = self.pw1.forward(&xs)?;
        let a = xs.narrow(1, 0, d)?;
        let b = xs.narrow(1, d, d)?;
        let gated = candle_nn::ops::sigmoid(&b)?;
        let xs = (a * gated)?;

        // depthwise conv
        let xs = self.dw.forward(&xs)?;
        let xs = self.bn.forward_t(&xs, train)?;
        let xs = xs.silu()?;

        // pointwise projection back to D
        let xs = self.pw2.forward(&xs)?;
        let xs = self.dropout.forward(&xs, train)?;
        let xs = xs.transpose(1, 2)?; // [B, T, D]
        Ok(xs)
    }
}

pub struct FastConformerBlock {
    ff1: FeedForward,
    ff2: FeedForward,
    self_attn: MultiHeadSelfAttention,
    conv_module: ConvModule,
    ln_ff1: LayerNorm,
    ln_mha: LayerNorm,
    ln_conv: LayerNorm,
    ln_ff2: LayerNorm,
    ln_out: LayerNorm,
}

impl FastConformerBlock {
    pub fn new(cfg: &FastConformerConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let d_model = cfg.d_model;
        let ln_cfg = LayerNormConfig {
            eps: 1e-5,
            affine: true,
            remove_mean: true,
        };

        Ok(Self {
            ff1: FeedForward::new(d_model, cfg.ff_mult, cfg.dropout, vb.pp("ff1"))?,
            ff2: FeedForward::new(d_model, cfg.ff_mult, cfg.dropout, vb.pp("ff2"))?,
            self_attn: MultiHeadSelfAttention::new(
                d_model,
                cfg.num_heads,
                cfg.dropout,
                vb.pp("mha"),
            )?,
            conv_module: ConvModule::new(d_model, cfg.conv_kernel_size, cfg.dropout, vb.pp("conv"))?,
            ln_ff1: layer_norm(d_model, ln_cfg, vb.pp("ln_ff1"))?,
            ln_mha: layer_norm(d_model, ln_cfg, vb.pp("ln_mha"))?,
            ln_conv: layer_norm(d_model, ln_cfg, vb.pp("ln_conv"))?,
            ln_ff2: layer_norm(d_model, ln_cfg, vb.pp("ln_ff2"))?,
            ln_out: layer_norm(d_model, ln_cfg, vb.pp("ln_out"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        // 1) Macaron FFN (scaled by 0.5 in literature; kept simple here).
        let y_ff1 = self.ff1.forward(&self.ln_ff1.forward(xs)?, train)?;
        let mut y = (xs + &y_ff1)?;

        // 2) Self-attention
        let y_attn = self.self_attn.forward(&self.ln_mha.forward(&y)?, train)?;
        y = (&y + &y_attn)?;

        // 3) Conv module
        let y_conv = self.conv_module.forward(&self.ln_conv.forward(&y)?, train)?;
        y = (&y + &y_conv)?;

        // 4) Second FFN
        let y_ff2 = self.ff2.forward(&self.ln_ff2.forward(&y)?, train)?;
        y = (&y + &y_ff2)?;

        // 5) Final layer norm
        let y_out = self.ln_out.forward(&y)?;
        Ok(y_out)
    }
}

pub struct FastConformerEncoder {
    subsampling: ConvSubsampling,
    blocks: Vec<FastConformerBlock>,
    pos_dropout: Dropout,
    cfg: FastConformerConfig,
}

impl FastConformerEncoder {
    pub fn new(cfg: FastConformerConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let subsampling = ConvSubsampling::new(&cfg, vb.pp("subsampling"))?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(FastConformerBlock::new(&cfg, vb.pp(format!("layers.{i}")))?);
        }
        Ok(Self {
            subsampling,
            blocks,
            pos_dropout: Dropout::new(cfg.dropout as f32),
            cfg,
        })
    }

    /// xs: [B, T, F]
    /// returns: [B, T', D_model]
    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let device = xs.device();
        let xs = self.subsampling.forward(xs)?; // [B, T', D]
        let (b, t, d) = xs.dims3()?;
        if d != self.cfg.d_model {
            return Err(anyhow!(
                "encoder expected d_model {}, got {}",
                self.cfg.d_model,
                d
            ));
        }

        // Add sinusoidal positional encoding
        let pe = sinusoidal_positional_encoding(t, d, device)?;
        let pe = pe.broadcast_as((b, t, d))?;
        let mut h = (xs + pe)?;
        h = self.pos_dropout.forward(&h, train)?;

        for blk in &self.blocks {
            h = blk.forward(&h, train)?;
        }
        Ok(h)
    }
}

pub struct ParakeetFastConformerCtc {
    pub encoder: FastConformerEncoder,
    pub proj: Linear, // D_model -> vocab_size
    pub cfg: FastConformerConfig,
    tokenizer: Option<Tokenizer>,
    id2token: Option<Vec<String>>,
}

impl ParakeetFastConformerCtc {
    pub fn new(cfg: FastConformerConfig, vb: VarBuilder<'_>, id2token: Vec<String>) -> Result<Self> {
        if id2token.len() != cfg.vocab_size {
            return Err(anyhow!(
                "id2token length {} must equal vocab_size {}",
                id2token.len(),
                cfg.vocab_size
            ));
        }
        let encoder = FastConformerEncoder::new(cfg.clone(), vb.pp("encoder"))?;
        let proj = linear(cfg.d_model, cfg.vocab_size, vb.pp("ctc_head"))?;
        Ok(Self {
            encoder,
            proj,
            cfg,
            tokenizer: None,
            id2token: Some(id2token),
        })
    }

    pub fn new_with_tokenizer(
        cfg: FastConformerConfig,
        vb: VarBuilder<'_>,
        tokenizer: Tokenizer,
    ) -> Result<Self> {
        let encoder = FastConformerEncoder::new(cfg.clone(), vb.pp("encoder"))?;
        let proj = linear(cfg.d_model, cfg.vocab_size, vb.pp("ctc_head"))?;
        Ok(Self {
            encoder,
            proj,
            cfg,
            tokenizer: Some(tokenizer),
            id2token: None,
        })
    }

    /// Forward pass:
    ///  input:  [B, T, F]
    ///  output: [B, T', V] (logits)
    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let h = self.encoder.forward(xs, train)?; // [B, T', D]
        let logits = self.proj.forward(&h)?; // [B, T', V]
        Ok(logits)
    }

    /// Greedy CTC decoding:
    ///  logits: [B, T', V]
    pub fn greedy_decode(&self, logits: &Tensor) -> Result<Vec<String>> {
        let (b, t, _v) = logits.dims3()?;
        let pred_ids = logits.argmax(D::Minus1)?; // [B, T']
        let pred_ids = pred_ids.to_vec2::<u32>()?;

        let mut transcripts = Vec::with_capacity(b);
        for bidx in 0..b {
            let mut prev = self.cfg.blank_id as u32;
            let mut tokens = Vec::new();
            for tidx in 0..t {
                let cur = pred_ids[bidx][tidx];
                if cur == self.cfg.blank_id as u32 {
                    prev = cur;
                    continue;
                }
                if cur == prev {
                    continue;
                }
                tokens.push(cur);
                prev = cur;
            }

            if let Some(ref tokenizer) = self.tokenizer {
                let text = tokenizer
                    .decode(tokens.as_slice(), true)
                    .map_err(|e| anyhow!("decode error: {e}"))?;
                transcripts.push(text);
            } else if let Some(ref vocab) = self.id2token {
                let mut pieces = Vec::with_capacity(tokens.len());
                for id in tokens {
                    let idx = id as usize;
                    if idx < vocab.len() {
                        pieces.push(vocab[idx].clone());
                    }
                }
                transcripts.push(pieces.join(""));
            } else {
                return Err(anyhow!("no tokenizer or id2token available for decoding"));
            }
        }
        Ok(transcripts)
    }
}

#[derive(Debug, Deserialize)]
pub struct HfEncoderConfig {
    pub activation_dropout: f64,
    pub attention_dropout: f64,
    pub conv_kernel_size: usize,
    pub dropout: f64,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_mel_bins: usize,
    pub subsampling_conv_channels: usize,
    pub subsampling_conv_stride: usize,
    pub subsampling_factor: usize,
}

#[derive(Debug, Deserialize)]
pub struct HfParakeetCtcConfig {
    pub encoder_config: HfEncoderConfig,
    pub vocab_size: usize,
    pub pad_token_id: usize,
}

impl FastConformerConfig {
    pub fn from_hf(hf: &HfParakeetCtcConfig) -> Self {
        let enc = &hf.encoder_config;
        Self {
            feat_in: enc.num_mel_bins,
            d_model: enc.hidden_size,
            num_heads: enc.num_attention_heads,
            ff_mult: enc.intermediate_size / enc.hidden_size,
            num_layers: enc.num_hidden_layers,
            conv_kernel_size: enc.conv_kernel_size,
            dropout: enc.dropout,
            subsampling_channels: enc.subsampling_conv_channels,
            vocab_size: hf.vocab_size,
            blank_id: hf.pad_token_id,
        }
    }
}

/// Download config/tokenizer/weights from the Hugging Face Hub and build a model.
pub fn load_parakeet_ctc_from_hf(repo_id: &str, device: &Device) -> Result<ParakeetFastConformerCtc> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());

    let config_path = repo.get("config.json")?;
    let weights_path = repo.get("model.safetensors")?;
    let tokenizer_path = repo.get("tokenizer.json")?;

    let cfg_json = std::fs::read_to_string(config_path)?;
    let hf_cfg: HfParakeetCtcConfig = serde_json::from_str(&cfg_json)?;
    let cfg = FastConformerConfig::from_hf(&hf_cfg);

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow!("tokenizer load error: {e}"))?;

    // Parakeet configs typically specify bf16 weights.
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::BF16, device)? };

    ParakeetFastConformerCtc::new_with_tokenizer(cfg, vb, tokenizer)
}
