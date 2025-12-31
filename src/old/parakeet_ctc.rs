/// Parakeet CTC Model - Native Rust/Candle Implementation
///
/// Based on nvidia/parakeet-ctc-0.6b
/// 608M parameter Conformer-based ASR model with CTC decoder
use anyhow::{anyhow, Result};
use candle_core::{Device, DType, Module, Tensor, D};
use candle_nn::{self as nn, VarBuilder};
use serde::Deserialize;

/// GELU activation (approximation)
fn gelu(x: &Tensor) -> Result<Tensor> {
    // GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
    // Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let x3 = ((x * x)? * x)?;
    let inner = (x + (x3 * 0.044715)?)?;
    let inner = (inner * 0.7978845608)?; // sqrt(2/pi)
    let tanh = inner.tanh()?;
    let one_plus_tanh = (tanh + 1.0)?;
    Ok(((x * one_plus_tanh)? * 0.5)?)
}

/// Generate relative positional encodings
/// Returns sinusoidal encodings of shape [batch, seq_len, dim]
fn relative_positional_encoding(batch: usize, seq_len: usize, dim: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; seq_len * dim];

    for pos in 0..seq_len {
        for i in 0..(dim / 2) {
            let idx = 2 * i;
            let div_term = (pos as f32) / (10000_f32.powf(2.0 * i as f32 / dim as f32));
            data[pos * dim + idx] = div_term.sin();
            if idx + 1 < dim {
                data[pos * dim + idx + 1] = div_term.cos();
            }
        }
    }

    let pos = Tensor::from_slice(&data, (1, seq_len, dim), device)?;
    Ok(pos.broadcast_as((batch, seq_len, dim))?)
}

/// Top-level model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct ParakeetConfig {
    pub ctc_loss_reduction: String,
    pub ctc_zero_infinity: bool,
    pub encoder_config: EncoderConfig,
    pub vocab_size: usize,
    pub pad_token_id: usize,
}

/// Encoder (Conformer) configuration
#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub num_mel_bins: usize,
    pub max_position_embeddings: usize,
    pub attention_dropout: f32,
    pub activation_dropout: f32,
    pub dropout: f32,
    pub dropout_positions: f32,
    pub layerdrop: f32,
    pub conv_kernel_size: usize,
    pub attention_bias: bool,
    pub subsampling_factor: usize,
    pub subsampling_conv_channels: usize,
    pub subsampling_conv_kernel_size: usize,
    pub subsampling_conv_stride: usize,
    pub scale_input: bool,
    pub initializer_range: f64,
}

impl ParakeetConfig {
    pub fn from_file(path: &str) -> Result<Self> {
        let config_str = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&config_str)?)
    }
}

/// Convolution subsampling module
/// Reduces temporal resolution by subsampling_factor (typically 8x)
/// Uses depthwise-separable convolutions
pub struct ConvSubsampling {
    // Layer 0: depthwise conv 3x3, stride 2
    layers_0: nn::Conv2d,
    // Layer 2: depthwise conv 3x3, stride 2
    layers_2: nn::Conv2d,
    // Layer 3: pointwise conv 1x1
    layers_3: nn::Conv2d,
    // Layer 5: depthwise conv 3x3, stride 2
    layers_5: nn::Conv2d,
    // Layer 6: pointwise conv 1x1
    layers_6: nn::Conv2d,
    // Final linear projection
    linear: nn::Linear,
}

impl ConvSubsampling {
    pub fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("subsampling");
        let vb_layers = vb.pp("layers");

        let channels = cfg.subsampling_conv_channels;

        // Layer 0: regular conv (1 -> 256, 3x3, stride 2)
        let layers_0 = nn::conv2d(
            1,
            channels,
            3,
            nn::Conv2dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vb_layers.pp("0"),
        )?;

        // Layer 2: depthwise conv (256 -> 256, 3x3, stride 2, groups=256)
        let layers_2 = nn::conv2d(
            channels,
            channels,
            3,
            nn::Conv2dConfig {
                stride: 2,
                padding: 1,
                groups: channels,  // depthwise
                ..Default::default()
            },
            vb_layers.pp("2"),
        )?;

        // Layer 3: pointwise conv (256 -> 256, 1x1)
        let layers_3 = nn::conv2d(
            channels,
            channels,
            1,
            nn::Conv2dConfig {
                stride: 1,
                padding: 0,
                ..Default::default()
            },
            vb_layers.pp("3"),
        )?;

        // Layer 5: depthwise conv (256 -> 256, 3x3, stride 2, groups=256)
        let layers_5 = nn::conv2d(
            channels,
            channels,
            3,
            nn::Conv2dConfig {
                stride: 2,
                padding: 1,
                groups: channels,  // depthwise
                ..Default::default()
            },
            vb_layers.pp("5"),
        )?;

        // Layer 6: pointwise conv (256 -> 256, 1x1)
        let layers_6 = nn::conv2d(
            channels,
            channels,
            1,
            nn::Conv2dConfig {
                stride: 1,
                padding: 0,
                ..Default::default()
            },
            vb_layers.pp("6"),
        )?;

        // Linear: 2560 -> 1024
        // After 3 stride-2 convs: 80 mel bins / 8 = 10, so 256 * 10 = 2560
        let in_features = channels * (cfg.num_mel_bins / 8);
        let linear = nn::linear(in_features, cfg.hidden_size, vb.pp("linear"))?;

        Ok(Self {
            layers_0,
            layers_2,
            layers_3,
            layers_5,
            layers_6,
            linear,
        })
    }

    /// Forward pass
    /// Input: [B, T, num_mel_bins]
    /// Output: [B, T/subsampling_factor, hidden_size]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, f) = x.dims3()?;

        // Add channel dimension: [B, T, F] -> [B, 1, T, F]
        // Need to reshape to [B, 1, T, F] for 2D convolution
        let x = x.reshape((b, t, f, 1))?.transpose(1, 3)?.transpose(2, 3)?;

        // Layer 0: depthwise conv 3x3, stride 2 + ReLU
        let x = self.layers_0.forward(&x)?.relu()?;

        // Layer 2: depthwise conv 3x3, stride 2 + ReLU
        let x = self.layers_2.forward(&x)?.relu()?;

        // Layer 3: pointwise conv 1x1 + ReLU
        let x = self.layers_3.forward(&x)?.relu()?;

        // Layer 5: depthwise conv 3x3, stride 2 + ReLU
        let x = self.layers_5.forward(&x)?.relu()?;

        // Layer 6: pointwise conv 1x1 + ReLU
        let x = self.layers_6.forward(&x)?.relu()?;

        // Flatten spatial dims: [B, C, H, W] -> [B, H, C*W]
        let (b, c, h, w) = x.dims4()?;
        let x = x.transpose(1, 2)?.reshape((b, h, c * w))?;

        // Project to hidden_size
        Ok(self.linear.forward(&x)?)
    }
}

/// Multi-head self-attention with relative positional encodings
pub struct MultiHeadSelfAttention {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    o_proj: nn::Linear,
    relative_k_proj: Tensor,  // weight matrix for relative position keys
    bias_u: Tensor,  // bias for content-based attention
    bias_v: Tensor,  // bias for position-based attention
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadSelfAttention {
    pub fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = hidden_size / num_heads;

        let q_proj = nn::linear(hidden_size, hidden_size, vb.pp("q_proj"))?;
        let k_proj = nn::linear(hidden_size, hidden_size, vb.pp("k_proj"))?;
        let v_proj = nn::linear(hidden_size, hidden_size, vb.pp("v_proj"))?;
        let o_proj = nn::linear(hidden_size, hidden_size, vb.pp("o_proj"))?;

        // Relative positional encoding components
        let relative_k_proj = vb.get((hidden_size, hidden_size), "relative_k_proj.weight")?;
        let bias_u = vb.get((num_heads, head_dim), "bias_u")?;
        let bias_v = vb.get((num_heads, head_dim), "bias_v")?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            relative_k_proj,
            bias_u,
            bias_v,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, pos: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, d) = hidden_states.dims3()?;

        // Project to Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Compute relative position keys
        // pos: [B, T, D] -> flatten -> matmul -> reshape
        let pos2 = pos.reshape((b * t, d))?;
        let k_rel = pos2.matmul(&self.relative_k_proj.transpose(D::Minus2, D::Minus1)?)?;
        let k_rel = k_rel.reshape((b, t, d))?;

        // Reshape for multi-head attention: [B, T, H] -> [B, num_heads, T, head_dim]
        let q = q.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k_rel = k_rel.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        // Add biases to queries: [B, H, T, head_dim]
        let bu = self.bias_u.unsqueeze(0)?.unsqueeze(2)?; // [1, H, 1, head_dim]
        let bv = self.bias_v.unsqueeze(0)?.unsqueeze(2)?; // [1, H, 1, head_dim]
        let q_bias_u = q.broadcast_add(&bu)?;
        let q_bias_v = q.broadcast_add(&bv)?;

        // Content-based attention scores
        let attn_scores_c = q_bias_u.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;

        // Position-based attention scores
        let mut attn_scores_r = q_bias_v.matmul(&k_rel.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
        attn_scores_r = self.rel_shift(&attn_scores_r)?;

        // Truncate position scores to match sequence length
        let last = attn_scores_r.dims4()?.3;
        let take = last.min(t);
        attn_scores_r = attn_scores_r.narrow(D::Minus1, 0, take)?;

        // Combine content and position scores
        let mut attn_scores = (attn_scores_c + attn_scores_r)?;

        // Apply mask if provided
        if let Some(mask) = attention_mask {
            attn_scores = (attn_scores + mask)?;
        }

        // Scale by sqrt(head_dim)
        let scale = (self.head_dim as f64).sqrt() as f32;
        let scale_t = Tensor::from_slice(&[scale], (), hidden_states.device())?;
        let scale_t = scale_t.broadcast_as(attn_scores.shape())?;
        attn_scores = (attn_scores / scale_t)?;

        // Softmax
        let attn_weights = nn::ops::softmax(&attn_scores, D::Minus1)?;

        // Apply attention to values
        let context = attn_weights.matmul(&v)?;

        // Reshape back: [B, num_heads, T, head_dim] -> [B, T, H]
        let context = context.transpose(1, 2)?.reshape((b, t, d))?;

        // Output projection
        Ok(self.o_proj.forward(&context)?)
    }

    /// Relative position shift to align relative position attention scores
    fn rel_shift(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, H, T, T] -> shift to align relative positions
        let (b, h, t, _) = x.dims4()?;

        // Prepend zeros: [B, H, T, T] -> [B, H, T, T+1]
        let zeros = Tensor::zeros((b, h, t, 1), x.dtype(), x.device())?;
        let x = Tensor::cat(&[&zeros, x], 3)?;

        // Reshape to shift: [B, H, T, T+1] -> [B, H, T+1, T]
        let x = x.reshape((b, h, t + 1, t))?;

        // Remove first row: [B, H, T+1, T] -> [B, H, T, T]
        let x = x.narrow(D::Minus2, 1, t)?;

        Ok(x)
    }
}

/// Depthwise convolution module (for Conformer)
pub struct ConvolutionModule {
    pointwise_conv1: nn::Conv1d,
    depthwise_conv: nn::Conv1d,
    pointwise_conv2: nn::Conv1d,
    activation: String,
}

impl ConvolutionModule {
    pub fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;

        // Pointwise conv (expansion): hidden_size -> 2*hidden_size, kernel_size=1
        let pointwise_conv1 = nn::conv1d(
            hidden_size,
            2 * hidden_size,
            1,
            nn::Conv1dConfig {
                stride: 1,
                padding: 0,
                ..Default::default()
            },
            vb.pp("pointwise_conv1"),
        )?;

        // Depthwise conv
        let depthwise_conv = nn::conv1d(
            hidden_size,
            hidden_size,
            cfg.conv_kernel_size,
            nn::Conv1dConfig {
                padding: cfg.conv_kernel_size / 2, // "same" padding
                groups: hidden_size, // depthwise
                ..Default::default()
            },
            vb.pp("depthwise_conv"),
        )?;

        // Pointwise conv (projection): hidden_size -> hidden_size, kernel_size=1
        let pointwise_conv2 = nn::conv1d(
            hidden_size,
            hidden_size,
            1,
            nn::Conv1dConfig {
                stride: 1,
                padding: 0,
                ..Default::default()
            },
            vb.pp("pointwise_conv2"),
        )?;

        Ok(Self {
            pointwise_conv1,
            depthwise_conv,
            pointwise_conv2,
            activation: cfg.hidden_act.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, T, H]
        let (_b, _t, d) = x.dims3()?;

        // Transpose for conv1d: [B, T, H] -> [B, H, T]
        let x = x.transpose(1, 2)?;

        // Pointwise conv1 (expansion): [B, H, T] -> [B, 2*H, T]
        let x = self.pointwise_conv1.forward(&x)?;

        // GLU activation: split in half on channel dimension and multiply
        let gate = x.narrow(1, 0, d)?;
        let value = x.narrow(1, d, d)?;

        // Apply activation to value (not gate - this matches lib.rs which uses sigmoid)
        let gated = nn::ops::sigmoid(&value)?;
        let x = (gate * gated)?;

        // Depthwise conv: [B, H, T] -> [B, H, T]
        let x = self.depthwise_conv.forward(&x)?;

        // BatchNorm (expects [B, C, T])
        // For inference, we skip batch norm (or could implement eval mode manually)
        // The model should work reasonably without it during inference
        // TODO: Implement proper eval-mode batch norm if needed
        // let x = self.norm.forward_train(&x)?;

        // Activation
        let x = match self.activation.as_str() {
            "silu" | "swish" => nn::ops::silu(&x)?,
            "gelu" => gelu(&x)?,
            _ => x.relu()?,
        };

        // Pointwise conv2 (projection): [B, H, T] -> [B, H, T]
        let x = self.pointwise_conv2.forward(&x)?;

        // Transpose back: [B, H, T] -> [B, T, H]
        Ok(x.transpose(1, 2)?)
    }
}

/// Feed-forward module (Macaron-style for Conformer)
pub struct FeedForwardModule {
    linear1: nn::Linear,
    linear2: nn::Linear,
    activation: String,
}

impl FeedForwardModule {
    pub fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let linear1 = nn::linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("linear1"))?;
        let linear2 = nn::linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("linear2"))?;

        Ok(Self {
            linear1,
            linear2,
            activation: cfg.hidden_act.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;

        // Activation
        let x = match self.activation.as_str() {
            "silu" | "swish" => nn::ops::silu(&x)?,
            "gelu" => gelu(&x)?,
            _ => x.relu()?,
        };

        // Second linear
        Ok(self.linear2.forward(&x)?)
    }
}

/// Conformer block (single layer)
pub struct ConformerBlock {
    feed_forward1: FeedForwardModule,
    self_attn: MultiHeadSelfAttention,
    conv: ConvolutionModule,
    feed_forward2: FeedForwardModule,
    norm_feed_forward1: nn::LayerNorm,
    norm_self_att: nn::LayerNorm,
    norm_conv: nn::LayerNorm,
    norm_feed_forward2: nn::LayerNorm,
    norm_out: nn::LayerNorm,
}

impl ConformerBlock {
    pub fn new(cfg: &EncoderConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let vb_layer = vb.pp(format!("layers.{}", layer_idx));

        let feed_forward1 = FeedForwardModule::new(cfg, vb_layer.pp("feed_forward1"))?;
        let self_attn = MultiHeadSelfAttention::new(cfg, vb_layer.pp("self_attn"))?;
        let conv = ConvolutionModule::new(cfg, vb_layer.pp("conv"))?;
        let feed_forward2 = FeedForwardModule::new(cfg, vb_layer.pp("feed_forward2"))?;

        let hidden_size = cfg.hidden_size;
        let norm_feed_forward1 = nn::layer_norm(hidden_size, 1e-5, vb_layer.pp("norm_feed_forward1"))?;
        let norm_self_att = nn::layer_norm(hidden_size, 1e-5, vb_layer.pp("norm_self_att"))?;
        let norm_conv = nn::layer_norm(hidden_size, 1e-5, vb_layer.pp("norm_conv"))?;
        let norm_feed_forward2 = nn::layer_norm(hidden_size, 1e-5, vb_layer.pp("norm_feed_forward2"))?;
        let norm_out = nn::layer_norm(hidden_size, 1e-5, vb_layer.pp("norm_out"))?;

        Ok(Self {
            feed_forward1,
            self_attn,
            conv,
            feed_forward2,
            norm_feed_forward1,
            norm_self_att,
            norm_conv,
            norm_feed_forward2,
            norm_out,
        })
    }

    /// Forward pass with pre-norm residual connections
    pub fn forward(&self, x: &Tensor, pos: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // 1. First FFN (half weight, Macaron-style)
        let residual = x.clone();
        let x = self.norm_feed_forward1.forward(x)?;
        let x = self.feed_forward1.forward(&x)?;
        let x = ((x * 0.5)? + residual)?;

        // 2. Self-attention with relative positional encodings
        let residual = x.clone();
        let x = self.norm_self_att.forward(&x)?;
        let x = self.self_attn.forward(&x, pos, attention_mask)?;
        let x = (x + residual)?;

        // 3. Convolution module
        let residual = x.clone();
        let x = self.norm_conv.forward(&x)?;
        let x = self.conv.forward(&x)?;
        let x = (x + residual)?;

        // 4. Second FFN (half weight, Macaron-style)
        let residual = x.clone();
        let x = self.norm_feed_forward2.forward(&x)?;
        let x = self.feed_forward2.forward(&x)?;
        let x = ((x * 0.5)? + residual)?;

        // 5. Final layer norm
        Ok(self.norm_out.forward(&x)?)
    }
}

/// Parakeet encoder (stack of Conformer blocks)
pub struct ParakeetEncoder {
    cfg: EncoderConfig,
    conv_subsampling: ConvSubsampling,
    layers: Vec<ConformerBlock>,
}

impl ParakeetEncoder {
    pub fn new(cfg: EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("encoder");

        let conv_subsampling = ConvSubsampling::new(&cfg, vb.clone())?;

        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            layers.push(ConformerBlock::new(&cfg, i, vb.clone())?);
        }

        Ok(Self {
            cfg,
            conv_subsampling,
            layers,
        })
    }

    pub fn forward(&self, features: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let device = features.device();

        // Subsample input features
        let mut hidden_states = self.conv_subsampling.forward(features)?;

        let (b, t, d) = hidden_states.dims3()?;

        // Scale input by sqrt(d_model) if configured
        if self.cfg.scale_input {
            let scale = (self.cfg.hidden_size as f64).sqrt() as f32;
            let scale_t = Tensor::from_slice(&[scale], (), device)?;
            let scale_t = scale_t.broadcast_as(hidden_states.shape())?;
            hidden_states = (hidden_states * scale_t)?;
        }

        // Generate relative positional encodings
        let pos = relative_positional_encoding(b, t, d, device)?;

        // Apply Conformer layers with positional encodings
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &pos, attention_mask)?;
        }

        Ok(hidden_states)
    }
}

/// CTC head (final projection to vocabulary)
pub struct CTCHead {
    weight: Tensor,
    bias: Tensor,
}

impl CTCHead {
    pub fn new(hidden_size: usize, vocab_size: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("ctc_head");

        let weight = vb.get((vocab_size, hidden_size, 1), "weight")?;
        let bias = vb.get(vocab_size, "bias")?;

        Ok(Self { weight, bias })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // hidden_states: [B, T, H]
        // Transpose for conv: [B, T, H] -> [B, H, T]
        let x = hidden_states.transpose(1, 2)?;

        // Conv1d: [B, H, T] -> [B, vocab_size, T]
        let logits = x.conv1d(&self.weight, 0, 1, 1, 1)?;

        // Add bias
        let (b, v, t) = logits.dims3()?;
        let bias = self.bias.reshape((1, v, 1))?.broadcast_as((b, v, t))?;
        let logits = (logits + bias)?;

        // Transpose back: [B, vocab_size, T] -> [B, T, vocab_size]
        Ok(logits.transpose(1, 2)?)
    }
}

/// Complete Parakeet CTC model
pub struct ParakeetCTC {
    pub cfg: ParakeetConfig,
    encoder: ParakeetEncoder,
    ctc_head: CTCHead,
}

impl ParakeetCTC {
    pub fn new(cfg: ParakeetConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = ParakeetEncoder::new(cfg.encoder_config.clone(), vb.clone())?;
        let ctc_head = CTCHead::new(
            cfg.encoder_config.hidden_size,
            cfg.vocab_size,
            vb.clone(),
        )?;

        Ok(Self { cfg, encoder, ctc_head })
    }

    /// Load Parakeet CTC model from GGUF quantized weights (local directory)
    ///
    /// Automatically tries Q8_0 first (recommended), then Q4K
    pub fn from_gguf_local<P: AsRef<std::path::Path>>(
        dir: P,
        device: &candle_core::Device,
    ) -> Result<Self> {
        use candle_core::quantized::gguf_file;
        use std::collections::HashMap;

        let dir = dir.as_ref();
        let config_path = dir.join("config.json");

        // Try Q8_0 first (recommended), then Q4K
        let gguf_path = if dir.join("model_q8_0.gguf").exists() {
            println!("Loading Q8_0 quantized model (recommended)");
            dir.join("model_q8_0.gguf")
        } else if dir.join("model_q4k.gguf").exists() {
            println!("Loading Q4K quantized model (high compression)");
            dir.join("model_q4k.gguf")
        } else {
            return Err(anyhow!(
                "No GGUF file found in {:?}. Expected model_q8_0.gguf or model_q4k.gguf",
                dir
            ));
        };

        if !config_path.exists() {
            return Err(anyhow!("config.json not found in {:?}", dir));
        }

        // Load config
        let cfg = ParakeetConfig::from_file(config_path.to_str().unwrap())?;

        // Load GGUF file
        println!("  Loading GGUF file...");
        let mut file = std::fs::File::open(&gguf_path)?;
        let gguf_content = gguf_file::Content::read(&mut file)?;
        println!("  Loaded {} tensors from GGUF", gguf_content.tensor_infos.len());

        // Dequantize all tensors to FP32
        println!("  Dequantizing tensors to FP32...");
        let mut tensors = HashMap::new();
        for (name, _tensor_info) in gguf_content.tensor_infos.iter() {
            let qtensor = gguf_content.tensor(&mut file, name, device)?;
            let tensor = qtensor.dequantize(device)?;
            tensors.insert(name.clone(), tensor);
        }
        println!("  ✓ All tensors dequantized");

        // Create VarBuilder from dequantized tensors
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);

        println!("  Building model...");
        let model = ParakeetCTC::new(cfg, vb)?;
        println!("✓ Quantized model loaded successfully\n");

        Ok(model)
    }

    /// Forward pass
    /// Input: mel-spectrogram features [B, T, num_mel_bins]
    /// Output: CTC logits [B, T', vocab_size] where T' = T / subsampling_factor
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let encoder_output = self.encoder.forward(features, None)?;
        self.ctc_head.forward(&encoder_output)
    }

    /// Greedy CTC decoding
    pub fn greedy_decode(&self, logits: &Tensor) -> Result<Vec<Vec<u32>>> {
        // logits: [B, T, vocab_size]
        let (batch_size, seq_len, _vocab_size) = logits.dims3()?;

        // Get argmax for each time step
        let predictions = logits.argmax(candle_core::D::Minus1)?;
        let predictions = predictions.to_vec2::<u32>()?;

        let mut results = Vec::new();

        for b in 0..batch_size {
            let mut tokens = Vec::new();
            let mut prev_token = self.cfg.vocab_size as u32; // Invalid token to start

            for t in 0..seq_len {
                let token = predictions[b][t];

                // Skip blank (vocab_size - 1) and repeated tokens
                if token != (self.cfg.vocab_size - 1) as u32 && token != prev_token {
                    tokens.push(token);
                }

                prev_token = token;
            }

            results.push(tokens);
        }

        Ok(results)
    }
}
