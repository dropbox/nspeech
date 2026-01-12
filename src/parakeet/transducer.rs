/// Transducer (RNN-T) decoder for Parakeet TDT models
///
/// A Transducer model consists of three networks:
/// 1. **Encoder** (FastConformer): Encodes acoustic features → [B, T, D_enc]
/// 2. **Predictor** (RNN): Predicts next token from history → [B, U, D_pred]
/// 3. **Joint Network**: Combines encoder and predictor → [B, T, U, vocab_size]
///
/// The joint network outputs logits for each (time, label) position, enabling
/// streaming inference and automatic alignment between audio and text.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{embedding, linear, rnn, Embedding, Linear, VarBuilder};
use serde::Deserialize;
use tokenizers::{Tokenizer, models::unigram::Unigram};

use super::fast_conformer::{FastConformerConfig, FastConformerEncoder, HfEncoderConfig};
use hf_hub::api::sync::Api;
use std::path::Path;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct TransducerConfig {
    pub vocab_size: usize,  // Predictor vocab size
    pub blank_id: usize,

    // Joint output vocab size (may differ from vocab_size)
    #[serde(default)]
    pub joint_vocab_size: Option<usize>,

    // Predictor (RNN) config
    pub pred_hidden: usize,
    pub pred_rnn_layers: usize,
    pub pred_dropout: f64,

    // Joint network config
    pub joint_hidden: usize,
    pub joint_dropout: f64,
}

impl Default for TransducerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8192,
            blank_id: 0,
            joint_vocab_size: None,  // Defaults to vocab_size if None
            pred_hidden: 512,
            pred_rnn_layers: 2,
            pred_dropout: 0.1,
            joint_hidden: 512,
            joint_dropout: 0.1,
        }
    }
}

/// Prediction Network (RNN)
///
/// Takes previous token predictions and produces context vectors.
/// Uses LSTM layers to model language dependencies.
pub struct PredictionNetwork {
    embedding: Embedding,
    lstms: Vec<rnn::LSTM>,
    projection: Option<Linear>,
    pred_hidden: usize,
    num_layers: usize,
}

impl PredictionNetwork {
    pub fn new(
        vocab_size: usize,
        pred_hidden: usize,
        num_layers: usize,
        _dropout: f64, // TODO: implement dropout between LSTM layers
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        // Embedding layer: vocab → hidden
        let embedding = embedding(vocab_size, pred_hidden, vb.pp("embed"))?;

        // Stack LSTM layers (NeMo uses single multi-layer LSTM with shared prefix)
        let mut lstms = Vec::new();
        for i in 0..num_layers {
            let config = rnn::LSTMConfig {
                layer_idx: i,
                ..Default::default()
            };
            // All layers share the same "lstm" prefix, layer_idx handles the _l{i} suffix
            let lstm = rnn::lstm(
                pred_hidden,
                pred_hidden,
                config,
                vb.pp("lstm"),
            )?;
            lstms.push(lstm);
        }

        // Project LSTM output to joint network dimensionality (optional - NeMo doesn't use this)
        let projection = match linear(pred_hidden, pred_hidden, vb.pp("proj")) {
            Ok(proj) => Some(proj),
            Err(_) => None,  // NeMo models don't have projection layer
        };

        Ok(Self {
            embedding,
            lstms,
            projection,
            pred_hidden,
            num_layers,
        })
    }

    /// Forward pass: [B, U] token IDs → [B, U, pred_hidden]
    pub fn forward(&self, tokens: &Tensor, states: Option<&Vec<rnn::LSTMState>>) -> Result<(Tensor, Vec<rnn::LSTMState>)> {
        use candle_nn::RNN;

        // Embed tokens: [B, U] → [B, U, pred_hidden]
        let embedded = self.embedding.forward(tokens)?;
        let (batch_size, seq_len, _hidden) = embedded.dims3()?;

        // Initialize states if not provided
        let current_states = if let Some(s) = states {
            s.clone()
        } else {
            self.init_states(batch_size, embedded.device())?
        };

        // Process sequence through LSTM layers
        // For each layer, process all timesteps and update states
        let mut layer_output = embedded;
        let mut new_states = Vec::new();

        for (layer_idx, lstm) in self.lstms.iter().enumerate() {
            let mut timestep_outputs = Vec::new();
            let mut state = current_states[layer_idx].clone();

            // Process each timestep through this LSTM layer
            for t in 0..seq_len {
                // Extract [B, pred_hidden] for this timestep
                let input_t = layer_output.narrow(1, t, 1)?.squeeze(1)?;

                // LSTM step: [B, pred_hidden] → new state
                state = lstm.step(&input_t, &state)?;

                // Collect hidden state output [B, pred_hidden]
                timestep_outputs.push(state.h().clone());
            }

            // Stack timestep outputs back to [B, U, pred_hidden]
            layer_output = Tensor::stack(&timestep_outputs, 1)?;
            new_states.push(state);
        }

        // Project: [B, U, pred_hidden] → [B, U, pred_hidden] (optional)
        let pred_output = if let Some(ref proj) = self.projection {
            proj.forward(&layer_output)?
        } else {
            layer_output  // No projection - use LSTM output directly
        };

        Ok((pred_output, new_states))
    }

    /// Initialize zero LSTM states for new sequences
    pub fn init_states(&self, batch_size: usize, device: &Device) -> Result<Vec<rnn::LSTMState>> {
        // Use same dtype as model (BF16 on GPU, F32 on CPU)
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        };

        let zeros = Tensor::zeros(
            (batch_size, self.pred_hidden),
            dtype,
            device,
        )?;

        // Create initial LSTM state (h0, c0) for each layer
        let mut states = Vec::new();
        for _ in 0..self.num_layers {
            states.push(rnn::LSTMState::new(zeros.clone(), zeros.clone()));
        }

        Ok(states)
    }
}

/// Joint Network
///
/// Combines encoder and predictor outputs to produce token logits.
/// Uses element-wise addition followed by MLP.
pub struct JointNetwork {
    encoder_proj: Linear,
    pred_proj: Linear,
    hidden: Option<Linear>,
    output: Linear,
    #[allow(dead_code)]
    joint_hidden: usize,
}

impl JointNetwork {
    pub fn new(
        enc_dim: usize,
        pred_dim: usize,
        joint_hidden: usize,
        vocab_size: usize,
        _dropout: f64, // TODO: implement dropout in joint network
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        // Project encoder output to joint dimensionality
        let encoder_proj = linear(enc_dim, joint_hidden, vb.pp("enc_proj"))?;

        // Project predictor output to joint dimensionality
        let pred_proj = linear(pred_dim, joint_hidden, vb.pp("pred_proj"))?;

        // Hidden layer with activation (optional - NeMo doesn't use this)
        let hidden = match linear(joint_hidden, joint_hidden, vb.pp("hidden")) {
            Ok(h) => Some(h),
            Err(_) => None,  // NeMo models don't have hidden layer
        };

        // Output layer: joint_hidden → vocab_size
        let output = linear(joint_hidden, vocab_size, vb.pp("output"))?;

        Ok(Self {
            encoder_proj,
            pred_proj,
            hidden,
            output,
            joint_hidden,
        })
    }

    /// Forward pass: (encoder, predictor) → logits
    ///
    /// encoder: [B, T, enc_dim]
    /// predictor: [B, U, pred_dim]
    /// output: [B, T, U, vocab_size]
    pub fn forward(&self, encoder_out: &Tensor, predictor_out: &Tensor) -> Result<Tensor> {
        let (_b, _t, _enc_dim) = encoder_out.dims3()?;
        let (_b2, _u, _pred_dim) = predictor_out.dims3()?;

        // Project encoder: [B, T, enc_dim] → [B, T, joint_hidden]
        let enc_proj = self.encoder_proj.forward(encoder_out)?;

        // Project predictor: [B, U, pred_dim] → [B, U, joint_hidden]
        let pred_proj = self.pred_proj.forward(predictor_out)?;

        // Add encoder and predictor with broadcasting
        // enc_proj: [B, T, 1, joint_hidden]
        // pred_proj: [B, 1, U, joint_hidden]
        // result: [B, T, U, joint_hidden]
        let enc_proj = enc_proj.unsqueeze(2)?; // [B, T, 1, joint_hidden]
        let pred_proj = pred_proj.unsqueeze(1)?; // [B, 1, U, joint_hidden]

        let joint = enc_proj.broadcast_add(&pred_proj)?; // [B, T, U, joint_hidden]

        // Apply activation and optional hidden layer
        let joint = joint.relu()?;
        let joint = if let Some(ref hidden) = self.hidden {
            let joint = hidden.forward(&joint)?;
            joint.tanh()?  // Tanh activation typical for joint networks
        } else {
            joint  // No hidden layer - NeMo uses direct output
        };

        // Output layer: [B, T, U, joint_hidden] → [B, T, U, vocab_size]
        let logits = self.output.forward(&joint)?;

        Ok(logits)
    }
}

/// Full Transducer Model
pub struct TransducerModel {
    pub encoder: FastConformerEncoder,
    pub predictor: PredictionNetwork,
    pub joint: JointNetwork,
    pub config: TransducerConfig,
    tokenizer: Option<Tokenizer>,
}

impl TransducerModel {
    pub fn new(
        encoder: FastConformerEncoder,
        tdt_config: TransducerConfig,
        enc_dim: usize,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let predictor = PredictionNetwork::new(
            tdt_config.vocab_size,
            tdt_config.pred_hidden,
            tdt_config.pred_rnn_layers,
            tdt_config.pred_dropout,
            vb.pp("predictor"),
        )?;

        // Use joint_vocab_size if specified, otherwise fall back to vocab_size
        let joint_vocab = tdt_config.joint_vocab_size.unwrap_or(tdt_config.vocab_size);

        let joint = JointNetwork::new(
            enc_dim,
            tdt_config.pred_hidden,
            tdt_config.joint_hidden,
            joint_vocab,
            tdt_config.joint_dropout,
            vb.pp("joint"),
        )?;

        Ok(Self {
            encoder,
            predictor,
            joint,
            config: tdt_config,
            tokenizer: None,
        })
    }

    /// Load tokenizer from directory
    ///
    /// Tries to load either tokenizer.json (HuggingFace format) or
    /// tokenizer.model (SentencePiece format).
    pub fn load_tokenizer<P: AsRef<Path>>(&mut self, dir: P) -> Result<()> {
        let dir = dir.as_ref();

        // Try HuggingFace tokenizer.json first
        let json_path = dir.join("tokenizer.json");
        if json_path.exists() {
            self.tokenizer = Some(Tokenizer::from_file(&json_path)
                .map_err(|e| anyhow!("Failed to load tokenizer.json: {}", e))?);
            return Ok(());
        }

        // Try SentencePiece tokenizer.model
        let sp_path = dir.join("tokenizer.model");
        if sp_path.exists() {
            // Load SentencePiece model using Unigram
            let model = Unigram::load(&sp_path)
                .map_err(|e| anyhow!("Failed to load tokenizer.model: {}", e))?;
            self.tokenizer = Some(Tokenizer::new(model));
            return Ok(());
        }

        Err(anyhow!(
            "No tokenizer found in {:?} (tried tokenizer.json and tokenizer.model)",
            dir
        ))
    }

    /// Decode token IDs to text
    ///
    /// Returns an error if tokenizer is not loaded.
    pub fn decode_tokens(&self, token_ids: &[u32]) -> Result<String> {
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow!("Tokenizer not loaded. Call load_tokenizer() first."))?;

        let text = tokenizer.decode(token_ids, true)
            .map_err(|e| anyhow!("Failed to decode tokens: {}", e))?;

        Ok(text)
    }

    /// Greedy decoding: Simple left-to-right decoding without beam search
    ///
    /// For each encoder timestep, predict the most likely token until blank is emitted.
    pub fn greedy_decode(&self, encoder_out: &Tensor) -> Result<Vec<u32>> {
        let (batch_size, time_steps, _enc_dim) = encoder_out.dims3()?;

        if batch_size != 1 {
            return Err(anyhow!("Greedy decode currently only supports batch_size=1"));
        }

        let mut decoded = Vec::new();
        let mut pred_states = None;

        // Start with blank token
        let mut last_token = self.config.blank_id as u32;

        // Decode all timesteps now that special tokens are handled correctly
        println!("  Decoding {} timesteps...", time_steps);

        for t in 0..time_steps {
            if t % 50 == 0 {
                println!("  Progress: {}/{} timesteps, {} tokens decoded", t, time_steps, decoded.len());
            }

            // Inner loop: keep predicting until blank
            // Add safety limit to prevent infinite loops
            let mut inner_steps = 0;
            const MAX_INNER_STEPS: usize = 50;
            let mut first_token_this_timestep = None;

            loop {
                inner_steps += 1;
                if inner_steps > MAX_INNER_STEPS {
                    println!("    WARNING: Hit max inner steps at timestep {}, forcing blank", t);
                    break;
                }

                // Get encoder output at current timestep: [1, 1, enc_dim]
                let enc_t = encoder_out.narrow(1, t, 1)?;

                // Predictor input: previous token [1, 1]
                // Note: Embedding layer expects integer input, dtype doesn't matter for indices
                let pred_input = Tensor::new(&[last_token], encoder_out.device())?
                    .unsqueeze(0)?;

                // Run predictor
                let (pred_out, new_states) = self.predictor.forward(&pred_input, pred_states.as_ref())?;
                pred_states = Some(new_states);

                // Joint network: [1, 1, enc_dim] + [1, 1, pred_dim] → [1, 1, 1, vocab_size]
                let logits = self.joint.forward(&enc_t, &pred_out)?;

                // Get most likely token: [vocab_size]
                let logits = logits.squeeze(0)?.squeeze(0)?.squeeze(0)?;

                // Convert to F32 for log_softmax (BF16 not supported)
                let logits_f32 = logits.to_dtype(DType::F32)?;

                // Apply log_softmax for proper probability distribution
                let log_probs = candle_nn::ops::log_softmax(&logits_f32, D::Minus1)?;

                // Mask out padding tokens 8193-8197 to prevent their use
                // Valid tokens: 0-8191 (content) + 8192 (blank)
                let mut masked_logits = logits_f32.clone();
                for i in 8193..8198 {
                    let mask_tensor = Tensor::new(&[-1e9_f32], masked_logits.device())?;
                    masked_logits = masked_logits.slice_assign(&[i..i+1], &mask_tensor)?;
                }
                let log_probs_masked = candle_nn::ops::log_softmax(&masked_logits, D::Minus1)?;
                let token_tensor = log_probs_masked.argmax(D::Minus1)?;
                let token = token_tensor.to_scalar::<u32>()?;

                // Debug first token of every 50th timestep
                if first_token_this_timestep.is_none() {
                    first_token_this_timestep = Some(token);
                    if t % 50 == 0 {
                        let blank_prob = log_probs.get(self.config.blank_id)?.to_scalar::<f32>()?;
                        let token_prob = log_probs.get(token as usize)?.to_scalar::<f32>()?;

                        // Find best content token (0-8191, excluding special tokens 8192-8197)
                        let mut best_content_token = 0;
                        let mut best_content_prob = f32::NEG_INFINITY;
                        for i in 0..8192 {
                            let prob = log_probs.get(i)?.to_scalar::<f32>()?;
                            if prob > best_content_prob {
                                best_content_prob = prob;
                                best_content_token = i;
                            }
                        }

                        println!("    t={}: top=tok{} ({:.3}), blank=tok{} ({:.3}), best_content=tok{} ({:.3})",
                                 t, token, token_prob, self.config.blank_id, blank_prob,
                                 best_content_token, best_content_prob);
                    }
                }

                if token == self.config.blank_id as u32 {
                    // Blank: move to next timestep
                    break;
                } else if token >= self.config.vocab_size as u32 {
                    // Special token beyond vocab (can't feed to predictor), treat as blank
                    if t % 50 == 0 && inner_steps == 1 {
                        println!("    (Special token {}, treating as blank)", token);
                    }
                    break;
                } else {
                    // Valid vocabulary token: emit and continue at same timestep
                    decoded.push(token);
                    last_token = token;
                }
            }
        }

        Ok(decoded)
    }
}

/// Remap NeMo tensor names to our expected format
///
/// NeMo uses different naming conventions:
/// - `encoder.pre_encode.*` → `encoder.subsampling.*`
/// - `decoder.prediction.*` → `predictor.*`
/// - `joint.enc` → `joint.encoder_proj`
/// - `self_attn.linear_q` → `self_attn.q_proj`
fn remap_nemo_tensor_name(nemo_name: &str) -> String {
    let name = nemo_name
        // Encoder subsampling: pre_encode → subsampling
        .replace("encoder.pre_encode.conv.", "encoder.subsampling.layers.")
        .replace("encoder.pre_encode.out.", "encoder.subsampling.linear.")
        // Conv module: batch_norm → norm
        .replace("conv.batch_norm.", "conv.norm.")
        // Attention projections: linear_* → *_proj
        .replace("self_attn.linear_q.", "self_attn.q_proj.")
        .replace("self_attn.linear_k.", "self_attn.k_proj.")
        .replace("self_attn.linear_v.", "self_attn.v_proj.")
        .replace("self_attn.linear_out.", "self_attn.o_proj.")
        .replace("self_attn.linear_pos.", "self_attn.relative_k_proj.")
        .replace("self_attn.pos_bias_u", "self_attn.bias_u")
        .replace("self_attn.pos_bias_v", "self_attn.bias_v")
        // Predictor: decoder.prediction → predictor
        .replace("decoder.prediction.embed", "predictor.embed")
        .replace("decoder.prediction.dec_rnn.lstm.", "predictor.lstm.")  // NeMo uses single multi-layer LSTM
        // Joint network
        .replace("joint.enc.", "joint.enc_proj.")
        .replace("joint.pred.", "joint.pred_proj.")
        .replace("joint.joint_net.2.", "joint.output.");  // NeMo only has output layer (no hidden)

    name
}

/// HuggingFace TDT model configuration format
#[derive(Debug, Deserialize)]
pub struct HfTransducerConfig {
    pub encoder_config: HfEncoderConfig,
    pub vocab_size: usize,
    pub blank_id: usize,
    #[serde(default)]
    pub joint_vocab_size: Option<usize>,
    pub predictor_config: HfPredictorConfig,
    pub joint_config: HfJointConfig,
}

#[derive(Debug, Deserialize)]
pub struct HfPredictorConfig {
    pub pred_hidden: usize,
    pub pred_rnn_layers: usize,
}

#[derive(Debug, Deserialize)]
pub struct HfJointConfig {
    pub joint_hidden: usize,
    #[allow(dead_code)]
    pub activation: Option<String>,
}

impl TransducerConfig {
    pub fn from_hf(hf: &HfTransducerConfig) -> Self {
        Self {
            vocab_size: hf.vocab_size,
            blank_id: hf.blank_id,
            joint_vocab_size: hf.joint_vocab_size,
            pred_hidden: hf.predictor_config.pred_hidden,
            pred_rnn_layers: hf.predictor_config.pred_rnn_layers,
            pred_dropout: 0.1,  // Default value
            joint_hidden: hf.joint_config.joint_hidden,
            joint_dropout: 0.1,  // Default value
        }
    }
}

/// Load Parakeet TDT (Transducer) model from local directory
///
/// # Arguments
/// * `dir` - Directory containing config.json, model.safetensors, and tokenizer files
/// * `device` - Device to load model on
///
/// Expected files in directory:
/// - `config.json` - Model configuration
/// - `model.safetensors` - Model weights
/// - `tokenizer.model` or `tokenizer.json` - Tokenizer
///
/// # Example
/// ```no_run
/// use speech::parakeet::transducer::{load_parakeet_tdt_from_local, TransducerModel};
/// use speech::parakeet::get_device;
/// let device = get_device()?;
/// let model = load_parakeet_tdt_from_local(".cache/parakeet-tdt", &device)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn load_parakeet_tdt_from_local<P: AsRef<Path>>(
    dir: P,
    device: &Device,
) -> Result<TransducerModel> {
    let dir = dir.as_ref();
    let config_path = dir.join("config.json");
    let weights_path = dir.join("model.safetensors");

    // Check for tokenizer files (try both SentencePiece .model and HuggingFace .json)
    // TODO: Add tokenizer support to TransducerModel for token decoding
    let _tokenizer_path = if dir.join("tokenizer.model").exists() {
        dir.join("tokenizer.model")
    } else {
        dir.join("tokenizer.json")
    };

    if !config_path.exists() || !weights_path.exists() {
        return Err(anyhow!(
            "missing files in {:?}, need config.json and model.safetensors",
            dir
        ));
    }

    // Load config
    let cfg_json = std::fs::read_to_string(&config_path)?;
    let hf_cfg: HfTransducerConfig = serde_json::from_str(&cfg_json)?;
    let tdt_cfg = TransducerConfig::from_hf(&hf_cfg);

    // Manually construct encoder config from TDT config structure
    let enc = &hf_cfg.encoder_config;
    let encoder_cfg = FastConformerConfig {
        feat_in: enc.num_mel_bins,
        d_model: enc.hidden_size,
        num_heads: enc.num_attention_heads,
        ff_mult: enc.intermediate_size / enc.hidden_size,
        num_layers: enc.num_hidden_layers,
        conv_kernel_size: enc.conv_kernel_size,
        dropout: enc.dropout,
        dropout_positions: enc.dropout_positions,
        subsampling_channels: enc.subsampling_conv_channels,
        subsampling_stride: enc.subsampling_conv_stride,
        subsampling_factor: enc.subsampling_factor,
        scale_input: enc.scale_input.unwrap_or(true),
        vocab_size: hf_cfg.vocab_size,
        blank_id: hf_cfg.blank_id,
    };

    // Load model weights
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16  // Use BF16 on GPU (matches training dtype)
    };

    println!("Loading TDT model with {:?} dtype", dtype);

    // Load safetensors and remap NeMo tensor names to our expected format
    println!("  Loading and remapping NeMo tensors...");
    let tensors_raw: HashMap<String, Tensor> = candle_core::safetensors::load(&weights_path, device)?;

    // Remap tensor names from NeMo format to our expected format
    let mut tensors = HashMap::new();
    for (nemo_name, tensor) in tensors_raw {
        let our_name = remap_nemo_tensor_name(&nemo_name);
        if our_name != nemo_name {
            println!("    {} -> {}", nemo_name, our_name);
        }
        // Convert tensors to target dtype
        let tensor_converted = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)?
        } else {
            tensor
        };
        tensors.insert(our_name.clone(), tensor_converted.clone());

        // NeMo models don't have biases for many layers - add zero biases where missing
        // This includes: feedforward (linear1/2), attention projections (q/k/v/o_proj),
        // relative position projection (relative_k_proj), joint network projections, and conv layers
        let needs_bias = our_name.contains(".linear1.weight")
            || our_name.contains(".linear2.weight")
            || our_name.contains(".q_proj.weight")
            || our_name.contains(".k_proj.weight")
            || our_name.contains(".v_proj.weight")
            || our_name.contains(".o_proj.weight")
            || our_name.contains(".relative_k_proj.weight")
            || our_name.contains(".enc_proj.weight")
            || our_name.contains(".pred_proj.weight")
            || our_name.contains(".hidden.weight")
            || our_name.contains(".output.weight")
            || our_name.contains(".pointwise_conv1.weight")
            || our_name.contains(".pointwise_conv2.weight")
            || our_name.contains(".depthwise_conv.weight");

        if needs_bias {
            let bias_name = our_name.replace(".weight", ".bias");
            if !tensors.contains_key(&bias_name) {
                // Create zero bias with appropriate shape
                let out_features = tensor_converted.dims()[0];
                let zero_bias = Tensor::zeros(out_features, dtype, device)?;
                tensors.insert(bias_name, zero_bias);
            }
        }
    }

    let vb = VarBuilder::from_tensors(tensors, dtype, device);

    // Build encoder
    let encoder = FastConformerEncoder::new(encoder_cfg.clone(), vb.pp("encoder"))?;

    // Build full transducer model
    let model = TransducerModel::new(
        encoder,
        tdt_cfg,
        encoder_cfg.d_model,
        vb,
    )?;

    Ok(model)
}

/// Load Parakeet TDT model from Hugging Face Hub
///
/// # Arguments
/// * `repo_id` - Hugging Face repository (e.g., "nvidia/parakeet-tdt-0.6b-v3")
/// * `device` - Device to load model on
///
/// # Example
/// ```no_run
/// use speech::parakeet::transducer::{load_parakeet_tdt_from_hf, TransducerModel};
/// use speech::parakeet::get_device;
/// let device = get_device()?;
/// let model = load_parakeet_tdt_from_hf("nvidia/parakeet-tdt-0.6b-v3", &device)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn load_parakeet_tdt_from_hf(
    repo_id: &str,
    device: &Device,
) -> Result<TransducerModel> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    let config_path = repo.get("config.json")?;
    let weights_path = repo.get("model.safetensors")?;

    // Try to get tokenizer (could be either format)
    // TODO: Add tokenizer support to TransducerModel for token decoding
    let _tokenizer_path = if let Ok(path) = repo.get("tokenizer.model") {
        path
    } else {
        repo.get("tokenizer.json")?
    };

    // Load config
    let cfg_json = std::fs::read_to_string(config_path)?;
    let hf_cfg: HfTransducerConfig = serde_json::from_str(&cfg_json)?;
    let tdt_cfg = TransducerConfig::from_hf(&hf_cfg);

    // Manually construct encoder config from TDT config structure
    let enc = &hf_cfg.encoder_config;
    let encoder_cfg = FastConformerConfig {
        feat_in: enc.num_mel_bins,
        d_model: enc.hidden_size,
        num_heads: enc.num_attention_heads,
        ff_mult: enc.intermediate_size / enc.hidden_size,
        num_layers: enc.num_hidden_layers,
        conv_kernel_size: enc.conv_kernel_size,
        dropout: enc.dropout,
        dropout_positions: enc.dropout_positions,
        subsampling_channels: enc.subsampling_conv_channels,
        subsampling_stride: enc.subsampling_conv_stride,
        subsampling_factor: enc.subsampling_factor,
        scale_input: enc.scale_input.unwrap_or(true),
        vocab_size: hf_cfg.vocab_size,
        blank_id: hf_cfg.blank_id,
    };

    // Load model weights
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16  // Use BF16 on GPU (matches training dtype)
    };

    println!("Loading TDT model from HF with {:?} dtype", dtype);

    // Load safetensors and remap NeMo tensor names to our expected format
    println!("  Loading and remapping NeMo tensors...");
    let tensors_raw: HashMap<String, Tensor> = candle_core::safetensors::load(&weights_path, device)?;

    // Remap tensor names from NeMo format to our expected format
    let mut tensors = HashMap::new();
    for (nemo_name, tensor) in tensors_raw {
        let our_name = remap_nemo_tensor_name(&nemo_name);
        // Convert tensors to target dtype
        let tensor_converted = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)?
        } else {
            tensor
        };
        tensors.insert(our_name.clone(), tensor_converted.clone());

        // NeMo models don't have biases for many layers - add zero biases where missing
        // This includes: feedforward (linear1/2), attention projections (q/k/v/o_proj),
        // relative position projection (relative_k_proj), joint network projections, and conv layers
        let needs_bias = our_name.contains(".linear1.weight")
            || our_name.contains(".linear2.weight")
            || our_name.contains(".q_proj.weight")
            || our_name.contains(".k_proj.weight")
            || our_name.contains(".v_proj.weight")
            || our_name.contains(".o_proj.weight")
            || our_name.contains(".relative_k_proj.weight")
            || our_name.contains(".enc_proj.weight")
            || our_name.contains(".pred_proj.weight")
            || our_name.contains(".hidden.weight")
            || our_name.contains(".output.weight")
            || our_name.contains(".pointwise_conv1.weight")
            || our_name.contains(".pointwise_conv2.weight")
            || our_name.contains(".depthwise_conv.weight");

        if needs_bias {
            let bias_name = our_name.replace(".weight", ".bias");
            if !tensors.contains_key(&bias_name) {
                // Create zero bias with appropriate shape
                let out_features = tensor_converted.dims()[0];
                let zero_bias = Tensor::zeros(out_features, dtype, device)?;
                tensors.insert(bias_name, zero_bias);
            }
        }
    }

    let vb = VarBuilder::from_tensors(tensors, dtype, device);

    // Build encoder
    let encoder = FastConformerEncoder::new(encoder_cfg.clone(), vb.pp("encoder"))?;

    // Build full transducer model
    let model = TransducerModel::new(
        encoder,
        tdt_cfg,
        encoder_cfg.d_model,
        vb,
    )?;

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transducer_config() {
        let config = TransducerConfig::default();
        assert_eq!(config.vocab_size, 8192);
        assert_eq!(config.blank_id, 0);
    }
}
