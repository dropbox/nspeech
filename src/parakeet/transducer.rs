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
use log::{info, warn, debug};

use super::fast_conformer::{FastConformerConfig, FastConformerEncoder, HfEncoderConfig};
#[cfg(feature = "triton-metal")]
use super::triton_encoder::TritonParakeetEncoder;
use std::path::Path;
use std::collections::HashMap;

// Embedded TDT model assets (compressed with zstd)
use crate::embed_zst_asset;
use crate::embed_asset;
embed_zst_asset!(pub TDT_CONFIG, "parakeet-tdt-config.json.zst");
embed_zst_asset!(pub TDT_MODEL, "parakeet-tdt-model.safetensors.zst");
embed_zst_asset!(pub TDT_MODEL_Q8_0_GGUF, "parakeet-tdt-model_q8_0.gguf.zst");
embed_asset!(pub TDT_MODEL_Q8_0_GGUF_MMAP, "parakeet-tdt-model_q8_0.gguf");  // Uncompressed for mmap
embed_zst_asset!(pub TDT_TOKENIZER, "parakeet-tdt-tokenizer.model.zst");
embed_zst_asset!(pub TDT_TOKENIZER_JSON, "parakeet-tdt-tokenizer.json.zst");

// Embedded Streaming TDT model assets (cache-aware variant)
embed_zst_asset!(pub STREAMING_TDT_CONFIG, "parakeet-streaming-tdt-config.json.zst");
embed_zst_asset!(pub STREAMING_TDT_MODEL, "parakeet-streaming-tdt-model.safetensors.zst");
//embed_zst_asset!(pub STREAMING_TDT_MODEL_Q8_0_GGUF, "parakeet-streaming-tdt-model_q8_0.gguf.zst");
embed_zst_asset!(pub STREAMING_TDT_TOKENIZER, "parakeet-streaming-tdt-tokenizer.model.zst");
embed_zst_asset!(pub STREAMING_TDT_TOKENIZER_JSON, "parakeet-streaming-tdt-tokenizer.json.zst");

/// Token with timestamp information from TDT alignment
#[derive(Debug, Clone)]
pub struct TokenWithTimestamp {
    pub token: u32,
    pub frame: usize,  // Encoder frame where this token was emitted
}

/// Beam search hypothesis for transducer decoding
#[derive(Debug, Clone)]
struct BeamHypothesis {
    /// Accumulated tokens
    tokens: Vec<u32>,
    /// Accumulated score (log probability)
    score: f32,
    /// Current predictor LSTM state
    pred_state: Vec<rnn::LSTMState>,
    /// Last predicted token
    last_token: u32,
    /// Current timestep in encoder output
    timestep: usize,
}

/// Streaming beam hypothesis (no timestep, for chunk-based processing)
#[derive(Debug, Clone)]
pub struct StreamingBeamHypothesis {
    /// Accumulated tokens for this chunk
    pub tokens: Vec<u32>,
    /// Accumulated score (log probability)
    pub score: f32,
    /// Current predictor LSTM state
    pub pred_state: Vec<rnn::LSTMState>,
    /// Last predicted token
    pub last_token: u32,
}

/// State for beam search across streaming chunks
#[derive(Debug, Clone)]
pub struct BeamStreamingState {
    /// Active beam hypotheses
    pub hypotheses: Vec<StreamingBeamHypothesis>,
    /// Beam size
    pub beam_size: usize,
}

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
        // NeMo includes blank token in predictor vocabulary, so vocab_size + 1
        let embedding = embedding(vocab_size + 1, pred_hidden, vb.pp("embed"))?;

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
    #[cfg(feature = "triton-metal")]
    triton_encoder: Option<TritonParakeetEncoder>,
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
            #[cfg(feature = "triton-metal")]
            triton_encoder: None,
            predictor,
            joint,
            config: tdt_config,
            tokenizer: None,
        })
    }

    /// Run encoder, dispatching to Triton when available.
    pub fn run_encoder(&self, features: &Tensor, train: bool) -> Result<Tensor> {
        let t0 = std::time::Instant::now();
        #[cfg(feature = "triton-metal")]
        if !std::env::var("NO_TRITON").is_ok() {
            if let Some(te) = &self.triton_encoder {
                let out = te.forward(features)?;
                eprintln!("  Triton encoder: {} ms", t0.elapsed().as_millis());
                return Ok(out);
            }
        }
        let out = self.encoder.forward(features, train)?;
        eprintln!("  Candle encoder: {} ms", t0.elapsed().as_millis());
        Ok(out)
    }

    /// Load tokenizer from directory
    ///
    /// Tries to load either tokenizer.json (HuggingFace format) or
    /// tokenizer.model (SentencePiece format).
    /// First tries embedded assets, then falls back to filesystem.
    pub fn load_tokenizer<P: AsRef<Path>>(&mut self, dir: P) -> Result<()> {
        let dir = dir.as_ref();
        let dir_pathbuf = dir.to_path_buf();

        // Try loading from embedded assets first
        if let Ok(tok_bytes) = TDT_TOKENIZER_JSON.bytes(&dir_pathbuf) {
            self.tokenizer = Some(Tokenizer::from_bytes(tok_bytes)
                .map_err(|e| anyhow!("Failed to parse embedded TDT_TOKENIZER_JSON: {}", e))?);
            return Ok(());
        }

        // Fall back to filesystem: Try HuggingFace tokenizer.json
        let json_path = dir.join("tokenizer.json");
        if json_path.exists() {
            self.tokenizer = Some(Tokenizer::from_file(&json_path)
                .map_err(|e| anyhow!("Failed to load tokenizer.json: {}", e))?);
            return Ok(());
        }

        // Fall back to SentencePiece tokenizer.model
        let sp_path = dir.join("tokenizer.model");
        if sp_path.exists() {
            let model = Unigram::load(&sp_path)
                .map_err(|e| anyhow!("Failed to load tokenizer.model: {}", e))?;
            self.tokenizer = Some(Tokenizer::new(model));
            return Ok(());
        }

        Err(anyhow!(
            "No tokenizer found (tried embedded assets and {:?})",
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

    /// Decode only new tokens incrementally
    ///
    /// This is more efficient for streaming as it only decodes new tokens
    /// since the last decode operation.
    ///
    /// # Arguments
    /// * `token_ids` - All accumulated token IDs
    /// * `already_decoded` - Number of tokens already decoded previously
    ///
    /// # Returns
    /// Text for the new tokens only
    pub fn decode_tokens_incremental(&self, token_ids: &[u32], already_decoded: usize) -> Result<String> {
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow!("Tokenizer not loaded. Call load_tokenizer() first."))?;

        if already_decoded >= token_ids.len() {
            // No new tokens
            return Ok(String::new());
        }

        // Decode only new tokens
        let new_tokens = &token_ids[already_decoded..];
        let new_text = tokenizer.decode(new_tokens, true)
            .map_err(|e| anyhow!("Failed to decode tokens: {}", e))?;

        Ok(new_text)
    }

    /// Greedy decoding with timestamps: Returns tokens with their frame-level alignment
    ///
    /// For each encoder timestep, predict the most likely token until blank is emitted.
    /// Each token is tagged with the encoder frame where it was produced.
    pub fn greedy_decode_with_timestamps(&self, encoder_out: &Tensor) -> Result<Vec<TokenWithTimestamp>> {
        let (batch_size, time_steps, _enc_dim) = encoder_out.dims3()?;

        if batch_size != 1 {
            return Err(anyhow!("Greedy decode currently only supports batch_size=1"));
        }

        let mut decoded = Vec::new();
        let mut pred_states = None;

        // Start with blank token
        let mut last_token = self.config.blank_id as u32;

        // Decode all timesteps
        debug!("  Decoding {} timesteps...", time_steps);

        for t in 0..time_steps {
            if t % 50 == 0 {
                debug!("  Progress: {}/{} timesteps, {} tokens decoded", t, time_steps, decoded.len());
            }

            // Inner loop: keep predicting until blank
            let mut inner_steps = 0;
            const MAX_INNER_STEPS: usize = 10;  // Reduced to prevent getting stuck
            let mut repetition_count = 0;
            let mut prev_inner_token = None;

            loop {
                inner_steps += 1;
                if inner_steps > MAX_INNER_STEPS {
                    // Reset predictor state and force blank to recover
                    pred_states = None;
                    last_token = self.config.blank_id as u32;
                    if t % 50 == 0 {
                        warn!("    WARNING: Hit max inner steps at timestep {}, resetting state", t);
                    }
                    break;
                }

                // Get encoder output at current timestep: [1, 1, enc_dim]
                let enc_t = encoder_out.narrow(1, t, 1)?;

                // Predictor input: previous token [1, 1]
                let pred_input = Tensor::new(&[last_token], encoder_out.device())?
                    .unsqueeze(0)?;

                // Run predictor
                let (pred_out, new_states) = self.predictor.forward(&pred_input, pred_states.as_ref())?;
                pred_states = Some(new_states);

                // Joint network
                let logits = self.joint.forward(&enc_t, &pred_out)?;
                let logits = logits.squeeze(0)?.squeeze(0)?.squeeze(0)?;
                let mut logits_f32 = logits.to_dtype(DType::F32)?;

                // Add small blank bias only after several inner steps to encourage termination
                let blank_bias = if inner_steps > 5 { 0.5 } else { 0.0 };
                if blank_bias > 0.0 {
                    let current_blank_logit = logits_f32.get(self.config.blank_id)?.to_scalar::<f32>()?;
                    let blank_tensor = Tensor::new(&[current_blank_logit + blank_bias], logits_f32.device())?;
                    logits_f32 = logits_f32.slice_assign(&[self.config.blank_id..self.config.blank_id+1], &blank_tensor)?;
                }

                // Mask out padding tokens 8193-8197 (only if they exist in this vocab)
                // Standard TDT has vocab_size=8192, streaming TDT has vocab_size=1024
                let joint_vocab = self.config.joint_vocab_size.unwrap_or(self.config.vocab_size);
                for i in 8193..8198 {
                    if i < joint_vocab {
                        let mask_tensor = Tensor::new(&[-1e9_f32], logits_f32.device())?;
                        logits_f32 = logits_f32.slice_assign(&[i..i+1], &mask_tensor)?;
                    }
                }

                let log_probs_masked = candle_nn::ops::log_softmax(&logits_f32, D::Minus1)?;
                let token_tensor = log_probs_masked.argmax(D::Minus1)?;
                let token = token_tensor.to_scalar::<u32>()?;

                // Repetition detection: if we see the same token 3+ times, force blank
                if let Some(prev_tok) = prev_inner_token {
                    if prev_tok == token && token != self.config.blank_id as u32 {
                        repetition_count += 1;
                        if repetition_count >= 3 {
                            // Force blank to break repetition loop
                            break;
                        }
                    } else {
                        repetition_count = 0;
                    }
                }
                prev_inner_token = Some(token);

                if token == self.config.blank_id as u32 {
                    // Blank: move to next timestep
                    break;
                } else if token >= self.config.vocab_size as u32 {
                    // Special token, treat as blank
                    break;
                } else {
                    // Valid token: emit with timestamp
                    decoded.push(TokenWithTimestamp {
                        token,
                        frame: t,  // Store the encoder frame
                    });
                    last_token = token;
                }
            }
        }

        debug!("  Decoded {} tokens total", decoded.len());
        Ok(decoded)
    }

    /// Add punctuation based on frame gaps between tokens
    ///
    /// Uses the TDT model's natural frame-level alignment to detect pauses:
    /// - Comma for short pauses (5-9 frames = 400-720ms)
    /// - Period for longer pauses (10+ frames = 800ms+)
    ///
    /// # Frame Timing
    /// - Mel frame: 10ms (160 samples / 16kHz)
    /// - Encoder frame: 80ms (8x downsampling)
    /// - So 5 frames = 400ms, 10 frames = 800ms
    pub fn add_punctuation_from_timestamps(&self, tokens_with_ts: &[TokenWithTimestamp]) -> Result<String> {
        if tokens_with_ts.is_empty() {
            return Ok(String::new());
        }

        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow!("Tokenizer not loaded. Call load_tokenizer() first."))?;

        // Configuration for punctuation insertion
        const COMMA_PAUSE_FRAMES: usize = 5;   // 400ms pause
        const PERIOD_PAUSE_FRAMES: usize = 10;  // 800ms pause

        // Group tokens into phrases based on pause locations
        // This allows proper subword decoding while inserting punctuation
        let mut phrases: Vec<Vec<u32>> = Vec::new();
        let mut current_phrase: Vec<u32> = Vec::new();
        let mut punctuation_marks: Vec<&str> = Vec::new();
        let mut prev_frame = tokens_with_ts[0].frame;

        for (i, token_with_ts) in tokens_with_ts.iter().enumerate() {
            let frame_gap = if i > 0 {
                token_with_ts.frame.saturating_sub(prev_frame)
            } else {
                0
            };

            // Check if we should end current phrase and add punctuation
            if frame_gap >= PERIOD_PAUSE_FRAMES {
                // Long pause - end phrase with period
                if !current_phrase.is_empty() {
                    phrases.push(current_phrase.clone());
                    punctuation_marks.push(". ");
                    current_phrase.clear();
                }
            } else if frame_gap >= COMMA_PAUSE_FRAMES {
                // Short pause - end phrase with comma
                if !current_phrase.is_empty() {
                    phrases.push(current_phrase.clone());
                    punctuation_marks.push(", ");
                    current_phrase.clear();
                }
            }

            current_phrase.push(token_with_ts.token);
            prev_frame = token_with_ts.frame;
        }

        // Add final phrase
        if !current_phrase.is_empty() {
            phrases.push(current_phrase);
            punctuation_marks.push("");  // No punctuation after last phrase (we'll add period at end)
        }

        // Decode each phrase and join with punctuation
        let mut result = String::new();
        for (phrase_tokens, punct) in phrases.iter().zip(punctuation_marks.iter()) {
            let phrase_text = tokenizer.decode(phrase_tokens, true)
                .map_err(|e| anyhow!("Failed to decode phrase: {}", e))?;

            result.push_str(phrase_text.trim());
            result.push_str(punct);
        }

        // Add final period if not already present
        let trimmed = result.trim();
        if !trimmed.is_empty() && !trimmed.ends_with('.') && !trimmed.ends_with('?') && !trimmed.ends_with('!') {
            result.push('.');
        }

        Ok(result)
    }

    /// Streaming greedy decode: Maintains predictor state across chunks
    ///
    /// This version accepts predictor state from previous chunks and returns
    /// the updated state for the next chunk.
    pub fn greedy_decode_streaming(
        &self,
        encoder_out: &Tensor,
        pred_states: Option<Vec<rnn::LSTMState>>,
        last_token: u32,
    ) -> Result<(Vec<u32>, Option<Vec<rnn::LSTMState>>, u32)> {
        let (batch_size, time_steps, _enc_dim) = encoder_out.dims3()?;

        if batch_size != 1 {
            return Err(anyhow!("Greedy decode currently only supports batch_size=1"));
        }

        let mut decoded = Vec::new();
        let mut pred_states = pred_states;
        let mut last_token = last_token;

        // Decode all timesteps
        for t in 0..time_steps {
            // Inner loop: keep predicting until blank
            let mut inner_steps = 0;
            const MAX_INNER_STEPS: usize = 10;
            let mut repetition_count = 0;
            let mut prev_inner_token = None;

            loop {
                inner_steps += 1;
                if inner_steps > MAX_INNER_STEPS {
                    // Reset predictor state and force blank to recover
                    pred_states = None;
                    last_token = self.config.blank_id as u32;
                    break;
                }

                // Get encoder output at current timestep: [1, 1, enc_dim]
                let enc_t = encoder_out.narrow(1, t, 1)?;

                // Predictor input: previous token [1, 1]
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
                let mut logits_f32 = logits.to_dtype(DType::F32)?;

                // Add small blank bias only after several inner steps to encourage termination
                let blank_bias = if inner_steps > 5 { 0.5 } else { 0.0 };
                if blank_bias > 0.0 {
                    let current_blank_logit = logits_f32.get(self.config.blank_id)?.to_scalar::<f32>()?;
                    let blank_tensor = Tensor::new(&[current_blank_logit + blank_bias], logits_f32.device())?;
                    logits_f32 = logits_f32.slice_assign(&[self.config.blank_id..self.config.blank_id+1], &blank_tensor)?;
                }

                // Mask out padding tokens (only if they exist in this vocab)
                let joint_vocab = self.config.joint_vocab_size.unwrap_or(self.config.vocab_size);
                for i in 8193..8198 {
                    if i < joint_vocab {
                        let mask_tensor = Tensor::new(&[-1e9_f32], logits_f32.device())?;
                        logits_f32 = logits_f32.slice_assign(&[i..i+1], &mask_tensor)?;
                    }
                }

                let log_probs_masked = candle_nn::ops::log_softmax(&logits_f32, D::Minus1)?;
                let token_tensor = log_probs_masked.argmax(D::Minus1)?;
                let token = token_tensor.to_scalar::<u32>()?;

                // Repetition detection: if we see the same token 3+ times, force blank
                if let Some(prev_tok) = prev_inner_token {
                    if prev_tok == token && token != self.config.blank_id as u32 {
                        repetition_count += 1;
                        if repetition_count >= 3 {
                            // Force blank to break repetition loop
                            break;
                        }
                    } else {
                        repetition_count = 0;
                    }
                }
                prev_inner_token = Some(token);

                if token == self.config.blank_id as u32 {
                    // Blank: move to next timestep
                    break;
                } else if token >= self.config.vocab_size as u32 {
                    // Special token beyond vocab (can't feed to predictor), treat as blank
                    break;
                } else {
                    // Valid vocabulary token: emit and continue at same timestep
                    decoded.push(token);
                    last_token = token;
                }
            }
        }

        Ok((decoded, pred_states, last_token))
    }

    /// Beam search streaming decode: Maintains beam hypotheses across chunks
    ///
    /// This version accepts beam state from previous chunks and returns
    /// the updated beam state for the next chunk. Returns tokens from the
    /// current best hypothesis.
    ///
    /// # Arguments
    /// * `encoder_out` - Encoder output for current chunk [1, T_chunk, D]
    /// * `beam_size` - Number of hypotheses to maintain
    /// * `beam_state` - Optional beam state from previous chunk
    ///
    /// # Returns
    /// * New tokens emitted in this chunk (from current best hypothesis)
    /// * Updated beam state for next chunk
    pub fn beam_decode_streaming(
        &self,
        encoder_out: &Tensor,
        beam_size: usize,
        beam_state: Option<BeamStreamingState>,
    ) -> Result<(Vec<u32>, BeamStreamingState)> {
        let (batch_size, time_steps, _enc_dim) = encoder_out.dims3()?;

        if batch_size != 1 {
            return Err(anyhow!("Beam decode streaming currently only supports batch_size=1"));
        }

        // Initialize or restore beam
        let mut beam = match beam_state {
            Some(state) => state.hypotheses,
            None => {
                // Start with blank token
                let init_state = self.predictor.init_states(1, encoder_out.device())?;
                vec![StreamingBeamHypothesis {
                    tokens: Vec::new(),
                    score: 0.0,
                    pred_state: init_state,
                    last_token: self.config.blank_id as u32,
                }]
            }
        };

        // Track how many tokens all hypotheses agreed on at the start
        // This is the "committed prefix" that we've already output
        let initial_token_count = beam.first().map(|h| h.tokens.len()).unwrap_or(0);

        // Process each timestep with beam search
        for t in 0..time_steps {
            let mut candidates = Vec::new();

            // Expand each hypothesis in the beam
            for hyp in &beam {
                // Get encoder output at current timestep
                let enc_t = encoder_out.narrow(1, t, 1)?;

                // Inner loop: predict tokens until blank
                let mut current_hyp = hyp.clone();
                const MAX_INNER_STEPS: usize = 10;

                for _inner_step in 0..MAX_INNER_STEPS {
                    // Predictor forward
                    let pred_input = Tensor::new(&[current_hyp.last_token], encoder_out.device())?
                        .unsqueeze(0)?;

                    let (pred_out, new_states) = self.predictor.forward(
                        &pred_input,
                        Some(&current_hyp.pred_state)
                    )?;

                    // Joint network
                    let logits = self.joint.forward(&enc_t, &pred_out)?;
                    let logits = logits.squeeze(0)?.squeeze(0)?.squeeze(0)?;
                    let mut logits_f32 = logits.to_dtype(DType::F32)?;

                    // Mask out padding tokens 8193-8197
                    let joint_vocab = self.config.joint_vocab_size.unwrap_or(self.config.vocab_size);
                    for i in 8193..8198 {
                        if i < joint_vocab {
                            let mask_tensor = Tensor::new(&[-1e9_f32], logits_f32.device())?;
                            logits_f32 = logits_f32.slice_assign(&[i..i+1], &mask_tensor)?;
                        }
                    }

                    let log_probs = candle_nn::ops::log_softmax(&logits_f32, D::Minus1)?;
                    let log_probs_vec: Vec<f32> = log_probs.to_vec1()?;

                    // Get best token (greedy within inner loop)
                    let (mut best_token, mut best_score) = (0usize, f32::NEG_INFINITY);
                    for (idx, &score) in log_probs_vec.iter().enumerate() {
                        if idx <= self.config.vocab_size && score > best_score {
                            best_token = idx;
                            best_score = score;
                        }
                    }

                    let token = best_token as u32;

                    if token == self.config.blank_id as u32 {
                        // Blank: add hypothesis to candidates and break inner loop
                        let mut blank_hyp = current_hyp.clone();
                        blank_hyp.score += best_score;
                        candidates.push(blank_hyp);
                        break;
                    } else if token < self.config.vocab_size as u32 {
                        // Non-blank: extend hypothesis and continue inner loop
                        current_hyp.tokens.push(token);
                        current_hyp.score += best_score;
                        current_hyp.pred_state = new_states;
                        current_hyp.last_token = token;
                    } else {
                        // Invalid token, treat as blank
                        let blank_hyp = current_hyp.clone();
                        candidates.push(blank_hyp);
                        break;
                    }
                }

                // Safety: if we didn't emit blank after MAX_INNER_STEPS, force it
                if candidates.is_empty() {
                    candidates.push(current_hyp);
                }
            }

            // Sort candidates by score and keep top beam_size
            candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            beam = candidates.into_iter().take(beam_size).collect();

            // Ensure we have at least one hypothesis
            if beam.is_empty() {
                let init_state = self.predictor.init_states(1, encoder_out.device())?;
                beam.push(StreamingBeamHypothesis {
                    tokens: Vec::new(),
                    score: 0.0,
                    pred_state: init_state,
                    last_token: self.config.blank_id as u32,
                });
            }
        }

        // Select best hypothesis
        let best_hyp = beam.iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .ok_or_else(|| anyhow!("No hypotheses in beam"))?;

        let new_tokens = if best_hyp.tokens.len() > initial_token_count {
            best_hyp.tokens[initial_token_count..].to_vec()
        } else {
            Vec::new()
        };

        // Normalize scores to prevent degradation across chunks
        // Find the best score and subtract it from all hypotheses
        let best_score = beam.iter().map(|h| h.score).fold(f32::NEG_INFINITY, f32::max);
        let mut normalized_beam = beam;
        for hyp in &mut normalized_beam {
            hyp.score -= best_score;
        }

        Ok((new_tokens, BeamStreamingState { hypotheses: normalized_beam, beam_size }))
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

        // Start with blank token from config (should be 0 for this model)
        let mut last_token = self.config.blank_id as u32;

        debug!("[DECODE] Starting greedy decode (blank_id={}, vocab_size={})",
                 self.config.blank_id, self.config.vocab_size);

        // Decode all timesteps now that special tokens are handled correctly
        debug!("  Decoding {} timesteps...", time_steps);

        for t in 0..time_steps {
            if t % 50 == 0 {
                debug!("  Progress: {}/{} timesteps, {} tokens decoded", t, time_steps, decoded.len());
            }

            // Inner loop: keep predicting until blank
            // Add safety limit to prevent infinite loops
            let mut inner_steps = 0;
            const MAX_INNER_STEPS: usize = 10;  // Reduced from 50 to prevent getting stuck
            let mut first_token_this_timestep = None;
            let mut repetition_count = 0;
            let mut prev_inner_token = None;

            loop {
                inner_steps += 1;
                if inner_steps > MAX_INNER_STEPS {
                    // Reset predictor state and force blank to recover
                    pred_states = None;
                    last_token = self.config.blank_id as u32;
                    if t % 50 == 0 {
                        warn!("    WARNING: Hit max inner steps at timestep {}, resetting state", t);
                    }
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
                let mut logits_f32 = logits.to_dtype(DType::F32)?;

                // Add small blank bias only after several inner steps to encourage termination
                // This helps prevent infinite loops while still allowing normal decoding
                let blank_bias = if inner_steps > 5 { 0.5 } else { 0.0 };
                if blank_bias > 0.0 {
                    let current_blank_logit = logits_f32.get(self.config.blank_id)?.to_scalar::<f32>()?;
                    let blank_tensor = Tensor::new(&[current_blank_logit + blank_bias], logits_f32.device())?;
                    logits_f32 = logits_f32.slice_assign(&[self.config.blank_id..self.config.blank_id+1], &blank_tensor)?;
                }

                // Mask out padding tokens 8193-8197 to prevent their use (only if they exist in this vocab)
                // Standard TDT: 0-8191 (content) + 8192 (blank), streaming TDT: 0-1023 (content) + 1024 (blank)
                let joint_vocab = self.config.joint_vocab_size.unwrap_or(self.config.vocab_size);
                for i in 8193..8198 {
                    if i < joint_vocab {
                        let mask_tensor = Tensor::new(&[-1e9_f32], logits_f32.device())?;
                        logits_f32 = logits_f32.slice_assign(&[i..i+1], &mask_tensor)?;
                    }
                }

                let log_probs_masked = candle_nn::ops::log_softmax(&logits_f32, D::Minus1)?;
                let token_tensor = log_probs_masked.argmax(D::Minus1)?;
                let token = token_tensor.to_scalar::<u32>()?;

                // Debug first 3 timesteps
                if t < 3 {
                    debug!("[DECODE t={} inner={}] Token selected: {} (blank={}, vocab_size={})",
                             t, inner_steps, token, self.config.blank_id, self.config.vocab_size);
                    if inner_steps == 1 {
                        match log_probs_masked.to_vec1() {
                            Ok(top_logits) => {
                                let mut indexed: Vec<(usize, f32)> = top_logits.iter().copied().enumerate().collect();
                                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                                debug!("  Top 5 predictions:");
                                for (idx, score) in indexed.iter().take(5) {
                                    debug!("    Token {}: {:.3}", idx, score);
                                }
                                // Check if token 800 is in top 20
                                let token_800_rank = indexed.iter().position(|(idx, _)| *idx == 800);
                                if let Some(rank) = token_800_rank {
                                    let score = indexed[rank].1;
                                    debug!("  Token 800 (NeMo's first): rank {}, score {:.3}", rank + 1, score);
                                } else {
                                    debug!("  Token 800 not in top predictions");
                                }
                            }
                            Err(e) => {
                                warn!("  ERROR extracting logits: {}", e);
                            }
                        }
                    }
                }

                // Repetition detection: if we see the same token 3+ times, force blank
                if let Some(prev_tok) = prev_inner_token {
                    if prev_tok == token && token != self.config.blank_id as u32 {
                        repetition_count += 1;
                        if repetition_count >= 3 {
                            // Force blank to break repetition loop
                            if t % 50 == 0 {
                                debug!("    Detected repetition of token {}, forcing blank", token);
                            }
                            break;
                        }
                    } else {
                        repetition_count = 0;
                    }
                }
                prev_inner_token = Some(token);

                // Debug first token of every 50th timestep
                if first_token_this_timestep.is_none() {
                    first_token_this_timestep = Some(token);
                }

                if token == self.config.blank_id as u32 {
                    // Blank: move to next timestep
                    if t < 3 {
                        debug!("[DECODE t={}] Emitted blank, moving to next timestep", t);
                    }
                    break;
                } else if token >= self.config.vocab_size as u32 {
                    // Special token beyond vocab (can't feed to predictor), treat as blank
                    break;
                } else {
                    // Valid vocabulary token: emit and continue at same timestep
                    decoded.push(token);
                    last_token = token;
                }
            }
        }

        debug!("  Decoded {} tokens total", decoded.len());
        Ok(decoded)
    }

    /// Beam search decoding: Explores multiple hypotheses in parallel
    ///
    /// Implements beam search with configurable beam size (typically 2).
    /// For each timestep, maintains the top-K best hypotheses.
    pub fn beam_decode(&self, encoder_out: &Tensor, beam_size: usize) -> Result<Vec<u32>> {
        let (batch_size, time_steps, _enc_dim) = encoder_out.dims3()?;

        if batch_size != 1 {
            return Err(anyhow!("Beam decode currently only supports batch_size=1"));
        }

        debug!("  Beam decoding (beam_size={}) {} timesteps...", beam_size, time_steps);

        // Initialize beam with single hypothesis
        let init_state = self.predictor.init_states(1, encoder_out.device())?;
        let mut beam = vec![BeamHypothesis {
            tokens: Vec::new(),
            score: 0.0,
            pred_state: init_state,
            last_token: self.config.blank_id as u32,
            timestep: 0,
        }];

        let mut completed = Vec::new();

        // Pre-allocate candidates vector to avoid repeated allocations
        // Worst case: beam_size hypotheses × MAX_INNER_STEPS candidates each
        let mut candidates: Vec<BeamHypothesis> = Vec::with_capacity(beam_size * 10);

        for t in 0..time_steps {
            /*
            if t % 50 == 0 {
                debug!("  Progress: {}/{} timesteps, beam size: {}, completed: {}",
                         t, time_steps, beam.len(), completed.len());
            }
            */

            candidates.clear();  // Reuse allocation from previous iteration

            // Expand each hypothesis in the beam
            for hyp in &beam {
                // Skip if this hypothesis has already passed this timestep
                if hyp.timestep > t {
                    candidates.push(hyp.clone());
                    continue;
                }

                // Get encoder output at current timestep
                let enc_t = encoder_out.narrow(1, t, 1)?;

                // Try extending with non-blank tokens (inner loop)
                let mut current_hyp = hyp.clone();
                const MAX_INNER_STEPS: usize = 10;
                let mut emitted_blank = false;

                for _inner_step in 0..MAX_INNER_STEPS {
                    // Predictor forward
                    let pred_input = Tensor::new(&[current_hyp.last_token], encoder_out.device())?
                        .unsqueeze(0)?;

                    let (pred_out, new_states) = self.predictor.forward(
                        &pred_input,
                        Some(&current_hyp.pred_state)
                    )?;

                    // Joint network
                    let logits = self.joint.forward(&enc_t, &pred_out)?;
                    let logits = logits.squeeze(0)?.squeeze(0)?.squeeze(0)?;
                    let mut logits_f32 = logits.to_dtype(DType::F32)?;

                    // Mask out padding tokens 8193-8197 (only if they exist in this vocab)
                    let joint_vocab = self.config.joint_vocab_size.unwrap_or(self.config.vocab_size);
                    for i in 8193..8198 {
                        if i < joint_vocab {
                            let mask_tensor = Tensor::new(&[-1e9_f32], logits_f32.device())?;
                            logits_f32 = logits_f32.slice_assign(&[i..i+1], &mask_tensor)?;
                        }
                    }

                    let log_probs = candle_nn::ops::log_softmax(&logits_f32, D::Minus1)?;
                    let log_probs_vec: Vec<f32> = log_probs.to_vec1()?;

                    // Get top token
                    let (mut best_token, mut best_score) = (0usize, f32::NEG_INFINITY);
                    for (idx, &score) in log_probs_vec.iter().enumerate() {
                        if idx <= self.config.vocab_size && score > best_score {
                            best_token = idx;
                            best_score = score;
                        }
                    }

                    let token = best_token as u32;

                    if token == self.config.blank_id as u32 {
                        // Blank: save current hypothesis with updated timestep
                        let mut blank_hyp = current_hyp.clone();
                        blank_hyp.timestep = t + 1;
                        blank_hyp.score += best_score;
                        candidates.push(blank_hyp);
                        emitted_blank = true;
                        break;
                    } else if token < self.config.vocab_size as u32 {
                        // Non-blank: update current hypothesis and continue
                        current_hyp.tokens.push(token);
                        current_hyp.score += best_score;
                        current_hyp.pred_state = new_states;
                        current_hyp.last_token = token;
                        // Continue inner loop to look for more tokens
                    } else {
                        // Invalid token, treat as blank
                        let mut blank_hyp = current_hyp.clone();
                        blank_hyp.timestep = t + 1;
                        candidates.push(blank_hyp);
                        emitted_blank = true;
                        break;
                    }
                }

                // If we didn't emit blank after MAX_INNER_STEPS, force it
                if !emitted_blank {
                    let mut forced_hyp = current_hyp;
                    forced_hyp.timestep = t + 1;
                    candidates.push(forced_hyp);
                }
            }

            // Keep top beam_size hypotheses
            candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            beam = candidates.drain(..beam_size.min(candidates.len())).collect();

            // Move completed hypotheses (reached end of encoder output)
            beam.retain(|hyp| {
                if hyp.timestep >= time_steps {
                    completed.push(hyp.clone());
                    false
                } else {
                    true
                }
            });

            // If beam is empty, we're done
            if beam.is_empty() {
                break;
            }
        }

        // Add remaining beam hypotheses to completed
        completed.extend(beam);

        // Return best hypothesis
        completed.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let best = completed.first()
            .ok_or_else(|| anyhow!("No hypotheses generated"))?;

        debug!("  Decoded {} tokens with beam search (score: {:.2})", best.tokens.len(), best.score);
        Ok(best.tokens.clone())
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
    #[serde(default)]
    pub blank_id: Option<usize>,  // Defaults to vocab_size if not present
    #[serde(default)]
    pub joint_vocab_size: Option<usize>,
    pub predictor_config: HfPredictorConfig,
    pub joint_config: HfJointConfig,
    #[serde(default)]
    pub streaming_config: Option<serde_json::Value>,  // Streaming-specific config (att_context_size, etc.)
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
            blank_id: hf.blank_id.unwrap_or(hf.vocab_size),  // Default: last position in vocab
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
    use std::io::{Error, ErrorKind};

    let dir = dir.as_ref();
    let dir_pathbuf = dir.to_path_buf();

    // Load config from embedded asset
    info!("  Loading config from embedded assets...");
    let cfg_bytes = TDT_CONFIG.bytes(&dir_pathbuf).map_err(|_| {
        Error::new(
            ErrorKind::Other,
            "failed to get decompressed bytes for TDT_CONFIG",
        )
    })?;
    let hf_cfg: HfTransducerConfig = serde_json::from_slice(cfg_bytes).map_err(|e| {
        Error::new(
            ErrorKind::Other,
            format!("failed to parse TDT_CONFIG as JSON: {e}"),
        )
    })?;
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
        blank_id: hf_cfg.blank_id.unwrap_or(hf_cfg.vocab_size),
    };

    // Load model weights
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16  // Use BF16 on GPU (matches training dtype)
    };

    info!("Loading TDT model with {:?} dtype", dtype);

    // Load model weights from embedded asset
    info!("  Loading model weights from embedded assets...");
    let model_bytes = TDT_MODEL.bytes(&dir_pathbuf).map_err(|_| {
        Error::new(
            ErrorKind::Other,
            "failed to get decompressed bytes for TDT_MODEL",
        )
    })?;

    // Load safetensors from memory
    let tensors_raw: HashMap<String, Tensor> =
        candle_core::safetensors::load_buffer(model_bytes, device)?;

    info!("  Loading and remapping NeMo tensors...");

    // Remap tensor names from NeMo format to our expected format
    let mut tensors = HashMap::new();
    for (nemo_name, tensor) in tensors_raw {
        let our_name = remap_nemo_tensor_name(&nemo_name);
        if our_name != nemo_name {
            debug!("    {} -> {}", nemo_name, our_name);
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
/// Load Parakeet TDT (Transducer) model from GGUF quantized format
///
/// # Arguments
/// * `dir` - Directory containing quantized model assets
/// * `device` - Device to load model on
///
/// Expected files in directory:
/// - `parakeet-tdt-config.json.zst` - Model configuration (compressed)
/// - `parakeet-tdt-model_q8_0.gguf.zst` - Quantized model weights (compressed)
/// - `parakeet-tdt-tokenizer.json.zst` - Tokenizer (compressed)
///
/// # Example
/// ```no_run
/// use speech::parakeet::transducer::{load_parakeet_tdt_from_gguf_local, TransducerModel};
/// use speech::parakeet::get_device;
/// let device = get_device()?;
/// let model = load_parakeet_tdt_from_gguf_local("assets", &device)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn load_parakeet_tdt_from_gguf_local<P: AsRef<Path>>(
    dir: P,
    device: &Device,
) -> Result<TransducerModel> {
    use std::io::{Error, ErrorKind};

    let assets = dir.as_ref().to_path_buf();

    info!("Loading TDT model with Q8_0 quantization (recommended, compressed)");

    // Load config from embedded asset
    info!("  Loading config from assets...");
    let cfg_bytes = TDT_CONFIG.bytes(&assets).map_err(|_| {
        Error::new(
            ErrorKind::Other,
            format!("Failed to load parakeet-tdt-config.json.zst from {:?}\n\
                     \nMissing model files? Download with:\n\
                     python scripts/download_parakeet_tdt.py",
                    assets.join("parakeet-tdt-config.json.zst")),
        )
    })?;
    let hf_cfg: HfTransducerConfig = serde_json::from_slice(cfg_bytes).map_err(|e| {
        Error::new(
            ErrorKind::Other,
            format!("failed to parse TDT_CONFIG as JSON: {e}"),
        )
    })?;
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
        blank_id: hf_cfg.blank_id.unwrap_or(hf_cfg.vocab_size),
    };

    // Load tokenizer from embedded asset
    info!("  Loading tokenizer from assets...");
    let tokenizer = if let Ok(tok_bytes) = TDT_TOKENIZER_JSON.bytes(&assets) {
        // JSON format available (preferred - faster loading)
        Tokenizer::from_bytes(tok_bytes)
            .map_err(|e| Error::new(ErrorKind::Other, format!("failed to parse TDT_TOKENIZER_JSON: {e}")))?
    } else {
        // JSON format not found - check if model format exists
        if TDT_TOKENIZER.bytes(&assets).is_ok() {
            return Err(anyhow!(
                "Found parakeet-tdt-tokenizer.model.zst but missing parakeet-tdt-tokenizer.json.zst\n\
                 \nThe JSON format is required for loading. Please install Python dependencies and re-run:\n\
                 \n  pip install sentencepiece tokenizers\n\
                 python scripts/download_parakeet_tdt.py\n\
                 \nThis will create the required parakeet-tdt-tokenizer.json.zst file."
            ));
        } else {
            return Err(anyhow!(
                "Failed to load tokenizer from {:?}\n\
                 \nMissing file: parakeet-tdt-tokenizer.json.zst\n\
                 \nDownload with:\n\
                 pip install sentencepiece tokenizers\n\
                 python scripts/download_parakeet_tdt.py",
                assets
            ));
        }
    };

    // Load GGUF from embedded asset (already decompressed by embed_zst_asset macro)
    info!("  Loading GGUF file from assets...");
    let gguf_bytes = TDT_MODEL_Q8_0_GGUF.bytes(&assets).map_err(|_| {
        Error::new(
            ErrorKind::Other,
            format!("Failed to load parakeet-tdt-model_q8_0.gguf.zst from {:?}\n\
                     \nMissing model files? Download with:\n\
                     python scripts/download_parakeet_tdt.py\n\
                     \nNote: This is a large file (~650MB compressed)",
                    assets.join("parakeet-tdt-model_q8_0.gguf.zst")),
        )
    })?;

    // For quantized models, we need to dequantize to FP32/BF16 for inference
    // TDT uses LSTM which doesn't support quantized operations yet
    info!("  Dequantizing GGUF to tensors...");

    // Determine target dtype
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16  // Use BF16 on GPU (matches training dtype)
    };

    debug!("    Target dtype: {:?}", dtype);

    // Load GGUF and dequantize tensors
    let gguf_file = candle_core::quantized::gguf_file::Content::read(&mut std::io::Cursor::new(gguf_bytes))?;
    let mut tensors = HashMap::new();

    for (nemo_name, _) in gguf_file.tensor_infos.iter() {
        let qtensor = gguf_file.tensor(&mut std::io::Cursor::new(gguf_bytes), nemo_name, device)?;
        let tensor = qtensor.dequantize(device)?;

        // Convert to target dtype if needed
        let tensor = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)?
        } else {
            tensor
        };

        // Remap NeMo tensor names to our format
        let our_name = remap_nemo_tensor_name(nemo_name);
        tensors.insert(our_name.clone(), tensor.clone());

        // Add zero biases for layers that need them (NeMo doesn't have biases)
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
                let out_features = tensor.dims()[0];
                let zero_bias = Tensor::zeros(out_features, dtype, device)?;
                tensors.insert(bias_name, zero_bias);
            }
        }

        // Add batch norm statistics if this is a norm.weight tensor (NeMo doesn't include them for inference)
        if our_name.ends_with(".norm.weight") && our_name.contains(".conv.norm") {
            let num_features = tensor.dims()[0];

            // Add running_mean (zeros for eval mode)
            let mean_name = our_name.replace(".weight", ".running_mean");
            if !tensors.contains_key(&mean_name) {
                let zeros = Tensor::zeros(num_features, dtype, device)?;
                tensors.insert(mean_name, zeros);
            }

            // Add running_var (ones for eval mode - std dev = 1)
            let var_name = our_name.replace(".weight", ".running_var");
            if !tensors.contains_key(&var_name) {
                let ones = Tensor::ones(num_features, dtype, device)?;
                tensors.insert(var_name, ones);
            }

            // Add num_batches_tracked (optional, but some implementations expect it)
            let num_batches_name = our_name.replace(".weight", ".num_batches_tracked");
            if !tensors.contains_key(&num_batches_name) {
                let zero = Tensor::zeros((), DType::I64, device)?;
                tensors.insert(num_batches_name, zero);
            }
        }
    }

    info!("    ✓ Dequantized {} tensors", tensors.len());
    info!("    ✓ Added zero biases and batch norm stats for NeMo compatibility");

    let vb = VarBuilder::from_tensors(tensors, dtype, device);

    // Build encoder
    info!("  Building encoder...");
    let encoder = FastConformerEncoder::new(encoder_cfg.clone(), vb.pp("encoder"))?;

    // Build Triton encoder (optional, Metal only)
    #[cfg(feature = "triton-metal")]
    let triton_encoder = {
        if let Device::Metal(_md) = device {
            match TritonParakeetEncoder::new(encoder_cfg.clone(), vb.pp("encoder")) {
                Ok(te) => {
                    info!("  Triton encoder loaded");
                    Some(te)
                }
                Err(e) => {
                    info!("  Triton encoder unavailable: {e}");
                    None
                }
            }
        } else {
            None
        }
    };

    // Build full transducer model
    info!("  Building transducer model...");
    let mut model = TransducerModel::new(
        encoder,
        tdt_cfg,
        encoder_cfg.d_model,
        vb,
    )?;

    #[cfg(feature = "triton-metal")]
    {
        model.triton_encoder = triton_encoder;
    }

    // Store tokenizer
    model.tokenizer = Some(tokenizer);

    info!("  ✓ TDT model loaded successfully (quantized)");

    Ok(model)
}

/// Load Parakeet TDT (Transducer) model from memory-mapped GGUF
///
/// This loader uses memory-mapping for efficient access to GGUF weights.
/// Instead of loading the entire compressed file and decompressing it,
/// this directly mmaps the uncompressed GGUF file for zero-copy access.
///
/// # Benefits
/// - Lower memory usage: GGUF tensors are read directly from disk
/// - Faster loading: No decompression overhead
/// - OS manages paging: Only loads needed portions into RAM
///
/// # Arguments
/// * `dir` - Directory containing model assets
/// * `device` - Device to load model on
///
/// # Example
/// ```no_run
/// use speech::parakeet::transducer::{load_parakeet_tdt_from_gguf_mmap_local, TransducerModel};
/// use speech::parakeet::get_device;
/// let device = get_device()?;
/// let model = load_parakeet_tdt_from_gguf_mmap_local("assets", &device)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn load_parakeet_tdt_from_gguf_mmap_local<P: AsRef<Path>>(
    dir: P,
    device: &Device,
) -> Result<TransducerModel> {
    use std::io::{Error, ErrorKind};

    let assets = dir.as_ref().to_path_buf();

    info!("Loading TDT model with Q8_0 quantization (mmap, zero-copy)");

    // Load config from embedded asset
    info!("  Loading config from assets...");
    let cfg_bytes = TDT_CONFIG.bytes(&assets).map_err(|_| {
        Error::new(
            ErrorKind::Other,
            "failed to get decompressed bytes for TDT_CONFIG",
        )
    })?;
    let hf_cfg: HfTransducerConfig = serde_json::from_slice(cfg_bytes).map_err(|e| {
        Error::new(
            ErrorKind::Other,
            format!("failed to parse TDT_CONFIG as JSON: {e}"),
        )
    })?;
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
        blank_id: hf_cfg.blank_id.unwrap_or(hf_cfg.vocab_size),
    };

    // Load tokenizer from embedded asset
    info!("  Loading tokenizer from assets...");
    let tokenizer = if let Ok(tok_bytes) = TDT_TOKENIZER_JSON.bytes(&assets) {
        Tokenizer::from_bytes(tok_bytes)
            .map_err(|e| Error::new(ErrorKind::Other, format!("failed to parse TDT_TOKENIZER_JSON: {e}")))?
    } else {
        if TDT_TOKENIZER.bytes(&assets).is_ok() {
            return Err(anyhow!(
                "Found parakeet-tdt-tokenizer.model.zst but missing parakeet-tdt-tokenizer.json.zst\n\
                 \nThe JSON format is required. Please install Python dependencies and re-run:\n\
                 pip install sentencepiece tokenizers\n\
                 python scripts/download_parakeet_tdt.py"
            ));
        } else {
            return Err(anyhow!(
                "Missing tokenizer file: parakeet-tdt-tokenizer.json.zst\n\
                 \nDownload with:\n\
                 pip install sentencepiece tokenizers\n\
                 python scripts/download_parakeet_tdt.py"
            ));
        }
    };

    // Memory-map GGUF file (uncompressed)
    info!("  Memory-mapping GGUF file from assets...");
    let gguf_bytes = TDT_MODEL_Q8_0_GGUF_MMAP.bytes(&assets).map_err(|_| {
        Error::new(
            ErrorKind::Other,
            "failed to mmap TDT_MODEL_Q8_0_GGUF_MMAP",
        )
    })?;

    // For quantized models, we need to dequantize to FP32/BF16 for inference
    // TDT uses LSTM which doesn't support quantized operations yet
    info!("  Dequantizing GGUF to tensors (from mmap)...");

    // Determine target dtype
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16  // Use BF16 on GPU (matches training dtype)
    };

    debug!("    Target dtype: {:?}", dtype);

    // Load GGUF from mmap'd bytes
    // Strategy: Only dequantize predictor (LSTM) weights upfront
    // Dequantize encoder/joint weights lazily to reduce peak memory
    let gguf_file = candle_core::quantized::gguf_file::Content::read(&mut std::io::Cursor::new(gguf_bytes))?;
    let mut tensors = HashMap::new();
    let mut deferred_tensors: Vec<(String, candle_core::quantized::QTensor)> = Vec::new();

    for (nemo_name, _) in gguf_file.tensor_infos.iter() {
        let qtensor = gguf_file.tensor(&mut std::io::Cursor::new(gguf_bytes), nemo_name, device)?;
        let our_name = remap_nemo_tensor_name(nemo_name);

        // Only dequantize predictor tensors immediately (LSTM requires it)
        let should_dequantize_now = our_name.contains("predictor.");

        let tensor = if should_dequantize_now {
            debug!("    Dequantizing now: {}", our_name);
            qtensor.dequantize(device)?
        } else {
            // Defer dequantization for encoder/joint - store QTensor for later
            debug!("    Deferring dequantization: {}", our_name);
            deferred_tensors.push((our_name.clone(), qtensor));
            continue;  // Skip further processing for now
        };

        // Convert to target dtype if needed
        let tensor = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)?
        } else {
            tensor
        };

        // Remap NeMo tensor names to our format
        let our_name = remap_nemo_tensor_name(nemo_name);
        tensors.insert(our_name.clone(), tensor.clone());

        // Add zero biases for layers that need them (NeMo doesn't have biases)
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
                let out_features = tensor.dims()[0];
                let zero_bias = Tensor::zeros(out_features, dtype, device)?;
                tensors.insert(bias_name, zero_bias);
            }
        }

        // Add batch norm statistics if this is a norm.weight tensor (NeMo doesn't include them for inference)
        if our_name.ends_with(".norm.weight") && our_name.contains(".conv.norm") {
            let num_features = tensor.dims()[0];

            // Add running_mean (zeros for eval mode)
            let mean_name = our_name.replace(".weight", ".running_mean");
            if !tensors.contains_key(&mean_name) {
                let zeros = Tensor::zeros(num_features, dtype, device)?;
                tensors.insert(mean_name, zeros);
            }

            // Add running_var (ones for eval mode - std dev = 1)
            let var_name = our_name.replace(".weight", ".running_var");
            if !tensors.contains_key(&var_name) {
                let ones = Tensor::ones(num_features, dtype, device)?;
                tensors.insert(var_name, ones);
            }

            // Add num_batches_tracked (optional, but some implementations expect it)
            let num_batches_name = our_name.replace(".weight", ".num_batches_tracked");
            if !tensors.contains_key(&num_batches_name) {
                let zero = Tensor::zeros((), DType::I64, device)?;
                tensors.insert(num_batches_name, zero);
            }
        }
    }

    info!("    ✓ Dequantized {} predictor tensors immediately", tensors.len());
    info!("    ✓ Deferred {} encoder/joint tensors for lazy dequantization", deferred_tensors.len());

    // Now dequantize deferred tensors (encoder/joint) lazily as needed
    // This reduces peak memory since we process them one at a time
    for (our_name, qtensor) in deferred_tensors {
        let tensor = qtensor.dequantize(device)?;

        // Convert to target dtype if needed
        let tensor = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)?
        } else {
            tensor
        };

        // Remap NeMo tensor names to our format (already done above)
        tensors.insert(our_name.clone(), tensor.clone());

        // Add zero biases for layers that need them (NeMo doesn't have biases)
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
                let out_features = tensor.dims()[0];
                let zero_bias = Tensor::zeros(out_features, dtype, device)?;
                tensors.insert(bias_name, zero_bias);
            }
        }

        // Add batch norm statistics if this is a norm.weight tensor (NeMo doesn't include them for inference)
        if our_name.ends_with(".norm.weight") && our_name.contains(".conv.norm") {
            let num_features = tensor.dims()[0];

            // Add running_mean (zeros for eval mode)
            let mean_name = our_name.replace(".weight", ".running_mean");
            if !tensors.contains_key(&mean_name) {
                let zeros = Tensor::zeros(num_features, dtype, device)?;
                tensors.insert(mean_name, zeros);
            }

            // Add running_var (ones for eval mode - std dev = 1)
            let var_name = our_name.replace(".weight", ".running_var");
            if !tensors.contains_key(&var_name) {
                let ones = Tensor::ones(num_features, dtype, device)?;
                tensors.insert(var_name, ones);
            }

            // Add num_batches_tracked (optional, but some implementations expect it)
            let num_batches_name = our_name.replace(".weight", ".num_batches_tracked");
            if !tensors.contains_key(&num_batches_name) {
                let zero = Tensor::zeros((), DType::I64, device)?;
                tensors.insert(num_batches_name, zero);
            }
        }
    }

    info!("    ✓ Total tensors: {}", tensors.len());
    info!("    ✓ Added zero biases and batch norm stats for NeMo compatibility");

    let vb = VarBuilder::from_tensors(tensors, dtype, device);

    // Build encoder
    info!("  Building encoder...");
    let encoder = FastConformerEncoder::new(encoder_cfg.clone(), vb.pp("encoder"))?;

    // Build Triton encoder (optional, Metal only)
    #[cfg(feature = "triton-metal")]
    let triton_encoder = {
        if let Device::Metal(_md) = device {
            match TritonParakeetEncoder::new(encoder_cfg.clone(), vb.pp("encoder")) {
                Ok(te) => {
                    info!("  Triton encoder loaded");
                    Some(te)
                }
                Err(e) => {
                    info!("  Triton encoder unavailable: {e}");
                    None
                }
            }
        } else {
            None
        }
    };

    // Build full transducer model
    info!("  Building transducer model...");
    let mut model = TransducerModel::new(
        encoder,
        tdt_cfg,
        encoder_cfg.d_model,
        vb,
    )?;

    #[cfg(feature = "triton-metal")]
    {
        model.triton_encoder = triton_encoder;
    }

    // Store tokenizer
    model.tokenizer = Some(tokenizer);

    info!("  ✓ TDT model loaded successfully (quantized, mmap)");

    Ok(model)
}

/// Load Parakeet Streaming TDT from local directory (BF16 safetensors)
///
/// Loads the full-precision streaming model for maximum accuracy.
/// Use this for debugging and comparison against NeMo.
///
/// # Arguments
/// * `dir` - Directory containing model files
/// * `device` - Device to load model on
///
/// Expected files:
/// - `config.json` - Model configuration
/// - `model.safetensors` - Model weights (BF16)
/// - `tokenizer.json` or `tokenizer.model` - Tokenizer
pub fn load_parakeet_streaming_tdt_from_local<P: AsRef<Path>>(
    dir: P,
    device: &Device,
) -> Result<TransducerModel> {
    let dir = dir.as_ref();
    let config_path = dir.join("config.json");

    // Load config
    let cfg_json = std::fs::read_to_string(&config_path)?;
    let hf_cfg: HfTransducerConfig = serde_json::from_str(&cfg_json)?;
    let tdt_cfg = TransducerConfig::from_hf(&hf_cfg);

    let enc = &hf_cfg.encoder_config;

    // Load model weights
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    };

    info!("Loading Streaming TDT model (BF16 safetensors)");
    debug!("  Device: {:?}, dtype: {:?}", device, dtype);

    // Load safetensors
    let weights_path = dir.join("model.safetensors");
    let tensors_raw: HashMap<String, Tensor> = candle_core::safetensors::load(&weights_path, device)?;
    debug!("  Loaded {} tensors from safetensors", tensors_raw.len());

    // Remap tensor names and detect actual parameters
    let mut tensors = HashMap::new();
    let mut actual_flatten_dim = None;
    let mut actual_joint_vocab = None;

    for (nemo_name, tensor) in tensors_raw {
        let our_name = remap_nemo_tensor_name(&nemo_name);

        // Detect parameters from tensor shapes
        if nemo_name == "encoder.pre_encode.out.weight" {
            let dims = tensor.dims();
            if dims.len() == 2 {
                actual_flatten_dim = Some(dims[1]);
            }
        }
        if nemo_name == "joint.joint_net.2.weight" {
            let dims = tensor.dims();
            if dims.len() == 2 {
                actual_joint_vocab = Some(dims[0]);
            }
        }

        // Convert to target dtype
        let tensor_converted = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)?
        } else {
            tensor
        };
        tensors.insert(our_name.clone(), tensor_converted.clone());

        // Add zero biases for NeMo compatibility
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
                let out_features = tensor_converted.dims()[0];
                let zero_bias = Tensor::zeros(out_features, dtype, device)?;
                tensors.insert(bias_name, zero_bias);
            }
        }

        // Add batch norm statistics if needed
        if our_name.ends_with(".norm.weight") && our_name.contains(".conv.norm") {
            let num_features = tensor_converted.dims()[0];

            let mean_name = our_name.replace(".weight", ".running_mean");
            if !tensors.contains_key(&mean_name) {
                let zeros = Tensor::zeros(num_features, dtype, device)?;
                tensors.insert(mean_name, zeros);
            }

            let var_name = our_name.replace(".weight", ".running_var");
            if !tensors.contains_key(&var_name) {
                let ones = Tensor::ones(num_features, dtype, device)?;
                tensors.insert(var_name, ones);
            }

            let num_batches_name = our_name.replace(".weight", ".num_batches_tracked");
            if !tensors.contains_key(&num_batches_name) {
                let zero = Tensor::zeros((), DType::I64, device)?;
                tensors.insert(num_batches_name, zero);
            }
        }
    }

    // Calculate correct feat_in from actual tensor dimensions
    let feat_in = if let Some(flatten_dim) = actual_flatten_dim {
        let features_per_channel = flatten_dim / enc.subsampling_conv_channels;
        let calculated = features_per_channel * enc.subsampling_factor;
        debug!("  Detected feat_in: {} (config says {})", calculated, enc.num_mel_bins);
        calculated
    } else {
        enc.num_mel_bins
    };

    // Adjust vocab size and blank_id based on actual model
    let mut tdt_cfg_adjusted = tdt_cfg.clone();
    if let Some(detected_vocab) = actual_joint_vocab {
        if detected_vocab != tdt_cfg.vocab_size {
            debug!("  Detected joint vocab: {} (config says {})", detected_vocab, tdt_cfg.vocab_size);
            tdt_cfg_adjusted.joint_vocab_size = Some(detected_vocab);

            // CRITICAL FIX: Config has wrong blank_id=0
            // NeMo uses blank_idx=1024 (last position in vocab)
            // Predictor embedding has 1025 entries [0-1024], where 1024 is blank
            tdt_cfg_adjusted.blank_id = detected_vocab - 1;  // 1025 - 1 = 1024
            debug!("  Fixed blank_id: {} (config says {})", tdt_cfg_adjusted.blank_id, tdt_cfg.blank_id);
        }
    }

    let encoder_cfg = FastConformerConfig {
        feat_in,
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
        blank_id: hf_cfg.blank_id.unwrap_or(hf_cfg.vocab_size),
    };

    let vb = VarBuilder::from_tensors(tensors, dtype, device);

    info!("  Building encoder...");
    let encoder = FastConformerEncoder::new(encoder_cfg.clone(), vb.pp("encoder"))?;

    info!("  Building transducer model...");
    let mut model = TransducerModel::new(
        encoder,
        tdt_cfg_adjusted,
        encoder_cfg.d_model,
        vb,
    )?;

    // Load tokenizer
    let json_path = dir.join("tokenizer.json");
    let sp_path = dir.join("tokenizer.model");

    if json_path.exists() {
        model.tokenizer = Some(Tokenizer::from_file(&json_path)
            .map_err(|e| anyhow!("Failed to load tokenizer.json: {}", e))?);
    } else if sp_path.exists() {
        let tokenizer_model = Unigram::load(&sp_path)
            .map_err(|e| anyhow!("Failed to load tokenizer.model: {}", e))?;
        model.tokenizer = Some(Tokenizer::new(tokenizer_model));
    } else {
        return Err(anyhow!("No tokenizer found in {:?}", dir));
    }

    info!("  ✓ Streaming TDT model loaded successfully (BF16)");

    Ok(model)
}

/// Load Parakeet Streaming TDT (Cache-Aware Transducer) model from GGUF quantized format
///
/// This loads the cache-aware streaming variant of Parakeet TDT:
/// - Zero overlapping computations via attention and convolution caching
/// - Configurable chunk sizes (70-1190ms via att_context_size parameter)
/// - Built-in punctuation and capitalization
///
/// # Arguments
/// * `dir` - Directory containing quantized model assets
/// * `device` - Device to load model on
///
/// Expected files in directory:
/// - `parakeet-streaming-tdt-config.json.zst` - Model configuration (compressed)
/// - `parakeet-streaming-tdt-model_q8_0.gguf.zst` - Quantized model weights (compressed)
/// - `parakeet-streaming-tdt-tokenizer.json.zst` - Tokenizer (compressed)
///
/// # Example
/// ```no_run
/// use speech::parakeet::{load_parakeet_streaming_tdt_from_gguf_local, get_device};
/// let device = get_device()?;
/// let model = load_parakeet_streaming_tdt_from_gguf_local("assets", &device)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
/*
pub fn load_parakeet_streaming_tdt_from_gguf_local<P: AsRef<Path>>(
    dir: P,
    device: &Device,
) -> Result<TransducerModel> {
    use std::io::{Error, ErrorKind};

    let assets = dir.as_ref().to_path_buf();

    info!("Loading Streaming TDT model with Q8_0 quantization (cache-aware, compressed)");

    // Load config from embedded asset
    info!("  Loading config from assets...");
    let cfg_bytes = STREAMING_TDT_CONFIG.bytes(&assets).map_err(|_| {
        Error::new(
            ErrorKind::Other,
            "failed to get decompressed bytes for STREAMING_TDT_CONFIG",
        )
    })?;
    let hf_cfg: HfTransducerConfig = serde_json::from_slice(cfg_bytes).map_err(|e| {
        Error::new(
            ErrorKind::Other,
            format!("failed to parse STREAMING_TDT_CONFIG as JSON: {e}"),
        )
    })?;
    let tdt_cfg = TransducerConfig::from_hf(&hf_cfg);

    // Store encoder config reference for later use
    let enc_config_ref = &hf_cfg.encoder_config;

    // Parse streaming config
    let streaming_config = hf_cfg.streaming_config.clone();
    if let Some(ref config) = streaming_config {
        if let Some(att_context_sizes) = config.get("att_context_size") {
            info!("  ✓ Cache-aware model with configurable chunk sizes:");
            if let Some(sizes) = att_context_sizes.as_array() {
                for size_arr in sizes {
                    if let Some(arr) = size_arr.as_array() {
                        if arr.len() == 2 {
                            let left = arr[0].as_i64().unwrap_or(0);
                            let right = arr[1].as_i64().unwrap_or(0);
                            let chunk_ms = (right + 1) * 80; // Each frame is 80ms after 8x subsampling
                            debug!("    [{}, {}] = {}ms chunks", left, right, chunk_ms);
                        }
                    }
                }
            }
        }
    }

    // Load tokenizer from embedded asset
    info!("  Loading tokenizer from assets...");
    let tokenizer = if let Ok(tok_bytes) = STREAMING_TDT_TOKENIZER_JSON.bytes(&assets) {
        Tokenizer::from_bytes(tok_bytes)
            .map_err(|e| Error::new(ErrorKind::Other, format!("failed to parse STREAMING_TDT_TOKENIZER_JSON: {e}")))?
    } else {
        if STREAMING_TDT_TOKENIZER.bytes(&assets).is_ok() {
            return Err(anyhow!(
                "Found parakeet-streaming-tdt-tokenizer.model.zst but missing parakeet-streaming-tdt-tokenizer.json.zst\n\
                 \nThe JSON format is required. Please install Python dependencies and re-run:\n\
                 pip install sentencepiece tokenizers\n\
                 python scripts/download_parakeet_tdt.py"
            ));
        } else {
            return Err(anyhow!(
                "Missing tokenizer file: parakeet-streaming-tdt-tokenizer.json.zst\n\
                 \nDownload with:\n\
                 pip install sentencepiece tokenizers\n\
                 python scripts/download_parakeet_tdt.py"
            ));
        }
    };

    // Load GGUF from embedded asset (already decompressed by embed_zst_asset macro)
    info!("  Loading GGUF file from assets...");
    let gguf_bytes = STREAMING_TDT_MODEL_Q8_0_GGUF.bytes(&assets).map_err(|_| {
        Error::new(
            ErrorKind::Other,
            "failed to load STREAMING_TDT_MODEL_Q8_0_GGUF",
        )
    })?;

    // For quantized models, we need to dequantize to FP32/BF16 for inference
    // TDT uses LSTM which doesn't support quantized operations yet
    info!("  Dequantizing GGUF to tensors...");

    // Determine target dtype
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16  // Use BF16 on GPU (matches training dtype)
    };

    debug!("    Target dtype: {:?}", dtype);

    // Load GGUF and dequantize tensors
    let gguf_file = candle_core::quantized::gguf_file::Content::read(&mut std::io::Cursor::new(gguf_bytes))?;
    let mut tensors = HashMap::new();

    for (nemo_name, _) in gguf_file.tensor_infos.iter() {
        let qtensor = gguf_file.tensor(&mut std::io::Cursor::new(gguf_bytes), nemo_name, device)?;
        let tensor = qtensor.dequantize(device)?;

        // Convert to target dtype if needed
        let tensor = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)?
        } else {
            tensor
        };

        // Remap NeMo tensor names to our format
        let our_name = remap_nemo_tensor_name(nemo_name);
        tensors.insert(our_name.clone(), tensor.clone());

        // Add zero biases for layers that need them (NeMo doesn't have biases)
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
                let out_features = tensor.dims()[0];
                let zero_bias = Tensor::zeros(out_features, dtype, device)?;
                tensors.insert(bias_name, zero_bias);
            }
        }

        // Add batch norm statistics if this is a norm.weight tensor (NeMo doesn't include them for inference)
        if our_name.ends_with(".norm.weight") && our_name.contains(".conv.norm") {
            let num_features = tensor.dims()[0];

            // Add running_mean (zeros for eval mode)
            let mean_name = our_name.replace(".weight", ".running_mean");
            if !tensors.contains_key(&mean_name) {
                let zeros = Tensor::zeros(num_features, dtype, device)?;
                tensors.insert(mean_name, zeros);
            }

            // Add running_var (ones for eval mode - std dev = 1)
            let var_name = our_name.replace(".weight", ".running_var");
            if !tensors.contains_key(&var_name) {
                let ones = Tensor::ones(num_features, dtype, device)?;
                tensors.insert(var_name, ones);
            }

            // Add num_batches_tracked (optional, but some implementations expect it)
            let num_batches_name = our_name.replace(".weight", ".num_batches_tracked");
            if !tensors.contains_key(&num_batches_name) {
                let zero = Tensor::zeros((), DType::I64, device)?;
                tensors.insert(num_batches_name, zero);
            }
        }
    }

    info!("    ✓ Dequantized {} tensors", tensors.len());
    info!("    ✓ Added zero biases and batch norm stats for NeMo compatibility");

    // Construct encoder config with corrected feat_in (calculated from actual tensor dimensions)
    // The NeMo streaming model uses feat_in=136 but config says 128
    let mut actual_flatten_dim = None;
    if let Some(info) = gguf_file.tensor_infos.get("encoder.pre_encode.out.weight") {
        let dims = info.shape.dims();
        if dims.len() == 2 {
            actual_flatten_dim = Some(dims[1]);
            debug!("  Detected actual flatten_dim from model: {}", dims[1]);
        }
    }

    let feat_in = if let Some(flatten_dim) = actual_flatten_dim {
        let features_per_channel = flatten_dim / enc_config_ref.subsampling_conv_channels;
        let calculated_feat_in = features_per_channel * enc_config_ref.subsampling_factor;
        debug!("  Calculated feat_in from actual dimensions: {}", calculated_feat_in);
        debug!("  Config says feat_in: {} (ignoring, using calculated value)", enc_config_ref.num_mel_bins);
        calculated_feat_in
    } else {
        debug!("  Using config feat_in: {}", enc_config_ref.num_mel_bins);
        enc_config_ref.num_mel_bins
    };

    // Detect actual joint vocab size from model weights
    // The streaming model has vocab_size=1025 but config says 8192
    let mut actual_joint_vocab = None;
    if let Some(info) = gguf_file.tensor_infos.get("joint.joint_net.2.weight") {
        let dims = info.shape.dims();
        if dims.len() == 2 {
            actual_joint_vocab = Some(dims[0]);
            debug!("  Detected actual joint vocab size from model: {}", dims[0]);
        }
    }

    // Override config vocab sizes with detected values if they differ significantly
    // Note: PredictionNetwork adds +1 to vocab_size for blank token, so we subtract 1 here
    // since the streaming model already includes blank in its tensor dimensions
    let mut tdt_cfg_adjusted = tdt_cfg.clone();
    if let Some(detected_vocab) = actual_joint_vocab {
        if detected_vocab != tdt_cfg.vocab_size && detected_vocab != tdt_cfg.joint_vocab_size.unwrap_or(tdt_cfg.vocab_size) {
            debug!("  Config says vocab_size: {}, joint_vocab_size: {:?}",
                     tdt_cfg.vocab_size, tdt_cfg.joint_vocab_size);
            debug!("  Detected vocab size from model (including blank): {}", detected_vocab);
            // Subtract 1 because PredictionNetwork will add it back (it expects vocab_size without blank)
            tdt_cfg_adjusted.vocab_size = detected_vocab - 1;
            tdt_cfg_adjusted.joint_vocab_size = Some(detected_vocab);
        }
    }

    let encoder_cfg = FastConformerConfig {
        feat_in,
        d_model: enc_config_ref.hidden_size,
        num_heads: enc_config_ref.num_attention_heads,
        ff_mult: enc_config_ref.intermediate_size / enc_config_ref.hidden_size,
        num_layers: enc_config_ref.num_hidden_layers,
        conv_kernel_size: enc_config_ref.conv_kernel_size,
        dropout: enc_config_ref.dropout,
        dropout_positions: enc_config_ref.dropout_positions,
        subsampling_channels: enc_config_ref.subsampling_conv_channels,
        subsampling_stride: enc_config_ref.subsampling_conv_stride,
        subsampling_factor: enc_config_ref.subsampling_factor,
        scale_input: enc_config_ref.scale_input.unwrap_or(true),
        vocab_size: hf_cfg.vocab_size,
        blank_id: hf_cfg.blank_id,
    };

    let vb = VarBuilder::from_tensors(tensors, dtype, device);

    // Build encoder
    info!("  Building encoder...");
    let encoder = FastConformerEncoder::new(encoder_cfg.clone(), vb.pp("encoder"))?;

    // Build full transducer model
    info!("  Building transducer model...");
    let mut model = TransducerModel::new(
        encoder,
        tdt_cfg_adjusted,
        encoder_cfg.d_model,
        vb,
    )?;

    // Store tokenizer
    model.tokenizer = Some(tokenizer);

    info!("  ✓ Streaming TDT model loaded successfully (cache-aware, quantized)");

    Ok(model)
}
*/

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
