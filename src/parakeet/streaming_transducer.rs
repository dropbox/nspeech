/// Streaming inference for Parakeet TDT (Transducer) models
///
/// Enables real-time transcription by processing audio in chunks and
/// maintaining encoder/predictor state between chunks.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::rnn;

use super::transducer::TransducerModel;

/// Configuration for streaming inference
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Number of audio frames per chunk (after feature extraction)
    /// Typical: 40-80 frames = 320-640ms at 50fps
    pub chunk_size: usize,

    /// Number of frames of context from previous chunk to include
    /// Needed for convolution modules (kernel_size - 1)
    pub context_size: usize,

    /// Whether to emit partial results during processing
    pub emit_partial: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 50,      // 400ms chunks at 50fps
            context_size: 8,     // Conv kernel size - 1
            emit_partial: true,
        }
    }
}

/// Streaming state maintained between chunks
pub struct StreamingState {
    /// Previous chunk's last frames (for convolution context)
    context_frames: Option<Tensor>,

    /// LSTM predictor states
    predictor_states: Option<Vec<rnn::LSTMState>>,

    /// Last emitted token (for predictor input)
    last_token: u32,

    /// Accumulated tokens so far
    tokens: Vec<u32>,

    /// Configuration
    config: StreamingConfig,

    /// Blank token ID
    blank_id: usize,
}

impl StreamingState {
    pub fn new(blank_id: usize, config: StreamingConfig) -> Self {
        Self {
            context_frames: None,
            predictor_states: None,
            last_token: blank_id as u32,
            tokens: Vec::new(),
            config,
            blank_id,
        }
    }

    /// Reset state for new audio stream
    pub fn reset(&mut self) {
        self.context_frames = None;
        self.predictor_states = None;
        self.last_token = self.blank_id as u32;
        self.tokens.clear();
    }

    /// Get accumulated tokens
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

/// Streaming transcriber
pub struct StreamingTransducer {
    model: TransducerModel,
    state: StreamingState,
}

impl StreamingTransducer {
    pub fn new(model: TransducerModel, config: StreamingConfig) -> Self {
        let blank_id = model.config.blank_id;
        Self {
            model,
            state: StreamingState::new(blank_id, config),
        }
    }

    /// Process a chunk of audio features
    ///
    /// # Arguments
    /// * `features` - Audio features [1, T, feat_dim]
    /// * `is_final` - Whether this is the last chunk
    ///
    /// # Returns
    /// New tokens decoded from this chunk
    pub fn process_chunk(&mut self, features: &Tensor, is_final: bool) -> Result<Vec<u32>> {
        let (batch_size, frames, _feat_dim) = features.dims3()?;
        assert_eq!(batch_size, 1, "Streaming only supports batch_size=1");

        // Prepend context from previous chunk if available
        let (features_with_context, context_enc_frames) = if let Some(ref context) = self.state.context_frames {
            let (_, context_frames, _) = context.dims3()?;
            // Context frames after 8x subsampling (approximate)
            let context_enc_frames = (context_frames + 7) / 8;
            (Tensor::cat(&[context, features], 1)?, context_enc_frames)
        } else {
            (features.clone(), 0)
        };

        // Run encoder on chunk
        let encoder_out = self.model.encoder.forward(&features_with_context, false)?;
        let (_, total_enc_frames, _) = encoder_out.dims3()?;

        // Calculate how many NEW encoder frames we got (excluding context)
        let new_enc_frames = total_enc_frames.saturating_sub(context_enc_frames);

        // Only decode the NEW frames, not the context frames
        let encoder_out_new = if context_enc_frames > 0 && context_enc_frames < total_enc_frames {
            encoder_out.narrow(1, context_enc_frames, new_enc_frames)?
        } else {
            encoder_out
        };

        // Save last frames as context for next chunk (if not final)
        if !is_final && frames >= self.state.config.context_size {
            let context_start = frames.saturating_sub(self.state.config.context_size);
            self.state.context_frames = Some(features.narrow(1, context_start, self.state.config.context_size)?);
        } else {
            self.state.context_frames = None;
        }

        // Decode tokens for this chunk (only NEW frames)
        let chunk_tokens = self.decode_chunk(&encoder_out_new, new_enc_frames)?;

        // Accumulate tokens
        self.state.tokens.extend(&chunk_tokens);

        Ok(chunk_tokens)
    }

    /// Decode tokens from encoder output using greedy decoding
    fn decode_chunk(&mut self, encoder_out: &Tensor, num_frames: usize) -> Result<Vec<u32>> {
        let mut decoded = Vec::new();

        for t in 0..num_frames {
            // Inner loop: keep predicting until blank
            let mut inner_steps = 0;
            const MAX_INNER_STEPS: usize = 20;  // Reduced for streaming

            loop {
                inner_steps += 1;
                if inner_steps > MAX_INNER_STEPS {
                    break;
                }

                // Get encoder output at current timestep: [1, 1, enc_dim]
                let enc_t = encoder_out.narrow(1, t, 1)?;

                // Predictor input: previous token [1, 1]
                let pred_input = Tensor::new(&[self.state.last_token], encoder_out.device())?
                    .unsqueeze(0)?;

                // Run predictor
                let (pred_out, new_states) = self.model.predictor.forward(
                    &pred_input,
                    self.state.predictor_states.as_ref()
                )?;
                self.state.predictor_states = Some(new_states);

                // Joint network: [1, 1, enc_dim] + [1, 1, pred_dim] → [1, 1, 1, vocab_size]
                let logits = self.model.joint.forward(&enc_t, &pred_out)?;

                // Get most likely token
                let logits = logits.squeeze(0)?.squeeze(0)?.squeeze(0)?;
                let logits_f32 = logits.to_dtype(DType::F32)?;

                // Mask out padding tokens 8193-8197
                let mut masked_logits = logits_f32.clone();
                for i in 8193..8198 {
                    let mask_tensor = Tensor::new(&[-1e9_f32], masked_logits.device())?;
                    masked_logits = masked_logits.slice_assign(&[i..i+1], &mask_tensor)?;
                }

                let log_probs_masked = candle_nn::ops::log_softmax(&masked_logits, candle_core::D::Minus1)?;
                let token_tensor = log_probs_masked.argmax(candle_core::D::Minus1)?;
                let token = token_tensor.to_scalar::<u32>()?;

                if token == self.state.blank_id as u32 {
                    // Blank: move to next timestep
                    break;
                } else if token >= self.model.config.vocab_size as u32 {
                    // Special token, treat as blank
                    break;
                } else {
                    // Valid vocabulary token: emit and continue at same timestep
                    decoded.push(token);
                    self.state.last_token = token;
                }
            }
        }

        Ok(decoded)
    }

    /// Decode accumulated tokens to text
    pub fn decode_text(&self) -> Result<String> {
        self.model.decode_tokens(&self.state.tokens)
    }

    /// Get accumulated tokens
    pub fn tokens(&self) -> &[u32] {
        self.state.tokens()
    }

    /// Reset state for new stream
    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Finalize stream and return complete transcription
    pub fn finalize(&mut self) -> Result<String> {
        let text = self.decode_text()?;
        self.reset();
        Ok(text)
    }
}
