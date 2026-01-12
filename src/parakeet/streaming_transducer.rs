/// Streaming inference for Parakeet TDT (Transducer) models
///
/// Provides practical streaming ASR using overlapping chunks with sufficient
/// context for the encoder. While not true frame-level streaming (which requires
/// attention caching), this approach enables low-latency transcription suitable
/// for real-time applications.

use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::rnn;
use std::collections::VecDeque;

use super::transducer::TransducerModel;

/// Configuration for streaming inference
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Size of each processing chunk (in audio samples at 16kHz)
    /// Larger chunks = better accuracy, higher latency
    /// Typical: 8000-16000 samples (0.5-1.0s)
    pub chunk_samples: usize,

    /// Overlap between chunks (in audio samples)
    /// Provides context for better accuracy at chunk boundaries
    /// Typical: 3200-6400 samples (0.2-0.4s)
    pub overlap_samples: usize,

    /// Whether to emit partial results during processing
    pub emit_partial: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_samples: 12800,  // 0.8s at 16kHz
            overlap_samples: 4800,  // 0.3s overlap
            emit_partial: true,
        }
    }
}

/// Streaming state maintained between chunks
pub struct StreamingState {
    /// Buffer of audio samples from previous chunk (for overlap)
    audio_buffer: VecDeque<f32>,

    /// LSTM predictor states
    predictor_states: Option<Vec<rnn::LSTMState>>,

    /// Last emitted token (for predictor input)
    last_token: u32,

    /// Accumulated tokens so far
    tokens: Vec<u32>,

    /// Number of tokens already decoded to text
    tokens_decoded: usize,

    /// Configuration
    config: StreamingConfig,

    /// Blank token ID
    blank_id: usize,

    /// Total audio samples processed
    total_samples: usize,

    /// Number of chunks processed (for tracking first vs subsequent chunks)
    chunks_processed: usize,
}

impl StreamingState {
    pub fn new(blank_id: usize, config: StreamingConfig) -> Self {
        Self {
            audio_buffer: VecDeque::with_capacity(config.overlap_samples),
            predictor_states: None,
            last_token: blank_id as u32,
            tokens: Vec::new(),
            tokens_decoded: 0,
            config,
            blank_id,
            total_samples: 0,
            chunks_processed: 0,
        }
    }

    /// Reset state for new audio stream
    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.predictor_states = None;
        self.last_token = self.blank_id as u32;
        self.tokens.clear();
        self.tokens_decoded = 0;
        self.total_samples = 0;
        self.chunks_processed = 0;
    }

    /// Get accumulated tokens
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Get number of tokens already decoded
    pub fn tokens_decoded(&self) -> usize {
        self.tokens_decoded
    }

    /// Mark tokens as decoded
    pub fn mark_decoded(&mut self, count: usize) {
        self.tokens_decoded = count;
    }
}

/// Streaming transcriber with overlapping chunks
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

    /// Process audio samples (not features!)
    ///
    /// This method handles raw audio samples and maintains overlap between chunks
    /// for better transcription quality.
    ///
    /// # Arguments
    /// * `samples` - Audio samples (f32, normalized to -1.0 to 1.0)
    /// * `is_final` - Whether this is the last chunk of audio
    ///
    /// # Returns
    /// New tokens decoded from this chunk
    pub fn process_samples(&mut self, samples: &[f32], is_final: bool) -> Result<Vec<u32>> {
        // Add new samples to buffer
        self.state.audio_buffer.extend(samples);
        self.state.total_samples += samples.len();

        // Check if we have enough samples to process
        let process_size = self.state.config.chunk_samples;
        let overlap_size = self.state.config.overlap_samples;

        if !is_final && self.state.audio_buffer.len() < process_size {
            // Not enough samples yet, wait for more
            return Ok(Vec::new());
        }

        // Extract samples to process (all if final, otherwise chunk_size)
        let samples_to_process: Vec<f32> = if is_final {
            self.state.audio_buffer.drain(..).collect()
        } else {
            self.state.audio_buffer.drain(..process_size).collect()
        };

        // Keep overlap for next chunk (unless final)
        if !is_final && samples_to_process.len() >= overlap_size {
            let overlap_start = samples_to_process.len() - overlap_size;
            self.state.audio_buffer.extend(&samples_to_process[overlap_start..]);
        }

        // Now process this chunk through the model
        // This requires feature extraction - we need access to the feature extractor
        // For now, return empty - this needs to be called from the example level
        // where we have access to ParakeetFeatureExtractor

        Ok(Vec::new())
    }

    /// Process audio features (lower-level interface)
    ///
    /// # Arguments
    /// * `features` - Audio features [1, T, feat_dim]
    ///
    /// # Returns
    /// Tokens decoded from these features
    pub fn process_features(&mut self, features: &Tensor) -> Result<Vec<u32>> {
        let (batch_size, _mel_frames, _feat_dim) = features.dims3()?;
        assert_eq!(batch_size, 1, "Streaming only supports batch_size=1");

        // Run encoder on features
        let encoder_out = self.model.encoder.forward(features, false)?;
        let (_, enc_frames, _) = encoder_out.dims3()?;

        // Decode tokens maintaining LSTM state across chunks
        // The overlap provides acoustic context to the encoder
        // Carrying predictor state maintains language model continuity
        let chunk_tokens = self.decode_chunk(&encoder_out, enc_frames)?;

        // Accumulate tokens
        self.state.tokens.extend(&chunk_tokens);

        // Increment chunk counter
        self.state.chunks_processed += 1;

        Ok(chunk_tokens)
    }

    /// Decode tokens from encoder output using greedy decoding
    ///
    /// # Arguments
    /// * `encoder_out` - Encoder output tensor [1, T, enc_dim]
    /// * `num_frames` - Number of encoder frames to process
    ///
    /// # Returns
    /// Decoded tokens
    fn decode_chunk(&mut self, encoder_out: &Tensor, num_frames: usize) -> Result<Vec<u32>> {
        let mut decoded = Vec::new();

        for t in 0..num_frames {
            // Inner loop: keep predicting until blank
            let mut inner_steps = 0;
            const MAX_INNER_STEPS: usize = 30;

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

    /// Decode only new tokens since last decode (streaming)
    ///
    /// This is efficient for streaming as it only decodes new tokens.
    /// Automatically tracks which tokens have been decoded.
    ///
    /// # Returns
    /// (new_text, total_tokens_decoded)
    pub fn decode_text_incremental(&mut self) -> Result<(String, usize)> {
        let already_decoded = self.state.tokens_decoded();
        let total_tokens = self.state.tokens.len();

        if already_decoded >= total_tokens {
            // No new tokens to decode
            return Ok((String::new(), total_tokens));
        }

        // Decode only new tokens
        let new_text = self.model.decode_tokens_incremental(
            &self.state.tokens,
            already_decoded
        )?;

        // Update decoded count
        self.state.mark_decoded(total_tokens);

        Ok((new_text, total_tokens))
    }

    /// Get accumulated tokens
    pub fn tokens(&self) -> &[u32] {
        self.state.tokens()
    }

    /// Get number of tokens already decoded to text
    pub fn tokens_decoded(&self) -> usize {
        self.state.tokens_decoded()
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

    /// Get reference to underlying model (for feature extraction)
    pub fn model(&self) -> &TransducerModel {
        &self.model
    }
}
