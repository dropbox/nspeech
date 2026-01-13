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

    /// Token buffer for LCS-based overlap deduplication
    /// Stores recent tokens to detect duplicates in overlapping regions
    token_buffer: Vec<u32>,

    /// Number of encoder frames consumed so far (for frame-level masking)
    /// Used to skip overlapping frames when decoding
    encoder_frames_consumed: usize,

    /// Chunk configuration for calculating overlap frames
    chunk_samples: usize,
    overlap_samples: usize,
}

impl StreamingState {
    pub fn new(blank_id: usize, config: StreamingConfig) -> Self {
        Self {
            audio_buffer: VecDeque::with_capacity(config.overlap_samples),
            predictor_states: None,
            last_token: blank_id as u32,
            tokens: Vec::new(),
            tokens_decoded: 0,
            chunk_samples: config.chunk_samples,
            overlap_samples: config.overlap_samples,
            config,
            blank_id,
            total_samples: 0,
            chunks_processed: 0,
            token_buffer: Vec::new(),
            encoder_frames_consumed: 0,
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
        self.token_buffer.clear();
        self.encoder_frames_consumed = 0;
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
        let (batch_size, mel_frames, _feat_dim) = features.dims3()?;
        assert_eq!(batch_size, 1, "Streaming only supports batch_size=1");

        // Run encoder on features (process full overlapping chunk for context)
        let encoder_out = self.model.encoder.forward(features, false)?;
        let (_, enc_frames, _) = encoder_out.dims3()?;

        // Calculate overlap in encoder frames
        // Encoder does 8x temporal downsampling (ConvSubsampling stride)
        let overlap_encoder_frames = if self.state.chunks_processed > 0 && self.state.overlap_samples > 0 {
            // Estimate: overlap_samples -> mel_frames -> encoder_frames
            // Mel: 160 hop -> overlap_samples/160 frames
            // Encoder: 8x downsample -> mel_frames/8 frames
            let overlap_mel_frames = (self.state.overlap_samples / 160) + 1;
            let overlap_enc = (overlap_mel_frames / 8) + 1;
            overlap_enc.min(enc_frames) // Clamp to actual frames
        } else {
            0 // First chunk - no overlap to skip
        };

        // IMPORTANT: Decode ALL frames to provide LSTM context
        // The LSTM needs overlapping frames as context to produce good predictions
        // We rely on LCS deduplication to remove duplicate tokens from overlap
        eprintln!("  [Chunk {}] Enc frames: {}, overlap: {} frames (decoded for context, LCS will dedupe)",
                  self.state.chunks_processed + 1, enc_frames, overlap_encoder_frames);

        let chunk_tokens = self.decode_chunk_masked(&encoder_out, 0, enc_frames)?;

        // Track encoder frames (skip overlap count for total)
        let novel_frames = if self.state.chunks_processed > 0 {
            enc_frames.saturating_sub(overlap_encoder_frames)
        } else {
            enc_frames
        };
        self.state.encoder_frames_consumed += novel_frames;

        // Reset LSTM after each chunk to prevent state accumulation issues
        // Why: The LSTM predictor state can get "stuck" predicting blanks after
        // processing silence or certain acoustic patterns. Resetting provides
        // a fresh start for each chunk while the overlap ensures acoustic continuity.
        // Trade-off: Sacrifices language model continuity for robustness
        self.state.predictor_states = None;
        self.state.last_token = self.state.blank_id as u32;

        // Deduplicate tokens using LCS (remove overlapping content from previous chunk)
        let deduplicated = self.deduplicate_tokens(chunk_tokens.clone());

        // Log LCS deduplication effect
        if chunk_tokens.len() != deduplicated.len() {
            eprintln!("  [LCS] Dedup: {} raw tokens → {} deduplicated ({} removed)",
                      chunk_tokens.len(), deduplicated.len(), chunk_tokens.len() - deduplicated.len());
        }

        // Update token buffer for next chunk's LCS (keep last N tokens as context)
        const BUFFER_SIZE: usize = 50; // Match MAX_SEARCH_LEN
        self.state.token_buffer.extend(&chunk_tokens);
        if self.state.token_buffer.len() > BUFFER_SIZE {
            let drain_count = self.state.token_buffer.len() - BUFFER_SIZE;
            self.state.token_buffer.drain(..drain_count);
        }

        // Accumulate only deduplicated tokens
        self.state.tokens.extend(&deduplicated);

        // Increment chunk counter
        self.state.chunks_processed += 1;

        Ok(deduplicated)
    }

    /// Decode tokens from encoder output using greedy decoding with frame masking
    ///
    /// # Arguments
    /// * `encoder_out` - Encoder output tensor [1, T, enc_dim]
    /// * `start_frame` - First frame to decode (skip overlap)
    /// * `num_frames` - Number of frames to decode
    ///
    /// # Returns
    /// Decoded tokens
    fn decode_chunk_masked(&mut self, encoder_out: &Tensor, start_frame: usize, num_frames: usize) -> Result<Vec<u32>> {
        let mut decoded = Vec::new();
        let mut blank_count = 0;

        for t in start_frame..(start_frame + num_frames) {
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

                // CRITICAL: Save state BEFORE prediction (for blank rollback)
                let saved_states = self.state.predictor_states.clone();

                // Run predictor
                let (pred_out, new_states) = self.model.predictor.forward(
                    &pred_input,
                    self.state.predictor_states.as_ref()
                )?;

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
                    // Blank: ROLLBACK state (don't corrupt with blank prediction)
                    // This is critical per NeMo - blanks should not update decoder state
                    self.state.predictor_states = saved_states;
                    blank_count += 1;
                    break;
                } else if token >= self.model.config.vocab_size as u32 {
                    // Special token, treat as blank (rollback state)
                    self.state.predictor_states = saved_states;
                    blank_count += 1;
                    break;
                } else {
                    // Non-blank token: UPDATE state (this is the only place we accept new_states)
                    self.state.predictor_states = Some(new_states);

                    // Emit token and update last_token
                    decoded.push(token);
                    self.state.last_token = token;
                }
            }
        }

        // Debug logging for 0-token chunks
        if decoded.is_empty() && num_frames > 0 {
            eprintln!("  [DEBUG] 0 tokens decoded from {} frames ({} blanks, {:.1}% blank ratio)",
                      num_frames, blank_count, (blank_count as f32 / num_frames as f32) * 100.0);
        }

        Ok(decoded)
    }

    /// Find longest common subsequence between buffer tail and new tokens
    /// Returns the index in new_tokens where novel content begins
    ///
    /// Based on NeMo's LCS algorithm for overlap deduplication
    fn find_lcs_slice_point(&self, buffer: &[u32], new_tokens: &[u32]) -> usize {
        if buffer.is_empty() || new_tokens.is_empty() {
            return 0; // No overlap possible
        }

        // Limit search to reasonable size (per NeMo)
        const MAX_SEARCH_LEN: usize = 50;
        let buffer_tail = if buffer.len() > MAX_SEARCH_LEN {
            &buffer[buffer.len() - MAX_SEARCH_LEN..]
        } else {
            buffer
        };

        let search_len = new_tokens.len().min(MAX_SEARCH_LEN);
        let new_head = &new_tokens[..search_len];

        // Find longest matching subsequence
        let mut best_match_len = 0;
        let mut best_slice_point = 0;

        // Try all possible starting points in buffer
        for buf_start in 0..buffer_tail.len() {
            let mut match_len = 0;
            let mut new_idx = 0;

            // Match as many tokens as possible
            for buf_idx in buf_start..buffer_tail.len() {
                if new_idx >= new_head.len() {
                    break;
                }
                if buffer_tail[buf_idx] == new_head[new_idx] {
                    match_len += 1;
                    new_idx += 1;
                } else if match_len > 0 {
                    // Allow some mismatches (diagonal expansion per NeMo)
                    new_idx += 1;
                }
            }

            if match_len > best_match_len {
                best_match_len = match_len;
                best_slice_point = new_idx;
            }
        }

        // Only use LCS if match is significant (per NeMo: MIN_MERGE_SUBSEQUENCE_LEN)
        const MIN_LCS_LENGTH: usize = 3;
        if best_match_len >= MIN_LCS_LENGTH {
            best_slice_point
        } else {
            0 // No significant overlap, use all tokens
        }
    }

    /// Deduplicate chunk tokens using LCS with token buffer
    fn deduplicate_tokens(&mut self, chunk_tokens: Vec<u32>) -> Vec<u32> {
        if self.state.token_buffer.is_empty() || chunk_tokens.is_empty() {
            // First chunk or empty chunk - no deduplication needed
            return chunk_tokens;
        }

        // Find where novel content begins
        let slice_point = self.find_lcs_slice_point(&self.state.token_buffer, &chunk_tokens);

        // Return only novel tokens
        chunk_tokens[slice_point..].to_vec()
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
