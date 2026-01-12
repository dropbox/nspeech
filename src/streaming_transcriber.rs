/// Streaming transcription with VAD-based segmentation and pause detection
///
/// This module provides a unified streaming transcription implementation used by both:
/// - Node.js bindings (src/lib.rs)
/// - CLI example (examples/transcribe_with_vad.rs)
///
/// Key features:
/// - VAD-based speech detection
/// - Pause detection for comma/period insertion
/// - Pre-buffering to capture start of speech
/// - Startup buffer for initial audio capture
/// - Optional Qwen3 text correction

use anyhow::Result;
use candle_core::Device;
use log::info;
use std::collections::VecDeque;

use crate::parakeet;
use crate::silero::VadStream;

#[cfg(feature = "qwen")]
use crate::qwen::QwenCorrector;

/// Configuration for streaming transcription
#[derive(Clone, Debug)]
pub struct StreamingConfig {
    /// VAD probability threshold for speech detection (default: 0.1)
    pub speech_threshold: f32,
    /// Minimum speech duration in milliseconds (default: 250.0)
    pub min_speech_duration_ms: f32,
    /// Pre-buffer duration in milliseconds (default: 1000.0)
    pub pre_buffer_ms: f32,
    /// Short pause duration for comma insertion (default: 150.0)
    pub comma_pause_duration_ms: f32,
    /// Long pause duration for period/segment end (default: 500.0)
    pub period_pause_duration_ms: f32,
    /// Very long pause for auto-flush (default: 2000.0)
    pub silence_timeout_ms: f32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            speech_threshold: 0.1,
            min_speech_duration_ms: 250.0,
            pre_buffer_ms: 1000.0,
            comma_pause_duration_ms: 150.0,
            period_pause_duration_ms: 500.0,
            silence_timeout_ms: 2000.0,
        }
    }
}

/// Transcription segment with timing information
#[derive(Clone, Debug)]
pub struct TranscriptionSegment {
    pub text: String,
    pub raw_text: String,
    pub start_time: f64,
    pub end_time: f64,
}

/// Streaming transcriber with VAD-based segmentation
pub struct StreamingTranscriber {
    vad_stream: VadStream,
    parakeet_model: parakeet::ParakeetCtc,
    device: Device,
    config: StreamingConfig,

    // Accumulated samples for current speech segment
    current_segment: Vec<f32>,
    current_segment_start: Option<f64>,

    // Pre-buffer to capture audio before speech detection
    pre_buffer: VecDeque<f32>,

    // Startup buffer to capture initial audio before first speech detection
    startup_buffer: Vec<f32>,
    first_speech_detected: bool,

    // Sub-segments (phrases) for comma insertion
    phrase_boundaries: Vec<usize>,

    // Tracking state
    total_samples_processed: usize,
    silence_frames: usize,
    was_speech_last_frame: bool,
    last_audio_time: std::time::Instant,

    // Optional Qwen3 text correction
    #[cfg(feature = "qwen")]
    qwen_corrector: Option<QwenCorrector>,
}

impl StreamingTranscriber {
    /// Create a new streaming transcriber
    pub fn new(
        vad_stream: VadStream,
        parakeet_model: parakeet::ParakeetCtc,
        device: Device,
        config: StreamingConfig,
        #[cfg(feature = "qwen")] qwen_corrector: Option<QwenCorrector>,
    ) -> Self {
        let pre_buffer_samples = (config.pre_buffer_ms * 16.0) as usize;
        let pre_buffer = VecDeque::with_capacity(pre_buffer_samples);

        Self {
            vad_stream,
            parakeet_model,
            device,
            config,
            current_segment: Vec::new(),
            current_segment_start: None,
            pre_buffer,
            startup_buffer: Vec::new(),
            first_speech_detected: false,
            phrase_boundaries: Vec::new(),
            total_samples_processed: 0,
            silence_frames: 0,
            was_speech_last_frame: false,
            last_audio_time: std::time::Instant::now(),
            #[cfg(feature = "qwen")]
            qwen_corrector,
        }
    }

    /// Process audio samples and return completed transcription segments
    pub fn process_samples(&mut self, samples: &[f32]) -> Result<Vec<TranscriptionSegment>> {
        let mut segments = Vec::new();

        // Check for silence timeout
        let time_since_last_audio = self.last_audio_time.elapsed().as_millis() as f32;
        if time_since_last_audio >= self.config.silence_timeout_ms {
            if let Some(segment) = self.flush_segment()? {
                info!(
                    "Silence timeout ({}ms) - auto-flushing segment",
                    time_since_last_audio
                );
                segments.push(segment);
            }
        }

        self.last_audio_time = std::time::Instant::now();

        // Accumulate to startup buffer if no speech detected yet
        if !self.first_speech_detected {
            self.startup_buffer.extend_from_slice(samples);
            // Limit startup buffer to first 3 seconds
            if self.startup_buffer.len() > 48000 {
                self.startup_buffer
                    .drain(0..(self.startup_buffer.len() - 48000));
            }
            info!(
                "VAD: Accumulating to startup buffer, total {} samples ({:.2}s)",
                self.startup_buffer.len(),
                self.startup_buffer.len() as f32 / 16000.0
            );
        }

        // Process through VAD in chunks
        const CHUNK_SIZE: usize = 160;
        let mut idx = 0;

        while idx < samples.len() {
            let end = (idx + CHUNK_SIZE).min(samples.len());
            let chunk = &samples[idx..end];

            let probs = self.vad_stream.push(chunk)?;

            for prob in probs {
                let is_speech = prob >= self.config.speech_threshold;

                if is_speech {
                    self.handle_speech_detected()?;
                } else {
                    if let Some(segment) = self.handle_silence_detected()? {
                        segments.push(segment);
                    }
                }
            }

            // Maintain pre-buffer during silence
            if !self.was_speech_last_frame {
                let pre_buffer_max = (self.config.pre_buffer_ms * 16.0) as usize;
                for &sample in chunk {
                    if self.pre_buffer.len() >= pre_buffer_max {
                        self.pre_buffer.pop_front();
                    }
                    self.pre_buffer.push_back(sample);
                }
            }

            // Accumulate chunk samples if in active speech segment
            if self.current_segment_start.is_some() {
                self.current_segment.extend_from_slice(chunk);
            }

            self.total_samples_processed += chunk.len();
            idx = end;
        }

        Ok(segments)
    }

    /// Flush any remaining segment
    pub fn flush(&mut self) -> Result<Option<TranscriptionSegment>> {
        self.flush_segment()
    }

    fn handle_speech_detected(&mut self) -> Result<()> {
        // Check if speech is resuming after silence
        if self.current_segment_start.is_some()
            && !self.was_speech_last_frame
            && self.silence_frames > 0
        {
            let pre_buffer_len = self.pre_buffer.len();
            if pre_buffer_len > 0 {
                info!(
                    "VAD: Speech resumed after {}ms pause, prepending {}ms pre-buffer",
                    self.silence_frames as f32 * 32.0,
                    pre_buffer_len as f32 / 16.0
                );
                let insert_pos = self.current_segment.len();
                self.current_segment.reserve(pre_buffer_len);
                self.current_segment
                    .extend(self.pre_buffer.iter().copied());
                self.current_segment[insert_pos..].rotate_right(pre_buffer_len);
            }
            self.pre_buffer.clear();
        }

        self.silence_frames = 0;
        self.was_speech_last_frame = true;

        if self.current_segment_start.is_none() {
            // Start new speech segment
            let start_time;
            self.current_segment.clear();

            if !self.first_speech_detected {
                start_time = 0.0;
                self.current_segment
                    .extend(self.startup_buffer.iter().copied());
                info!(
                    "VAD: First speech detected, using startup buffer ({} samples from beginning)",
                    self.startup_buffer.len()
                );
                self.startup_buffer.clear();
                self.first_speech_detected = true;
            } else {
                let pre_buffer_duration = self.pre_buffer.len() as f64 / 16000.0;
                start_time =
                    (self.total_samples_processed as f64 / 16000.0) - pre_buffer_duration;
                self.current_segment
                    .extend(self.pre_buffer.iter().copied());
                info!(
                    "VAD: Speech started at {:.3}s (pre-buffer={}ms)",
                    start_time,
                    pre_buffer_duration * 1000.0
                );
            }

            self.current_segment_start = Some(start_time);
            self.pre_buffer.clear();
        }

        Ok(())
    }

    fn handle_silence_detected(&mut self) -> Result<Option<TranscriptionSegment>> {
        self.was_speech_last_frame = false;

        if self.current_segment_start.is_some() {
            self.silence_frames += 1;
            let silence_duration_ms = self.silence_frames as f32 * 32.0;

            // Check for comma pause
            if silence_duration_ms >= self.config.comma_pause_duration_ms
                && silence_duration_ms < self.config.period_pause_duration_ms
                && self.silence_frames
                    == (self.config.comma_pause_duration_ms / 32.0).ceil() as usize
            {
                let boundary_pos = self.current_segment.len();
                if boundary_pos > 0 {
                    self.phrase_boundaries.push(boundary_pos);
                    info!(
                        "VAD: Comma pause detected at sample {} ({}ms pause)",
                        boundary_pos, silence_duration_ms
                    );
                }
            }

            // Check for period pause (end segment)
            if silence_duration_ms >= self.config.period_pause_duration_ms {
                let start_time = self.current_segment_start.unwrap();
                let end_time = self.total_samples_processed as f64 / 16000.0;
                let duration_ms = (end_time - start_time) * 1000.0;

                info!(
                    "VAD: Period pause - speech ended at {:.3}s (duration={:.0}ms, samples={}, phrases={})",
                    end_time,
                    duration_ms,
                    self.current_segment.len(),
                    self.phrase_boundaries.len() + 1
                );

                if duration_ms >= self.config.min_speech_duration_ms as f64 {
                    let segment = self.transcribe_current_segment(start_time, end_time)?;
                    self.reset_segment();
                    return Ok(Some(segment));
                } else {
                    info!(
                        "VAD: Segment too short ({:.0}ms < {:.0}ms), skipping",
                        duration_ms, self.config.min_speech_duration_ms
                    );
                    self.reset_segment();
                }
            }
        }

        Ok(None)
    }

    fn flush_segment(&mut self) -> Result<Option<TranscriptionSegment>> {
        if let Some(start_time) = self.current_segment_start {
            if !self.current_segment.is_empty() {
                let end_time = self.total_samples_processed as f64 / 16000.0;
                let duration_ms = (end_time - start_time) * 1000.0;

                info!(
                    "Flush: VAD segment {:.3}s-{:.3}s (duration={:.0}ms, samples={}, phrases={})",
                    start_time,
                    end_time,
                    duration_ms,
                    self.current_segment.len(),
                    self.phrase_boundaries.len() + 1
                );

                if duration_ms >= self.config.min_speech_duration_ms as f64 {
                    let segment = self.transcribe_current_segment(start_time, end_time)?;
                    self.reset_segment();
                    return Ok(Some(segment));
                } else {
                    info!(
                        "Flush: Segment too short ({:.0}ms < {:.0}ms), skipping",
                        duration_ms, self.config.min_speech_duration_ms
                    );
                    self.reset_segment();
                }
            }
        }

        Ok(None)
    }

    fn transcribe_current_segment(
        &mut self,
        start_time: f64,
        end_time: f64,
    ) -> Result<TranscriptionSegment> {
        info!(
            "Transcribe: Processing {} samples with {} phrase boundaries",
            self.current_segment.len(),
            self.phrase_boundaries.len()
        );

        let (raw_text, text) = if !self.phrase_boundaries.is_empty() {
            self.transcribe_with_phrases()?
        } else {
            self.transcribe_single_phrase()?
        };

        Ok(TranscriptionSegment {
            text,
            raw_text,
            start_time,
            end_time,
        })
    }

    fn transcribe_with_phrases(&mut self) -> Result<(String, String)> {
        let mut phrases = Vec::new();
        let mut start_idx = 0;

        for &boundary_pos in &self.phrase_boundaries {
            if boundary_pos > start_idx && boundary_pos <= self.current_segment.len() {
                let phrase_samples = &self.current_segment[start_idx..boundary_pos];
                let raw_phrase = parakeet::transcribe_streaming_chunk(
                    phrase_samples,
                    None,
                    None,
                    &self.parakeet_model,
                    &self.device,
                )?;

                if !raw_phrase.is_empty() {
                    phrases.push(raw_phrase);
                }
                start_idx = boundary_pos;
            }
        }

        // Transcribe final phrase
        if start_idx < self.current_segment.len() {
            let final_phrase_samples = &self.current_segment[start_idx..];
            let raw_phrase = parakeet::transcribe_streaming_chunk(
                final_phrase_samples,
                None,
                None,
                &self.parakeet_model,
                &self.device,
            )?;

            if !raw_phrase.is_empty() {
                phrases.push(raw_phrase);
            }
        }

        let raw_text = phrases.join(" , ");
        let text = self.apply_correction(&raw_text, true)?;

        info!("Transcribe: Raw phrases: {} -> \"{}\"", phrases.len(), raw_text);
        info!("Transcribe: With punctuation: \"{}\"", text);

        Ok((raw_text, text))
    }

    fn transcribe_single_phrase(&mut self) -> Result<(String, String)> {
        let raw_text = parakeet::transcribe_streaming_chunk(
            &self.current_segment,
            None,
            None,
            &self.parakeet_model,
            &self.device,
        )?;

        let text = self.apply_correction(&raw_text, false)?;

        info!("Transcribe: Raw: \"{}\"", raw_text);
        info!("Transcribe: With punctuation: \"{}\"", text);

        Ok((raw_text.clone(), text))
    }

    fn apply_correction(&mut self, raw_text: &str, has_commas: bool) -> Result<String> {
        #[cfg(feature = "qwen")]
        {
            if let Some(ref mut corrector) = self.qwen_corrector {
                match corrector.correct_text(raw_text) {
                    Ok(corrected) => return Ok(corrected),
                    Err(e) => {
                        info!(
                            "Qwen3 correction failed: {}, falling back to rule-based",
                            e
                        );
                    }
                }
            }
        }

        // Fall back to rule-based
        if has_commas {
            Ok(parakeet::add_punctuation_internal(raw_text, true))
        } else {
            Ok(parakeet::add_punctuation(raw_text))
        }
    }

    fn reset_segment(&mut self) {
        self.current_segment.clear();
        self.current_segment_start = None;
        self.phrase_boundaries.clear();
        self.silence_frames = 0;
    }
}
