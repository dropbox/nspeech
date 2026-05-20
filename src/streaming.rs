//! Shared streaming transcription logic.
//!
//! `StreamingTranscriber` owns the Moonshine model, VAD, and all segment/streaming
//! state. It mirrors the proven logic from `examples/mic_streaming.rs` — push 16kHz
//! mono audio in, get transcription events out.
//!
//! Used by both the NAPI binding and the mic_streaming example.

use std::collections::VecDeque;

use anyhow::Result;
use candle_core::Device;

use crate::moonshine::{MoonshineModel, MoonshineStream};
use crate::silero::VadStream;

const VAD_CHUNK_SIZE: usize = 512; // 32ms at 16kHz

/// A transcription event emitted by the streaming transcriber.
pub struct TranscriptionEvent {
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
    pub is_partial: bool,
    pub segment_index: u32,
    /// Longest prefix of `text` unchanged from the previous partial.
    /// For final transcriptions, equals `text`.
    pub stable_text: String,
}

/// Configuration for the streaming transcriber.
pub struct StreamingConfig {
    /// VAD speech probability threshold (0.0–1.0).
    pub speech_threshold: f32,
    /// Minimum speech duration (ms) to emit a final transcription.
    pub min_speech_duration_ms: f32,
    /// Pre-buffer duration (ms) — audio kept before speech onset.
    pub pre_buffer_ms: f32,
    /// Silence duration (ms) after speech to trigger pause/segment end.
    pub pause_duration_ms: f32,
    /// Minimum ms between streaming partial updates.
    pub stream_update_interval_ms: usize,
    /// Minimum audio ms before first partial transcription.
    pub stream_min_audio_ms: usize,
    /// Automatically emit a final transcription when a pause is detected.
    /// When false, pause still resets segment state but does not transcribe —
    /// call `flush()` to get the transcription.
    pub auto_transcribe_on_pause: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            speech_threshold: 0.3,
            min_speech_duration_ms: 250.0,
            pre_buffer_ms: 500.0,
            pause_duration_ms: 800.0,
            stream_update_interval_ms: 500,
            stream_min_audio_ms: 500,
            auto_transcribe_on_pause: true,
        }
    }
}

/// Streaming transcriber that processes 16kHz mono audio and emits transcription events.
///
/// Internally manages VAD, segment accumulation, Moonshine streaming state,
/// and stable-text computation. The logic mirrors `examples/mic_streaming.rs`.
pub struct StreamingTranscriber {
    model: MoonshineModel,
    vad_stream: VadStream,
    moonshine_stream: MoonshineStream,
    device: Device,

    // Segment state (mirrors mic_streaming.rs exactly)
    current_segment: Vec<f32>,
    pre_buffer: VecDeque<f32>,
    in_speech: bool,
    silence_frames: usize,
    segment_index: u32,
    last_partial_text: String,
    total_samples_processed: usize,
    current_segment_start: Option<f64>,

    // Sub-chunk audio accumulator
    audio_buf: Vec<f32>,

    // Configuration
    config: StreamingConfig,
    pre_buffer_max: usize,
    pause_frames: usize,
}

impl StreamingTranscriber {
    pub fn new(
        model: MoonshineModel,
        vad_stream: VadStream,
        device: Device,
        config: StreamingConfig,
    ) -> Self {
        let moonshine_stream = model.stream_new(
            config.stream_update_interval_ms,
            config.stream_min_audio_ms,
        );
        let pre_buffer_max = (config.pre_buffer_ms * 16.0) as usize;
        let pause_frames = (config.pause_duration_ms / 32.0) as usize;

        Self {
            model,
            vad_stream,
            moonshine_stream,
            device,
            current_segment: Vec::new(),
            pre_buffer: VecDeque::new(),
            in_speech: false,
            silence_frames: 0,
            segment_index: 0,
            last_partial_text: String::new(),
            total_samples_processed: 0,
            current_segment_start: None,
            audio_buf: Vec::new(),
            config,
            pre_buffer_max,
            pause_frames,
        }
    }

    /// Push 16kHz mono audio samples. Returns any transcription events generated.
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<Vec<TranscriptionEvent>> {
        self.audio_buf.extend_from_slice(samples);
        let mut events = Vec::new();

        while self.audio_buf.len() >= VAD_CHUNK_SIZE {
            let chunk: Vec<f32> = self.audio_buf.drain(..VAD_CHUNK_SIZE).collect();
            let probs = self.vad_stream.push(&chunk)?;

            for prob in &probs {
                let is_speech = *prob >= self.config.speech_threshold;

                if is_speech {
                    self.silence_frames = 0;

                    if !self.in_speech {
                        // Speech started — prepend pre-buffer, reset stream
                        self.in_speech = true;
                        self.current_segment.clear();
                        self.current_segment.extend(self.pre_buffer.iter());
                        self.segment_index += 1;
                        self.last_partial_text.clear();
                        self.moonshine_stream.reset();

                        let start = (self.total_samples_processed as f64
                            - self.pre_buffer.len() as f64)
                            / 16000.0;
                        self.current_segment_start = Some(start.max(0.0));
                    }

                    self.current_segment.extend_from_slice(&chunk);

                    // Try streaming partial transcription
                    match self.model.stream_try_update(
                        &mut self.moonshine_stream,
                        &self.current_segment,
                        &self.device,
                    )? {
                        Some(text) => {
                            let trimmed = text.trim();
                            if !trimmed.is_empty() && trimmed != self.last_partial_text {
                                let stable_text = longest_common_prefix(
                                    &self.last_partial_text,
                                    trimmed,
                                )
                                .to_string();
                                self.last_partial_text = trimmed.to_string();
                                let start = self.current_segment_start.unwrap_or(0.0);
                                events.push(TranscriptionEvent {
                                    text: trimmed.to_string(),
                                    start_time: start,
                                    end_time: start
                                        + self.current_segment.len() as f64 / 16000.0,
                                    is_partial: true,
                                    segment_index: self.segment_index,
                                    stable_text,
                                });
                            }
                        }
                        None => {}
                    }
                } else {
                    if self.in_speech {
                        // Silence during active speech — accumulate and check for pause
                        self.current_segment.extend_from_slice(&chunk);
                        self.silence_frames += 1;

                        if self.silence_frames >= self.pause_frames {
                            if self.config.auto_transcribe_on_pause {
                                // Pause detected — finalize segment and reset
                                let duration_ms = self.current_segment.len() as f32 / 16.0;
                                if duration_ms >= self.config.min_speech_duration_ms {
                                    if let Some(evt) = self.finalize_current_segment()? {
                                        events.push(evt);
                                    }
                                }

                                self.in_speech = false;
                                self.silence_frames = 0;
                                self.current_segment.clear();
                                self.current_segment_start = None;
                                self.last_partial_text.clear();
                            } else {
                                // No auto-transcribe: keep segment, reset silence counter.
                                // Audio accumulates until explicit flush().
                                self.silence_frames = 0;
                            }
                        }
                    }

                    // Update pre-buffer only during silence outside speech
                    if !self.in_speech {
                        for &s in &chunk {
                            if self.pre_buffer.len() >= self.pre_buffer_max {
                                self.pre_buffer.pop_front();
                            }
                            self.pre_buffer.push_back(s);
                        }
                    }
                }

                self.total_samples_processed += VAD_CHUNK_SIZE;
            }
        }

        Ok(events)
    }

    /// Flush any remaining audio and return a final transcription if available.
    pub fn flush(&mut self) -> Result<Option<TranscriptionEvent>> {
        let result = if !self.current_segment.is_empty() {
            self.finalize_current_segment()?
        } else {
            None
        };

        // Reset all state
        self.current_segment.clear();
        self.current_segment_start = None;
        self.pre_buffer.clear();
        self.in_speech = false;
        self.silence_frames = 0;
        self.last_partial_text.clear();
        self.audio_buf.clear();
        self.moonshine_stream.reset();
        self.vad_stream.reset()?;
        self.total_samples_processed = 0;

        Ok(result)
    }

    /// Discard all buffered audio and state without finalizing.
    pub fn reset(&mut self) -> Result<()> {
        self.current_segment.clear();
        self.current_segment_start = None;
        self.pre_buffer.clear();
        self.in_speech = false;
        self.silence_frames = 0;
        self.last_partial_text.clear();
        self.audio_buf.clear();
        self.moonshine_stream.reset();
        self.vad_stream.reset()?;
        self.total_samples_processed = 0;
        Ok(())
    }

    /// Access the current segment index.
    pub fn segment_index(&self) -> u32 {
        self.segment_index
    }

    /// Whether we're currently inside a speech segment.
    pub fn in_speech(&self) -> bool {
        self.in_speech
    }

    fn finalize_current_segment(&mut self) -> Result<Option<TranscriptionEvent>> {
        if self.current_segment.len() < 4000 {
            // Skip very short segments (< 250ms)
            return Ok(None);
        }

        let start = self.current_segment_start.unwrap_or(0.0);
        let end = start + self.current_segment.len() as f64 / 16000.0;

        let text = self
            .model
            .stream_finalize(&mut self.moonshine_stream, &self.current_segment, &self.device)?;
        let trimmed = text.trim();

        if trimmed.is_empty() {
            return Ok(None);
        }

        Ok(Some(TranscriptionEvent {
            stable_text: trimmed.to_string(),
            text: trimmed.to_string(),
            start_time: start,
            end_time: end,
            is_partial: false,
            segment_index: self.segment_index,
        }))
    }
}

/// Returns the longest common prefix of two strings, splitting on a char boundary.
fn longest_common_prefix<'a>(a: &'a str, b: &str) -> &'a str {
    let len = a
        .bytes()
        .zip(b.bytes())
        .take_while(|(x, y)| x == y)
        .count();
    match a.get(..len) {
        Some(s) => s,
        None => {
            let mut end = len;
            while end > 0 && !a.is_char_boundary(end) {
                end -= 1;
            }
            &a[..end]
        }
    }
}
