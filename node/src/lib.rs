/// NAPI Node module for streaming transcription with Silero VAD + Parakeet CTC
///
/// Usage from Node.js:
/// ```js
/// const { TranscriptionStream } = require('./parakeet-node');
///
/// const stream = new TranscriptionStream('./assets', (transcription) => {
///   console.log('Transcription:', transcription);
/// });
///
/// // Stream audio samples (16kHz mono, normalized [-1, 1])
/// await stream.input(Float64Array.from([...]));
/// ```

use napi::{Env, JsFunction, Result, Status};
use napi_derive::napi;
use once_cell::sync::OnceCell;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use log::{info, warn};

mod assets;
mod silero {
    include!("../../src/silero.rs");
}

use silero::{SileroVad, VadStream};

/// Transcription result with timestamp
#[napi(object)]
pub struct Transcription {
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
}

/// Inner state for transcription stream
struct StreamInner {
    vad_stream: VadStream,
    parakeet_model: parakeet::ParakeetFastConformerCtc,
    device: candle_core::Device,

    // Accumulated samples for current speech segment
    current_segment: Vec<f32>,
    current_segment_start: Option<f64>, // Start time in seconds

    // Tracking
    total_samples_processed: usize,
    silence_frames: usize,

    // Configuration
    speech_threshold: f32,
    min_speech_duration_ms: f32,
    min_silence_duration_ms: f32,
}

impl StreamInner {
    fn new(
        vad: SileroVad,
        parakeet_model: parakeet::ParakeetFastConformerCtc,
        device: candle_core::Device,
    ) -> Result<Self> {
        let vad_stream = VadStream::new(vad, &device)
            .map_err(|e| napi::Error::from_reason(format!("Failed to create VAD stream: {}", e)))?;

        Ok(Self {
            vad_stream,
            parakeet_model,
            device,
            current_segment: Vec::new(),
            current_segment_start: None,
            total_samples_processed: 0,
            silence_frames: 0,
            speech_threshold: 0.5,
            min_speech_duration_ms: 250.0,
            min_silence_duration_ms: 300.0,
        })
    }

    fn process_samples(&mut self, samples: &[f32]) -> Result<Vec<Transcription>> {
        let mut transcriptions = Vec::new();

        // Process through VAD in 160-sample chunks (10ms at 16kHz)
        const CHUNK_SIZE: usize = 160;
        let mut idx = 0;

        while idx < samples.len() {
            let end = (idx + CHUNK_SIZE).min(samples.len());
            let chunk = &samples[idx..end];

            let probs = self.vad_stream.push(chunk)
                .map_err(|e| napi::Error::from_reason(format!("VAD error: {}", e)))?;

            // Each probability corresponds to ~512 samples (32ms at 16kHz)
            for prob in probs {
                let is_speech = prob >= self.speech_threshold;

                if is_speech {
                    self.silence_frames = 0;

                    if self.current_segment_start.is_none() {
                        // Start new speech segment
                        self.current_segment_start = Some(self.total_samples_processed as f64 / 16000.0);
                        self.current_segment.clear();
                    }

                    // Add samples to current segment (approximate - we add the chunk)
                    if self.current_segment_start.is_some() {
                        let segment_start = self.total_samples_processed.saturating_sub(chunk.len());
                        let segment_end = self.total_samples_processed;
                        if segment_start < samples.len() && segment_end <= samples.len() {
                            self.current_segment.extend_from_slice(chunk);
                        }
                    }
                } else {
                    // Silence detected
                    if self.current_segment_start.is_some() {
                        self.silence_frames += 1;
                        let silence_duration_ms = self.silence_frames as f32 * 32.0; // 32ms per frame

                        if silence_duration_ms >= self.min_silence_duration_ms {
                            // End current segment and transcribe
                            let start_time = self.current_segment_start.unwrap();
                            let end_time = self.total_samples_processed as f64 / 16000.0;
                            let duration_ms = (end_time - start_time) * 1000.0;

                            if duration_ms >= self.min_speech_duration_ms as f64 {
                                // Transcribe the accumulated segment
                                if let Ok(text) = self.transcribe_segment() {
                                    if !text.is_empty() {
                                        transcriptions.push(Transcription {
                                            text,
                                            start_time,
                                            end_time,
                                        });
                                    }
                                }
                            }

                            self.current_segment.clear();
                            self.current_segment_start = None;
                            self.silence_frames = 0;
                        }
                    }
                }

                self.total_samples_processed += 512;
            }

            idx = end;
        }

        Ok(transcriptions)
    }

    fn transcribe_segment(&self) -> Result<String> {
        if self.current_segment.is_empty() {
            return Ok(String::new());
        }

        // Extract features from segment
        let features = parakeet::extract_features_from_samples(
            &self.current_segment,
            self.parakeet_model.cfg.feat_in,
            &self.device,
        )
        .map_err(|e| napi::Error::from_reason(format!("Feature extraction error: {}", e)))?;

        // Run inference
        let logits = self.parakeet_model.forward(&features, false)
            .map_err(|e| napi::Error::from_reason(format!("Inference error: {}", e)))?;

        // Decode
        let transcriptions = self.parakeet_model.greedy_decode(&logits)
            .map_err(|e| napi::Error::from_reason(format!("Decoding error: {}", e)))?;

        Ok(transcriptions.first().cloned().unwrap_or_default())
    }
}

#[napi]
pub struct TranscriptionStream {
    inner: Arc<Mutex<StreamInner>>,
    callback: ThreadsafeFunction<Transcription>,
}

#[napi]
impl TranscriptionStream {
    /// Create a new transcription stream
    ///
    /// @param assets_path - Path to directory containing model files
    /// @param callback - Function called with each transcription result
    #[napi(constructor)]
    pub fn new(
        env: Env,
        assets_path: String,
        #[napi(ts_arg_type = "(transcription: Transcription) => void")]
        callback: JsFunction,
    ) -> Result<Self> {
        info!("Initializing TranscriptionStream with assets: {}", assets_path);

        let assets = PathBuf::from(&assets_path);

        // Get device
        let device = parakeet::get_device()
            .map_err(|e| napi::Error::from_reason(format!("Device error: {}", e)))?;

        info!("Loading Silero VAD...");
        // Load VAD model (expect files in assets directory)
        let vad_path = assets.join("vad16.safetensors");
        let vad_config_path = assets.join("vad16.config.json");

        let vad = SileroVad::load(
            &device,
            vad_path.to_str().unwrap(),
            vad_config_path.to_str().unwrap(),
        )
        .map_err(|e| napi::Error::from_reason(format!("Failed to load VAD: {}", e)))?;

        info!("Loading Parakeet model...");
        // Load Parakeet model (expect hf_parakeet directory in assets)
        let parakeet_dir = assets.join("hf_parakeet");
        let parakeet_model = parakeet::load_parakeet_ctc_from_gguf_local(
            parakeet_dir.to_str().unwrap(),
            &device,
        )
        .map_err(|e| napi::Error::from_reason(format!("Failed to load Parakeet: {}", e)))?;

        info!("Models loaded successfully");

        // Create inner state
        let inner = StreamInner::new(vad, parakeet_model, device)?;

        // Create threadsafe callback
        let tsfn: ThreadsafeFunction<Transcription> = callback
            .build_threadsafe_function()
            .build()?;

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
            callback: tsfn,
        })
    }

    /// Process audio samples and emit transcriptions via callback
    ///
    /// @param samples - Audio samples (16kHz mono, normalized to [-1, 1])
    #[napi]
    pub fn input(&self, samples: Float64Array) -> Result<()> {
        // Convert f64 to f32
        let samples_f32: Vec<f32> = samples.to_vec()
            .iter()
            .map(|&x| x as f32)
            .collect();

        // Process samples
        let transcriptions = {
            let mut inner = self.inner.lock()
                .map_err(|e| napi::Error::from_reason(format!("Lock error: {}", e)))?;
            inner.process_samples(&samples_f32)?
        };

        // Emit transcriptions via callback
        for transcription in transcriptions {
            self.callback.call(transcription, ThreadsafeFunctionCallMode::NonBlocking);
        }

        Ok(())
    }

    /// Flush any remaining audio and get final transcription
    #[napi]
    pub fn flush(&self) -> Result<Option<Transcription>> {
        let mut inner = self.inner.lock()
            .map_err(|e| napi::Error::from_reason(format!("Lock error: {}", e)))?;

        // Transcribe any remaining segment
        if inner.current_segment_start.is_some() && !inner.current_segment.is_empty() {
            let start_time = inner.current_segment_start.unwrap();
            let end_time = inner.total_samples_processed as f64 / 16000.0;

            if let Ok(text) = inner.transcribe_segment() {
                if !text.is_empty() {
                    let transcription = Transcription {
                        text,
                        start_time,
                        end_time,
                    };

                    // Emit via callback
                    self.callback.call(transcription.clone(), ThreadsafeFunctionCallMode::NonBlocking);

                    return Ok(Some(transcription));
                }
            }
        }

        Ok(None)
    }
}

/// Simple logging setup for Node.js
#[napi]
pub fn init_logging() {
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init();
}
