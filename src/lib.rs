/* Node module API for Parakeet Speech Recognition */

use napi::{Env, Result};
use napi_derive::napi;

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use log::{info, LevelFilter, Log, Metadata, Record};
use napi::bindgen_prelude::{Function, Unknown, Float64Array};
use napi::threadsafe_function::{ThreadsafeFunctionCallMode, ThreadsafeCallContext};
use once_cell::sync::OnceCell;

#[napi(object)]
pub struct LogEvent {
    pub level: String,
    pub message: String,
    pub file: Option<String>,
    pub line: Option<u32>,
}

// Store the threadsafe function using a boxed type-erased version
static LOGFN: OnceCell<Box<dyn Fn(LogEvent) + Send + Sync>> = OnceCell::new();

struct JsLogger;

impl Log for JsLogger {
    fn enabled(&self, _metadata: &Metadata) -> bool {
        true
    }

    fn log(&self, record: &Record) {
        if let Some(callback) = LOGFN.get() {
            let evt = LogEvent {
                level: record.level().to_string().to_lowercase(),
                message: record.args().to_string(),
                file: record.file().map(|s| s.to_string()),
                line: record.line(),
            };
            callback(evt);
        } else {
            println!(
                "[{}] {}: {}",
                record.level(),
                record.target(),
                record.args()
            );
        }
    }

    fn flush(&self) {}
}

static LOGGER: JsLogger = JsLogger;

#[napi]
pub fn set_log_callback(
    callback: Function<(LogEvent,), Unknown>,
    max_level: Option<String>,
) -> Result<()> {
    // Safety: We're intentionally leaking the callback reference to make it 'static
    // This is okay because we only set the callback once and it lives for the duration of the program
    let callback_static: Function<'static, (LogEvent,), Unknown> =
        unsafe { std::mem::transmute(callback) };

    let tsfn = callback_static.build_threadsafe_function().build_callback(
        |ctx: ThreadsafeCallContext<(LogEvent,)>| {
            // Extract the first element from the tuple (LogEvent,) to pass just the LogEvent
            Ok(ctx.value.0)
        },
    )?;

    // Wrap the threadsafe function in a closure that we can store
    let _ = LOGFN.set(Box::new(move |evt: LogEvent| {
        let _ = tsfn.call((evt,), ThreadsafeFunctionCallMode::NonBlocking);
    }));

    let lvl = match max_level.as_deref().map(str::to_ascii_lowercase).as_deref() {
        Some("error") => LevelFilter::Error,
        Some("warn") => LevelFilter::Warn,
        Some("info") => LevelFilter::Info,
        Some("debug") => LevelFilter::Debug,
        Some("trace") => LevelFilter::Trace,
        Some("off") => LevelFilter::Off,
        _ => LevelFilter::Trace,
    };
    let _ = log::set_logger(&LOGGER).map(|_| log::set_max_level(lvl));
    Ok(())
}

#[napi(object)]
pub struct StatsEvent {
    pub name: String,
    pub number: f64,
}

// Store the threadsafe function using a boxed type-erased version
static STATSFN: OnceCell<Box<dyn Fn(StatsEvent) + Send + Sync>> = OnceCell::new();

#[napi]
pub fn set_stats_callback(callback: Function<(StatsEvent,), Unknown>) -> Result<()> {
    // Safety: We're intentionally leaking the callback reference to make it 'static
    // This is okay because we only set the callback once and it lives for the duration of the program
    let callback_static: Function<'static, (StatsEvent,), Unknown> =
        unsafe { std::mem::transmute(callback) };

    let tsfn = callback_static.build_threadsafe_function().build_callback(
        |ctx: ThreadsafeCallContext<(StatsEvent,)>| {
            // Extract the first element from the tuple (StatsEvent,) to pass just the StatsEvent
            Ok(ctx.value.0)
        },
    )?;

    // Wrap the threadsafe function in a closure that we can store
    let _ = STATSFN.set(Box::new(move |evt: StatsEvent| {
        let _ = tsfn.call((evt,), ThreadsafeFunctionCallMode::NonBlocking);
    }));

    Ok(())
}

pub fn stats(name: &str, number: f64) {
    if let Some(callback) = STATSFN.get() {
        let evt = StatsEvent {
            name: name.to_string(),
            number: number,
        };
        callback(evt);
    }
}

// Import Silero VAD
mod silero;
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
    callback: Box<dyn Fn(Transcription) + Send + Sync>,
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
        callback: Function<Transcription, Unknown>,
    ) -> Result<Self> {
        info!("Initializing TranscriptionStream with assets: {}", assets_path);

        let assets = PathBuf::from(assets_path);

        // Get device
        let device = parakeet::get_device()
            .map_err(|e| napi::Error::from_reason(format!("Device error: {}", e)))?;

        info!("Loading Silero VAD...");
        // Load VAD model (expect files in assets directory)
        let vad_path = assets.join("vad16.safetensors");
        let vad_config_path = assets.join("vad16.config.json");

        let vad = SileroVad::load(
            &device,
            &vad_path,
            &vad_config_path,
        )
        .map_err(|e| napi::Error::from_reason(format!("Failed to load VAD: {}", e)))?;

        info!("Loading Parakeet model...");
        // Load Parakeet model from assets directory
        let parakeet_model = parakeet::load_parakeet_ctc_from_gguf_local(
            &assets,
            &device,
        )
        .map_err(|e| napi::Error::from_reason(format!("Failed to load Parakeet: {}", e)))?;

        info!("Models loaded successfully");

        // Create inner state
        let inner = StreamInner::new(vad, parakeet_model, device)?;

        // Create threadsafe callback and wrap in a closure
        // Safety: We're intentionally leaking the callback reference to make it 'static
        // This is okay because the callback lives for the duration of the TranscriptionStream
        let callback_static: Function<'static, Transcription, Unknown> =
            unsafe { std::mem::transmute(callback) };
        let tsfn = callback_static.build_threadsafe_function().build()?;
        let callback_fn = Box::new(move |t: Transcription| {
            let _ = tsfn.call(t, ThreadsafeFunctionCallMode::NonBlocking);
        });

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
            callback: callback_fn,
        })
    }

    /// Process audio samples and emit transcriptions via callback
    ///
    /// Transcriptions are automatically emitted when VAD detects pauses in speech.
    /// No explicit flushing needed - just keep feeding audio samples.
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
            (self.callback)(transcription);
        }

        Ok(())
    }
}

// Parakeet speech recognition library
pub mod parakeet;
