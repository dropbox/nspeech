/* Node module API for Parakeet Speech Recognition */

use napi::{Env, Result, Task};
use napi_derive::napi;

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use log::{info, LevelFilter, Log, Metadata, Record};
use napi::bindgen_prelude::{Function, Unknown};
use napi::threadsafe_function::{ThreadsafeFunctionCallMode, ThreadsafeCallContext};
use once_cell::sync::OnceCell;
use hound::{WavWriter, WavSpec, SampleFormat};

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

// Parakeet speech recognition library (must be declared before silero, which imports VAD assets from it)
pub mod parakeet;

// Silero VAD module
pub mod silero;
use silero::{SileroVad, VadStream};

/// Transcription result with timestamp
#[napi(object)]
pub struct Transcription {
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
}

/// Inner state for transcription stream
struct SpeechInner {
    vad_stream: VadStream,
    parakeet_model: parakeet::ParakeetFastConformerCtc,
    device: candle_core::Device,

    // Accumulated samples for current speech segment
    current_segment: Vec<f32>,
    current_segment_start: Option<f64>, // Start time in seconds

    // Tracking
    total_samples_processed: usize,
    silence_frames: usize,

    // Configuration (VAD mode)
    speech_threshold: f32,
    min_speech_duration_ms: f32,
    min_silence_duration_ms: f32,

    // Debug WAV writer
    debug_wav_writer: Option<WavWriter<std::io::BufWriter<std::fs::File>>>,
}

impl SpeechInner {
    fn new(
        vad: SileroVad,
        parakeet_model: parakeet::ParakeetFastConformerCtc,
        device: candle_core::Device,
    ) -> Result<Self> {
        let vad_stream = VadStream::new(vad, &device)
            .map_err(|e| napi::Error::from_reason(format!("Failed to create VAD stream: {}", e)))?;

        // Create debug WAV writer
        let debug_wav_writer = match Self::create_debug_wav_writer() {
            Ok(writer) => {
                info!("Debug WAV writer created: debug_input.wav");
                Some(writer)
            }
            Err(e) => {
                info!("Failed to create debug WAV writer: {}", e);
                None
            }
        };

        info!("VAD mode: enabled (speech detection)");
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
            debug_wav_writer,
        })
    }

    fn create_debug_wav_writer() -> std::result::Result<WavWriter<std::io::BufWriter<std::fs::File>>, hound::Error> {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        WavWriter::create("debug_input.wav", spec)
    }

    fn process_samples(&mut self, samples: &[f32]) -> Result<Vec<Transcription>> {
        // Write incoming samples to debug WAV file
        if let Some(ref mut writer) = self.debug_wav_writer {
            for &sample in samples {
                let _ = writer.write_sample(sample);
            }
            // Flush constantly as requested for debugging
            let _ = writer.flush();
        }

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
                        let start_time = self.total_samples_processed as f64 / 16000.0;
                        self.current_segment_start = Some(start_time);
                        self.current_segment.clear();
                        info!("VAD: Speech started at {:.3}s (prob={:.3})", start_time, prob);
                    }

                    // Add samples to current segment
                    // Note: We accumulate the full chunk even though VAD probs correspond to ~512 samples
                    // This ensures we capture all speech audio
                    if self.current_segment_start.is_some() {
                        self.current_segment.extend_from_slice(chunk);
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

                            info!("VAD: Speech ended at {:.3}s (duration={:.0}ms, samples={})",
                                  end_time, duration_ms, self.current_segment.len());

                            if duration_ms >= self.min_speech_duration_ms as f64 {
                                // Transcribe the accumulated segment
                                info!("VAD: Transcribing segment {:.3}s-{:.3}s", start_time, end_time);
                                match self.transcribe_segment() {
                                    Ok(text) => {
                                        if !text.is_empty() {
                                            info!("VAD: Generated transcription: \"{}\"", text);
                                            transcriptions.push(Transcription {
                                                text,
                                                start_time,
                                                end_time,
                                            });
                                        } else {
                                            info!("VAD: Transcription was empty (likely silence/noise)");
                                        }
                                    }
                                    Err(e) => {
                                        info!("VAD: Transcription FAILED: {}", e);
                                    }
                                }
                            } else {
                                info!("VAD: Segment too short ({:.0}ms < {:.0}ms), skipping",
                                      duration_ms, self.min_speech_duration_ms);
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
            info!("Transcribe: Segment is empty, returning empty string");
            return Ok(String::new());
        }

        info!("Transcribe: Processing {} samples", self.current_segment.len());

        // Use streaming transcription helper (no context for VAD segments)
        let text = parakeet::transcribe_streaming_chunk(
            &self.current_segment,
            None, // No left context
            None, // No right context
            &self.parakeet_model,
            &self.device,
        )
        .map_err(|e| {
            let err = format!("Transcription error: {}", e);
            info!("Transcribe: ERROR - {}", err);
            napi::Error::from_reason(err)
        })?;

        info!("Transcribe: Result length: {} chars", text.len());
        Ok(text)
    }
}

/// Async task for processing audio samples in the background
struct TranscribeTask {
    inner: Arc<Mutex<SpeechInner>>,
    samples: Vec<f32>,
    callback: Arc<Box<dyn Fn(Transcription) + Send + Sync>>,
}

impl Task for TranscribeTask {
    type Output = Vec<Transcription>;
    type JsValue = ();

    /// Runs on background thread - does the heavy processing
    fn compute(&mut self) -> Result<Self::Output> {
        let mut inner = self.inner.lock()
            .map_err(|e| napi::Error::from_reason(format!("Lock error: {}", e)))?;
        inner.process_samples(&self.samples)
    }

    /// Runs on main JS thread - emits results via callback
    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        info!("Callback: Sending {} transcription(s) to JavaScript", output.len());
        for transcription in output {
            info!("Callback: Emitting \"{}\", {:.3}s-{:.3}s",
                  transcription.text, transcription.start_time, transcription.end_time);
            (self.callback)(transcription);
        }
        Ok(())
    }
}

#[napi(js_name = "Speech")]
pub struct Speech {
    inner: Arc<Mutex<SpeechInner>>,
    callback: Arc<Box<dyn Fn(Transcription) + Send + Sync>>,
}

#[napi]
impl Speech {
    #[napi(constructor)]
    pub fn new(
        assets: String,
        callback: Function<Transcription, Unknown>,
    ) -> Self {
        info!(
            "speech running assets=`{assets}"
        );

        let assets = PathBuf::from(assets);

        // Get device
        let device = parakeet::get_device()
            .map_err(|e| napi::Error::from_reason(format!("Device error: {}", e))).unwrap();

        info!("Loading Silero VAD...");
        // Load VAD model using embed_zst_asset (with optional binary embedding)
        let vad = SileroVad::load(&assets, &device)
            .map_err(|e| napi::Error::from_reason(format!("Failed to load VAD: {}", e))).unwrap();

        info!("Loading Parakeet model...");
        // Load Parakeet model from assets directory
        let parakeet_model = parakeet::load_parakeet_ctc_from_gguf_local(
            &assets,
            &device,
        )
        .map_err(|e| napi::Error::from_reason(format!("Failed to load Parakeet: {}", e))).unwrap();

        info!("Models loaded successfully");

        // Create inner state with VAD
        let inner = SpeechInner::new(vad, parakeet_model, device).unwrap();

        // Create threadsafe callback and wrap in Arc so it can be cloned for async tasks
        // Safety: We're intentionally leaking the callback reference to make it 'static
        // This is okay because the callback lives for the duration of the Speech
        let callback_static: Function<'static, Transcription, Unknown> =
            unsafe { std::mem::transmute(callback) };
        let tsfn = callback_static.build_threadsafe_function().build().unwrap();
        let callback_fn = Arc::new(Box::new(move |t: Transcription| {
            let _ = tsfn.call(t, ThreadsafeFunctionCallMode::NonBlocking);
        }) as Box<dyn Fn(Transcription) + Send + Sync>);

        Self {
            inner: Arc::new(Mutex::new(inner)),
            callback: callback_fn,
        }
    }

    #[napi]
    pub fn input(&self, env: Env, samples: Vec<f64>) -> Result<()> {
        info!("input {} samples (async)", samples.len());

        // Convert f64 to f32
        let samples_f32: Vec<f32> = samples.iter().map(|&x| x as f32).collect();

        // Create async task with cloned Arc references
        let task = TranscribeTask {
            inner: Arc::clone(&self.inner),
            samples: samples_f32,
            callback: Arc::clone(&self.callback),
        };

        // Spawn task on background thread - returns immediately
        env.spawn(task)?;

        Ok(())
    }

    #[napi]
    pub fn shutdown(&self) {
    }
}
