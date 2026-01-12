/* Node module API for Parakeet Speech Recognition */

use napi::Result;
use napi_derive::napi;

use std::{
    path::PathBuf,
    sync::{Arc, Mutex, mpsc},
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

// Streaming buffer module (shared between Node.js and CLI examples)
pub mod streaming_buffer;

// Streaming transcription with VAD-based segmentation
pub mod streaming_transcriber;

// Qwen3 model for text correction (punctuation, capitalization)
// Only available when "qwen" feature is enabled
#[cfg(feature = "qwen")]
pub mod qwen;

/// Transcription result with timestamp
#[napi(object)]
pub struct Transcription {
    pub text: String,
    pub raw_text: String,
    pub start_time: f64,
    pub end_time: f64,
}

/// Inner state for transcription stream
struct SpeechInner {
    transcriber: streaming_transcriber::StreamingTranscriber,

    // Debug WAV writer
    debug_wav_writer: Option<WavWriter<std::io::BufWriter<std::fs::File>>>,
}

impl SpeechInner {
    fn new(
        vad: SileroVad,
        parakeet_model: parakeet::ParakeetCtc,
        device: candle_core::Device,
        #[cfg(feature = "qwen")]
        qwen_corrector: Option<qwen::QwenCorrector>,
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

        info!("VAD mode: enabled (speech detection with pause-based punctuation)");

        // Create streaming transcriber with default config
        let config = streaming_transcriber::StreamingConfig::default();
        let transcriber = streaming_transcriber::StreamingTranscriber::new(
            vad_stream,
            parakeet_model,
            device,
            config,
            #[cfg(feature = "qwen")]
            qwen_corrector,
        );

        Ok(Self {
            transcriber,
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

    fn process_samples<F>(&mut self, samples: &[f32], callback: &F) -> Result<()>
    where
        F: Fn(Transcription),
    {
        // Process samples through the streaming transcriber
        let segments = self.transcriber.process_samples(samples)
            .map_err(|e| napi::Error::from_reason(format!("Transcription error: {}", e)))?;

        // Handle completed segments
        for segment in segments {
            // Write segment to debug WAV file (after VAD processing)
            self.write_segment_to_debug_wav(&segment);

            if !segment.text.is_empty() {
                info!("Generated transcription: \"{}\"", segment.text);
                callback(Transcription {
                    text: segment.text,
                    raw_text: segment.raw_text,
                    start_time: segment.start_time,
                    end_time: segment.end_time,
                });
            }
        }

        Ok(())
    }

    fn write_segment_to_debug_wav(&mut self, _segment: &streaming_transcriber::TranscriptionSegment) {
        // Note: We can't access the raw audio samples from the segment
        // The debug WAV writing would need to be integrated into StreamingTranscriber
        // For now, we'll skip this feature or implement it differently
        // TODO: Consider passing debug_wav_writer to StreamingTranscriber
    }

    fn flush<F>(&mut self, callback: &F) -> Result<()>
    where
        F: Fn(Transcription),
    {
        // Flush any remaining segment from the transcriber
        if let Some(segment) = self.transcriber.flush()
            .map_err(|e| napi::Error::from_reason(format!("Flush error: {}", e)))? {
            // Write segment to debug WAV file
            self.write_segment_to_debug_wav(&segment);

            if !segment.text.is_empty() {
                info!("Flush: Generated transcription: \"{}\"", segment.text);
                callback(Transcription {
                    text: segment.text,
                    raw_text: segment.raw_text,
                    start_time: segment.start_time,
                    end_time: segment.end_time,
                });
            }
        }

        Ok(())
    }
}

/// Work item for the background processing queue
enum WorkItem {
    Samples(Vec<f32>),
    Flush,
    Shutdown,
}

#[napi(js_name = "Speech")]
pub struct Speech {
    work_sender: mpsc::SyncSender<WorkItem>,
    work_receiver: Arc<Mutex<mpsc::Receiver<WorkItem>>>,
    worker_thread: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
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

        // Load Qwen3 text correction model if feature is enabled
        #[cfg(feature = "qwen")]
        let qwen_corrector = {
            info!("Loading Qwen3 text correction model...");
            match qwen::QwenCorrector::load(&assets, &device) {
                Ok(corrector) => {
                    info!("✓ Qwen3 loaded (text correction enabled)");
                    Some(corrector)
                }
                Err(e) => {
                    info!("⚠ Failed to load Qwen3: {}", e);
                    info!("  Falling back to rule-based punctuation");
                    None
                }
            }
        };

        info!("Models loaded successfully");

        // Create inner state with VAD
        let inner = SpeechInner::new(
            vad,
            parakeet_model,
            device,
            #[cfg(feature = "qwen")]
            qwen_corrector,
        ).unwrap();

        // Create threadsafe callback and wrap in Arc so it can be cloned for async tasks
        // Safety: We're intentionally leaking the callback reference to make it 'static
        // This is okay because the callback lives for the duration of the Speech
        let callback_static: Function<'static, Transcription, Unknown> =
            unsafe { std::mem::transmute(callback) };
        let tsfn = callback_static.build_threadsafe_function().build().unwrap();
        let callback_fn = Arc::new(Box::new(move |t: Transcription| {
            let _ = tsfn.call(t, ThreadsafeFunctionCallMode::NonBlocking);
        }) as Box<dyn Fn(Transcription) + Send + Sync>);

        // Create bounded work queue (buffer up to 1000 chunks)
        // If queue is full, queue will be drained to admit new samples
        let (tx, rx) = mpsc::sync_channel::<WorkItem>(1000);
        let rx = Arc::new(Mutex::new(rx));

        // Spawn background worker thread
        let mut inner = inner;
        let callback_clone = Arc::clone(&callback_fn);
        let rx_clone = Arc::clone(&rx);

        let worker_thread = std::thread::spawn(move || {
            info!("Background worker thread started");
            loop {
                let work_item = {
                    let rx = rx_clone.lock().unwrap();
                    match rx.recv() {
                        Ok(item) => item,
                        Err(_) => {
                            info!("Worker: Channel closed, exiting");
                            break;
                        }
                    }
                };

                match work_item {
                    WorkItem::Samples(samples) => {
                        let callback = Arc::clone(&callback_clone);
                        let callback_fn = |transcription: Transcription| {
                            info!("Callback: Emitting \"{}\", {:.3}s-{:.3}s",
                                  transcription.text, transcription.start_time, transcription.end_time);
                            callback(transcription);
                        };

                        if let Err(e) = inner.process_samples(&samples, &callback_fn) {
                            info!("Worker: process_samples error: {:?}", e);
                        }
                    }
                    WorkItem::Flush => {
                        let callback = Arc::clone(&callback_clone);
                        let callback_fn = |transcription: Transcription| {
                            info!("Flush callback: Emitting \"{}\", {:.3}s-{:.3}s",
                                  transcription.text, transcription.start_time, transcription.end_time);
                            callback(transcription);
                        };

                        if let Err(e) = inner.flush(&callback_fn) {
                            info!("Worker: flush error: {:?}", e);
                        }
                    }
                    WorkItem::Shutdown => {
                        info!("Worker: Shutdown requested, exiting");
                        break;
                    }
                }
            }
            info!("Background worker thread exited");
        });

        Self {
            work_sender: tx,
            work_receiver: rx,
            worker_thread: Arc::new(Mutex::new(Some(worker_thread))),
        }
    }

    #[napi]
    pub fn input(&self, samples: Vec<f64>) -> Result<()> {
        // Convert f64 to f32
        let samples_f32: Vec<f32> = samples.iter().map(|&x| x as f32).collect();

        // Try to send to queue
        match self.work_sender.try_send(WorkItem::Samples(samples_f32)) {
            Ok(_) => Ok(()),
            Err(mpsc::TrySendError::Full(work_item)) => {
                // Queue is full - we're falling behind
                // Drain the queue to skip to current audio (non-blocking)
                info!("Work queue full, draining stale audio and admitting new samples");

                let mut drained = 0;
                if let Ok(rx) = self.work_receiver.try_lock() {
                    // Drain all queued items using try_recv (non-blocking)
                    while let Ok(_) = rx.try_recv() {
                        drained += 1;
                    }
                }

                if drained > 0 {
                    info!("Drained {} stale items from queue", drained);
                }

                // Now try to send again
                match self.work_sender.try_send(work_item) {
                    Ok(_) => {
                        info!("Sent new samples after draining queue");
                        Ok(())
                    }
                    Err(mpsc::TrySendError::Full(_)) => {
                        // Still full (worker might be processing slowly)
                        info!("Queue still full after drain, dropping samples");
                        Ok(())
                    }
                    Err(mpsc::TrySendError::Disconnected(_)) => {
                        Err(napi::Error::from_reason("Worker thread disconnected"))
                    }
                }
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                Err(napi::Error::from_reason("Worker thread disconnected"))
            }
        }
    }

    #[napi]
    pub fn flush(&self) -> Result<()> {
        info!("flush: queueing flush command");

        // Send flush command to queue
        match self.work_sender.try_send(WorkItem::Flush) {
            Ok(_) => Ok(()),
            Err(mpsc::TrySendError::Full(_)) => {
                info!("Work queue full, dropping flush command");
                Ok(()) // Don't error
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                Err(napi::Error::from_reason("Worker thread disconnected"))
            }
        }
    }

    #[napi]
    pub fn shutdown(&self) {
        info!("Shutdown requested - sending shutdown signal to worker");

        // Send shutdown message
        let _ = self.work_sender.send(WorkItem::Shutdown);

        // Wait for worker thread to finish
        if let Ok(mut guard) = self.worker_thread.lock() {
            if let Some(handle) = guard.take() {
                info!("Waiting for worker thread to finish...");
                if let Err(e) = handle.join() {
                    info!("Worker thread panicked: {:?}", e);
                }
                info!("Worker thread joined successfully");
            }
        }
    }
}
