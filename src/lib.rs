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

// Parakeet speech recognition library
pub mod parakeet;

// Silero VAD (Voice Activity Detection)
pub mod silero;

// Streaming buffer module (shared between Node.js and CLI examples)
pub mod streaming_buffer;

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
    transcriber: parakeet::StreamingTransducer,
    feat_extractor: parakeet::ParakeetFeatureExtractor,
    device: candle_core::Device,

    // Accumulated samples and timing
    accumulated_samples: Vec<f32>,
    total_samples_processed: usize,
    last_transcription_time: f64,

    // Debug WAV writer
    debug_wav_writer: Option<WavWriter<std::io::BufWriter<std::fs::File>>>,
}

impl SpeechInner {
    fn new(
        model: parakeet::TransducerModel,
        device: candle_core::Device,
    ) -> Result<Self> {
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

        info!("TDT streaming mode: enabled (automatic alignment, no VAD needed)");

        // Create streaming transcriber with config optimized for Node.js use
        let config = parakeet::StreamingConfig {
            chunk_samples: 16000,    // 1.0s chunks for reasonable latency
            overlap_samples: 4800,   // 0.3s overlap
            emit_partial: true,
        };
        let transcriber = parakeet::StreamingTransducer::new(model, config);

        // Feature extractor for TDT (128 mel bins)
        let feat_extractor = parakeet::ParakeetFeatureExtractor::new(128);

        Ok(Self {
            transcriber,
            feat_extractor,
            device: device.clone(),
            accumulated_samples: Vec::new(),
            total_samples_processed: 0,
            last_transcription_time: 0.0,
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
        use candle_core::DType;

        // Accumulate samples
        self.accumulated_samples.extend_from_slice(samples);
        let chunk_size = 16000; // 1.0s chunks

        // Process complete chunks
        while self.accumulated_samples.len() >= chunk_size {
            let chunk: Vec<f32> = self.accumulated_samples.drain(..chunk_size).collect();

            // Extract features
            let features = self.feat_extractor.extract_to_tensor(&chunk, &self.device)
                .map_err(|e| napi::Error::from_reason(format!("Feature extraction error: {}", e)))?;

            // Convert to BF16 if on GPU
            let features = if !self.device.is_cpu() {
                features.to_dtype(DType::BF16)
                    .map_err(|e| napi::Error::from_reason(format!("DType conversion error: {}", e)))?
            } else {
                features
            };

            // Process through streaming transcriber
            let _new_tokens = self.transcriber.process_features(&features)
                .map_err(|e| napi::Error::from_reason(format!("Transcription error: {}", e)))?;

            // Decode incrementally
            match self.transcriber.decode_text_incremental() {
                Ok((new_text, _)) => {
                    if !new_text.is_empty() {
                        // Calculate timing
                        let start_time = self.last_transcription_time;
                        let end_time = self.total_samples_processed as f64 / 16000.0;

                        info!("Generated transcription: \"{}\" ({:.2}s-{:.2}s)",
                              new_text.trim(), start_time, end_time);

                        callback(Transcription {
                            text: new_text.trim().to_string(),
                            raw_text: new_text.trim().to_string(),
                            start_time,
                            end_time,
                        });

                        self.last_transcription_time = end_time;
                    }
                }
                Err(e) => {
                    info!("Decode error: {}", e);
                }
            }

            self.total_samples_processed += chunk_size;
        }

        Ok(())
    }

    fn flush<F>(&mut self, callback: &F) -> Result<()>
    where
        F: Fn(Transcription),
    {
        use candle_core::DType;

        // Process any remaining accumulated samples
        if !self.accumulated_samples.is_empty() {
            let chunk = self.accumulated_samples.clone();
            self.accumulated_samples.clear();

            // Extract features
            let features = self.feat_extractor.extract_to_tensor(&chunk, &self.device)
                .map_err(|e| napi::Error::from_reason(format!("Feature extraction error: {}", e)))?;

            // Convert to BF16 if on GPU
            let features = if !self.device.is_cpu() {
                features.to_dtype(DType::BF16)
                    .map_err(|e| napi::Error::from_reason(format!("DType conversion error: {}", e)))?
            } else {
                features
            };

            // Process through streaming transcriber
            let _new_tokens = self.transcriber.process_features(&features)
                .map_err(|e| napi::Error::from_reason(format!("Transcription error: {}", e)))?;

            self.total_samples_processed += chunk.len();
        }

        // Get final transcription
        match self.transcriber.decode_text() {
            Ok(final_text) => {
                if !final_text.is_empty() {
                    let start_time = self.last_transcription_time;
                    let end_time = self.total_samples_processed as f64 / 16000.0;

                    info!("Flush: Final transcription: \"{}\" ({:.2}s-{:.2}s)",
                          final_text.trim(), start_time, end_time);

                    callback(Transcription {
                        text: final_text.trim().to_string(),
                        raw_text: final_text.trim().to_string(),
                        start_time,
                        end_time,
                    });
                }
            }
            Err(e) => {
                info!("Flush decode error: {}", e);
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

        info!("Loading Parakeet TDT model...");
        // Load TDT model from assets directory
        let mut model = parakeet::load_parakeet_tdt_from_local(
            &assets,
            &device,
        )
        .map_err(|e| napi::Error::from_reason(format!("Failed to load TDT model: {}", e))).unwrap();

        info!("Loading tokenizer...");
        // Load tokenizer
        model.load_tokenizer(&assets)
            .map_err(|e| napi::Error::from_reason(format!("Failed to load tokenizer: {}", e))).unwrap();

        info!("Models loaded successfully");

        // Create inner state with TDT streaming
        let inner = SpeechInner::new(
            model,
            device,
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
