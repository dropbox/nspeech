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
    parakeet_model: parakeet::ParakeetCtc,
    device: candle_core::Device,

    // Streaming buffer for continuous updates
    streaming_buffer: streaming_buffer::StreamingBuffer,
    block_duration_samples: usize, // How often to transcribe (750ms default)
    samples_since_transcribe: usize,

    // Accumulated samples for current speech segment (VAD-based)
    current_segment: Vec<f32>,
    current_segment_start: Option<f64>, // Start time in seconds

    // Pre-buffer to capture audio before speech detection (300ms)
    pre_buffer: std::collections::VecDeque<f32>,

    // Sub-segments (phrases) for comma insertion
    // Tracks sample positions where pauses occurred (comma boundaries)
    phrase_boundaries: Vec<usize>, // Sample indices where commas should go

    // Tracking
    total_samples_processed: usize,
    silence_frames: usize,
    was_speech_last_frame: bool,  // Track speech->silence transitions
    last_audio_time: std::time::Instant, // Track when we last received audio

    // Configuration (VAD mode)
    speech_threshold: f32,
    min_speech_duration_ms: f32,
    pre_buffer_ms: f32,            // Pre-buffer duration (300ms)
    comma_pause_duration_ms: f32, // Short pause → comma (150ms)
    period_pause_duration_ms: f32, // Long pause → period (500ms - longer to avoid breaking natural speech)
    silence_timeout_ms: f32,       // Very long pause → auto-flush (2000ms - end of turn)

    // Debug WAV writer
    debug_wav_writer: Option<WavWriter<std::io::BufWriter<std::fs::File>>>,
}

impl SpeechInner {
    fn new(
        vad: SileroVad,
        parakeet_model: parakeet::ParakeetCtc,
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

        info!("VAD mode: enabled (speech detection with pause-based punctuation)");
        info!("Streaming mode: enabled (continuous buffer updates every 750ms)");

        // Pre-buffer to capture audio before speech detection
        let pre_buffer_ms = 300.0;
        let pre_buffer_samples = (pre_buffer_ms * 16.0) as usize; // 4800 samples at 16kHz
        let pre_buffer = std::collections::VecDeque::with_capacity(pre_buffer_samples);

        // Streaming buffer configuration (matches index.html)
        let max_buffer_secs = 10.0; // 10 second rolling window
        let overlap_secs = 0.25;     // 250ms overlap
        let streaming_buffer = streaming_buffer::StreamingBuffer::new(max_buffer_secs, overlap_secs, 16000);

        // Transcribe every 750ms (matches index.html BLOCK_DURATION)
        let block_duration_samples = (0.75 * 16000.0) as usize; // 12000 samples

        Ok(Self {
            vad_stream,
            parakeet_model,
            device,
            streaming_buffer,
            block_duration_samples,
            samples_since_transcribe: 0,
            current_segment: Vec::new(),
            current_segment_start: None,
            pre_buffer,
            phrase_boundaries: Vec::new(),
            total_samples_processed: 0,
            silence_frames: 0,
            was_speech_last_frame: false,
            last_audio_time: std::time::Instant::now(),
            speech_threshold: 0.5,
            min_speech_duration_ms: 250.0,
            pre_buffer_ms,
            comma_pause_duration_ms: 150.0,  // Short pause → comma
            period_pause_duration_ms: 500.0, // Long pause → period (increased to avoid breaking natural speech)
            silence_timeout_ms: 2000.0,      // Very long pause → auto-flush segment
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
        // Write incoming samples to debug WAV file
        if let Some(ref mut writer) = self.debug_wav_writer {
            for &sample in samples {
                let _ = writer.write_sample(sample);
            }
            // Flush constantly as requested for debugging
            let _ = writer.flush();
        }
        //return Ok(());

        // Check if too much time has passed since last audio (silence timeout)
        let time_since_last_audio = self.last_audio_time.elapsed().as_millis() as f32;

        if time_since_last_audio >= self.silence_timeout_ms {
            // Very long pause detected - auto-flush any active segment before processing new audio
            if let Some(start_time) = self.current_segment_start {
                let end_time = self.total_samples_processed as f64 / 16000.0;
                let duration_ms = (end_time - start_time) * 1000.0;

                info!("Silence timeout ({}ms) - auto-flushing segment {:.3}s-{:.3}s (duration={:.0}ms)",
                      time_since_last_audio, start_time, end_time, duration_ms);

                if duration_ms >= self.min_speech_duration_ms as f64 {
                    // Transcribe the segment before clearing
                    match self.transcribe_segment() {
                        Ok(text) => {
                            if !text.is_empty() {
                                info!("Auto-flush: Generated transcription: \"{}\"", text);
                                callback(Transcription {
                                    text,
                                    start_time,
                                    end_time,
                                });
                            }
                        }
                        Err(e) => {
                            info!("Auto-flush: Transcription FAILED: {}", e);
                        }
                    }
                }

                // Clear the segment state for fresh start
                self.current_segment.clear();
                self.current_segment_start = None;
                self.phrase_boundaries.clear();
                self.silence_frames = 0;
            }
        }

        // Update last audio time
        self.last_audio_time = std::time::Instant::now();

        // Add samples to streaming buffer for continuous updates
        let should_commit = self.streaming_buffer.push_samples(samples);
        self.samples_since_transcribe += samples.len();

        // Check if it's time to transcribe the rolling buffer (every 750ms)
        if self.samples_since_transcribe >= self.block_duration_samples {
            let buffer = self.streaming_buffer.get_buffer();
            if !buffer.is_empty() {
                let buffer_duration = self.streaming_buffer.buffer_duration_secs(16000);
                let start_time = (self.total_samples_processed as f64 / 16000.0) - buffer_duration as f64;
                let end_time = self.total_samples_processed as f64 / 16000.0;

                info!("Streaming: Transcribing buffer ({:.2}s)", buffer_duration);

                // Transcribe the rolling buffer
                match parakeet::transcribe_streaming_chunk(
                    &buffer,
                    None,
                    None,
                    &self.parakeet_model,
                    &self.device,
                ) {
                    Ok(raw_text) => {
                        if !raw_text.is_empty() {
                            let text = parakeet::add_punctuation(&raw_text);
                            info!("Streaming: Update - \"{}\"", text);

                            // Update streaming buffer's current line
                            self.streaming_buffer.update_current_line(text.clone());

                            // Invoke callback immediately
                            callback(Transcription {
                                text,
                                start_time,
                                end_time,
                            });
                        }
                    }
                    Err(e) => {
                        info!("Streaming: Transcription error: {}", e);
                    }
                }
            }

            self.samples_since_transcribe = 0;

            // Handle buffer commit if rolling window is full
            if should_commit {
                info!("Streaming: Buffer full, committing line");
                self.streaming_buffer.commit_and_trim(samples.len());
            }
        }

        // Process through VAD in 160-sample chunks (10ms at 16kHz)
        // Key: VAD probabilities indicate speech state, but we accumulate samples
        // independently based on that state to avoid duplication
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
                    // Check if speech is resuming after silence within an active segment
                    if self.current_segment_start.is_some() && !self.was_speech_last_frame && self.silence_frames > 0 {
                        // Speech resumed after a brief pause - prepend pre-buffer to capture start of resumed speech
                        let pre_buffer_len = self.pre_buffer.len();
                        if pre_buffer_len > 0 {
                            info!("VAD: Speech resumed after {}ms pause, prepending {}ms pre-buffer",
                                  self.silence_frames as f32 * 32.0, pre_buffer_len as f32 / 16.0);
                            // Insert pre-buffer before the current position
                            let insert_pos = self.current_segment.len();
                            self.current_segment.reserve(pre_buffer_len);
                            self.current_segment.extend(self.pre_buffer.iter().copied());
                            // Rotate to put pre-buffer before current position
                            self.current_segment[insert_pos..].rotate_right(pre_buffer_len);
                        }
                        self.pre_buffer.clear();
                    }

                    self.silence_frames = 0;
                    self.was_speech_last_frame = true;

                    if self.current_segment_start.is_none() {
                        // Start new speech segment
                        // Account for pre-buffer when calculating start time
                        let pre_buffer_duration = self.pre_buffer.len() as f64 / 16000.0;
                        let start_time = (self.total_samples_processed as f64 / 16000.0) - pre_buffer_duration;
                        self.current_segment_start = Some(start_time);
                        self.current_segment.clear();

                        // Include pre-buffered audio to capture start of speech
                        self.current_segment.extend(self.pre_buffer.iter().copied());
                        info!("VAD: Speech started at {:.3}s (prob={:.3}, pre-buffer={}ms)",
                              start_time, prob, pre_buffer_duration * 1000.0);
                        self.pre_buffer.clear();
                    }
                } else {
                    self.was_speech_last_frame = false;
                    // Silence detected
                    if self.current_segment_start.is_some() {
                        self.silence_frames += 1;
                        let silence_duration_ms = self.silence_frames as f32 * 32.0; // 32ms per frame

                        // Check for comma pause (short pause)
                        if silence_duration_ms >= self.comma_pause_duration_ms
                            && silence_duration_ms < self.period_pause_duration_ms
                            && self.silence_frames == (self.comma_pause_duration_ms / 32.0).ceil() as usize {
                            // Mark phrase boundary for comma insertion
                            let boundary_pos = self.current_segment.len();
                            if boundary_pos > 0 {
                                self.phrase_boundaries.push(boundary_pos);
                                info!("VAD: Comma pause detected at sample {} ({}ms pause)",
                                      boundary_pos, silence_duration_ms);
                            }
                        }

                        // Check for period pause (long pause - end segment)
                        if silence_duration_ms >= self.period_pause_duration_ms {
                            // End current segment and transcribe
                            let start_time = self.current_segment_start.unwrap();
                            let end_time = self.total_samples_processed as f64 / 16000.0;
                            let duration_ms = (end_time - start_time) * 1000.0;

                            info!("VAD: Period pause - speech ended at {:.3}s (duration={:.0}ms, samples={}, phrases={})",
                                  end_time, duration_ms, self.current_segment.len(), self.phrase_boundaries.len() + 1);

                            if duration_ms >= self.min_speech_duration_ms as f64 {
                                // Transcribe the accumulated segment with phrase boundaries
                                info!("VAD: Transcribing segment {:.3}s-{:.3}s", start_time, end_time);
                                match self.transcribe_segment() {
                                    Ok(text) => {
                                        if !text.is_empty() {
                                            info!("VAD: Generated transcription: \"{}\"", text);
                                            callback(Transcription {
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
                            self.phrase_boundaries.clear();
                            self.silence_frames = 0;
                        }
                    }
                }

                self.total_samples_processed += 512;
            }

            // Always maintain pre-buffer during silence (even within an active segment)
            // This ensures we capture the start of resumed speech after brief pauses
            if !self.was_speech_last_frame {
                let pre_buffer_max = (self.pre_buffer_ms * 16.0) as usize;
                for &sample in chunk {
                    if self.pre_buffer.len() >= pre_buffer_max {
                        self.pre_buffer.pop_front();
                    }
                    self.pre_buffer.push_back(sample);
                }
            }

            // Accumulate chunk samples if we're in an active speech segment
            // This happens outside the probability loop to avoid duplication
            if self.current_segment_start.is_some() {
                self.current_segment.extend_from_slice(chunk);
            }

            idx = end;
        }

        Ok(())
    }

    fn transcribe_segment(&self) -> Result<String> {
        if self.current_segment.is_empty() {
            info!("Transcribe: Segment is empty, returning empty string");
            return Ok(String::new());
        }

        info!("Transcribe: Processing {} samples with {} phrase boundaries",
              self.current_segment.len(), self.phrase_boundaries.len());

        // If we have phrase boundaries (comma pauses), split and transcribe each phrase
        if !self.phrase_boundaries.is_empty() {
            let mut phrases = Vec::new();
            let mut start_idx = 0;

            // Transcribe each phrase between boundaries
            for &boundary_pos in &self.phrase_boundaries {
                if boundary_pos > start_idx && boundary_pos <= self.current_segment.len() {
                    let phrase_samples = &self.current_segment[start_idx..boundary_pos];
                    let raw_phrase = parakeet::transcribe_streaming_chunk(
                        phrase_samples,
                        None,
                        None,
                        &self.parakeet_model,
                        &self.device,
                    )
                    .map_err(|e| {
                        let err = format!("Phrase transcription error: {}", e);
                        info!("Transcribe: ERROR - {}", err);
                        napi::Error::from_reason(err)
                    })?;

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
                )
                .map_err(|e| {
                    let err = format!("Final phrase transcription error: {}", e);
                    info!("Transcribe: ERROR - {}", err);
                    napi::Error::from_reason(err)
                })?;

                if !raw_phrase.is_empty() {
                    phrases.push(raw_phrase);
                }
            }

            // Join phrases with comma marker
            let raw_text = phrases.join(" , ");
            let text = parakeet::add_punctuation_internal(&raw_text, true);

            info!("Transcribe: Raw phrases: {} -> \"{}\"", phrases.len(), raw_text);
            info!("Transcribe: With punctuation: \"{}\"", text);
            Ok(text)
        } else {
            // No phrase boundaries - single phrase
            let raw_text = parakeet::transcribe_streaming_chunk(
                &self.current_segment,
                None,
                None,
                &self.parakeet_model,
                &self.device,
            )
            .map_err(|e| {
                let err = format!("Transcription error: {}", e);
                info!("Transcribe: ERROR - {}", err);
                napi::Error::from_reason(err)
            })?;

            let text = parakeet::add_punctuation(&raw_text);

            info!("Transcribe: Raw: \"{}\"", raw_text);
            info!("Transcribe: With punctuation: \"{}\"", text);
            Ok(text)
        }
    }

    fn flush<F>(&mut self, callback: &F) -> Result<()>
    where
        F: Fn(Transcription),
    {
        let mut transcription_count = 0;

        // First, flush the streaming buffer if it has content
        let buffer = self.streaming_buffer.get_buffer();
        if !buffer.is_empty() {
            let buffer_duration = self.streaming_buffer.buffer_duration_secs(16000);
            let start_time = (self.total_samples_processed as f64 / 16000.0) - buffer_duration as f64;
            let end_time = self.total_samples_processed as f64 / 16000.0;

            info!("Flush: Transcribing streaming buffer ({:.2}s)", buffer_duration);

            match parakeet::transcribe_streaming_chunk(
                &buffer,
                None,
                None,
                &self.parakeet_model,
                &self.device,
            ) {
                Ok(raw_text) => {
                    if !raw_text.is_empty() {
                        let text = parakeet::add_punctuation(&raw_text);
                        info!("Flush: Streaming buffer - \"{}\"", text);

                        callback(Transcription {
                            text,
                            start_time,
                            end_time,
                        });
                        transcription_count += 1;
                    }
                }
                Err(e) => {
                    info!("Flush: Streaming buffer transcription error: {}", e);
                }
            }
        }

        // Also flush VAD-based segment if present
        if let Some(start_time) = self.current_segment_start {
            if !self.current_segment.is_empty() {
                let end_time = self.total_samples_processed as f64 / 16000.0;
                let duration_ms = (end_time - start_time) * 1000.0;

                info!("Flush: VAD segment {:.3}s-{:.3}s (duration={:.0}ms, samples={}, phrases={})",
                      start_time, end_time, duration_ms, self.current_segment.len(), self.phrase_boundaries.len() + 1);

                if duration_ms >= self.min_speech_duration_ms as f64 {
                    match self.transcribe_segment() {
                        Ok(text) => {
                            if !text.is_empty() {
                                info!("Flush: VAD segment - \"{}\"", text);
                                callback(Transcription {
                                    text,
                                    start_time,
                                    end_time,
                                });
                                transcription_count += 1;
                            }
                        }
                        Err(e) => {
                            info!("Flush: VAD segment transcription error: {}", e);
                        }
                    }
                }
            }

            // Clear VAD state
            self.current_segment.clear();
            self.current_segment_start = None;
            self.phrase_boundaries.clear();
            self.silence_frames = 0;
        }

        // Clear streaming buffer state
        self.streaming_buffer.clear();
        self.samples_since_transcribe = 0;

        if transcription_count == 0 {
            info!("Flush: No content to flush");
        } else {
            info!("Flush: Invoked callback {} time(s)", transcription_count);
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

        // Create bounded work queue (buffer up to 150 chunks ~ 9.6MB)
        // If queue is full, queue will be drained to admit new samples
        let (tx, rx) = mpsc::sync_channel::<WorkItem>(150);
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
