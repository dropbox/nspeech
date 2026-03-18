/* Node module API for Speech Recognition
 *
 * ASR backend is selected at compile time:
 *   cargo build --release --lib                        # Parakeet TDT (default)
 *   cargo build --release --lib --features use-moonshine  # Moonshine V2
 */

use napi::Result;
use napi_derive::napi;

use std::{
    path::PathBuf,
    sync::{Arc, Mutex, mpsc},
};

use log::{info, warn, LevelFilter, Log, Metadata, Record};
use napi::bindgen_prelude::{Function, Unknown};
use napi::threadsafe_function::{ThreadsafeFunctionCallMode, ThreadsafeCallContext};
use once_cell::sync::OnceCell;
use hound::{WavWriter, WavSpec, SampleFormat};

#[cfg(not(feature = "use-moonshine"))]
use crate::parakeet;
#[cfg(feature = "use-moonshine")]
use crate::moonshine;
use crate::silero;

/// Returns the longest common prefix of two strings, splitting on a char boundary.
#[allow(dead_code)]
fn longest_common_prefix<'a>(a: &'a str, b: &str) -> &'a str {
    let len = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
    // Ensure we don't split a multi-byte UTF-8 character
    match a.get(..len) {
        Some(s) => s,
        None => {
            // Back up to the previous char boundary
            let mut end = len;
            while end > 0 && !a.is_char_boundary(end) {
                end -= 1;
            }
            &a[..end]
        }
    }
}

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
    println!("speech set_logger {:?}", max_level);
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

#[allow(dead_code)]
pub fn stats(name: &str, number: f64) {
    if let Some(callback) = STATSFN.get() {
        let evt = StatsEvent {
            name: name.to_string(),
            number: number,
        };
        callback(evt);
    }
}

/// Transcription result with timestamp
#[napi(object)]
pub struct Transcription {
    pub text: String,
    pub raw_text: String,
    pub start_time: f64,
    pub end_time: f64,
    /// True for streaming partial results that may change; false for final results.
    pub is_partial: bool,
    /// Monotonically increasing index identifying which speech segment this belongs to.
    /// Same index + is_partial=true → update existing element; new index → new element.
    pub segment_index: u32,
    /// Longest prefix of `text` that is stable (unchanged from previous partial).
    /// For final transcriptions, equals `text`. Enables two-tone rendering (confirmed vs tentative).
    pub stable_text: String,
}

/// Inner state for transcription stream
struct SpeechInner {
    vad_stream: Option<silero::VadStream>,
    #[cfg(not(feature = "use-moonshine"))]
    tdt_model: Option<parakeet::TransducerModel>,
    #[cfg(not(feature = "use-moonshine"))]
    feat_extractor: parakeet::ParakeetFeatureExtractor,
    #[cfg(feature = "use-moonshine")]
    moonshine_model: Option<moonshine::MoonshineModel>,
    #[cfg(feature = "use-moonshine")]
    moonshine_stream: Option<moonshine::MoonshineStream>,
    device: candle_core::Device,

    // VAD state
    current_segment: Vec<f32>,
    current_segment_start: Option<f64>,
    pre_buffer: std::collections::VecDeque<f32>,
    silence_frames: usize,
    was_speech_last_frame: bool,
    total_samples_processed: usize,

    // Configuration
    speech_threshold: f32,
    #[allow(dead_code)]
    period_pause_frames: usize,  // Frames of silence to trigger segment end (used with feature flag)

    // Segment tracking for web UX
    segment_index: u32,
    last_partial_text: String,

    // Debug WAV writer
    #[allow(dead_code)]
    debug_wav_writer: Option<WavWriter<std::io::BufWriter<std::fs::File>>>,
}

impl SpeechInner {
    #[cfg(not(feature = "use-moonshine"))]
    fn new(
        vad: Option<silero::SileroVad>,
        model: Option<parakeet::TransducerModel>,
        device: candle_core::Device,
    ) -> Self {
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

        info!("TDT + VAD streaming mode: enabled (VAD segmentation + TDT transcription)");

        // Create VAD stream if VAD was loaded successfully
        let vad_stream = vad.and_then(|v| match silero::VadStream::new(v, &device) {
            Ok(stream) => Some(stream),
            Err(e) => {
                warn!("VAD stream could not be created: {}", e);
                None
            }
        });

        // Feature extractor for TDT (128 mel bins for streaming TDT GGUF model)
        let feat_extractor = parakeet::ParakeetFeatureExtractor::new(128);

        // Pre-buffer: 1 second of audio
        let pre_buffer_samples = 16000;
        let pre_buffer = std::collections::VecDeque::with_capacity(pre_buffer_samples);

        // Configuration
        let speech_threshold = 0.1;  // Low threshold for better capture
        let period_pause_ms = 500.0;  // 500ms silence = segment end
        let period_pause_frames = (period_pause_ms / 32.0) as usize;  // 32ms per VAD frame (512 samples @ 16kHz)

        Self {
            vad_stream,
            tdt_model: model,
            feat_extractor,
            device: device.clone(),
            current_segment: Vec::new(),
            current_segment_start: None,
            pre_buffer,
            silence_frames: 0,
            was_speech_last_frame: false,
            total_samples_processed: 0,
            speech_threshold,
            period_pause_frames,
            segment_index: 0,
            last_partial_text: String::new(),
            debug_wav_writer,
        }
    }

    #[cfg(feature = "use-moonshine")]
    fn new(
        vad: Option<silero::SileroVad>,
        model: Option<moonshine::MoonshineModel>,
        device: candle_core::Device,
    ) -> Self {
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

        info!("Moonshine + VAD streaming mode: enabled (VAD segmentation + Moonshine transcription)");

        // Create VAD stream if VAD was loaded successfully
        let vad_stream = vad.and_then(|v| match silero::VadStream::new(v, &device) {
            Ok(stream) => Some(stream),
            Err(e) => {
                warn!("VAD stream could not be created: {}", e);
                None
            }
        });

        // Pre-buffer: 1 second of audio
        let pre_buffer_samples = 16000;
        let pre_buffer = std::collections::VecDeque::with_capacity(pre_buffer_samples);

        // Configuration
        let speech_threshold = 0.1;  // Low threshold for better capture
        let period_pause_ms = 500.0;  // 500ms silence = segment end
        let period_pause_frames = (period_pause_ms / 32.0) as usize;  // 32ms per VAD frame (512 samples @ 16kHz)

        let moonshine_stream = model.as_ref().map(|m| m.stream_new(500, 500));

        Self {
            vad_stream,
            moonshine_model: model,
            moonshine_stream,
            device: device.clone(),
            current_segment: Vec::new(),
            current_segment_start: None,
            pre_buffer,
            silence_frames: 0,
            was_speech_last_frame: false,
            total_samples_processed: 0,
            speech_threshold,
            period_pause_frames,
            segment_index: 0,
            last_partial_text: String::new(),
            debug_wav_writer,
        }
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

    fn process_samples<F>(&mut self, samples: &[f32], _callback: &F) -> Result<()>
    where
        F: Fn(Transcription),
    {
        info!("process_samples {}", samples.len());

        // Write raw audio to debug WAV
        if let Some(writer) = self.debug_wav_writer.as_mut() {
            for &s in samples {
                let _ = writer.write_sample(s);
            }
        }

        // Check if VAD stream is available
        let vad_stream = match self.vad_stream.as_mut() {
            Some(stream) => stream,
            None => {
                warn!("VAD not available");
                return Ok(()); // Fail silently if VAD not available
            }
        };

        // Push samples to VAD stream - it will process them in chunks and return probabilities
        let speech_probs = vad_stream.push(samples)
            .map_err(|e| napi::Error::from_reason(format!("VAD error: {}", e)))?;

        // VAD processes in 512-sample chunks (32ms at 16kHz), so we must align with that
        // Previously this was 160 samples (10ms), but that didn't match VAD's chunk size
        const CHUNK_SIZE: usize = 512;  // Must match VAD chunk size!
        let mut offset = 0;
        let mut prob_idx = 0;

        // Track speech detection stats for this batch
        let mut speech_chunks = 0;
        let mut total_chunks = 0;

        while offset + CHUNK_SIZE <= samples.len() && prob_idx < speech_probs.len() {
            let chunk = &samples[offset..offset + CHUNK_SIZE];
            let speech_prob = speech_probs[prob_idx];
            let is_speech = speech_prob >= self.speech_threshold;

            total_chunks += 1;
            if is_speech {
                speech_chunks += 1;
            }

            // Update pre-buffer
            self.pre_buffer.extend(chunk.iter().copied());
            if self.pre_buffer.len() > 16000 {  // Keep 1s
                self.pre_buffer.drain(0..(self.pre_buffer.len() - 16000));
            }

            if is_speech {
                // Start new segment if needed
                if self.current_segment.is_empty() {
                    self.segment_index += 1;
                    self.last_partial_text.clear();
                    // Add pre-buffer to catch start of speech
                    self.current_segment.extend(self.pre_buffer.iter().copied());
                    // Use saturating_sub to prevent underflow
                    let start_sample = self.total_samples_processed.saturating_sub(self.pre_buffer.len());
                    self.current_segment_start = Some(start_sample as f64 / 16000.0);
                    info!("Speech started at {:.2}s (segment {})", self.current_segment_start.unwrap(), self.segment_index);
                }

                // Add current chunk
                self.current_segment.extend_from_slice(chunk);
                self.silence_frames = 0;

                // Streaming: try partial transcription (Moonshine only)
                #[cfg(feature = "use-moonshine")]
                if let (Some(model), Some(stream)) = (&self.moonshine_model, &mut self.moonshine_stream) {
                    let start_time = self.current_segment_start.unwrap_or(0.0);
                    match model.stream_try_update(stream, &self.current_segment, &self.device) {
                        Ok(Some(text)) => {
                            let trimmed = text.trim();
                            if !trimmed.is_empty() {
                                let stable_text = longest_common_prefix(&self.last_partial_text, trimmed).to_string();
                                self.last_partial_text = trimmed.to_string();
                                info!("Streaming partial: \"{}\" (stable: \"{}\")", trimmed, stable_text);
                                _callback(Transcription {
                                    text: trimmed.to_string(),
                                    raw_text: trimmed.to_string(),
                                    start_time,
                                    end_time: start_time + (self.current_segment.len() as f64 / 16000.0),
                                    is_partial: true,
                                    segment_index: self.segment_index,
                                    stable_text,
                                });
                            }
                        }
                        Ok(None) => {}
                        Err(e) => {
                            warn!("Streaming partial transcription error: {}", e);
                        }
                    }
                }
            } else if !self.current_segment.is_empty() {
                // In silence but have active segment
                self.current_segment.extend_from_slice(chunk);
                self.silence_frames += 1;

                // Automatic pause-based transcription (only with "auto-transcribe-on-pause" feature)
                // When disabled: segment accumulates until explicit flush() call
                // When enabled: segment auto-transcribes after period_pause_frames of silence
                #[cfg(feature = "auto-transcribe-on-pause")]
                if self.silence_frames >= self.period_pause_frames {
                    info!("Segment ended at {:.2}s after {} frames of silence",
                          self.total_samples_processed as f64 / 16000.0, self.silence_frames);

                    if let Some(transcription) = self.transcribe_segment()? {
                        info!("Transcription: \"{}\" ({:.2}s-{:.2}s)",
                              transcription.text, transcription.start_time, transcription.end_time);
                        _callback(transcription);
                    }

                    // Reset for next segment
                    self.current_segment.clear();
                    self.current_segment_start = None;
                    self.silence_frames = 0;
                    self.last_partial_text.clear();
                    info!("Ready for next segment");
                }
            }

            self.was_speech_last_frame = is_speech;
            self.total_samples_processed += CHUNK_SIZE;
            offset += CHUNK_SIZE;
            prob_idx += 1;
        }

        // Log batch summary
        info!("Batch processed: {} chunks ({} speech, {} silence), total time: {:.1}s",
              total_chunks, speech_chunks, total_chunks - speech_chunks,
              self.total_samples_processed as f64 / 16000.0);

        Ok(())
    }

    #[cfg(not(feature = "use-moonshine"))]
    fn transcribe_segment(&mut self) -> Result<Option<Transcription>> {
        use candle_core::DType;

        // Check if TDT model is available
        let tdt_model = match self.tdt_model.as_ref() {
            Some(model) => model,
            None => return Ok(None),
        };

        let segment_duration_ms = self.current_segment.len() as f64 / 16.0;

        if self.current_segment.len() < 4000 {  // Skip very short segments (< 250ms)
            info!("Skipping short segment: {:.0}ms ({} samples)", segment_duration_ms, self.current_segment.len());
            return Ok(None);
        }

        let start_time = self.current_segment_start.unwrap_or(0.0);
        let end_time = start_time + (self.current_segment.len() as f64 / 16000.0);

        info!("Transcribing segment: {:.2}s-{:.2}s ({:.1}s, {} samples)",
              start_time, end_time, end_time - start_time, self.current_segment.len());

        // Extract features
        let features = self.feat_extractor.extract_to_tensor(&self.current_segment, &self.device)
            .map_err(|e| napi::Error::from_reason(format!("Feature extraction error: {}", e)))?;

        // Convert to BF16 if on GPU
        let features = if !self.device.is_cpu() {
            features.to_dtype(DType::BF16)
                .map_err(|e| napi::Error::from_reason(format!("DType conversion error: {}", e)))?
        } else {
            features
        };

        // Run encoder
        let encoder_out = tdt_model.encoder.forward(&features, false)
            .map_err(|e| napi::Error::from_reason(format!("Encoder error: {}", e)))?;

        // Run TDT beam decode with beam_size=2 for quality
        let tokens = tdt_model.beam_decode(&encoder_out, 2)
            .map_err(|e| napi::Error::from_reason(format!("TDT decode error: {}", e)))?;

        if tokens.is_empty() {
            return Ok(None);
        }

        // Decode tokens to text
        let text = tdt_model.decode_tokens(&tokens)
            .map_err(|e| napi::Error::from_reason(format!("Token decode error: {}", e)))?;

        if text.trim().is_empty() {
            return Ok(None);
        }

        let text = text.trim().to_string();
        Ok(Some(Transcription {
            stable_text: text.clone(),
            text: text.clone(),
            raw_text: text,
            start_time,
            end_time,
            is_partial: false,
            segment_index: self.segment_index,
        }))
    }

    #[cfg(feature = "use-moonshine")]
    fn transcribe_segment(&mut self) -> Result<Option<Transcription>> {
        // Check if Moonshine model is available
        let model = match self.moonshine_model.as_ref() {
            Some(model) => model,
            None => return Ok(None),
        };

        let segment_duration_ms = self.current_segment.len() as f64 / 16.0;

        if self.current_segment.len() < 4000 {  // Skip very short segments (< 250ms)
            info!("Skipping short segment: {:.0}ms ({} samples)", segment_duration_ms, self.current_segment.len());
            return Ok(None);
        }

        let start_time = self.current_segment_start.unwrap_or(0.0);
        let end_time = start_time + (self.current_segment.len() as f64 / 16000.0);

        info!("Transcribing segment: {:.2}s-{:.2}s ({:.1}s, {} samples)",
              start_time, end_time, end_time - start_time, self.current_segment.len());

        // Use stream_finalize if streaming state exists, otherwise fall back to transcribe
        let text = if let Some(stream) = self.moonshine_stream.as_mut() {
            model.stream_finalize(stream, &self.current_segment, &self.device)
                .map_err(|e| napi::Error::from_reason(format!("Moonshine transcription error: {}", e)))?
        } else {
            model.transcribe(&self.current_segment, &self.device)
                .map_err(|e| napi::Error::from_reason(format!("Moonshine transcription error: {}", e)))?
        };

        if text.trim().is_empty() {
            return Ok(None);
        }

        let text = text.trim().to_string();
        Ok(Some(Transcription {
            stable_text: text.clone(),
            text: text.clone(),
            raw_text: text,
            start_time,
            end_time,
            is_partial: false,
            segment_index: self.segment_index,
        }))
    }

    fn flush<F>(&mut self, callback: &F) -> Result<()>
    where
        F: Fn(Transcription),
    {
        info!("Flush called: current_segment has {} samples, total processed: {:.2}s",
              self.current_segment.len(), self.total_samples_processed as f64 / 16000.0);

        // Transcribe any remaining segment
        if !self.current_segment.is_empty() {
            if let Some(transcription) = self.transcribe_segment()? {
                info!("Flush transcription: \"{}\" ({:.2}s-{:.2}s)",
                      transcription.text, transcription.start_time, transcription.end_time);
                callback(transcription);
            }
        } else {
            info!("Flush: no remaining segment to transcribe");
        }

        // Flush debug WAV writer so the file is readable
        if let Some(writer) = self.debug_wav_writer.as_mut() {
            let _ = writer.flush();
        }

        // Clear all state buffers to prevent audio bleed between utterances
        self.current_segment.clear();
        self.current_segment_start = None;
        self.pre_buffer.clear();  // Critical: prevents previous utterance from bleeding into next
        self.silence_frames = 0;
        self.was_speech_last_frame = false;
        self.last_partial_text.clear();

        // Reset Moonshine streaming state
        #[cfg(feature = "use-moonshine")]
        if let Some(stream) = self.moonshine_stream.as_mut() {
            stream.reset();
            info!("Moonshine stream state reset");
        }

        // Reset VAD stream LSTM states to prevent context bleeding
        if let Some(vad_stream) = self.vad_stream.as_mut() {
            if let Err(e) = vad_stream.reset() {
                warn!("Failed to reset VAD stream: {}", e);
            } else {
                info!("VAD stream reset successfully");
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
        #[cfg(not(feature = "use-moonshine"))]
        let device = match parakeet::get_device() {
            Ok(device) => device,
            Err(e) => {
                warn!("Device could not be initialized: {}", e);
                candle_core::Device::Cpu
            }
        };
        #[cfg(feature = "use-moonshine")]
        let device = {
            // Moonshine uses the same device selection logic
            if std::env::var("PARAKEET_DEVICE").as_deref() == Ok("cpu") {
                info!("Using CPU (forced by PARAKEET_DEVICE=cpu)");
                candle_core::Device::Cpu
            } else {
                #[cfg(target_os = "macos")]
                match candle_core::Device::new_metal(0) {
                    Ok(d) => { info!("Using Metal GPU acceleration"); d }
                    Err(_) => { info!("Metal not available, using CPU"); candle_core::Device::Cpu }
                }
                #[cfg(not(target_os = "macos"))]
                candle_core::Device::Cpu
            }
        };

        info!("Loading Silero VAD (memory-mapped GGUF)...");
        let vad = match silero::SileroVad::load_from_gguf_mmap(&assets, &device) {
            Ok(vad) => {
                info!("VAD loaded (Q8_0 storage format)");
                Some(vad)
            }
            Err(e) => {
                warn!("Failed to load VAD: {}", e);
                None
            }
        };

        #[cfg(not(feature = "use-moonshine"))]
        let model = {
            info!("Loading Parakeet TDT model (quantized GGUF, mmap)...");
            match parakeet::load_parakeet_tdt_from_gguf_mmap_local(&assets, &device) {
                Ok(mut model) => {
                    info!("Loading tokenizer...");
                    match model.load_tokenizer(&assets) {
                        Ok(_) => {
                            info!("TDT model and tokenizer loaded");
                            Some(model)
                        }
                        Err(e) => {
                            warn!("Failed to load tokenizer: {}", e);
                            None
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to load TDT model: {}", e);
                    None
                }
            }
        };

        #[cfg(feature = "use-moonshine")]
        let model = {
            info!("Loading Moonshine V2 model (quantized GGUF, mmap)...");
            match moonshine::MoonshineModel::load_from_gguf_mmap(&assets, &device) {
                Ok(model) => {
                    info!("Moonshine V2 model loaded");
                    Some(model)
                }
                Err(e) => {
                    warn!("Failed to load Moonshine model: {}", e);
                    None
                }
            }
        };

        if vad.is_some() && model.is_some() {
            info!("Models loaded successfully");
        } else {
            warn!("Some models failed to load - transcription will not work");
        }

        let inner = SpeechInner::new(vad, model, device);

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
            let mut work_items_processed = 0;
            loop {
                let work_item = {
                    let rx = rx_clone.lock().unwrap();
                    match rx.recv() {
                        Ok(item) => item,
                        Err(_) => {
                            info!("Worker: Channel closed, exiting (processed {} items)", work_items_processed);
                            break;
                        }
                    }
                };

                work_items_processed += 1;
                match work_item {
                    WorkItem::Samples(samples) => {
                        info!("Worker: Processing samples chunk #{} ({} samples)", work_items_processed, samples.len());
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
        match self.work_sender.try_send(WorkItem::Samples(samples_f32.clone())) {
            Ok(_) => {
                info!("input: Queued {} samples successfully", samples_f32.len());
                Ok(())
            },
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
