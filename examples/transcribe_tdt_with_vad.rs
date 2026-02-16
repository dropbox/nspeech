/// VAD-based streaming transcription using Parakeet TDT (Transducer)
///
/// This example demonstrates real-time streaming transcription using VAD
/// to detect natural speech boundaries. Processes audio in chunks and
/// transcribes segments as they complete.
///
/// Key features:
/// - Streaming audio processing (500ms chunks)
/// - VAD-based speech detection
/// - High-quality beam search decoding (matches transcribe_tdt.rs)
/// - Natural pause detection for segment boundaries
///
/// Usage:
///   cargo run --example transcribe_tdt_with_vad --release -- dots.wav
///   cargo run --example transcribe_tdt_with_vad --release -- MLKDream_16k.wav
///   PARAKEET_DEVICE=cpu cargo run --example transcribe_tdt_with_vad --release -- audio.wav

use anyhow::Result;
use speech::parakeet::{get_device, load_parakeet_tdt_from_gguf_mmap_local, TransducerModel};
use speech::silero::{SileroVad, VadStream};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;

/// Performance metrics tracking
#[derive(Debug, Default)]
struct PerformanceMetrics {
    // Model loading times
    vad_load_ms: u128,
    tdt_load_ms: u128,
    audio_load_ms: u128,

    // Processing times
    vad_processing_ms: u128,
    transcription_ms: u128,

    // Per-segment transcription times
    segment_times: Vec<(usize, f64, u128)>, // (segment_id, audio_duration_sec, processing_ms)

    // Speech detection stats
    total_speech_sec: f64,
    total_segments: usize,
}

impl PerformanceMetrics {
    fn add_segment_time(&mut self, segment_id: usize, audio_duration: f64, processing_ms: u128) {
        self.segment_times.push((segment_id, audio_duration, processing_ms));
        self.total_speech_sec += audio_duration;
        self.total_segments += 1;
    }

    fn print_report(&self, total_audio_sec: f64) {
        println!("\n=== PERFORMANCE METRICS ===");

        // Model loading
        println!("\nModel Loading:");
        println!("  VAD load:           {:>6} ms", self.vad_load_ms);
        println!("  TDT load:           {:>6} ms", self.tdt_load_ms);
        println!("  Audio load:         {:>6} ms", self.audio_load_ms);
        println!("  Total setup:        {:>6} ms",
                 self.vad_load_ms + self.tdt_load_ms + self.audio_load_ms);

        // VAD processing
        println!("\nVAD Processing:");
        println!("  Processing time:    {:>6} ms", self.vad_processing_ms);
        println!("  Audio processed:    {:>6.2} s", total_audio_sec);
        let vad_rtf = self.vad_processing_ms as f64 / 1000.0 / total_audio_sec;
        println!("  Real-time factor:   {:>6.3}x {}", vad_rtf,
                 if vad_rtf < 1.0 { "(faster than real-time)" } else { "" });

        // Speech detection
        let silence_sec = total_audio_sec - self.total_speech_sec;
        let speech_percent = (self.total_speech_sec / total_audio_sec) * 100.0;
        println!("\nSpeech Detection:");
        println!("  Total audio:        {:>6.2} s", total_audio_sec);
        println!("  Speech detected:    {:>6.2} s ({:.1}%)", self.total_speech_sec, speech_percent);
        println!("  Silence filtered:   {:>6.2} s ({:.1}%)",
                 silence_sec, 100.0 - speech_percent);
        println!("  Segments found:     {:>6}", self.total_segments);

        // Transcription
        println!("\nTranscription:");
        println!("  Processing time:    {:>6} ms", self.transcription_ms);
        println!("  Speech transcribed: {:>6.2} s", self.total_speech_sec);
        let trans_rtf = self.transcription_ms as f64 / 1000.0 / self.total_speech_sec;
        println!("  Real-time factor:   {:>6.3}x {}", trans_rtf,
                 if trans_rtf < 1.0 { "(faster than real-time)" } else { "" });

        if !self.segment_times.is_empty() {
            println!("\n  Per-segment performance:");
            for (seg_id, audio_dur, proc_ms) in &self.segment_times {
                let rtf = *proc_ms as f64 / 1000.0 / audio_dur;
                println!("    Segment {:2}: {:.2}s audio → {:>5} ms (RTF: {:.3}x)",
                         seg_id, audio_dur, proc_ms, rtf);
            }
        }

        // Overall pipeline
        let total_processing_ms = self.vad_processing_ms + self.transcription_ms;
        let overall_rtf = total_processing_ms as f64 / 1000.0 / total_audio_sec;
        println!("\nOverall Pipeline:");
        println!("  Total processing:   {:>6} ms", total_processing_ms);
        println!("  Total audio:        {:>6.2} s", total_audio_sec);
        println!("  Real-time factor:   {:>6.3}x {}", overall_rtf,
                 if overall_rtf < 1.0 { "(faster than real-time)" } else { "" });

        // Efficiency summary
        let processing_speedup = 1.0 / overall_rtf;
        println!("\nEfficiency Summary:");
        if overall_rtf < 1.0 {
            println!("  ✓ Processing is {:.1}x faster than real-time", processing_speedup);
        } else {
            println!("  ⚠ Processing is {:.1}x slower than real-time", overall_rtf);
        }
        println!("  ✓ VAD filtered {:.1}% of audio (silence skipped)", 100.0 - speech_percent);
    }
}

/// Configuration for VAD-based streaming
#[derive(Debug, Clone)]
struct StreamConfig {
    /// VAD probability threshold for speech detection
    speech_threshold: f32,
    /// Minimum speech duration in milliseconds
    min_speech_duration_ms: f32,
    /// Pre-buffer duration to capture start of speech (ms)
    pre_buffer_ms: f32,
    /// Pause duration to trigger transcription (ms)
    pause_duration_ms: f32,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            speech_threshold: 0.3,        // Lower threshold to catch more speech
            min_speech_duration_ms: 250.0,
            pre_buffer_ms: 1000.0,        // 1s pre-buffer to capture start
            pause_duration_ms: 1000.0,    // 1s pause triggers transcription
        }
    }
}

/// Streaming transcriber with VAD
struct StreamingTranscriber {
    vad_stream: VadStream,
    config: StreamConfig,
    #[allow(dead_code)]
    device: candle_core::Device,

    // Current segment accumulation
    current_segment: Vec<f32>,
    current_segment_start_time: Option<f64>,

    // Pre-buffer to capture audio before speech detection
    pre_buffer: VecDeque<f32>,

    // State tracking
    total_samples_processed: usize,
    silence_frames: usize,
    in_speech: bool,
}

impl StreamingTranscriber {
    fn new(
        vad_stream: VadStream,
        config: StreamConfig,
        device: candle_core::Device,
    ) -> Self {
        let pre_buffer_samples = (config.pre_buffer_ms * 16.0) as usize;

        Self {
            vad_stream,
            config,
            device,
            current_segment: Vec::new(),
            current_segment_start_time: None,
            pre_buffer: VecDeque::with_capacity(pre_buffer_samples),
            total_samples_processed: 0,
            silence_frames: 0,
            in_speech: false,
        }
    }

    /// Process a chunk of audio samples
    /// Returns completed speech segments ready for transcription
    fn process_chunk(&mut self, samples: &[f32]) -> Result<Vec<(Vec<f32>, f64, f64)>> {
        let mut completed_segments = Vec::new();

        // Process through VAD in 10ms chunks (160 samples at 16kHz)
        const VAD_CHUNK_SIZE: usize = 160;
        let mut idx = 0;

        while idx < samples.len() {
            let end = (idx + VAD_CHUNK_SIZE).min(samples.len());
            let chunk = &samples[idx..end];

            // Get VAD probabilities
            let probs = self.vad_stream.push(chunk)?;

            for prob in probs {
                let is_speech = prob >= self.config.speech_threshold;

                if is_speech {
                    self.silence_frames = 0;

                    // Speech detected - start new segment if needed
                    if !self.in_speech {
                        self.in_speech = true;

                        // Calculate start time (accounting for pre-buffer)
                        let start_time = (self.total_samples_processed as f64
                                        - self.pre_buffer.len() as f64) / 16000.0;
                        self.current_segment_start_time = Some(start_time);

                        // Prepend pre-buffer to capture start of speech
                        self.current_segment.clear();
                        self.current_segment.extend(self.pre_buffer.iter());

                        eprintln!("  [VAD] Speech started at {:.2}s", start_time);
                    }
                } else {
                    // Silence detected
                    self.silence_frames += 1;

                    if self.in_speech {
                        // Check if pause is long enough to end segment
                        let pause_ms = (self.silence_frames * 10) as f32;

                        if pause_ms >= self.config.pause_duration_ms {
                            // Long pause - end segment
                            let segment_duration_ms = self.current_segment.len() as f32 / 16.0;

                            if segment_duration_ms >= self.config.min_speech_duration_ms {
                                // Valid segment - queue for transcription
                                let start = self.current_segment_start_time.unwrap();
                                let end = self.total_samples_processed as f64 / 16000.0;

                                eprintln!("  [VAD] Speech ended at {:.2}s (duration: {:.2}s)",
                                         end, segment_duration_ms / 1000.0);

                                completed_segments.push((
                                    self.current_segment.clone(),
                                    start,
                                    end,
                                ));
                            }

                            // Reset for next segment
                            self.in_speech = false;
                            self.silence_frames = 0;
                            self.current_segment.clear();
                            self.current_segment_start_time = None;
                        }
                    }
                }
            }

            // Maintain pre-buffer during silence
            if !self.in_speech {
                let pre_buffer_max = (self.config.pre_buffer_ms * 16.0) as usize;
                for &sample in chunk {
                    if self.pre_buffer.len() >= pre_buffer_max {
                        self.pre_buffer.pop_front();
                    }
                    self.pre_buffer.push_back(sample);
                }
            }

            // Accumulate to current segment if in speech
            if self.in_speech {
                self.current_segment.extend_from_slice(chunk);
            }

            self.total_samples_processed += chunk.len();
            idx = end;
        }

        Ok(completed_segments)
    }

    /// Flush any remaining segment (call at end of stream)
    fn flush(&mut self) -> Result<Option<(Vec<f32>, f64, f64)>> {
        if self.in_speech && !self.current_segment.is_empty() {
            let segment_duration_ms = self.current_segment.len() as f32 / 16.0;

            if segment_duration_ms >= self.config.min_speech_duration_ms {
                let start = self.current_segment_start_time.unwrap();
                let end = self.total_samples_processed as f64 / 16000.0;

                eprintln!("  [VAD] Flushing final segment ({:.2}s)", segment_duration_ms / 1000.0);

                let segment = self.current_segment.clone();
                self.current_segment.clear();
                self.current_segment_start_time = None;
                self.in_speech = false;

                return Ok(Some((segment, start, end)));
            }
        }

        Ok(None)
    }
}

/// Transcribe a speech segment using beam search
fn transcribe_segment(
    audio_samples: &[f32],
    model: &TransducerModel,
    device: &candle_core::Device,
) -> Result<(String, usize)> {
    // Save to temp file for feature extraction
    let temp_path = format!("/tmp/segment_{}.wav", std::process::id());
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&temp_path, spec)?;
        for &sample in audio_samples {
            writer.write_sample((sample * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
    }

    // Extract features
    let features = speech::parakeet::load_wav_as_features(&temp_path, 128, device)?;

    // Clean up temp file
    std::fs::remove_file(&temp_path).ok();

    // Convert to BF16 on GPU
    let features = if !device.is_cpu() {
        features.to_dtype(candle_core::DType::BF16)?
    } else {
        features
    };

    // Run encoder
    let encoder_out = model.encoder.forward(&features, false)?;

    // Beam decode with beam_size=2 (same as transcribe_tdt.rs)
    let tokens = model.beam_decode(&encoder_out, 2)?;
    let text = model.decode_tokens(&tokens)?;

    Ok((text, tokens.len()))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        eprintln!("\nThis example demonstrates streaming VAD-based transcription");
        eprintln!("with the same quality as the non-streaming version.");
        return Ok(());
    }

    let audio_path = &args[1];

    println!("Streaming VAD-Based TDT Transcription");
    println!("======================================\n");
    println!("Audio: {}\n", audio_path);

    // Get device
    let device = get_device()?;
    println!("Device: {:?}", device);
    println!("  (If you encounter errors, try: PARAKEET_DEVICE=cpu)\n");

    let assets = PathBuf::from("assets");

    // Initialize performance tracking
    let mut metrics = PerformanceMetrics::default();

    // Load Silero VAD (memory-mapped GGUF)
    println!("Loading Silero VAD...");
    let vad_start = Instant::now();
    let vad = SileroVad::load_from_gguf_mmap(&assets, &device)?;
    let vad_stream = VadStream::new(vad, &device)?;
    metrics.vad_load_ms = vad_start.elapsed().as_millis();
    println!("✓ VAD loaded ({} ms)\n", metrics.vad_load_ms);

    // Load Parakeet TDT model (memory-mapped GGUF)
    println!("Loading Parakeet TDT model...");
    let tdt_start = Instant::now();
    let mut model = load_parakeet_tdt_from_gguf_mmap_local(&assets, &device)?;
    model.load_tokenizer(&assets)?;
    metrics.tdt_load_ms = tdt_start.elapsed().as_millis();
    println!("✓ TDT model loaded ({} ms)\n", metrics.tdt_load_ms);

    // Load audio
    println!("Loading audio...");
    let audio_start = Instant::now();
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    if spec.channels != 1 {
        return Err(anyhow::anyhow!("Expected mono audio, got {} channels", spec.channels));
    }
    if spec.sample_rate != 16000 {
        return Err(anyhow::anyhow!("Expected 16kHz audio, got {} Hz", spec.sample_rate));
    }

    let all_samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;

    let total_duration_sec = all_samples.len() as f64 / 16000.0;
    metrics.audio_load_ms = audio_start.elapsed().as_millis();
    println!("✓ Loaded: {:.2}s ({} samples, {} ms)\n",
             total_duration_sec, all_samples.len(), metrics.audio_load_ms);

    // Create streaming transcriber
    let stream_config = StreamConfig::default();
    println!("Stream Configuration:");
    println!("  Speech threshold: {}", stream_config.speech_threshold);
    println!("  Min speech: {}ms", stream_config.min_speech_duration_ms);
    println!("  Pre-buffer: {}ms", stream_config.pre_buffer_ms);
    println!("  Pause threshold: {}ms\n", stream_config.pause_duration_ms);

    let mut streaming_transcriber = StreamingTranscriber::new(
        vad_stream,
        stream_config,
        device.clone(),
    );

    println!("=== STREAMING TRANSCRIPTION ===\n");

    // Process audio in streaming chunks (500ms chunks simulate real-time)
    const STREAM_CHUNK_SIZE: usize = 8000; // 500ms chunks at 16kHz
    let mut idx = 0;
    let mut segment_count = 0;
    let mut total_tokens = 0;
    let mut all_transcriptions = Vec::new();

    // Start VAD processing timer
    let vad_start = Instant::now();

    while idx < all_samples.len() {
        let end = (idx + STREAM_CHUNK_SIZE).min(all_samples.len());
        let chunk = &all_samples[idx..end];

        // Process through VAD and get completed segments
        let completed = streaming_transcriber.process_chunk(chunk)?;

        // Transcribe completed segments
        for (audio_samples, start_time, end_time) in completed {
            segment_count += 1;
            let segment_duration = end_time - start_time;
            println!("\n[Segment {}] Transcribing {:.2}s - {:.2}s ({:.2}s)",
                   segment_count, start_time, end_time, segment_duration);

            // Time the transcription
            let trans_start = Instant::now();
            let (text, token_count) = transcribe_segment(&audio_samples, &model, &device)?;
            let trans_ms = trans_start.elapsed().as_millis();

            total_tokens += token_count;
            metrics.transcription_ms += trans_ms;
            metrics.add_segment_time(segment_count, segment_duration, trans_ms);

            println!("  Tokens: {} ({} ms, RTF: {:.3}x)",
                     token_count, trans_ms, trans_ms as f64 / 1000.0 / segment_duration);
            println!("  \x1b[1;36mText: {}\x1b[0m\n", text.trim());

            all_transcriptions.push((start_time, end_time, text.trim().to_string()));
        }

        idx = end;
    }

    // Record VAD processing time (before final flush)
    metrics.vad_processing_ms = vad_start.elapsed().as_millis();

    // Flush any remaining segment
    if let Some((audio_samples, start_time, end_time)) = streaming_transcriber.flush()? {
        segment_count += 1;
        let segment_duration = end_time - start_time;
        println!("\n[Segment {}] Transcribing {:.2}s - {:.2}s (final)",
               segment_count, start_time, end_time);

        // Time the transcription
        let trans_start = Instant::now();
        let (text, token_count) = transcribe_segment(&audio_samples, &model, &device)?;
        let trans_ms = trans_start.elapsed().as_millis();

        total_tokens += token_count;
        metrics.transcription_ms += trans_ms;
        metrics.add_segment_time(segment_count, segment_duration, trans_ms);

        println!("  Tokens: {} ({} ms, RTF: {:.3}x)",
                 token_count, trans_ms, trans_ms as f64 / 1000.0 / segment_duration);
        println!("  \x1b[1;36mText: {}\x1b[0m\n", text.trim());

        all_transcriptions.push((start_time, end_time, text.trim().to_string()));
    }

    // Print final results
    println!("===================================\n");
    println!("=== FINAL TRANSCRIPT ===\n");

    for (start, end, text) in &all_transcriptions {
        if all_transcriptions.len() > 1 {
            println!("[{:.2}s - {:.2}s] {}", start, end, text);
        } else {
            println!("{}", text);
        }
    }

    println!("\n=== STATISTICS ===");
    println!("  Total audio: {:.2}s", total_duration_sec);
    println!("  Speech segments: {}", segment_count);
    println!("  Total tokens: {}", total_tokens);

    // Compare with baseline if using dots.wav
    if audio_path.contains("dots.wav") {
        let baseline_tokens = 187;  // From transcribe_tdt.rs (beam_size=2)
        let quality_percent = (total_tokens as f32 / baseline_tokens as f32) * 100.0;
        println!("\n  Baseline (transcribe_tdt.rs): {} tokens", baseline_tokens);
        println!("  Streaming VAD: {} tokens ({:.1}%)", total_tokens, quality_percent);

        if quality_percent >= 95.0 && quality_percent <= 105.0 {
            println!("\n✓ Quality matches baseline!");
        } else if total_tokens == baseline_tokens {
            println!("\n✓ Perfect match!");
        }
    }

    // Print performance metrics report
    metrics.print_report(total_duration_sec);

    println!("\n✓ Streaming transcription complete!");

    Ok(())
}
