/// Streaming transcription using Silero VAD + Parakeet CTC
///
/// This example demonstrates low-latency streaming transcription:
/// 1. Reads audio file in chunks (simulates live streaming)
/// 2. Uses Silero VAD to detect speech segments in real-time
/// 3. Transcribes segments immediately when pauses are detected
/// 4. Optionally uses Qwen3 for text correction (punctuation/capitalization)
/// 5. Target latency: ~1 second from speech end to transcript
///
/// Usage:
///   cargo run --example transcribe_with_vad --release -- dots.wav
///   cargo run --example transcribe_with_vad --release --features qwen -- dots.wav --use-qwen
///   cargo run --example transcribe_with_vad --release -- MLKDream_16k.wav
///   PARAKEET_DEVICE=cpu cargo run --example transcribe_with_vad --release -- audio.wav
///
/// Note: Qwen3 text correction requires the "qwen" feature to be enabled at build time.

use anyhow::Result;
use speech::parakeet;
#[cfg(feature = "qwen")]
use speech::qwen::QwenCorrector;
use std::path::PathBuf;
use std::time::Instant;
use log::info;

// Import Silero VAD from library
use speech::silero::{SileroVad, VadStream};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav> [--use-qwen]", args[0]);
        eprintln!("\nThis example uses Silero VAD to detect speech segments,");
        eprintln!("then transcribes only the speech portions with Parakeet.");
        eprintln!("\nOptions:");
        eprintln!("  --use-qwen    Use Qwen3 for text correction (requires \"qwen\" feature)");
        eprintln!("\nRequired files:");
        eprintln!("  assets/vad16.safetensors.zst, assets/vad16.config.json.zst");
        eprintln!("  assets/config.json.zst, assets/model_q8_0.gguf.zst, assets/tokenizer.json.zst");
        eprintln!("\nFor Qwen support:");
        eprintln!("  1. Build with: cargo build --example transcribe_with_vad --release --features qwen");
        eprintln!("  2. Download model: python scripts/download_qwen3.py");
        return Ok(());
    }

    let audio_path = &args[1];
    let use_qwen = args.iter().any(|arg| arg == "--use-qwen");

    println!("Streaming Transcription with Silero VAD");
    println!("========================================\n");
    println!("Audio: {}\n", audio_path);

    // Get device
    let device = parakeet::get_device()?;

    let assets = PathBuf::from("assets");

    // Load Silero VAD
    println!("Loading Silero VAD...");
    let vad = SileroVad::load(&assets, &device)?;
    let mut vad_stream = VadStream::new(vad, &device)?;
    println!("✓ VAD loaded");

    // Load Parakeet model
    println!("Loading Parakeet model...");
    let model = parakeet::load_parakeet_ctc_from_gguf_local(&assets, &device)?;
    println!("✓ Parakeet loaded");

    // Load Qwen model if requested (only available with "qwen" feature)
    #[cfg(feature = "qwen")]
    let mut qwen_corrector = if use_qwen {
        println!("Loading Qwen3 text correction model...");
        match QwenCorrector::load(&assets, &device) {
            Ok(corrector) => {
                println!("✓ Qwen3 loaded (text correction enabled)");
                Some(corrector)
            }
            Err(e) => {
                eprintln!("⚠ Failed to load Qwen3: {}", e);
                eprintln!("  Run: python scripts/download_qwen3.py");
                eprintln!("  Falling back to rule-based punctuation");
                None
            }
        }
    } else {
        None
    };

    #[cfg(not(feature = "qwen"))]
    let _qwen_corrector: Option<()> = if use_qwen {
        eprintln!("⚠ Qwen3 support not enabled!");
        eprintln!("  Rebuild with: cargo build --example transcribe_with_vad --release --features qwen");
        eprintln!("  Falling back to rule-based punctuation");
        None
    } else {
        None
    };

    println!();

    // Open audio file for streaming
    println!("Starting streaming transcription...");
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    if spec.channels != 1 {
        return Err(anyhow::anyhow!("Expected mono audio, got {} channels", spec.channels));
    }
    if spec.sample_rate != 16000 {
        return Err(anyhow::anyhow!("Expected 16kHz audio, got {} Hz", spec.sample_rate));
    }

    let total_samples = reader.duration() as usize;
    let total_duration_sec = total_samples as f32 / 16000.0;
    println!("  Duration: {:.2}s ({} samples)", total_duration_sec, total_samples);
    println!();

    // Streaming configuration with pause-based punctuation
    const VAD_CHUNK_SIZE: usize = 160; // 10ms at 16kHz for VAD processing
    const SPEECH_THRESHOLD: f32 = 0.1; // Very low threshold to detect speech as early as possible
    const MIN_SPEECH_DURATION_MS: f32 = 250.0;
    const PRE_BUFFER_MS: f32 = 1000.0; // Large pre-buffer to capture start of speech
    const COMMA_PAUSE_DURATION_MS: f32 = 150.0; // Short pause → comma
    const PERIOD_PAUSE_DURATION_MS: f32 = 500.0; // Long pause → period (increased to avoid breaking natural speech)

    println!("Configuration:");
    println!("  Speech threshold: {}", SPEECH_THRESHOLD);
    println!("  Min speech: {}ms", MIN_SPEECH_DURATION_MS);
    println!("  Pre-buffer: {}ms (captures start of speech + resumed speech)", PRE_BUFFER_MS);
    println!("  Comma pause: {}ms (short pause)", COMMA_PAUSE_DURATION_MS);
    println!("  Period pause: {}ms (long pause - triggers transcription)\n", PERIOD_PAUSE_DURATION_MS);

    println!("=== STREAMING TRANSCRIPTION ===\n");

    // State for streaming processing with pause-based punctuation
    let mut current_segment: Vec<f32> = Vec::new();
    let mut current_segment_start: Option<usize> = None;
    let mut phrase_boundaries: Vec<usize> = Vec::new(); // Sample positions for comma insertion
    let mut silence_frames = 0;
    let mut was_speech_last_frame = false; // Track speech->silence transitions
    let mut total_samples_processed: usize = 0;
    let mut segment_count = 0;
    let mut total_speech_duration = 0.0f32;

    // Pre-buffer to capture audio before speech detection (prevents cutting off start of speech)
    // Keep 300ms of audio = 4800 samples at 16kHz
    const PRE_BUFFER_SAMPLES: usize = (PRE_BUFFER_MS * 16.0) as usize;
    let mut pre_buffer: std::collections::VecDeque<f32> = std::collections::VecDeque::with_capacity(PRE_BUFFER_SAMPLES);

    // Load all samples (but process/transcribe incrementally)
    let all_samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => {
            reader.samples::<i16>()
                .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
                .collect::<Result<Vec<_>, _>>()?
        },
        (hound::SampleFormat::Int, 24) => {
            reader.samples::<i32>()
                .map(|s| s.map(|v| v as f32 / 8_388_608.0))
                .collect::<Result<Vec<_>, _>>()?
        },
        (hound::SampleFormat::Int, 32) => {
            reader.samples::<i32>()
                .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
                .collect::<Result<Vec<_>, _>>()?
        },
        (hound::SampleFormat::Float, 32) => {
            reader.samples::<f32>()
                .collect::<Result<Vec<_>, _>>()?
        },
        _ => return Err(anyhow::anyhow!("Unsupported audio format")),
    };

    // Process through VAD incrementally with correct sample accumulation
    // Key insight: VAD probabilities indicate speech state, but we accumulate
    // samples independently based on that state, not per-probability
    let mut idx = 0;
    let mut sample_idx: usize = 0; // Track position in audio stream for sample accumulation

    while idx < all_samples.len() {
        let end = (idx + VAD_CHUNK_SIZE).min(all_samples.len());
        let chunk = &all_samples[idx..end];

        let _vad_start = Instant::now();
        let probs = vad_stream.push(chunk)?;
        // Uncomment to log VAD timing:
        // let vad_elapsed = vad_start.elapsed();
        // info!("VAD inference: {:.2}ms ({} samples)", vad_elapsed.as_secs_f64() * 1000.0, chunk.len());

        // Process VAD probabilities to update speech state
        for prob in probs {
            let is_speech = prob >= SPEECH_THRESHOLD;

            if is_speech {
                // Check if speech is resuming after silence within an active segment
                if current_segment_start.is_some() && !was_speech_last_frame && silence_frames > 0 {
                    // Speech resumed after a brief pause - prepend pre-buffer to capture start of resumed speech
                    let pre_buffer_len = pre_buffer.len();
                    if pre_buffer_len > 0 {
                        // Insert pre-buffer before the current position
                        let insert_pos = current_segment.len();
                        current_segment.reserve(pre_buffer_len);
                        current_segment.extend(pre_buffer.iter().copied());
                        // Rotate to put pre-buffer before current position
                        current_segment[insert_pos..].rotate_right(pre_buffer_len);
                    }
                    pre_buffer.clear();
                }

                silence_frames = 0;
                was_speech_last_frame = true;

                if current_segment_start.is_none() {
                    // Start new speech segment
                    // Calculate actual start position accounting for pre-buffer
                    let pre_buffer_len = pre_buffer.len();
                    current_segment_start = Some(sample_idx.saturating_sub(pre_buffer_len));
                    current_segment.clear();

                    // Include pre-buffered audio to capture start of speech
                    current_segment.extend(pre_buffer.iter().copied());
                    pre_buffer.clear();
                }
            } else {
                was_speech_last_frame = false;
                // Silence detected
                if current_segment_start.is_some() {
                    silence_frames += 1;
                    let silence_duration_ms = silence_frames as f32 * 32.0;

                    // Check for comma pause (short pause)
                    if silence_duration_ms >= COMMA_PAUSE_DURATION_MS
                        && silence_duration_ms < PERIOD_PAUSE_DURATION_MS
                        && silence_frames == (COMMA_PAUSE_DURATION_MS / 32.0).ceil() as usize {
                        // Mark phrase boundary for comma insertion
                        let boundary_pos = current_segment.len();
                        if boundary_pos > 0 {
                            phrase_boundaries.push(boundary_pos);
                        }
                    }

                    // Check for period pause (long pause - end segment)
                    if silence_duration_ms >= PERIOD_PAUSE_DURATION_MS {
                        // End current segment and transcribe immediately
                        let start_sample = current_segment_start.unwrap();
                        let end_sample = total_samples_processed;
                        let duration_ms = (end_sample - start_sample) as f32 / 16.0;

                        if duration_ms >= MIN_SPEECH_DURATION_MS {
                            segment_count += 1;
                            let start_time = start_sample as f32 / 16000.0;
                            let end_time = end_sample as f32 / 16000.0;
                            let duration = end_time - start_time;
                            total_speech_duration += duration;

                            // Transcribe with comma support
                            print!("[Segment {}] {:.2}s - {:.2}s ({:.2}s, {} phrases) - ",
                                   segment_count, start_time, end_time, duration, phrase_boundaries.len() + 1);

                            // If we have phrase boundaries, split and transcribe each phrase
                            let text = if !phrase_boundaries.is_empty() {
                                let mut phrases = Vec::new();
                                let mut start_idx = 0;

                                for &boundary_pos in &phrase_boundaries {
                                    if boundary_pos > start_idx && boundary_pos <= current_segment.len() {
                                        let phrase_samples = &current_segment[start_idx..boundary_pos];

                                        let asr_start = Instant::now();
                                        let raw_phrase = parakeet::transcribe_streaming_chunk(
                                            phrase_samples, None, None, &model, &device
                                        )?;
                                        let asr_elapsed = asr_start.elapsed();
                                        info!("Parakeet ASR (phrase): {:.2}ms ({} samples)", asr_elapsed.as_secs_f64() * 1000.0, phrase_samples.len());

                                        if !raw_phrase.is_empty() {
                                            phrases.push(raw_phrase);
                                        }
                                        start_idx = boundary_pos;
                                    }
                                }

                                // Final phrase
                                if start_idx < current_segment.len() {
                                    let final_phrase_samples = &current_segment[start_idx..];

                                    let asr_start = Instant::now();
                                    let raw_phrase = parakeet::transcribe_streaming_chunk(
                                        final_phrase_samples, None, None, &model, &device
                                    )?;
                                    let asr_elapsed = asr_start.elapsed();
                                    info!("Parakeet ASR (final phrase): {:.2}ms ({} samples)", asr_elapsed.as_secs_f64() * 1000.0, final_phrase_samples.len());

                                    if !raw_phrase.is_empty() {
                                        phrases.push(raw_phrase);
                                    }
                                }

                                let raw_text = phrases.join(" , ");
                                eprintln!("DEBUG: Raw model output: \"{}\"", raw_text);

                                // Use Qwen for correction if available, otherwise fall back to rule-based
                                #[cfg(feature = "qwen")]
                                let corrected = if let Some(ref mut corrector) = qwen_corrector {
                                    let qwen_start = Instant::now();
                                    let result = corrector.correct_text(&raw_text)?;
                                    let qwen_elapsed = qwen_start.elapsed();
                                    info!("Qwen3 text correction: {:.2}ms ({} chars)", qwen_elapsed.as_secs_f64() * 1000.0, raw_text.len());
                                    result
                                } else {
                                    parakeet::add_punctuation_internal(&raw_text, true)
                                };
                                #[cfg(not(feature = "qwen"))]
                                let corrected = parakeet::add_punctuation_internal(&raw_text, true);
                                corrected
                            } else {
                                // Single phrase
                                let asr_start = Instant::now();
                                let raw_text = parakeet::transcribe_streaming_chunk(
                                    &current_segment, None, None, &model, &device
                                )?;
                                let asr_elapsed = asr_start.elapsed();
                                info!("Parakeet ASR (single phrase): {:.2}ms ({} samples)", asr_elapsed.as_secs_f64() * 1000.0, current_segment.len());
                                eprintln!("DEBUG: Raw model output: \"{}\"", raw_text);

                                // Use Qwen for correction if available, otherwise fall back to rule-based
                                #[cfg(feature = "qwen")]
                                let corrected = if let Some(ref mut corrector) = qwen_corrector {
                                    let qwen_start = Instant::now();
                                    let result = corrector.correct_text(&raw_text)?;
                                    let qwen_elapsed = qwen_start.elapsed();
                                    info!("Qwen3 text correction: {:.2}ms ({} chars)", qwen_elapsed.as_secs_f64() * 1000.0, raw_text.len());
                                    result
                                } else {
                                    parakeet::add_punctuation(&raw_text)
                                };
                                #[cfg(not(feature = "qwen"))]
                                let corrected = parakeet::add_punctuation(&raw_text);
                                corrected
                            };

                            if !text.is_empty() {
                                println!("\"{}\"", text);
                            } else {
                                println!("(empty)");
                            }
                        }

                        current_segment.clear();
                        current_segment_start = None;
                        phrase_boundaries.clear();
                        silence_frames = 0;
                    }
                }
            }

            total_samples_processed += 512;
        }

        // Always maintain pre-buffer during silence (even within an active segment)
        // This ensures we capture the start of resumed speech after brief pauses
        if !was_speech_last_frame {
            for &sample in chunk {
                if pre_buffer.len() >= PRE_BUFFER_SAMPLES {
                    pre_buffer.pop_front();
                }
                pre_buffer.push_back(sample);
            }
        }

        // Accumulate chunk samples if we're in an active speech segment
        if current_segment_start.is_some() {
            current_segment.extend_from_slice(chunk);
        }

        // Update sample position
        sample_idx += chunk.len();
        idx = end;
    }

    // Transcribe any final segment
    if let Some(start_sample) = current_segment_start {
        let duration_ms = current_segment.len() as f32 / 16.0;
        if duration_ms >= MIN_SPEECH_DURATION_MS {
            segment_count += 1;
            let start_time = start_sample as f32 / 16000.0;
            let end_time = total_samples_processed as f32 / 16000.0;
            let duration = end_time - start_time;
            total_speech_duration += duration;

            print!("[Segment {}] {:.2}s - {:.2}s ({:.2}s, {} phrases) - ",
                   segment_count, start_time, end_time, duration, phrase_boundaries.len() + 1);

            // Transcribe with comma support
            let text = if !phrase_boundaries.is_empty() {
                let mut phrases = Vec::new();
                let mut start_idx = 0;

                for &boundary_pos in &phrase_boundaries {
                    if boundary_pos > start_idx && boundary_pos <= current_segment.len() {
                        let phrase_samples = &current_segment[start_idx..boundary_pos];

                        let asr_start = Instant::now();
                        let raw_phrase = parakeet::transcribe_streaming_chunk(
                            phrase_samples, None, None, &model, &device
                        )?;
                        let asr_elapsed = asr_start.elapsed();
                        info!("Parakeet ASR (final segment phrase): {:.2}ms ({} samples)", asr_elapsed.as_secs_f64() * 1000.0, phrase_samples.len());

                        if !raw_phrase.is_empty() {
                            phrases.push(raw_phrase);
                        }
                        start_idx = boundary_pos;
                    }
                }

                // Final phrase
                if start_idx < current_segment.len() {
                    let final_phrase_samples = &current_segment[start_idx..];

                    let asr_start = Instant::now();
                    let raw_phrase = parakeet::transcribe_streaming_chunk(
                        final_phrase_samples, None, None, &model, &device
                    )?;
                    let asr_elapsed = asr_start.elapsed();
                    info!("Parakeet ASR (final segment final phrase): {:.2}ms ({} samples)", asr_elapsed.as_secs_f64() * 1000.0, final_phrase_samples.len());

                    if !raw_phrase.is_empty() {
                        phrases.push(raw_phrase);
                    }
                }

                let raw_text = phrases.join(" , ");
                eprintln!("DEBUG: Raw model output: \"{}\"", raw_text);

                // Use Qwen for correction if available, otherwise fall back to rule-based
                #[cfg(feature = "qwen")]
                let corrected = if let Some(ref mut corrector) = qwen_corrector {
                    let qwen_start = Instant::now();
                    let result = corrector.correct_text(&raw_text)?;
                    let qwen_elapsed = qwen_start.elapsed();
                    info!("Qwen3 text correction (final segment): {:.2}ms ({} chars)", qwen_elapsed.as_secs_f64() * 1000.0, raw_text.len());
                    result
                } else {
                    parakeet::add_punctuation_internal(&raw_text, true)
                };
                #[cfg(not(feature = "qwen"))]
                let corrected = parakeet::add_punctuation_internal(&raw_text, true);
                corrected
            } else {
                // Single phrase
                let asr_start = Instant::now();
                let raw_text = parakeet::transcribe_streaming_chunk(
                    &current_segment, None, None, &model, &device
                )?;
                let asr_elapsed = asr_start.elapsed();
                info!("Parakeet ASR (final segment single phrase): {:.2}ms ({} samples)", asr_elapsed.as_secs_f64() * 1000.0, current_segment.len());
                eprintln!("DEBUG: Raw model output: \"{}\"", raw_text);

                // Use Qwen for correction if available, otherwise fall back to rule-based
                #[cfg(feature = "qwen")]
                let corrected = if let Some(ref mut corrector) = qwen_corrector {
                    let qwen_start = Instant::now();
                    let result = corrector.correct_text(&raw_text)?;
                    let qwen_elapsed = qwen_start.elapsed();
                    info!("Qwen3 text correction (final segment): {:.2}ms ({} chars)", qwen_elapsed.as_secs_f64() * 1000.0, raw_text.len());
                    result
                } else {
                    parakeet::add_punctuation(&raw_text)
                };
                #[cfg(not(feature = "qwen"))]
                let corrected = parakeet::add_punctuation(&raw_text);
                corrected
            };

            if !text.is_empty() {
                println!("\"{}\"", text);
            } else {
                println!("(empty)");
            }
        }
    }

    println!("\n===============================\n");

    let speech_percentage = (total_speech_duration / total_duration_sec) * 100.0;

    println!("Statistics:");
    println!("  Total audio: {:.2}s", total_duration_sec);
    println!("  Speech detected: {:.2}s ({:.1}%)", total_speech_duration, speech_percentage);
    println!("  Number of segments: {}", segment_count);
    println!("  Average latency: ~{}ms (period pause threshold)\n", PERIOD_PAUSE_DURATION_MS);
    println!("✓ Streaming transcription complete with pause-based punctuation!");

    Ok(())
}
