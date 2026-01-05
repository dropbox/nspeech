/// Streaming transcription using Silero VAD + Parakeet CTC
///
/// This example demonstrates low-latency streaming transcription:
/// 1. Reads audio file in chunks (simulates live streaming)
/// 2. Uses Silero VAD to detect speech segments in real-time
/// 3. Transcribes segments immediately when pauses are detected
/// 4. Target latency: ~1 second from speech end to transcript
///
/// Usage:
///   cargo run --example transcribe_with_vad --release -- dots.wav
///   cargo run --example transcribe_with_vad --release -- MLKDream_16k.wav
///   PARAKEET_DEVICE=cpu cargo run --example transcribe_with_vad --release -- audio.wav

use anyhow::Result;
use speech::parakeet;
use std::path::PathBuf;

// Import Silero VAD from library
use speech::silero::{SileroVad, VadStream};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        eprintln!("\nThis example uses Silero VAD to detect speech segments,");
        eprintln!("then transcribes only the speech portions with Parakeet.");
        eprintln!("\nRequired files:");
        eprintln!("  assets/vad16.safetensors.zst, assets/vad16.config.json.zst");
        eprintln!("  assets/config.json.zst, assets/model_q8_0.gguf.zst, assets/tokenizer.json.zst");
        return Ok(());
    }

    let audio_path = &args[1];

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

    // Streaming configuration
    const STREAM_CHUNK_SIZE: usize = 8000; // 500ms chunks for streaming (simulates network packets)
    const VAD_CHUNK_SIZE: usize = 160; // 10ms at 16kHz for VAD processing
    const SPEECH_THRESHOLD: f32 = 0.5;
    const MIN_SPEECH_DURATION_MS: f32 = 250.0;
    const MIN_SILENCE_DURATION_MS: f32 = 300.0; // Target ~300ms for low latency

    println!("Configuration:");
    println!("  Stream chunk: {}ms", STREAM_CHUNK_SIZE as f32 / 16.0);
    println!("  Speech threshold: {}", SPEECH_THRESHOLD);
    println!("  Min speech: {}ms", MIN_SPEECH_DURATION_MS);
    println!("  Silence threshold: {}ms (triggers transcription)\n", MIN_SILENCE_DURATION_MS);

    println!("=== STREAMING TRANSCRIPTION ===\n");

    // State for streaming processing
    let mut current_segment: Vec<f32> = Vec::new();
    let mut current_segment_start: Option<usize> = None;
    let mut silence_frames = 0;
    let mut total_samples_processed: usize = 0;
    let mut segment_count = 0;
    let mut total_speech_duration = 0.0f32;

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
    let mut sample_idx = 0; // Track position in audio stream for sample accumulation

    while idx < all_samples.len() {
        let end = (idx + VAD_CHUNK_SIZE).min(all_samples.len());
        let chunk = &all_samples[idx..end];
        let probs = vad_stream.push(chunk)?;

        // Process VAD probabilities to update speech state
        for prob in probs {
            let is_speech = prob >= SPEECH_THRESHOLD;

            if is_speech {
                silence_frames = 0;

                if current_segment_start.is_none() {
                    // Start new speech segment at current sample position
                    current_segment_start = Some(sample_idx);
                    current_segment.clear();
                }
            } else {
                // Silence detected
                if current_segment_start.is_some() {
                    silence_frames += 1;
                    let silence_duration_ms = silence_frames as f32 * 32.0;

                    if silence_duration_ms >= MIN_SILENCE_DURATION_MS {
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

                            // Transcribe immediately (streaming!)
                            print!("[Segment {}] {:.2}s - {:.2}s ({:.2}s) - ",
                                   segment_count, start_time, end_time, duration);

                            let text = parakeet::transcribe_streaming_chunk(
                                &current_segment,
                                None,
                                None,
                                &model,
                                &device,
                            )?;

                            if !text.is_empty() {
                                println!("\"{}\"", text);
                            } else {
                                println!("(empty)");
                            }
                        }

                        current_segment.clear();
                        current_segment_start = None;
                        silence_frames = 0;
                    }
                }
            }

            total_samples_processed += 512;
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

            print!("[Segment {}] {:.2}s - {:.2}s ({:.2}s) - ",
                   segment_count, start_time, end_time, duration);

            let text = parakeet::transcribe_streaming_chunk(
                &current_segment,
                None,
                None,
                &model,
                &device,
            )?;

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
    println!("  Average latency: ~{}ms (silence threshold)\n", MIN_SILENCE_DURATION_MS);
    println!("✓ Streaming transcription complete!");

    Ok(())
}
