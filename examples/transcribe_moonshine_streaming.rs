/// Streaming transcription example using Moonshine V2.
///
/// Simulates real-time audio input by feeding audio in chunks and emitting
/// partial transcriptions during speech. Uses VAD to detect speech segments
/// and stream_try_update/stream_finalize for streaming results.
///
/// Partial results replace previous partials (standard streaming ASR behavior).
/// Final results are emitted on pause or end of audio.
///
/// Usage:
///   cargo run --example transcribe_moonshine_streaming --release -- MLKDream_16k.wav
///   cargo run --example transcribe_moonshine_streaming --release -- dots.wav
///   cargo run --example transcribe_moonshine_streaming --release -- dots.wav assets
///   PARAKEET_DEVICE=cpu cargo run --example transcribe_moonshine_streaming --release -- audio.wav

use anyhow::Result;
use speech::moonshine::MoonshineModel;
use speech::parakeet::get_device;
use speech::silero::{SileroVad, VadStream};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;

const SPEECH_THRESHOLD: f32 = 0.3;
const MIN_SPEECH_DURATION_MS: f32 = 250.0;
const PRE_BUFFER_MS: f32 = 500.0;
const PAUSE_DURATION_MS: f32 = 800.0;

/// Streaming update interval in ms — how often to emit partial results.
const STREAM_UPDATE_INTERVAL_MS: usize = 500;
/// Minimum audio before first partial transcription.
const STREAM_MIN_AUDIO_MS: usize = 500;

fn load_wav_samples(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        anyhow::bail!("expected mono wav, got {} channels", spec.channels);
    }
    if spec.sample_rate != 16000 {
        anyhow::bail!("expected 16kHz audio, got {} Hz", spec.sample_rate);
    }
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()?,
        _ => anyhow::bail!("unsupported WAV format"),
    };
    Ok(samples)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let wav_path = args.get(1).map(|s| s.as_str()).unwrap_or("dots.wav");
    let model_dir = args.get(2).map(|s| s.as_str()).unwrap_or("assets");

    println!("=== Moonshine V2 Streaming Transcription ===\n");

    // Load audio
    let t0 = Instant::now();
    let samples = load_wav_samples(wav_path)?;
    let total_duration = samples.len() as f64 / 16000.0;
    println!("Audio: {} ({:.2}s, {} samples)", wav_path, total_duration, samples.len());
    println!("  Loaded in {:.0}ms", t0.elapsed().as_millis());

    // Get device
    let device = get_device()?;

    // Load VAD
    let assets = PathBuf::from(model_dir);
    let vad = SileroVad::load_from_gguf_mmap(&assets, &device)?;
    let mut vad_stream = VadStream::new(vad, &device)?;

    // Load Moonshine model
    let t1 = Instant::now();
    println!("Loading Moonshine from GGUF assets ({})...", model_dir);
    let model = MoonshineModel::load_from_gguf_mmap(model_dir, &device)?;
    println!("Model loaded in {:.0}ms\n", t1.elapsed().as_millis());

    // Create streaming state
    let mut stream = model.stream_new(STREAM_UPDATE_INTERVAL_MS, STREAM_MIN_AUDIO_MS);

    // Simulate real-time streaming: feed audio in VAD-sized chunks
    const VAD_CHUNK_SIZE: usize = 512; // 32ms at 16kHz
    let pre_buffer_max = (PRE_BUFFER_MS * 16.0) as usize;
    let pause_frames = (PAUSE_DURATION_MS / 32.0) as usize;

    let mut pre_buffer: VecDeque<f32> = VecDeque::new();
    let mut current_segment: Vec<f32> = Vec::new();
    let mut segment_start: Option<f64> = None;
    let mut in_speech = false;
    let mut silence_frames: usize = 0;
    let mut segment_idx: usize = 0;

    let mut total_partials: usize = 0;
    let mut total_finals: usize = 0;
    let mut full_text: Vec<String> = Vec::new();

    let t_start = Instant::now();
    let mut pos = 0;

    while pos < samples.len() {
        let end = (pos + VAD_CHUNK_SIZE).min(samples.len());
        let chunk = &samples[pos..end];

        // Run VAD
        let probs = vad_stream.push(chunk)?;

        for prob in &probs {
            let is_speech = *prob >= SPEECH_THRESHOLD;

            if is_speech {
                silence_frames = 0;

                if !in_speech {
                    // Speech started — prepend pre-buffer
                    in_speech = true;
                    let start = (pos as f64 - pre_buffer.len() as f64) / 16000.0;
                    segment_start = Some(start.max(0.0));
                    current_segment.clear();
                    current_segment.extend(pre_buffer.iter());
                    segment_idx += 1;
                    println!("\n--- Segment {} (starts at {:.2}s) ---", segment_idx, start.max(0.0));
                }

                // Accumulate speech audio
                current_segment.extend_from_slice(chunk);

                // Try streaming partial transcription
                match model.stream_try_update(&mut stream, &current_segment, &device)? {
                    Some(text) => {
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            total_partials += 1;
                            let seg_dur = current_segment.len() as f64 / 16000.0;
                            print!("\r\x1b[K  [partial {:>2}] ({:.1}s) {}", total_partials, seg_dur, trimmed);
                            // Flush stdout so the partial shows up immediately
                            use std::io::Write;
                            std::io::stdout().flush()?;
                        }
                    }
                    None => {}
                }
            } else {
                // Silence
                if in_speech {
                    current_segment.extend_from_slice(chunk);
                    silence_frames += 1;

                    if silence_frames >= pause_frames {
                        // Pause detected — finalize segment
                        let duration_ms = current_segment.len() as f32 / 16.0;
                        if duration_ms >= MIN_SPEECH_DURATION_MS {
                            let t_fin = Instant::now();
                            let text = model.stream_finalize(&mut stream, &current_segment, &device)?;
                            let fin_ms = t_fin.elapsed().as_millis();
                            let trimmed = text.trim();

                            if !trimmed.is_empty() {
                                total_finals += 1;
                                let start = segment_start.unwrap_or(0.0);
                                let end_sec = start + current_segment.len() as f64 / 16000.0;
                                // Clear the partial line and print final
                                println!("\r\x1b[K  [final]     [{:.2}s-{:.2}s] ({:.0}ms) {}",
                                    start, end_sec, fin_ms, trimmed);
                                full_text.push(trimmed.to_string());
                            }
                        }

                        in_speech = false;
                        silence_frames = 0;
                        current_segment.clear();
                        segment_start = None;
                    }
                }

                // Update pre-buffer during silence
                if !in_speech {
                    for &s in chunk {
                        if pre_buffer.len() >= pre_buffer_max {
                            pre_buffer.pop_front();
                        }
                        pre_buffer.push_back(s);
                    }
                }
            }
        }

        pos = end;
    }

    // Flush final segment if still in speech
    if in_speech && !current_segment.is_empty() {
        let duration_ms = current_segment.len() as f32 / 16.0;
        if duration_ms >= MIN_SPEECH_DURATION_MS {
            let t_fin = Instant::now();
            let text = model.stream_finalize(&mut stream, &current_segment, &device)?;
            let fin_ms = t_fin.elapsed().as_millis();
            let trimmed = text.trim();

            if !trimmed.is_empty() {
                total_finals += 1;
                let start = segment_start.unwrap_or(0.0);
                let end_sec = start + current_segment.len() as f64 / 16000.0;
                println!("\r\x1b[K  [final]     [{:.2}s-{:.2}s] ({:.0}ms) {}",
                    start, end_sec, fin_ms, trimmed);
                full_text.push(trimmed.to_string());
            }
        }
    }

    let total_ms = t_start.elapsed().as_millis();

    println!("\n--- Full transcription ---");
    println!("{}", full_text.join(" "));

    println!("\n=== Stats ===");
    println!("Segments:     {}", total_finals);
    println!("Partials:     {}", total_partials);
    println!("Total:        {:.0}ms ({:.2}x realtime)",
        total_ms, (total_ms as f64 / 1000.0) / total_duration);

    Ok(())
}
