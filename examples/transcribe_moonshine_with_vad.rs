/// VAD-based transcription using Moonshine V2 medium streaming model.
///
/// Uses Silero VAD to detect speech segments and transcribes each
/// segment independently with Moonshine. Encoder/decoder weights stay
/// quantized (Q8_0) for reduced memory and faster inference.
///
/// Usage:
///   cargo run --example transcribe_moonshine_with_vad --release -- dots.wav
///   cargo run --example transcribe_moonshine_with_vad --release -- dots.wav assets
///   cargo run --example transcribe_moonshine_with_vad --release -- MLKDream_16k.wav
///   PARAKEET_DEVICE=cpu cargo run --example transcribe_moonshine_with_vad --release -- audio.wav

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

struct SpeechSegment {
    samples: Vec<f32>,
    start_sec: f64,
    end_sec: f64,
}

/// Detect speech segments using Silero VAD.
fn detect_speech_segments(
    samples: &[f32],
    device: &candle_core::Device,
) -> Result<Vec<SpeechSegment>> {
    let assets = PathBuf::from("assets");
    let vad = SileroVad::load_from_gguf_mmap(&assets, device)?;
    let mut stream = VadStream::new(vad, device)?;

    let mut segments = Vec::new();
    let mut current_segment = Vec::new();
    let mut segment_start: Option<f64> = None;
    let mut in_speech = false;
    let mut silence_frames = 0usize;
    let mut pre_buffer: VecDeque<f32> = VecDeque::new();
    let pre_buffer_max = (PRE_BUFFER_MS * 16.0) as usize;

    const VAD_CHUNK_SIZE: usize = 160; // 10ms at 16kHz
    let mut pos = 0;

    while pos < samples.len() {
        let end = (pos + VAD_CHUNK_SIZE).min(samples.len());
        let chunk = &samples[pos..end];

        let probs = stream.push(chunk)?;
        for prob in probs {
            let is_speech = prob >= SPEECH_THRESHOLD;

            if is_speech {
                silence_frames = 0;
                if !in_speech {
                    in_speech = true;
                    let start = (pos as f64 - pre_buffer.len() as f64) / 16000.0;
                    segment_start = Some(start.max(0.0));
                    current_segment.clear();
                    current_segment.extend(pre_buffer.iter());
                }
            } else {
                silence_frames += 1;
                if in_speech {
                    let pause_ms = (silence_frames * 10) as f32;
                    if pause_ms >= PAUSE_DURATION_MS {
                        let duration_ms = current_segment.len() as f32 / 16.0;
                        if duration_ms >= MIN_SPEECH_DURATION_MS {
                            segments.push(SpeechSegment {
                                samples: current_segment.clone(),
                                start_sec: segment_start.unwrap(),
                                end_sec: pos as f64 / 16000.0,
                            });
                        }
                        in_speech = false;
                        silence_frames = 0;
                        current_segment.clear();
                        segment_start = None;
                    }
                }
            }
        }

        if !in_speech {
            for &s in chunk {
                if pre_buffer.len() >= pre_buffer_max {
                    pre_buffer.pop_front();
                }
                pre_buffer.push_back(s);
            }
        }

        if in_speech {
            current_segment.extend_from_slice(chunk);
        }

        pos = end;
    }

    // Flush final segment
    if in_speech && !current_segment.is_empty() {
        let duration_ms = current_segment.len() as f32 / 16.0;
        if duration_ms >= MIN_SPEECH_DURATION_MS {
            segments.push(SpeechSegment {
                samples: current_segment,
                start_sec: segment_start.unwrap(),
                end_sec: samples.len() as f64 / 16000.0,
            });
        }
    }

    Ok(segments)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let wav_path = args.get(1).map(|s| s.as_str()).unwrap_or("dots.wav");
    let model_dir = args.get(2).map(|s| s.as_str()).unwrap_or("assets");

    println!("=== Moonshine V2 + VAD Transcription (Quantized) ===\n");

    // Load audio
    let t0 = Instant::now();
    let samples = load_wav_samples(wav_path)?;
    let total_duration = samples.len() as f64 / 16000.0;
    println!("Audio: {} ({:.2}s, {} samples)", wav_path, total_duration, samples.len());
    println!("  Loaded in {:.0}ms", t0.elapsed().as_millis());

    // Get device
    let device = get_device()?;

    // VAD segmentation
    let t1 = Instant::now();
    let segments = detect_speech_segments(&samples, &device)?;
    let vad_ms = t1.elapsed().as_millis();
    let speech_duration: f64 = segments.iter().map(|s| s.samples.len() as f64 / 16000.0).sum();
    println!("\nVAD: {} segments, {:.2}s speech ({:.1}% of total) in {:.0}ms",
        segments.len(), speech_duration, speech_duration / total_duration * 100.0, vad_ms);

    // Load model from GGUF assets
    let t2 = Instant::now();
    println!("Loading Moonshine from GGUF assets ({})...", model_dir);
    let model = MoonshineModel::load_from_gguf_mmap(model_dir, &device)?;
    println!("Model loaded in {:.0}ms\n", t2.elapsed().as_millis());

    // Transcribe each segment
    let t3 = Instant::now();
    let mut full_text = Vec::new();

    for (_i, segment) in segments.iter().enumerate() {
        let seg_duration = segment.samples.len() as f64 / 16000.0;
        let t_seg = Instant::now();

        let text = model.transcribe(&segment.samples, &device)?;
        let seg_ms = t_seg.elapsed().as_millis();

        println!("[{:6.2}s - {:6.2}s] ({:.2}s, {:.0}ms) {}",
            segment.start_sec, segment.end_sec, seg_duration, seg_ms, text);

        if !text.is_empty() {
            full_text.push(text);
        }
    }

    let total_transcribe_ms = t3.elapsed().as_millis();

    println!("\n--- Full transcription ---");
    println!("{}", full_text.join(" "));

    println!("\n=== Performance ===");
    println!("VAD:          {:.0}ms", vad_ms);
    println!("Transcribe:   {:.0}ms ({:.2}x realtime)",
        total_transcribe_ms, (total_transcribe_ms as f64 / 1000.0) / speech_duration);
    println!("Total:        {:.0}ms ({:.2}x realtime)",
        vad_ms as u128 + total_transcribe_ms,
        ((vad_ms + total_transcribe_ms) as f64 / 1000.0) / total_duration);

    Ok(())
}
