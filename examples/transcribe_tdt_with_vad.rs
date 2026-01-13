/// VAD-based transcription using Parakeet TDT (Transducer)
///
/// Uses Silero VAD to detect speech regions, then transcribes each region
/// using the same high-quality beam search decoder as transcribe_tdt.rs.
///
/// Usage:
///   cargo run --example transcribe_tdt_with_vad --release -- dots.wav
///   cargo run --example transcribe_tdt_with_vad --release -- MLKDream_16k.wav
///   PARAKEET_DEVICE=cpu cargo run --example transcribe_tdt_with_vad --release -- audio.wav

use anyhow::Result;
use speech::parakeet::{get_device, load_parakeet_tdt_from_local, load_wav_as_features};
use speech::silero::{SileroVad, VadStream};
use std::path::PathBuf;

/// Speech segment with timing
#[derive(Debug, Clone)]
struct SpeechSegment {
    start_sample: usize,
    end_sample: usize,
    start_time: f64,
    end_time: f64,
}

/// Detect speech segments using Silero VAD
fn detect_speech_segments(
    samples: &[f32],
    vad: SileroVad,
    device: &candle_core::Device,
    speech_threshold: f32,
    min_speech_samples: usize,
    min_silence_samples: usize,
) -> Result<Vec<SpeechSegment>> {
    let mut vad_stream = VadStream::new(vad, device)?;
    let mut segments = Vec::new();

    // Track speech regions
    let mut in_speech = false;
    let mut speech_start = 0;
    let mut silence_count = 0;

    // Process in 10ms chunks (160 samples at 16kHz)
    const CHUNK_SIZE: usize = 160;
    let mut sample_idx = 0;

    for chunk_start in (0..samples.len()).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(samples.len());
        let chunk = &samples[chunk_start..chunk_end];

        let probs = vad_stream.push(chunk)?;

        for prob in probs {
            let is_speech = prob >= speech_threshold;

            if is_speech {
                if !in_speech {
                    // Speech started
                    speech_start = sample_idx;
                    in_speech = true;
                }
                silence_count = 0;
            } else if in_speech {
                // In silence during speech
                silence_count += CHUNK_SIZE;

                if silence_count >= min_silence_samples {
                    // End of speech segment
                    let speech_length = sample_idx - speech_start;

                    if speech_length >= min_speech_samples {
                        segments.push(SpeechSegment {
                            start_sample: speech_start,
                            end_sample: sample_idx - silence_count,
                            start_time: speech_start as f64 / 16000.0,
                            end_time: (sample_idx - silence_count) as f64 / 16000.0,
                        });
                    }

                    in_speech = false;
                    silence_count = 0;
                }
            }

            sample_idx += CHUNK_SIZE;
        }
    }

    // Handle final segment
    if in_speech {
        let speech_length = samples.len() - speech_start;
        if speech_length >= min_speech_samples {
            segments.push(SpeechSegment {
                start_sample: speech_start,
                end_sample: samples.len(),
                start_time: speech_start as f64 / 16000.0,
                end_time: samples.len() as f64 / 16000.0,
            });
        }
    }

    Ok(segments)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        eprintln!("\nThis example uses VAD to detect speech regions, then transcribes");
        eprintln!("each region with the same quality as the non-VAD version.");
        return Ok(());
    }

    let audio_path = &args[1];

    println!("VAD-Based TDT Transcription");
    println!("============================\n");
    println!("Audio: {}\n", audio_path);

    // Get device
    let device = get_device()?;
    println!("Device: {:?}", device);
    println!("  (If you encounter errors, try: PARAKEET_DEVICE=cpu)\n");

    let assets = PathBuf::from("assets");

    // Load Silero VAD
    println!("Loading Silero VAD...");
    let vad = SileroVad::load(&assets, &device)?;
    println!("✓ VAD loaded\n");

    // Load Parakeet TDT model
    println!("Loading Parakeet TDT model...");
    let mut model = load_parakeet_tdt_from_local(".cache/parakeet-tdt", &device)?;
    model.load_tokenizer(".cache/parakeet-tdt")?;
    println!("✓ TDT model loaded\n");

    // Load audio
    println!("Loading audio...");
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
    println!("✓ Loaded: {:.2}s ({} samples)\n", total_duration_sec, all_samples.len());

    // Detect speech segments
    println!("Detecting speech segments...");
    let speech_threshold = 0.3;       // Lower threshold to catch more speech
    let min_speech_ms = 250.0;        // Minimum 250ms of speech
    let min_silence_ms = 1000.0;      // 1000ms silence ends segment (tolerate pauses)

    let min_speech_samples = (min_speech_ms * 16.0) as usize;
    let min_silence_samples = (min_silence_ms * 16.0) as usize;

    let segments = detect_speech_segments(
        &all_samples,
        vad,
        &device,
        speech_threshold,
        min_speech_samples,
        min_silence_samples,
    )?;

    println!("✓ Detected {} speech segment(s)\n", segments.len());

    // Show segment details
    for (i, seg) in segments.iter().enumerate() {
        println!("  Segment {}: {:.2}s - {:.2}s ({:.2}s)",
                 i + 1, seg.start_time, seg.end_time, seg.end_time - seg.start_time);
    }
    println!();

    // Transcribe each segment
    println!("=== TRANSCRIPTION ===\n");
    let mut all_texts = Vec::new();
    let mut total_tokens = 0;

    for (i, segment) in segments.iter().enumerate() {
        println!("Segment {}: {:.2}s - {:.2}s", i + 1, segment.start_time, segment.end_time);

        // Extract segment audio
        let segment_audio = &all_samples[segment.start_sample..segment.end_sample];

        // Save to temp file for feature extraction
        let temp_path = format!("/tmp/segment_{}.wav", i);
        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&temp_path, spec)?;
            for &sample in segment_audio {
                writer.write_sample((sample * i16::MAX as f32) as i16)?;
            }
            writer.finalize()?;
        }

        // Extract features using the same method as transcribe_tdt.rs
        let features = load_wav_as_features(&temp_path, 128, &device)?;

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

        total_tokens += tokens.len();

        println!("  Tokens: {}", tokens.len());
        println!("  Text: {}\n", text.trim());

        all_texts.push(text.trim().to_string());
    }

    // Combine results
    println!("=====================\n");
    println!("=== FINAL TRANSCRIPT ===\n");

    for (segment, text) in segments.iter().zip(all_texts.iter()) {
        if segments.len() > 1 {
            println!("[{:.2}s - {:.2}s] {}", segment.start_time, segment.end_time, text);
        } else {
            println!("{}", text);
        }
    }

    println!("\n=== STATISTICS ===");
    println!("  Total audio: {:.2}s", total_duration_sec);
    println!("  Speech segments: {}", segments.len());
    println!("  Total tokens: {}", total_tokens);

    // Compare with baseline if using dots.wav
    if audio_path.contains("dots.wav") {
        let baseline_tokens = 187;  // From transcribe_tdt.rs (beam_size=2)
        let quality_percent = (total_tokens as f32 / baseline_tokens as f32) * 100.0;
        println!("\n  Baseline (transcribe_tdt.rs): {} tokens", baseline_tokens);
        println!("  VAD-based: {} tokens ({:.1}%)", total_tokens, quality_percent);

        if quality_percent >= 95.0 && quality_percent <= 105.0 {
            println!("\n✓ Quality matches baseline!");
        } else if total_tokens == baseline_tokens {
            println!("\n✓ Perfect match!");
        }
    }

    println!("\n✓ Transcription complete!");

    Ok(())
}
