/// Transcribe audio using Silero VAD + Parakeet CTC integration
///
/// This example demonstrates:
/// 1. Using Silero VAD to detect speech segments
/// 2. Only running Parakeet on actual speech (not silence/noise)
/// 3. Transcribing accumulated speech when pauses are detected
///
/// Usage:
///   cargo run --example transcribe_with_vad --release -- dots.wav
///   cargo run --example transcribe_with_vad --release -- MLKDream_16k.wav
///   PARAKEET_DEVICE=cpu cargo run --example transcribe_with_vad --release -- audio.wav

use anyhow::Result;
use std::path::Path;

// Import Silero VAD (module is in src/)
mod silero {
    include!("../src/silero.rs");
}
use silero::{SileroVad, VadStream};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        eprintln!("\nThis example uses Silero VAD to detect speech segments,");
        eprintln!("then transcribes only the speech portions with Parakeet.");
        eprintln!("\nRequired files:");
        eprintln!("  VAD: vad16.safetensors, vad16.config.json");
        eprintln!("  Parakeet: hf_parakeet/config.json, hf_parakeet/model_q8_0.gguf, hf_parakeet/tokenizer.json");
        return Ok(());
    }

    let audio_path = &args[1];

    println!("Parakeet CTC Transcription with Silero VAD");
    println!("===========================================\n");
    println!("Audio: {}\n", audio_path);

    // Get device
    let device = parakeet::get_device()?;

    // Load Silero VAD
    println!("Loading Silero VAD...");
    let vad = SileroVad::load(&device, "vad16.safetensors", "vad16.config.json")?;
    let mut vad_stream = VadStream::new(vad, &device)?;
    println!("✓ VAD loaded\n");

    // Load Parakeet model
    println!("Loading Parakeet model...");
    let model = parakeet::load_parakeet_ctc_from_gguf_local("hf_parakeet", &device)?;
    println!("✓ Parakeet loaded\n");

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

    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Int, 24) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / 8_388_608.0))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()?,
        _ => return Err(anyhow::anyhow!("Unsupported audio format")),
    };

    let total_duration_sec = samples.len() as f32 / 16000.0;
    println!("✓ Audio loaded: {:.2}s, {} samples\n", total_duration_sec, samples.len());

    // VAD parameters
    const CHUNK_SIZE: usize = 160; // 10ms at 16kHz
    const SPEECH_THRESHOLD: f32 = 0.5; // Probability threshold for speech detection
    const MIN_SPEECH_DURATION_MS: f32 = 250.0; // Minimum speech duration to transcribe
    const MIN_SILENCE_DURATION_MS: f32 = 300.0; // Silence duration to trigger transcription

    println!("Running VAD and transcription...");
    println!("  Speech threshold: {}", SPEECH_THRESHOLD);
    println!("  Min speech duration: {}ms", MIN_SPEECH_DURATION_MS);
    println!("  Min silence for pause: {}ms\n", MIN_SILENCE_DURATION_MS);

    let mut idx = 0;
    let mut speech_segments: Vec<(usize, usize)> = Vec::new(); // (start_sample, end_sample)
    let mut current_segment_start: Option<usize> = None;
    let mut silence_frames = 0;
    let mut sample_idx = 0;

    // Process audio through VAD
    while idx < samples.len() {
        let end = (idx + CHUNK_SIZE).min(samples.len());
        let probs = vad_stream.push(&samples[idx..end])?;

        // Each probability corresponds to ~512 samples (32ms at 16kHz)
        for prob in probs {
            let is_speech = prob >= SPEECH_THRESHOLD;

            if is_speech {
                // Speech detected
                silence_frames = 0;

                if current_segment_start.is_none() {
                    // Start new speech segment
                    current_segment_start = Some(sample_idx);
                }
            } else {
                // Silence detected
                if current_segment_start.is_some() {
                    silence_frames += 1;

                    // Check if we've had enough silence to consider it a pause
                    let silence_duration_ms = silence_frames as f32 * 32.0; // 32ms per frame

                    if silence_duration_ms >= MIN_SILENCE_DURATION_MS {
                        // End current segment
                        let start = current_segment_start.unwrap();
                        let end = sample_idx;

                        // Check if segment is long enough
                        let duration_ms = (end - start) as f32 / 16.0; // samples to ms
                        if duration_ms >= MIN_SPEECH_DURATION_MS {
                            speech_segments.push((start, end));
                        }

                        current_segment_start = None;
                        silence_frames = 0;
                    }
                }
            }

            sample_idx += 512;
        }

        idx = end;
    }

    // Handle any remaining speech segment
    if let Some(start) = current_segment_start {
        let duration_ms = (samples.len() - start) as f32 / 16.0;
        if duration_ms >= MIN_SPEECH_DURATION_MS {
            speech_segments.push((start, samples.len()));
        }
    }

    println!("✓ VAD complete: Found {} speech segment(s)\n", speech_segments.len());

    if speech_segments.is_empty() {
        println!("No speech detected in audio.");
        return Ok(());
    }

    // Transcribe each speech segment
    println!("=== TRANSCRIPTION ===\n");

    for (seg_idx, (start_sample, end_sample)) in speech_segments.iter().enumerate() {
        let segment_samples = &samples[*start_sample..*end_sample];
        let start_time = *start_sample as f32 / 16000.0;
        let end_time = *end_sample as f32 / 16000.0;
        let duration = end_time - start_time;

        println!("[Segment {}] {:.2}s - {:.2}s (duration: {:.2}s)",
                 seg_idx + 1, start_time, end_time, duration);

        // Extract features directly from in-memory samples (no temp files!)
        let features = parakeet::extract_features_from_samples(
            segment_samples,
            model.cfg.feat_in,
            &device
        )?;

        // Transcribe
        let logits = model.forward(&features, false)?;
        let transcriptions = model.greedy_decode(&logits)?;

        if !transcriptions.is_empty() {
            println!("  \"{}\"", transcriptions[0]);
        }
        println!();
    }

    println!("=====================\n");

    // Calculate statistics
    let total_speech_duration: f32 = speech_segments.iter()
        .map(|(start, end)| (*end - *start) as f32 / 16000.0)
        .sum();

    let speech_percentage = (total_speech_duration / total_duration_sec) * 100.0;

    println!("Statistics:");
    println!("  Total audio: {:.2}s", total_duration_sec);
    println!("  Speech detected: {:.2}s ({:.1}%)", total_speech_duration, speech_percentage);
    println!("  Number of segments: {}", speech_segments.len());
    println!("\n✓ Transcription complete!");

    Ok(())
}
