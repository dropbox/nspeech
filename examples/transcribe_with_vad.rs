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
///   cargo run --example transcribe_with_vad --release --features qwen -- dots.wav
///   cargo run --example transcribe_with_vad --release -- MLKDream_16k.wav
///   PARAKEET_DEVICE=cpu cargo run --example transcribe_with_vad --release -- audio.wav
///
/// Note: Build with --features qwen to enable Qwen3 text correction (punctuation/capitalization)

use anyhow::Result;
use speech::parakeet;
#[cfg(feature = "qwen")]
use speech::qwen::QwenCorrector;
use std::path::PathBuf;

// Import Silero VAD and streaming transcriber from library
use speech::silero::{SileroVad, VadStream};
use speech::streaming_transcriber::{StreamingConfig, StreamingTranscriber};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        eprintln!("\nThis example uses Silero VAD to detect speech segments,");
        eprintln!("then transcribes only the speech portions with Parakeet.");
        eprintln!("\nRequired files:");
        eprintln!("  assets/vad16.safetensors.zst, assets/vad16.config.json.zst");
        eprintln!("  assets/config.json.zst, assets/model_q8_0.gguf.zst, assets/tokenizer.json.zst");
        eprintln!("\nFor Qwen3 text correction:");
        eprintln!("  1. Download model: python scripts/download_qwen3.py");
        eprintln!("  2. Build with: cargo build --example transcribe_with_vad --release --features qwen");
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
    let vad_stream = VadStream::new(vad, &device)?;
    println!("✓ VAD loaded");

    // Load Parakeet model
    println!("Loading Parakeet model...");
    let parakeet_model = parakeet::load_parakeet_ctc_from_gguf_local(&assets, &device)?;
    println!("✓ Parakeet loaded");

    // Load Qwen model if "qwen" feature is enabled
    #[cfg(feature = "qwen")]
    let qwen_corrector = {
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

    // Create streaming configuration
    let config = StreamingConfig::default();

    println!("Configuration:");
    println!("  Speech threshold: {}", config.speech_threshold);
    println!("  Min speech: {}ms", config.min_speech_duration_ms);
    println!("  Pre-buffer: {}ms (captures start of speech + resumed speech)", config.pre_buffer_ms);
    println!("  Comma pause: {}ms (short pause)", config.comma_pause_duration_ms);
    println!("  Period pause: {}ms (long pause - triggers transcription)\n", config.period_pause_duration_ms);

    // Save period pause value for statistics
    let period_pause_ms = config.period_pause_duration_ms;

    // Create streaming transcriber
    let mut transcriber = StreamingTranscriber::new(
        vad_stream,
        parakeet_model,
        device.clone(),
        config,
        #[cfg(feature = "qwen")]
        qwen_corrector,
    );

    println!("=== STREAMING TRANSCRIPTION ===\n");

    // Statistics tracking
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

    // Process audio in chunks through the streaming transcriber
    const CHUNK_SIZE: usize = 8000; // Process 500ms chunks (8000 samples at 16kHz)
    let mut idx = 0;

    while idx < all_samples.len() {
        let end = (idx + CHUNK_SIZE).min(all_samples.len());
        let chunk = &all_samples[idx..end];

        // Process chunk and get any completed segments
        let segments = transcriber.process_samples(chunk)?;

        // Print completed segments
        for segment in segments {
            segment_count += 1;
            let duration = segment.end_time - segment.start_time;
            total_speech_duration += duration as f32;

            print!("[Segment {}] {:.2}s - {:.2}s ({:.2}s) - ",
                   segment_count, segment.start_time, segment.end_time, duration);

            if !segment.text.is_empty() {
                println!("\"{}\"", segment.text);
            } else {
                println!("(empty)");
            }
        }

        idx = end;
    }

    // Flush any remaining segment
    if let Some(segment) = transcriber.flush()? {
        segment_count += 1;
        let duration = segment.end_time - segment.start_time;
        total_speech_duration += duration as f32;

        print!("[Segment {}] {:.2}s - {:.2}s ({:.2}s) - ",
               segment_count, segment.start_time, segment.end_time, duration);

        if !segment.text.is_empty() {
            println!("\"{}\"", segment.text);
        } else {
            println!("(empty)");
        }
    }

    println!("\n===============================\n");

    let speech_percentage = (total_speech_duration / total_duration_sec) * 100.0;

    println!("Statistics:");
    println!("  Total audio: {:.2}s", total_duration_sec);
    println!("  Speech detected: {:.2}s ({:.1}%)", total_speech_duration, speech_percentage);
    println!("  Number of segments: {}", segment_count);
    println!("  Average latency: ~{}ms (period pause threshold)\n", period_pause_ms);
    println!("✓ Streaming transcription complete with pause-based punctuation!");

    Ok(())
}
