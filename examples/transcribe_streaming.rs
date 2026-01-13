/// Streaming transcription example using rolling buffer (like index.html)
///
/// This demonstrates the streaming approach from index.html:
/// - Rolling buffer of last N seconds
/// - Continuous transcription of the buffer
/// - Commits lines when buffer fills
/// - Keeps overlap for context

use anyhow::Result;
use speech::{parakeet, streaming_buffer::StreamingBuffer};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        eprintln!();
        eprintln!("Streams audio through a rolling buffer like index.html:");
        eprintln!("  - Maintains last 10 seconds of audio");
        eprintln!("  - Transcribes every 0.75 seconds");
        eprintln!("  - Commits lines when buffer fills");
        eprintln!("  - Keeps 0.25s overlap for context");
        std::process::exit(1);
    }

    let audio_path = PathBuf::from(&args[1]);

    println!("=== STREAMING TRANSCRIPTION (Rolling Buffer) ===\n");
    println!("Loading models...");

    // Load Parakeet model
    let device = parakeet::get_device()?;
    let model = parakeet::load_parakeet_ctc_from_gguf_local("assets", &device)?;

    println!("✓ Models loaded\n");

    // Load audio file
    println!("Loading audio: {:?}", audio_path);
    let mut reader = hound::WavReader::open(&audio_path)?;
    let spec = reader.spec();

    println!("  Sample rate: {} Hz", spec.sample_rate);
    println!("  Channels: {}", spec.channels);
    println!("  Bits per sample: {}", spec.bits_per_sample);
    println!("  Format: {:?}", spec.sample_format);

    if spec.sample_rate != 16000 {
        eprintln!("ERROR: Audio must be 16kHz. Use ffmpeg to convert:");
        eprintln!("  ffmpeg -i input.wav -ar 16000 -ac 1 output.wav");
        std::process::exit(1);
    }

    // Read all samples
    let all_samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect::<Result<Vec<_>, _>>()?,
        (hound::SampleFormat::Int, 24) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / 8_388_608.0))
            .collect::<Result<Vec<_>, _>>()?,
        (hound::SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
            .collect::<Result<Vec<_>, _>>()?,
        (hound::SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        _ => {
            eprintln!("Unsupported audio format");
            std::process::exit(1);
        }
    };

    let total_duration_sec = all_samples.len() as f32 / 16000.0;
    println!("  Duration: {:.2}s ({} samples)\n", total_duration_sec, all_samples.len());

    // Streaming configuration (matching index.html defaults)
    const BLOCK_DURATION_SECS: f32 = 0.75; // How often to transcribe
    const MAX_BUFFER_SECS: f32 = 10.0;      // Rolling window size
    const OVERLAP_SECS: f32 = 0.25;         // Context overlap after commit
    const SAMPLE_RATE: usize = 16000;

    println!("Configuration:");
    println!("  Block duration: {}s (transcription interval)", BLOCK_DURATION_SECS);
    println!("  Max buffer: {}s (rolling window)", MAX_BUFFER_SECS);
    println!("  Overlap: {}s (context after commit)\n", OVERLAP_SECS);

    // Create streaming buffer
    let mut buffer = StreamingBuffer::new(MAX_BUFFER_SECS, OVERLAP_SECS, SAMPLE_RATE);

    // Simulate streaming by processing in blocks
    let block_samples = (BLOCK_DURATION_SECS * SAMPLE_RATE as f32) as usize;
    let mut transcription_count = 0;

    println!("=== STREAMING ===\n");

    let mut idx = 0;
    while idx < all_samples.len() {
        let end = (idx + block_samples).min(all_samples.len());
        let chunk = &all_samples[idx..end];

        // Push samples to buffer
        let should_commit = buffer.push_samples(chunk);

        // Transcribe current buffer
        let buffer_audio = buffer.get_buffer();
        if !buffer_audio.is_empty() {
            transcription_count += 1;
            let time_secs = idx as f32 / SAMPLE_RATE as f32;

            print!("[{:.2}s] Buffer: {:.2}s → ",
                   time_secs,
                   buffer.buffer_duration_secs(SAMPLE_RATE));

            // Transcribe
            let text = parakeet::transcribe_streaming_chunk(
                &buffer_audio,
                None,
                None,
                &model,
                &device,
            )?;

            let text_with_punct = parakeet::add_punctuation(&text);
            println!("\"{}\"", text_with_punct);

            // Update current line
            buffer.update_current_line(text_with_punct);

            // Commit if buffer is full
            if should_commit {
                println!("  → Committing line (buffer full)");
                buffer.commit_and_trim(chunk.len());
                println!();
            }
        }

        idx = end;
    }

    // Final output
    println!("\n===============================\n");
    println!("Full transcript:");
    println!("  Committed lines: {}", buffer.num_committed_lines());
    println!("  Total transcriptions: {}", transcription_count);
    println!();
    println!("{}", buffer.get_full_transcript());

    Ok(())
}
