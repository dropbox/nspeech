/// Diagnose LSTM state corruption in streaming transcription
///
/// This example adds detailed logging to understand why maintaining LSTM state
/// causes worse quality (30 tokens) than resetting every chunk (99 tokens).
///
/// Tracks:
/// - When chunks produce 0 tokens
/// - LSTM state statistics (mean, std, min, max)
/// - Token predictions per chunk
/// - State device and shape
///
/// Usage:
///   cargo run --example diagnose_lstm_state --release -- dots.wav

use anyhow::Result;
use speech::parakeet::{
    get_device, load_parakeet_tdt_from_local,
    ParakeetFeatureExtractor, StreamingConfig, StreamingTransducer,
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        return Ok(());
    }

    let audio_path = &args[1];

    println!("LSTM State Corruption Diagnostic");
    println!("==================================\n");
    println!("Audio: {}\n", audio_path);

    // Load model
    let device = get_device()?;
    println!("Loading TDT model...");
    let mut model = load_parakeet_tdt_from_local(".cache/parakeet-tdt", &device)?;
    model.load_tokenizer(".cache/parakeet-tdt")?;

    // Load audio
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    let samples = if spec.channels == 2 {
        samples.chunks(2).map(|c| (c[0] + c[1]) / 2.0).collect()
    } else {
        samples
    };

    println!("Audio: {} samples, {:.2}s\n", samples.len(), samples.len() as f32 / 16000.0);

    let feat_extractor = ParakeetFeatureExtractor::new(128);

    // Configuration
    const CHUNK_SECONDS: f32 = 3.0;
    const OVERLAP_SECONDS: f32 = 0.5;
    const SAMPLES_PER_CHUNK: usize = (16000.0 * CHUNK_SECONDS) as usize;
    const OVERLAP_SAMPLES: usize = (16000.0 * OVERLAP_SECONDS) as usize;
    let stride = SAMPLES_PER_CHUNK - OVERLAP_SAMPLES;
    let total_chunks = (samples.len() + stride - 1) / stride;

    println!("Chunk configuration:");
    println!("  Chunk size: {:.1}s ({} samples)", CHUNK_SECONDS, SAMPLES_PER_CHUNK);
    println!("  Overlap: {:.1}s ({} samples)", OVERLAP_SECONDS, OVERLAP_SAMPLES);
    println!("  Total chunks: {}\n", total_chunks);

    // Enable state preservation (no reset)
    unsafe { std::env::set_var("NO_LSTM_RESET", "1"); }
    unsafe { std::env::set_var("LSTM_STATE_DEBUG", "1"); }  // Trigger detailed logging

    let streaming_config = StreamingConfig {
        chunk_samples: SAMPLES_PER_CHUNK,
        overlap_samples: OVERLAP_SAMPLES,
        emit_partial: true,
    };
    let mut transcriber = StreamingTransducer::new(model, streaming_config);

    println!("=== Processing with State Preservation ===\n");

    for chunk_idx in 0..total_chunks {
        let chunk_start = chunk_idx * stride;
        let chunk_end = (chunk_start + SAMPLES_PER_CHUNK).min(samples.len());
        let chunk_samples = &samples[chunk_start..chunk_end];

        println!("--- Chunk {} ---", chunk_idx + 1);

        let features = feat_extractor.extract_to_tensor(chunk_samples, &device)?;
        let features = if !device.is_cpu() {
            features.to_dtype(candle_core::DType::BF16)?
        } else {
            features
        };

        let new_tokens = transcriber.process_features(&features)?;

        println!("  Tokens produced: {}", new_tokens.len());
        println!("  Total tokens: {}", transcriber.tokens().len());

        if new_tokens.is_empty() {
            println!("  ⚠️  WARNING: 0 tokens produced - LSTM may be stuck");
        }

        // Decode and show progress
        if !new_tokens.is_empty() {
            match transcriber.decode_text() {
                Ok(text) => println!("  Current text: {}", text.trim()),
                Err(e) => println!("  Decode error: {}", e),
            }
        }

        println!();
    }

    let final_text = transcriber.decode_text()?;

    println!("=====================================");
    println!("\n=== FINAL RESULTS ===");
    println!("Total tokens: {}", transcriber.tokens().len());
    println!("Transcript: {}\n", final_text.trim());

    // Compare with baseline
    println!("Expected (baseline): 140 tokens");
    println!("Actual: {} tokens ({:.0}% of baseline)",
             transcriber.tokens().len(),
             (transcriber.tokens().len() as f32 / 140.0) * 100.0);

    unsafe { std::env::remove_var("NO_LSTM_RESET"); }
    unsafe { std::env::remove_var("LSTM_STATE_DEBUG"); }

    Ok(())
}
