/// Transcribe with punctuation using TDT model's built-in timestamps
///
/// This example demonstrates using the Parakeet TDT model's natural frame-level
/// alignment to add punctuation based on pauses between tokens.
///
/// Unlike VAD-based approaches that use external timing, this uses the TDT
/// model's inherent timestamp information from the transducer alignment.
///
/// Usage:
///   cargo run --example transcribe_tdt_with_punctuation --release -- dots.wav
///   PARAKEET_DEVICE=cpu cargo run --example transcribe_tdt_with_punctuation --release -- dots.wav

use anyhow::Result;
use speech::parakeet::{
    get_device, load_parakeet_tdt_from_local, ParakeetFeatureExtractor,
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        eprintln!("\nThis example uses TDT's built-in frame-level alignment");
        eprintln!("to add punctuation based on pauses between tokens.");
        return Ok(());
    }

    let audio_path = &args[1];

    println!("Parakeet TDT with Timestamp-Based Punctuation");
    println!("==============================================\n");
    println!("Audio: {}\n", audio_path);

    // Load model
    let device = get_device()?;
    println!("Device: {:?}", device);

    println!("Loading TDT model...");
    let mut model = load_parakeet_tdt_from_local(".cache/parakeet-tdt", &device)?;
    model.load_tokenizer(".cache/parakeet-tdt")?;
    println!("✓ Model loaded\n");

    // Load audio
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;

    let samples = if spec.channels == 2 {
        samples.chunks(2).map(|c| (c[0] + c[1]) / 2.0).collect()
    } else {
        samples
    };

    let duration = samples.len() as f32 / 16000.0;
    println!("Audio: {:.2}s ({} samples)\n", duration, samples.len());

    // Extract features
    println!("Extracting features...");
    let feat_extractor = ParakeetFeatureExtractor::new(128);
    let features = feat_extractor.extract_to_tensor(&samples, &device)?;
    let features = if !device.is_cpu() {
        features.to_dtype(candle_core::DType::BF16)?
    } else {
        features
    };

    let (_, mel_frames, _) = features.dims3()?;
    println!("  Mel frames: {}", mel_frames);

    // Run encoder
    println!("\nRunning encoder...");
    let encoder_out = model.encoder.forward(&features, false)?;
    let (_, enc_frames, _) = encoder_out.dims3()?;
    println!("  Encoder frames: {} (8x downsampling)", enc_frames);
    println!("  Frame duration: 80ms per encoder frame");

    // Decode with timestamps
    println!("\nDecoding with timestamps...");
    let tokens_with_ts = model.greedy_decode_with_timestamps(&encoder_out)?;
    println!("  Decoded {} tokens with timestamps", tokens_with_ts.len());

    // Show some timestamp examples
    if tokens_with_ts.len() > 0 {
        println!("\n  Sample timestamps:");
        for (i, token_ts) in tokens_with_ts.iter().take(10).enumerate() {
            let time_sec = token_ts.frame as f32 * 0.08; // 80ms per frame
            println!("    Token {}: frame={}, time={:.3}s",
                     i, token_ts.frame, time_sec);
        }
        if tokens_with_ts.len() > 10 {
            println!("    ... ({} more)", tokens_with_ts.len() - 10);
        }
    }

    // Add punctuation based on frame gaps
    println!("\nAdding punctuation from frame gaps...");
    println!("  Comma threshold: 5 frames (400ms pause)");
    println!("  Period threshold: 10 frames (800ms pause)");

    let text_with_punct = model.add_punctuation_from_timestamps(&tokens_with_ts)?;

    // Also get text without punctuation for comparison
    let tokens: Vec<u32> = tokens_with_ts.iter().map(|t| t.token).collect();
    let text_no_punct = model.decode_tokens(&tokens)?;

    // Print results
    println!("\n{}", "=".repeat(70));
    println!("RESULTS");
    println!("{}", "=".repeat(70));
    println!("\nWithout punctuation:");
    println!("{}\n", text_no_punct.trim());

    println!("With timestamp-based punctuation:");
    println!("{}\n", text_with_punct.trim());

    // Statistics
    println!("{}", "=".repeat(70));
    println!("Statistics:");
    println!("  Total tokens: {}", tokens.len());
    println!("  Total encoder frames: {}", enc_frames);
    println!("  Audio duration: {:.2}s", duration);
    println!("  Tokens/second: {:.1}", tokens.len() as f32 / duration);

    // Count commas and periods added
    let comma_count = text_with_punct.matches(',').count();
    let period_count = text_with_punct.matches('.').count();
    println!("  Punctuation added: {} commas, {} periods", comma_count, period_count);

    println!("\n✓ Transcription with timestamp-based punctuation complete!");

    Ok(())
}
