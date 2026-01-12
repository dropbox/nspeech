/// Transcribe audio using Parakeet TDT (Transducer) model
///
/// This example demonstrates loading and using the Parakeet TDT v3 model.
/// The Transducer architecture provides streaming ASR with automatic alignment.
///
/// Usage:
///   cargo run --example transcribe_tdt --release -- dots.wav
///   cargo run --example transcribe_tdt --release -- MLKDream_16k.wav
///   PARAKEET_DEVICE=cpu cargo run --example transcribe_tdt --release -- dots.wav

use anyhow::Result;
use speech::parakeet::{get_device, load_parakeet_tdt_from_local};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        eprintln!("\nThis example uses Parakeet TDT (Transducer) model.");
        eprintln!("Expected files in .cache/parakeet-tdt/:");
        eprintln!("  - config.json");
        eprintln!("  - model.safetensors");
        eprintln!("  - tokenizer.model or tokenizer.json");
        eprintln!("\nTo download and convert the model:");
        eprintln!("  python scripts/download_parakeet_tdt.py --cache .cache/parakeet-tdt");
        return Ok(());
    }

    let audio_path = &args[1];

    println!("Parakeet TDT Transcription (Transducer)");
    println!("=========================================\n");
    println!("Audio: {}\n", audio_path);

    // Get device
    let device = get_device()?;

    // Load TDT model
    println!("Loading TDT model...");
    let mut model = load_parakeet_tdt_from_local(".cache/parakeet-tdt", &device)?;
    println!("  Model: nvidia/parakeet-tdt-0.6b-v3");
    println!("  Architecture: Transducer (RNN-T)");
    println!("  Predictor: {} LSTM layers", model.config.pred_rnn_layers);
    println!("  Vocab size: {}\n", model.config.vocab_size);

    // Load tokenizer
    println!("Loading tokenizer...");
    model.load_tokenizer(".cache/parakeet-tdt")?;
    println!("  Tokenizer loaded\n");

    // Load audio and extract features
    // TDT model uses 128 mel bins (different from CTC which uses 80)
    println!("Processing audio...");
    let features = speech::parakeet::load_wav_as_features(
        audio_path,
        128,  // TDT uses 128 mel bins
        &device
    )?;
    let (batch, frames, feat_dim) = features.dims3()?;
    println!("  Features: batch={}, frames={}, feat_dim={}\n", batch, frames, feat_dim);

    // Convert features to BF16 to match model dtype
    let features = if !device.is_cpu() {
        features.to_dtype(candle_core::DType::BF16)?
    } else {
        features
    };

    // Run encoder
    println!("Running encoder...");
    let start = std::time::Instant::now();
    let encoder_out = model.encoder.forward(&features, false)?;
    let encoder_time = start.elapsed();

    let (b, t, d) = encoder_out.dims3()?;
    println!("  Encoder output: [{}, {}, {}]", b, t, d);
    println!("  Encoder time: {:.2}s\n", encoder_time.as_secs_f32());

    // Run transducer decoding
    println!("Running transducer decoding...");
    let decode_start = std::time::Instant::now();
    let token_ids = model.greedy_decode(&encoder_out)?;
    let decode_time = decode_start.elapsed();

    println!("  Decoded {} tokens", token_ids.len());
    println!("  Decode time: {:.2}s", decode_time.as_secs_f32());
    println!("  Total time: {:.2}s\n", (encoder_time + decode_time).as_secs_f32());

    // Decode tokens to text using Rust tokenizer
    println!("\n=== TRANSCRIPTION ===");

    match model.decode_tokens(&token_ids) {
        Ok(text) => {
            println!("{}", text.trim());
        }
        Err(e) => {
            println!("(Token decoding failed: {})", e);
            println!("Token IDs: {:?}", token_ids);
        }
    }

    println!("=====================\n");

    println!("✓ Transcription complete!");
    println!("\nModel features:");
    println!("  - Streaming-capable architecture");
    println!("  - Automatic alignment (no forced alignment needed)");
    println!("  - Joint encoder-predictor network");

    Ok(())
}
