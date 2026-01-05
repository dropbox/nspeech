/// Transcribe audio using GGUF quantized model
///
/// This example demonstrates loading and using a GGUF quantized Parakeet model.
/// GGUF provides fast inference with optimized kernels while reducing model size.
///
/// Usage:
///   cargo run --example transcribe_quantized --release -- dots.wav
///   cargo run --example transcribe_quantized --release -- MLKDream_16k.wav
///   PARAKEET_DEVICE=cpu cargo run --example transcribe_quantized --release -- dots.wav

use anyhow::Result;
use speech::parakeet::{get_device, load_parakeet_ctc_from_gguf_local};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        eprintln!("\nThis example uses GGUF quantized weights for fast inference.");
        eprintln!("Expected files in assets/:");
        eprintln!("  - config.json.zst");
        eprintln!("  - model_q8_0.gguf.zst (recommended) or model_q4k.gguf.zst");
        eprintln!("  - tokenizer.json.zst");
        return Ok(());
    }

    let audio_path = &args[1];

    println!("Parakeet CTC Transcription (Quantized)");
    println!("=======================================\n");
    println!("Audio: {}\n", audio_path);

    // Get device
    let device = get_device()?;

    // Load quantized model (tries Q8_0 first, then Q4K)
    println!("Loading model from GGUF...");
    let model = load_parakeet_ctc_from_gguf_local("assets", &device)?;

    // Load audio and extract features
    println!("Processing audio...");
    let features = speech::parakeet::load_wav_as_features(
        audio_path,
        model.cfg.feat_in,
        &device
    )?;
    let (batch, frames, feat_dim) = features.dims3()?;
    println!("  Features: batch={}, frames={}, feat_dim={}\n", batch, frames, feat_dim);

    // Run inference
    println!("Running inference...");
    let start = std::time::Instant::now();
    let logits = model.forward(&features, false)?;
    let elapsed = start.elapsed();

    let (b, t, v) = logits.dims3()?;
    println!("  Logits shape: [{}, {}, {}]", b, t, v);
    println!("  Inference time: {:.2}s\n", elapsed.as_secs_f32());

    // Decode transcription
    println!("Decoding transcription...");
    let transcripts = model.greedy_decode(&logits)?;

    println!("\n=== TRANSCRIPTION ===");
    for (idx, transcript) in transcripts.iter().enumerate() {
        println!("[{}] {}", idx, transcript);
    }
    println!("=====================\n");

    println!("✓ Transcription complete!");
    println!("\nModel info:");
    println!("  Format: GGUF Quantized (Q8_0 or Q4K)");
    println!("  Compression: 2.65x (Q8_0) or 3.8x (Q4K)");
    println!("  Accuracy: Excellent (Q8_0) or Good (Q4K)");

    Ok(())
}
