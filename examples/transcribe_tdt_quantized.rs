/// Test TDT model with GGUF quantization (Q8_0)
///
/// This example demonstrates loading and using the quantized TDT model.
///
/// Usage:
///   cargo run --example transcribe_tdt_quantized --release -- jfk.wav
///   cargo run --example transcribe_tdt_quantized --release -- dots.wav

use anyhow::Result;
use speech::parakeet::{get_device, load_parakeet_tdt_from_gguf_local, load_wav_as_features};
use std::env;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        std::process::exit(1);
    }

    let audio_path = &args[1];
    println!("Testing TDT Quantized Inference");
    println!("===============================");
    println!("Audio: {}\n", audio_path);

    // Get device
    let device = get_device()?;

    // Load quantized TDT model from assets
    println!("Loading quantized TDT model...");
    let load_start = Instant::now();
    let model = load_parakeet_tdt_from_gguf_local("assets", &device)?;
    println!("✓ Model loaded in {:.2}s\n", load_start.elapsed().as_secs_f32());

    // Extract features from audio
    println!("Extracting features from audio...");
    let features = load_wav_as_features(audio_path, 128, &device)?; // TDT uses 128 mel bins

    // Convert to BF16 on GPU for faster inference (dequantized model uses BF16)
    let features = if !device.is_cpu() {
        features.to_dtype(candle_core::DType::BF16)?
    } else {
        features
    };

    let (_, num_frames, _) = features.dims3()?;
    println!("✓ Features extracted: {} frames\n", num_frames);

    // Run encoder
    println!("Running encoder...");
    let encoder_start = Instant::now();
    let encoder_out = model.encoder.forward(&features, false)?;
    let encoder_time = encoder_start.elapsed().as_secs_f32();
    println!("✓ Encoder complete in {:.2}s\n", encoder_time);

    // Decode
    println!("Decoding (greedy)...");
    let decode_start = Instant::now();
    let tokens = model.greedy_decode(&encoder_out)?;
    let decode_time = decode_start.elapsed().as_secs_f32();
    println!("✓ Decoded {} tokens in {:.2}s\n", tokens.len(), decode_time);

    // Convert to text
    let text = model.decode_tokens(&tokens)?;

    // Print results
    println!("Transcription:");
    println!("─────────────────────────────────────────────────────");
    println!("{}", text);
    println!("─────────────────────────────────────────────────────");

    // Summary
    let total_time = load_start.elapsed().as_secs_f32();
    let audio_duration = num_frames as f32 * 0.01; // 10ms per frame
    let rtf = total_time / audio_duration;

    println!("\nPerformance Summary:");
    println!("  Audio duration: {:.2}s", audio_duration);
    println!("  Total time:     {:.2}s", total_time);
    println!("  Encoder time:   {:.2}s", encoder_time);
    println!("  Decode time:    {:.2}s", decode_time);
    println!("  RTF:            {:.2}x", rtf);

    Ok(())
}
