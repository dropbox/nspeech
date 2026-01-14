/// Test streaming TDT model on full audio (non-streaming) to verify model works
///
/// This tests if the streaming TDT model can produce good transcriptions
/// when given full audio context, before we tackle the streaming implementation.

use anyhow::Result;
use speech::parakeet::{
    get_device, load_parakeet_streaming_tdt_from_local,
    ParakeetFeatureExtractor,
};
use candle_core::DType;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        return Ok(());
    }
    let audio_path = &args[1];

    println!("Testing Streaming TDT Model (Full Audio, BF16, Non-Streaming)");
    println!("===============================================================\n");

    // Get device
    let device = get_device()?;
    println!("Device: {:?}\n", device);

    // Load model (BF16 safetensors)
    println!("Loading Streaming TDT model...");
    let model_dir = PathBuf::from(".cache/parakeet-streaming-tdt");
    let model = load_parakeet_streaming_tdt_from_local(&model_dir, &device)?;
    println!("✓ Model loaded\n");

    // Load audio
    println!("Loading audio: {}", audio_path);
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    if spec.channels != 1 || spec.sample_rate != 16000 {
        return Err(anyhow::anyhow!("Expected mono 16kHz audio"));
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;

    let duration_sec = samples.len() as f64 / 16000.0;
    println!("✓ Loaded: {:.2}s ({} samples)\n", duration_sec, samples.len());

    // Extract features for full audio
    println!("Extracting features (feat_in={})...", model.encoder.cfg.feat_in);
    let feat_extractor = ParakeetFeatureExtractor::new(model.encoder.cfg.feat_in);
    let features = feat_extractor.extract_to_tensor(&samples, &device)?;

    let features = if !device.is_cpu() {
        features.to_dtype(DType::BF16)?
    } else {
        features
    };
    println!("✓ Features: {:?}\n", features.dims());

    // Run encoder (no caches - full context)
    println!("Running encoder...");
    let start = std::time::Instant::now();
    let encoder_out = model.encoder.forward(&features, false)?;
    println!("✓ Encoder output: {:?} ({:.2}s)\n", encoder_out.dims(), start.elapsed().as_secs_f64());

    // Run greedy decoding
    println!("Running greedy decode...");
    let start = std::time::Instant::now();
    let tokens = model.greedy_decode(&encoder_out)?;
    println!("✓ Decoded {} tokens ({:.2}s)\n", tokens.len(), start.elapsed().as_secs_f64());

    // Decode to text
    let text = model.decode_tokens(&tokens)?;
    println!("=== TRANSCRIPTION ===\n");
    println!("{}\n", text.trim());

    println!("=== STATISTICS ===");
    println!("  Audio duration: {:.2}s", duration_sec);
    println!("  Tokens: {}", tokens.len());
    println!("  Vocab size: {} (joint: {})",
             model.config.vocab_size,
             model.config.joint_vocab_size.unwrap_or(model.config.vocab_size));

    Ok(())
}
