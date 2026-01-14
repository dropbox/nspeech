/// Test standard TDT model with NO normalization (to match streaming model config)
///
/// This tests if the blank domination issue is caused by normalization mismatch.
/// Streaming model config has normalize='NA' (no normalization), but we were using per-feature normalization.

use anyhow::Result;
use speech::parakeet::{
    get_device, load_parakeet_tdt_from_local,
    ParakeetFeatureExtractor,
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        return Ok(());
    }

    let device = get_device()?;
    println!("Device: {:?}\n", device);

    // Load standard TDT model
    println!("Loading TDT model (BF16 safetensors)...");
    let mut model = load_parakeet_tdt_from_local(".cache/parakeet-tdt", &device)?;
    println!("✓ Model loaded\n");

    // Load tokenizer
    println!("Loading tokenizer...");
    model.load_tokenizer(".cache/parakeet-tdt")?;
    println!("✓ Tokenizer loaded\n");

    // Load audio
    println!("Loading audio...");
    let audio_path = &args[1];
    let mut reader = hound::WavReader::open(&audio_path)?;
    let audio_samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;

    let duration = audio_samples.len() as f64 / 16000.0;
    println!("✓ Audio loaded: {:.2}s ({} samples)\n", duration, audio_samples.len());

    // Test 1: WITH normalization (default, should work)
    println!("=== Test 1: WITH per-feature normalization (baseline) ===");
    let feat_extractor_norm = ParakeetFeatureExtractor::new(80);
    let features_norm = feat_extractor_norm.extract_to_tensor(&audio_samples, &device)?;

    // Convert to BF16 if on GPU
    let features_norm = if !device.is_cpu() {
        features_norm.to_dtype(candle_core::DType::BF16)?
    } else {
        features_norm
    };

    let tokens_norm = model.greedy_decode(&features_norm)?;
    let text_norm = model.decode_tokens(&tokens_norm)?;

    println!("  Tokens: {}", tokens_norm.len());
    println!("  Text: {}\n", text_norm);

    // Test 2: WITHOUT normalization (to match streaming model)
    println!("=== Test 2: WITHOUT normalization (streaming model style) ===");
    let feat_extractor_no_norm = ParakeetFeatureExtractor::new_with_config(80, false);
    let features_no_norm = feat_extractor_no_norm.extract_to_tensor(&audio_samples, &device)?;

    // Convert to BF16 if on GPU
    let features_no_norm = if !device.is_cpu() {
        features_no_norm.to_dtype(candle_core::DType::BF16)?
    } else {
        features_no_norm
    };

    let tokens_no_norm = model.greedy_decode(&features_no_norm)?;
    let text_no_norm = model.decode_tokens(&tokens_no_norm)?;

    println!("  Tokens: {}", tokens_no_norm.len());
    println!("  Text: {}\n", text_no_norm);

    // Compare
    println!("=== Comparison ===");
    println!("  With normalization: {} tokens", tokens_norm.len());
    println!("  Without normalization: {} tokens", tokens_no_norm.len());

    if tokens_no_norm.len() < 50 {
        println!("\n⚠️  WARNING: Without normalization produces very few tokens!");
        println!("  This suggests the model was NOT trained without normalization.");
    } else if tokens_no_norm.len() as f32 > tokens_norm.len() as f32 * 0.8 {
        println!("\n✓ Both approaches produce reasonable results.");
    }

    Ok(())
}
