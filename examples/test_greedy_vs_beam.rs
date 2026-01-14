/// Test greedy vs beam search decoding to isolate quality issue
///
/// Compares:
/// 1. Beam search (beam_size=2) - baseline
/// 2. Greedy decode - same encoder output, different decoder

use anyhow::Result;
use speech::parakeet::{
    get_device, load_parakeet_tdt_from_gguf_local,
    ParakeetFeatureExtractor,
};

fn main() -> Result<()> {
    let device = get_device()?;
    println!("Device: {:?}\n", device);

    // Load model
    println!("Loading TDT model...");
    let model = load_parakeet_tdt_from_gguf_local("assets", &device)?;
    println!("✓ Model loaded\n");

    // Load audio
    println!("Loading audio...");
    let mut reader = hound::WavReader::open("dots.wav")?;
    let audio_samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;
    println!("✓ Audio loaded: {:.2}s\n", audio_samples.len() as f64 / 16000.0);

    // Extract features
    let num_mel_bins = model.encoder.cfg.feat_in;
    let feat_extractor = ParakeetFeatureExtractor::new(num_mel_bins);
    let features = feat_extractor.extract_to_tensor(&audio_samples, &device)?;

    // Convert to model dtype if needed
    let features = if !device.is_cpu() {
        features.to_dtype(candle_core::DType::BF16)?
    } else {
        features
    };

    println!("Running encoder (full context)...");
    let encoder_out = model.encoder.forward(&features, false)?;
    let (_, time_steps, _) = encoder_out.dims3()?;
    println!("✓ Encoder output: {} timesteps\n", time_steps);

    // Test 1: Beam search (beam_size=2)
    println!("=== Test 1: Beam Search (beam_size=2) ===");
    let beam_tokens = model.beam_decode(&encoder_out, 2)?;
    let beam_text = model.decode_tokens(&beam_tokens)?;
    println!("Tokens: {}", beam_tokens.len());
    println!("Text: {}\n", beam_text);

    // Test 2: Greedy decode
    println!("=== Test 2: Greedy Decode ===");
    let greedy_tokens = model.greedy_decode(&encoder_out)?;
    let greedy_text = model.decode_tokens(&greedy_tokens)?;
    println!("Tokens: {}", greedy_tokens.len());
    println!("Text: {}\n", greedy_text);

    // Compare
    println!("=== Comparison ===");
    println!("Beam search: {} tokens", beam_tokens.len());
    println!("Greedy:      {} tokens", greedy_tokens.len());
    println!("Quality loss from greedy: {:.1}%",
        (1.0 - greedy_tokens.len() as f32 / beam_tokens.len() as f32) * 100.0);

    Ok(())
}
