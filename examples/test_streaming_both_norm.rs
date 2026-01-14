/// Test streaming model with BOTH normalization approaches
/// Compare: normalize=true (per-feature) vs normalize=false (none)

use anyhow::Result;
use speech::parakeet::{
    get_device, load_parakeet_streaming_tdt_from_local,
    ParakeetFeatureExtractor,
};
use candle_core::DType;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        return Ok(());
    }

    let device = get_device()?;
    println!("Device: {:?}\n", device);

    // Load model
    let mut model = load_parakeet_streaming_tdt_from_local(".cache/parakeet-streaming-tdt", &device)?;
    model.load_tokenizer(".cache/parakeet-streaming-tdt")?;
    println!("Model loaded (vocab={}, blank_id={}, mel_bins={})\n",
             model.config.vocab_size, model.config.blank_id, model.encoder.cfg.feat_in);

    // Load audio
    let audio_path = &args[1];
    let mut reader = hound::WavReader::open(&audio_path)?;
    let audio_samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;
    println!("Audio: {:.2}s\n", audio_samples.len() as f64 / 16000.0);

    let num_mel_bins = model.encoder.cfg.feat_in;

    // Test 1: WITHOUT normalization (config says normalize='NA')
    println!("=== Test 1: WITHOUT normalization (config='NA') ===");
    let feat_no_norm = ParakeetFeatureExtractor::new_with_config(num_mel_bins, false);
    let features_no_norm = feat_no_norm.extract_to_tensor(&audio_samples, &device)?;
    let features_no_norm = if !device.is_cpu() {
        features_no_norm.to_dtype(DType::BF16)?
    } else {
        features_no_norm
    };

    // Non-streaming decode (full context)
    let encoder_out_no_norm = model.encoder.forward(&features_no_norm, false)?;
    let tokens_no_norm = model.greedy_decode_streaming(
        &encoder_out_no_norm,
        None,
        model.config.blank_id as u32,
    )?.0;

    let text_no_norm = model.decode_tokens(&tokens_no_norm)?;
    println!("  Tokens: {}", tokens_no_norm.len());
    println!("  Text: {}\n", text_no_norm);

    // Test 2: WITH per-feature normalization
    println!("=== Test 2: WITH per-feature normalization ===");
    let feat_with_norm = ParakeetFeatureExtractor::new_with_config(num_mel_bins, true);
    let features_with_norm = feat_with_norm.extract_to_tensor(&audio_samples, &device)?;
    let features_with_norm = if !device.is_cpu() {
        features_with_norm.to_dtype(DType::BF16)?
    } else {
        features_with_norm
    };

    let encoder_out_with_norm = model.encoder.forward(&features_with_norm, false)?;
    let tokens_with_norm = model.greedy_decode_streaming(
        &encoder_out_with_norm,
        None,
        model.config.blank_id as u32,
    )?.0;

    let text_with_norm = model.decode_tokens(&tokens_with_norm)?;
    println!("  Tokens: {}", tokens_with_norm.len());
    println!("  Text: {}\n", text_with_norm);

    // Compare
    println!("=== Comparison ===");
    println!("  WITHOUT normalization: {} tokens ({:.1}% quality)",
             tokens_no_norm.len(), tokens_no_norm.len() as f32 / 225.0 * 100.0);
    println!("  WITH normalization: {} tokens ({:.1}% quality)",
             tokens_with_norm.len(), tokens_with_norm.len() as f32 / 225.0 * 100.0);
    println!("  NeMo reference: 225 tokens (100%)");

    if tokens_with_norm.len() > tokens_no_norm.len() {
        println!("\n✅ per-feature normalization is BETTER (despite config saying 'NA')");
    } else {
        println!("\n✅ No normalization matches config");
    }

    Ok(())
}
