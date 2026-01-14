/// Test streaming TDT model on full audio (non-streaming baseline)
/// This verifies the model works correctly with fixed blank_id=1024

use anyhow::Result;
use speech::parakeet::{
    get_device, load_parakeet_streaming_tdt_from_local,
    ParakeetFeatureExtractor,
};
use candle_core::DType;

fn main() -> Result<()> {
    let device = get_device()?;
    println!("Device: {:?}\n", device);

    // Load model
    println!("Loading Streaming TDT model...");
    let model = load_parakeet_streaming_tdt_from_local(".cache/parakeet-streaming-tdt", &device)?;
    println!("✓ Model loaded");
    println!("  Vocab size: {}", model.config.vocab_size);
    println!("  Blank ID: {}\n", model.config.blank_id);

    // Load audio
    println!("Loading audio...");
    let mut reader = hound::WavReader::open("dots.wav")?;
    let audio_samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;

    let duration = audio_samples.len() as f64 / 16000.0;
    println!("✓ Audio loaded: {:.2}s ({} samples)\n", duration, audio_samples.len());

    // Extract features for full audio
    println!("Extracting features...");
    let num_mel_bins = model.encoder.cfg.feat_in;
    let feat_extractor = ParakeetFeatureExtractor::new(num_mel_bins);
    let features = feat_extractor.extract_to_tensor(&audio_samples, &device)?;

    // Convert to model dtype
    let features = if !device.is_cpu() {
        features.to_dtype(DType::BF16)?
    } else {
        features
    };

    let (batch, time, feat) = features.dims3()?;
    println!("✓ Features extracted: [{}, {}, {}]\n", batch, time, feat);

    // Run encoder (no caching for baseline)
    println!("Running encoder...");
    let start = std::time::Instant::now();
    let encoder_out = model.encoder.forward(&features, false)?;
    let encoder_time = start.elapsed();

    let (_, enc_time, enc_dim) = encoder_out.dims3()?;
    println!("✓ Encoder output: [{}, {}, {}] ({:.2}s)\n", batch, enc_time, enc_dim, encoder_time.as_secs_f64());

    // Run greedy decode
    println!("Running greedy decode with blank_id={}...", model.config.blank_id);
    let start = std::time::Instant::now();
    let tokens = model.greedy_decode(&encoder_out)?;
    let decode_time = start.elapsed();

    println!("✓ Decoded {} tokens ({:.2}s)\n", tokens.len(), decode_time.as_secs_f64());

    // Decode to text
    println!("Decoding to text...");
    let text = model.decode_tokens(&tokens)?;

    println!("\n=== TRANSCRIPTION ===");
    println!("{}\n", text);

    println!("=== STATISTICS ===");
    println!("  Audio duration: {:.2}s", duration);
    println!("  Total time: {:.2}s", (encoder_time + decode_time).as_secs_f64());
    println!("  Tokens: {}", tokens.len());
    println!("  First 10 token IDs: {:?}", &tokens[..tokens.len().min(10)]);

    println!("\n  NeMo reference: 225 tokens");
    println!("  Our result: {} tokens ({:.1}%)", tokens.len(), tokens.len() as f32 / 225.0 * 100.0);

    Ok(())
}
