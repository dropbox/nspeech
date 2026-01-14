/// Compare encoder outputs between standard and streaming models
use anyhow::Result;
use speech::parakeet::{
    get_device,
    load_parakeet_tdt_from_local,
    load_parakeet_streaming_tdt_from_local,
    ParakeetFeatureExtractor,
};
use candle_core::DType;

fn main() -> Result<()> {
    let device = get_device()?;

    // Load both models
    println!("Loading models...");
    let mut standard_model = load_parakeet_tdt_from_local(".cache/parakeet-tdt", &device)?;
    standard_model.load_tokenizer(".cache/parakeet-tdt")?;
    let streaming_model = load_parakeet_streaming_tdt_from_local(".cache/parakeet-streaming-tdt", &device)?;
    println!("✓ Models loaded\n");

    // Load audio (first chunk)
    let mut reader = hound::WavReader::open("dots.wav")?;
    let audio_samples: Vec<f32> = reader
        .samples::<i16>()
        .take(32000)  // 2 seconds
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;
    println!("Loaded {} samples ({:.2}s)\n", audio_samples.len(), audio_samples.len() as f64 / 16000.0);

    // Extract features for STANDARD model (80 mel bins)
    println!("=== Standard Model ===");
    let feat_extractor_80 = ParakeetFeatureExtractor::new(80);
    let features_80 = feat_extractor_80.extract_to_tensor(&audio_samples, &device)?;
    let features_80 = if !device.is_cpu() {
        features_80.to_dtype(DType::BF16)?
    } else {
        features_80
    };
    println!("Features (80 mel bins): {:?}", features_80.dims());

    let encoder_out_standard = standard_model.encoder.forward(&features_80, false)?;
    println!("Encoder output: {:?}", encoder_out_standard.dims());

    // Statistics
    let encoder_f32 = encoder_out_standard.to_dtype(DType::F32)?;
    let encoder_vec: Vec<f32> = encoder_f32.flatten_all()?.to_vec1()?;
    let mean: f32 = encoder_vec.iter().sum::<f32>() / encoder_vec.len() as f32;
    let std: f32 = (encoder_vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / encoder_vec.len() as f32).sqrt();
    let min = encoder_vec.iter().copied().fold(f32::INFINITY, f32::min);
    let max = encoder_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    println!("Encoder statistics:");
    println!("  Mean: {:.6}", mean);
    println!("  Std: {:.6}", std);
    println!("  Min: {:.6}", min);
    println!("  Max: {:.6}\n", max);

    // Extract features for STREAMING model (136 mel bins)
    println!("=== Streaming Model ===");
    let feat_extractor_136 = ParakeetFeatureExtractor::new(136);
    let features_136 = feat_extractor_136.extract_to_tensor(&audio_samples, &device)?;
    let features_136 = if !device.is_cpu() {
        features_136.to_dtype(DType::BF16)?
    } else {
        features_136
    };
    println!("Features (136 mel bins): {:?}", features_136.dims());

    let encoder_out_streaming = streaming_model.encoder.forward(&features_136, false)?;
    println!("Encoder output: {:?}", encoder_out_streaming.dims());

    // Statistics
    let encoder_f32 = encoder_out_streaming.to_dtype(DType::F32)?;
    let encoder_vec: Vec<f32> = encoder_f32.flatten_all()?.to_vec1()?;
    let mean: f32 = encoder_vec.iter().sum::<f32>() / encoder_vec.len() as f32;
    let std: f32 = (encoder_vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / encoder_vec.len() as f32).sqrt();
    let min = encoder_vec.iter().copied().fold(f32::INFINITY, f32::min);
    let max = encoder_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    println!("Encoder statistics:");
    println!("  Mean: {:.6}", mean);
    println!("  Std: {:.6}", std);
    println!("  Min: {:.6}", min);
    println!("  Max: {:.6}", max);

    Ok(())
}
