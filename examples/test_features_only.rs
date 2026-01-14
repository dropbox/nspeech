/// Test feature extraction for streaming TDT model
/// Print statistics to compare with NeMo

use anyhow::Result;
use speech::parakeet::{get_device, ParakeetFeatureExtractor};
use candle_core::IndexOp;

fn main() -> Result<()> {
    let device = get_device()?;

    // Load audio
    println!("Loading audio...");
    let mut reader = hound::WavReader::open("dots.wav")?;
    let audio_samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;

    println!("✓ Audio: {} samples\n", audio_samples.len());

    // Extract features with 136 mel bins (streaming TDT)
    println!("Extracting features (136 mel bins)...");
    let feat_extractor = ParakeetFeatureExtractor::new(136);
    let features = feat_extractor.extract_to_tensor(&audio_samples, &device)?;

    let (batch, time, feat) = features.dims3()?;
    println!("✓ Features shape: [{}, {}, {}]\n", batch, time, feat);

    // Get statistics
    let features_f32 = features.to_dtype(candle_core::DType::F32)?;
    let mean = features_f32.mean_all()?.to_scalar::<f32>()?;
    let var = features_f32.var_keepdim(0)?.mean_all()?.to_scalar::<f32>()?;
    let std = var.sqrt();
    let min = features_f32.min(0)?.min(0)?.min(0)?.to_scalar::<f32>()?;
    let max = features_f32.max(0)?.max(0)?.max(0)?.to_scalar::<f32>()?;

    println!("=== Feature Statistics ===");
    println!("  Mean: {:.6}", mean);
    println!("  Std:  {:.6}", std);
    println!("  Min:  {:.6}", min);
    println!("  Max:  {:.6}", max);
    println!();

    // Print first frame (first 10 values)
    println!("First frame (first 10 values):");
    let first_frame = features_f32.i((0, 0))?;
    for i in 0..10.min(136) {
        let val = first_frame.i(i)?.to_scalar::<f32>()?;
        println!("  [{}]: {:.6}", i, val);
    }

    Ok(())
}
