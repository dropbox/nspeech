/// Debug feature extraction - check raw and normalized values

use anyhow::Result;

fn main() -> Result<()> {
    // Load audio
    println!("Loading audio...");
    let mut reader = hound::WavReader::open("dots.wav")?;
    let audio_samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;

    println!("✓ Audio: {} samples\n", audio_samples.len());

    // Extract features with 136 mel bins
    println!("Extracting features (136 mel bins)...");
    let feat_extractor = speech::parakeet::ParakeetFeatureExtractor::new(136);
    let (frames, feats_flat) = feat_extractor.extract_flat(&audio_samples);

    println!("✓ Features: {} frames x 136 bins = {} total values\n", frames, feats_flat.len());

    // Print first frame (first 10 bins)
    println!("First frame (first 10 bins):");
    for i in 0..10 {
        let val = feats_flat[i];  // First frame is feats_flat[0..136]
        println!("  Bin {}: {:.6}", i, val);
    }
    println!();

    // Calculate statistics over all features
    let sum: f32 = feats_flat.iter().sum();
    let mean = sum / feats_flat.len() as f32;

    let sq_diff_sum: f32 = feats_flat.iter().map(|x| (x - mean).powi(2)).sum();
    let variance = sq_diff_sum / feats_flat.len() as f32;
    let std = variance.sqrt();

    let min = feats_flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = feats_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("=== Overall Statistics ===");
    println!("  Mean: {:.6}", mean);
    println!("  Std:  {:.6}", std);
    println!("  Min:  {:.6}", min);
    println!("  Max:  {:.6}", max);

    Ok(())
}
