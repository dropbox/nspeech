/// Test FP16 vs FP32 inference to identify where numerical issues occur
///
/// This example compares FP32 and FP16 model outputs layer-by-layer to identify
/// where precision loss causes problems.

use anyhow::Result;
use candle_core::DType;
use speech::parakeet::{self, ParakeetCtcModel};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        return Ok(());
    }

    let audio_path = &args[1];
    let device = parakeet::get_device()?;

    println!("FP16 Numerical Stability Test");
    println!("==============================\n");

    // Load audio and extract features (always F32)
    println!("Loading audio: {}", audio_path);
    let features = parakeet::load_wav_as_features(audio_path, 80, &device)?;
    println!("Features shape: {:?}", features.dims());
    println!("Features dtype: {:?}\n", features.dtype());

    // Check feature statistics
    let feat_flat = features.flatten_all()?.to_vec1::<f32>()?;
    let feat_mean = feat_flat.iter().sum::<f32>() / feat_flat.len() as f32;
    let feat_min = feat_flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let feat_max = feat_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("Feature statistics:");
    println!("  Mean: {:.6}", feat_mean);
    println!("  Min:  {:.6}", feat_min);
    println!("  Max:  {:.6}", feat_max);
    println!();

    // Load F32 model
    println!("Loading F32 model...");
    let model_f32 = parakeet::load_parakeet_ctc_from_gguf_local("assets", &device)?;
    println!("✓ F32 model loaded\n");

    // F32 inference
    println!("Running F32 inference...");
    let features_f32 = features.to_dtype(DType::F32)?;
    let logits_f32 = model_f32.forward(&features_f32, false)?;
    println!("  Logits shape: {:?}", logits_f32.dims());

    let logits_f32_flat = logits_f32.flatten_all()?.to_vec1::<f32>()?;
    let logits_f32_mean = logits_f32_flat.iter().sum::<f32>() / logits_f32_flat.len() as f32;
    let logits_f32_min = logits_f32_flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let logits_f32_max = logits_f32_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  Logits mean: {:.6}", logits_f32_mean);
    println!("  Logits min:  {:.6}", logits_f32_min);
    println!("  Logits max:  {:.6}", logits_f32_max);

    let transcripts_f32 = model_f32.greedy_decode(&logits_f32)?;
    println!("  F32 transcript: \"{}\"", transcripts_f32[0]);
    println!();

    // F16 inference - convert features to F16
    println!("Running F16 inference (converted features)...");
    let features_f16 = features.to_dtype(DType::F16)?;

    // Check if F16 conversion changed values significantly
    let features_f16_back = features_f16.to_dtype(DType::F32)?;
    let feat_f16_flat = features_f16_back.flatten_all()?.to_vec1::<f32>()?;
    let max_diff = feat_flat.iter().zip(feat_f16_flat.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  Max feature F32->F16->F32 diff: {:.10}", max_diff);

    let logits_f16 = model_f32.forward(&features_f16, false)?;
    let logits_f16_f32 = logits_f16.to_dtype(DType::F32)?;
    let logits_f16_flat = logits_f16_f32.flatten_all()?.to_vec1::<f32>()?;
    let logits_f16_mean = logits_f16_flat.iter().sum::<f32>() / logits_f16_flat.len() as f32;
    let logits_f16_min = logits_f16_flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let logits_f16_max = logits_f16_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  Logits mean: {:.6}", logits_f16_mean);
    println!("  Logits min:  {:.6}", logits_f16_min);
    println!("  Logits max:  {:.6}", logits_f16_max);

    // Compare logits
    let logits_diff = logits_f32_flat.iter().zip(logits_f16_flat.iter())
        .map(|(a, b)| (a - b).abs())
        .collect::<Vec<_>>();
    let max_logit_diff = logits_diff.iter().cloned().fold(0.0f32, f32::max);
    let mean_logit_diff = logits_diff.iter().sum::<f32>() / logits_diff.len() as f32;
    println!("  Max logit diff (F32 vs F16): {:.6}", max_logit_diff);
    println!("  Mean logit diff: {:.6}", mean_logit_diff);

    let transcripts_f16 = model_f32.greedy_decode(&logits_f16_f32)?;
    println!("  F16 transcript: \"{}\"", transcripts_f16[0]);
    println!();

    // Check if transcripts match
    if transcripts_f32[0] == transcripts_f16[0] {
        println!("✓ F32 and F16 transcripts MATCH");
    } else {
        println!("✗ F32 and F16 transcripts DIFFER");
        println!("\nThis suggests the model is sensitive to F16 precision in:");
        println!("  - Attention softmax operations");
        println!("  - Layer/batch normalization");
        println!("  - Scale factors in conformer blocks");
    }

    Ok(())
}
