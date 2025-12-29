/// Test GGUF quantized model inference
///
/// Usage:
///   cargo run --example test_gguf_inference --release -- --model hf_parakeet/model_q8_0.gguf
///   cargo run --example test_gguf_inference --release -- --model hf_parakeet/model_q4k.gguf
///   PARAKEET_DEVICE=cpu cargo run --example test_gguf_inference --release -- --model hf_parakeet/model_q8_0.gguf

use anyhow::Result;
use candle_core::{quantized::gguf_file, DType, Tensor};
use candle_nn::VarBuilder;
use parakeet::parakeet_ctc::{ParakeetConfig, ParakeetCTC};
use std::collections::HashMap;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut gguf_path = "hf_parakeet/model_q8_0.gguf".to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" if i + 1 < args.len() => {
                gguf_path = args[i + 1].clone();
                i += 2;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    println!("GGUF Quantized Model Inference Test");
    println!("====================================\n");
    println!("Model: {}\n", gguf_path);

    let device = parakeet::get_device()?;
    println!("Device: {:?}\n", device);

    // Load config
    let config = ParakeetConfig::from_file("hf_parakeet/config.json")?;

    // Load GGUF quantized weights
    println!("Loading GGUF model...");
    let mut file = std::fs::File::open(&gguf_path)?;
    let gguf_content = gguf_file::Content::read(&mut file)?;

    println!("  Loaded {} tensors from GGUF", gguf_content.tensor_infos.len());

    // Convert GGUF tensors to Candle tensors
    let mut tensors = HashMap::new();
    for (name, _tensor_info) in gguf_content.tensor_infos.iter() {
        // Load the quantized tensor and dequantize it to FP32
        let qtensor = gguf_content.tensor(&mut file, name, &device)?;
        let tensor = qtensor.dequantize(&device)?;
        tensors.insert(name.clone(), tensor);
    }

    println!("  Dequantized all tensors to FP32");

    // Create VarBuilder from the loaded tensors
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

    let model = ParakeetCTC::new(config.clone(), vb)?;
    println!("✓ Model loaded with GGUF quantized weights\n");

    // Create test input (3 seconds of audio = ~300 frames)
    println!("Creating test input (3s audio)...");
    let batch_size = 1;
    let num_frames = 300;
    let num_mel_bins = config.encoder_config.num_mel_bins;

    let features = Tensor::randn(
        0f32,
        1f32,
        (batch_size, num_frames, num_mel_bins),
        &device,
    )?;

    println!("  Input shape: [{}, {}, {}]\n", batch_size, num_frames, num_mel_bins);

    // Run inference
    println!("Running inference...");
    let start = std::time::Instant::now();
    let logits = model.forward(&features)?;
    let elapsed = start.elapsed();

    let (b, t, v) = logits.dims3()?;
    println!("  Output shape: [{}, {}, {}]", b, t, v);
    println!("  Time: {:.2}s\n", elapsed.as_secs_f32());

    // Check logit statistics
    let logit_vec = logits.flatten_all()?.to_vec1::<f32>()?;
    let mean: f32 = logit_vec.iter().sum::<f32>() / logit_vec.len() as f32;
    let min = logit_vec.iter().copied().fold(f32::INFINITY, f32::min);
    let max = logit_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    println!("Logit statistics:");
    println!("  Mean: {:.4}", mean);
    println!("  Min: {:.4}", min);
    println!("  Max: {:.4}", max);

    // Check first frame
    println!("\nFirst frame top-5 predictions:");
    let first_frame = logits.get(0)?.get(0)?;
    let first_frame_vec = first_frame.to_vec1::<f32>()?;

    let mut indexed: Vec<(usize, f32)> = first_frame_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (idx, (token_id, logit)) in indexed.iter().take(5).enumerate() {
        println!("  {}: token {} = {:.4}", idx + 1, token_id, logit);
    }

    println!("\n✓ GGUF quantized inference completed successfully!");
    println!("\nNOTE: Use PARAKEET_DEVICE=cpu to test CPU inference with optimized GGUF kernels");

    Ok(())
}
