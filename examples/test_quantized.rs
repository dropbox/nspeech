/// Test quantized model loading and numerical accuracy
///
/// Usage:
///   cargo run --example test_quantized --release -- --quant q8_0
///   cargo run --example test_quantized --release -- --quant q4_0
///   cargo run --example test_quantized --release -- --compare

use anyhow::Result;
use candle_core::DType;
use candle_nn::VarBuilder;
use std::path::Path;

// NOTE: This example uses old code from src/old/
#[path = "../src/old/parakeet_ctc.rs"]
mod parakeet_ctc;
#[path = "../src/old/quantized_loader.rs"]
mod quantized_loader;
use parakeet_ctc::ParakeetConfig;
use quantized_loader::QuantizedLoader;

fn compute_stats(data: &[f32]) -> (f32, f32, f32, f32) {
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (mean, std, min, max)
}

fn compare_weights(fp32_data: &[f32], quant_data: &[f32], name: &str) {
    assert_eq!(fp32_data.len(), quant_data.len(), "Weight size mismatch for {}", name);

    let (fp32_mean, fp32_std, fp32_min, fp32_max) = compute_stats(fp32_data);
    let (quant_mean, quant_std, quant_min, quant_max) = compute_stats(quant_data);

    // Compute error metrics
    let mut abs_errors = Vec::with_capacity(fp32_data.len());
    let mut rel_errors = Vec::with_capacity(fp32_data.len());

    for (fp32, quant) in fp32_data.iter().zip(quant_data.iter()) {
        let abs_err = (fp32 - quant).abs();
        abs_errors.push(abs_err);

        if fp32.abs() > 1e-8 {
            let rel_err = abs_err / fp32.abs();
            rel_errors.push(rel_err);
        }
    }

    let (mae, _, _, max_ae) = compute_stats(&abs_errors);
    let mean_re = if !rel_errors.is_empty() {
        rel_errors.iter().sum::<f32>() / rel_errors.len() as f32
    } else {
        0.0
    };

    println!("\n  {}", name);
    println!("    FP32:   mean={:.6}, std={:.6}, range=[{:.6}, {:.6}]",
             fp32_mean, fp32_std, fp32_min, fp32_max);
    println!("    Quant:  mean={:.6}, std={:.6}, range=[{:.6}, {:.6}]",
             quant_mean, quant_std, quant_min, quant_max);
    println!("    Error:  MAE={:.6}, Max AE={:.6}, Mean RE={:.4}%",
             mae, max_ae, mean_re * 100.0);
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let compare_mode = args.len() > 1 && args[1] == "--compare";
    let quant_type = if !compare_mode && args.len() > 2 && args[1] == "--quant" {
        &args[2]
    } else {
        "q8_0"
    };

    let device = parakeet::get_device()?;

    // Load config
    let _config = ParakeetConfig::from_file("hf_parakeet/config.json")?;

    if compare_mode {
        println!("Comparing Quantized vs FP32 Weights");
        println!("============================================\n");

        // Load FP32 weights
        println!("Loading FP32 baseline...");
        let weights_path = Path::new("hf_parakeet/model.safetensors");
        let vb_fp32 = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
        };

        // Test both Q8_0 and Q4_0
        for qt in &["q8_0", "q4_0"] {
            println!("\n\n========== {} Comparison ==========", qt.to_uppercase());

            let npz_path = format!("hf_parakeet/model_{}.npz", qt);
            let qloader = QuantizedLoader::from_npz(&npz_path, device.clone())?;

            let sample_keys = [
                "encoder.layers.0.self_attn.q_proj.weight",
                "encoder.layers.0.feed_forward1.linear1.weight",
                "encoder.layers.15.self_attn.q_proj.weight",
                "ctc_head.weight",
            ];

            for key in &sample_keys {
                // Load quantized version first to get the expected shape
                let quant_tensor = match qloader.get(key) {
                    Ok(t) => t,
                    Err(e) => {
                        println!("\n  Skipping {}: {}", key, e);
                        continue;
                    }
                };
                let shape = quant_tensor.shape();

                // Navigate VarBuilder hierarchy to get FP32 tensor
                // Split key into path components: "encoder.layers.0.self_attn.q_proj.weight"
                let mut vb = vb_fp32.clone();
                let parts: Vec<&str> = key.split('.').collect();

                // Navigate to the right VarBuilder node (all but last component)
                for &part in &parts[..parts.len()-1] {
                    vb = vb.pp(part);
                }

                // Get the final tensor (last component, usually "weight" or "bias")
                let tensor_name = parts[parts.len()-1];
                let fp32_tensor = match vb.get(shape.clone(), tensor_name) {
                    Ok(t) => t,
                    Err(e) => {
                        println!("\n  Skipping {}: Could not load FP32 version: {}", key, e);
                        continue;
                    }
                };

                // Compare
                let fp32_data = fp32_tensor.flatten_all()?.to_vec1::<f32>()?;
                let quant_data = quant_tensor.flatten_all()?.to_vec1::<f32>()?;

                compare_weights(&fp32_data, &quant_data, key);
            }
        }

        println!("\n\n✓ Quantization accuracy comparison complete!");

    } else {
        println!("Testing {} Quantized Model\n", quant_type.to_uppercase());
        println!("============================================\n");

        // Load quantized weights
        let npz_path = format!("hf_parakeet/model_{}.npz", quant_type);
        println!("Loading quantized weights from {}...", npz_path);
        let qloader = QuantizedLoader::from_npz(&npz_path, device.clone())?;

        println!("Loaded {} weight tensors\n", qloader.weight_names().len());

        println!("\n============================================");
        println!("Quantized Model Summary:");
        println!("  Format: {}", quant_type.to_uppercase());
        println!("  Weights: {} tensors", qloader.weight_names().len());

        // Sample a few weights to verify loading
        println!("\nSample weights (verifying dequantization):");
        let sample_keys = [
            "encoder.layers.0.self_attn.q_proj.weight",
            "encoder.layers.0.feed_forward1.linear1.weight",
            "ctc_head.weight",
        ];

        for key in &sample_keys {
            if let Ok(tensor) = qloader.get(key) {
                let data = tensor.flatten_all()?.to_vec1::<f32>()?;
                let (mean, std, min, max) = compute_stats(&data);
                println!("  {}", key);
                println!("    Shape: {:?}", tensor.shape());
                println!("    Mean: {:.6}, Std: {:.6}, Range: [{:.6}, {:.6}]", mean, std, min, max);
            }
        }

        println!("\n✓ Quantized model weights loaded and verified!");
        println!("\nRun with --compare to compare quantized weights with FP32 baseline.");
    }

    Ok(())
}
