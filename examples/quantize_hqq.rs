/// Quantize Parakeet model using HQQ (Half-Quadratic Quantization)
///
/// **Status**: Research/proof-of-concept tool. Output is NOT compatible with
/// production QMatMul inference. See HQQ_QUANTIZATION.md for details.
///
/// HQQ is an advanced quantization method that optimizes scales and zero-points
/// to minimize quantization error. This typically provides better accuracy than
/// standard round-to-nearest quantization.
///
/// Usage:
///   cargo run --example quantize_hqq --release -- \
///     hf_parakeet/model.safetensors \
///     hf_parakeet/model_hqq4.safetensors \
///     --nbits 4 --group-size 128
///
///   cargo run --example quantize_hqq --release -- \
///     hf_parakeet/model.safetensors \
///     hf_parakeet/model_hqq3.safetensors \
///     --nbits 3 --group-size 64 --symmetric

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

// Include HQQ module directly (not part of lib.rs)
#[path = "../src/hqq.rs"]
mod hqq;
use hqq::{HqqConfig, HqqTensor};

/// Determine if a layer should be quantized with HQQ
fn should_quantize_tensor(name: &str, tensor: &Tensor, group_size: usize) -> bool {
    // Don't quantize biases, norms, or embeddings
    if name.contains("bias")
        || name.contains("norm")
        || name.contains("layernorm")
        || name.contains("batchnorm")
    {
        return false;
    }

    // Only quantize 2D weight matrices
    if tensor.rank() != 2 {
        return false;
    }

    // Check if dimensions are compatible with group size
    let cols = tensor.dim(1).unwrap_or(0);
    if cols < group_size {
        return false;
    }

    // Quantize if it's a weight matrix
    name.contains("weight")
}

/// Compute quantization error metrics
fn compute_error_metrics(original: &Tensor, quantized: &Tensor) -> Result<(f32, f32, f32)> {
    let orig_flat = original.flatten_all()?.to_vec1::<f32>()?;
    let quant_flat = quantized.flatten_all()?.to_vec1::<f32>()?;

    let mut mse = 0.0f32;
    let mut max_error = 0.0f32;
    let mut sum_orig = 0.0f32;

    for (o, q) in orig_flat.iter().zip(quant_flat.iter()) {
        let err = (o - q).abs();
        mse += (o - q).powi(2);
        max_error = max_error.max(err);
        sum_orig += o.abs();
    }

    let n = orig_flat.len() as f32;
    mse /= n;
    let rmse = mse.sqrt();
    let mean_orig = sum_orig / n;
    let relative_error = rmse / mean_orig.max(1e-10);

    Ok((rmse, max_error, relative_error * 100.0))
}

fn quantize_model(
    in_file: PathBuf,
    out_file: PathBuf,
    config: HqqConfig,
) -> Result<()> {
    println!("HQQ Quantization for Parakeet Model");
    println!("====================================");
    println!("Input:  {:?}", in_file);
    println!("Output: {:?}", out_file);
    println!("Config: {:?}\n", config);

    // Load safetensors on CPU
    println!("Loading model from safetensors...");
    let tensors = candle_core::safetensors::load(&in_file, &Device::Cpu)?;
    println!("Loaded {} tensors\n", tensors.len());

    println!("Quantizing tensors with HQQ...");
    let mut quantized_tensors: HashMap<String, Tensor> = HashMap::new();
    let mut quantized_count = 0;
    let mut skipped_count = 0;
    let mut total_rmse = 0.0f32;
    let mut total_max_error = 0.0f32;

    for (name, tensor) in tensors.iter() {
        // Skip scalar tensors
        if tensor.rank() == 0 {
            println!("  - Excluding {} [] (scalar)", name);
            continue;
        }

        let should_quantize = should_quantize_tensor(name, tensor, config.group_size);

        if should_quantize {
            quantized_count += 1;
            let shape = tensor.shape();
            print!("  ✓ Quantizing {} {:?} with HQQ ... ", name, shape.dims());
            std::io::stdout().flush().ok();

            // Quantize with HQQ
            let hqq = HqqTensor::quantize(tensor, config.clone())?;

            // Dequantize to get reconstruction
            let reconstructed = hqq.dequantize(&Device::Cpu)?;

            // Compute error metrics
            let (rmse, max_err, rel_err) = compute_error_metrics(tensor, &reconstructed)?;
            total_rmse += rmse;
            total_max_error = total_max_error.max(max_err);

            println!(
                "RMSE={:.6}, Max={:.6}, Rel={:.2}%",
                rmse, max_err, rel_err
            );

            // Store the dequantized tensor (for compatibility with safetensors format)
            // In production, you'd want a custom format that stores the HQQ parameters
            quantized_tensors.insert(name.clone(), reconstructed);
        } else {
            skipped_count += 1;
            let shape = tensor.shape();
            println!("  - Skipping {} {:?} (keeping as F32)", name, shape.dims());
            quantized_tensors.insert(name.clone(), tensor.clone());
        }
    }

    let avg_rmse = if quantized_count > 0 {
        total_rmse / quantized_count as f32
    } else {
        0.0
    };

    println!(
        "\nQuantization summary: {} quantized, {} kept as F32",
        quantized_count, skipped_count
    );
    println!("Average RMSE: {:.6}", avg_rmse);
    println!("Max error across all layers: {:.6}\n", total_max_error);

    // Save to safetensors
    println!("Saving quantized model...");
    candle_core::safetensors::save(&quantized_tensors, &out_file)?;

    // Report file size
    let in_size = std::fs::metadata(&in_file)?.len();
    let out_size = std::fs::metadata(&out_file)?.len();
    let in_mb = in_size as f64 / (1024.0 * 1024.0);
    let out_mb = out_size as f64 / (1024.0 * 1024.0);
    let compression = in_size as f64 / out_size as f64;

    println!("\n✓ Quantization complete!");
    println!("Input size:  {:.2} MB", in_mb);
    println!("Output size: {:.2} MB", out_mb);
    println!("Compression: {:.2}x", compression);
    println!("\nNote: This saves dequantized weights for compatibility.");
    println!("For production, use a custom format to store HQQ parameters directly.");

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <input.safetensors> <output.safetensors> [OPTIONS]", args[0]);
        eprintln!("\nOptions:");
        eprintln!("  --nbits N           Number of bits (2, 3, 4, 8) [default: 4]");
        eprintln!("  --group-size N      Group size for quantization [default: 128]");
        eprintln!("  --symmetric         Use symmetric quantization (no zero-point)");
        eprintln!("  --optimize-iters N  Number of optimization iterations [default: 20]");
        eprintln!("\nExamples:");
        eprintln!("  {} model.safetensors model_hqq4.safetensors --nbits 4", args[0]);
        eprintln!("  {} model.safetensors model_hqq3.safetensors --nbits 3 --group-size 64", args[0]);
        return Err(anyhow!("Invalid arguments"));
    }

    let in_file: PathBuf = args[1].clone().into();
    let out_file: PathBuf = args[2].clone().into();

    // Parse optional arguments
    let mut config = HqqConfig::default();

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--nbits" if i + 1 < args.len() => {
                config.nbits = args[i + 1].parse()?;
                if ![2, 3, 4, 8].contains(&config.nbits) {
                    return Err(anyhow!("nbits must be 2, 3, 4, or 8"));
                }
                i += 2;
            }
            "--group-size" if i + 1 < args.len() => {
                config.group_size = args[i + 1].parse()?;
                i += 2;
            }
            "--symmetric" => {
                config.symmetric = true;
                i += 1;
            }
            "--optimize-iters" if i + 1 < args.len() => {
                config.optimize_iters = args[i + 1].parse()?;
                i += 2;
            }
            _ => {
                eprintln!("Warning: Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    quantize_model(in_file, out_file, config)
}
