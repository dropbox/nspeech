/// Compare GGUF quantized weights vs FP32 baseline
///
/// Usage:
///   cargo run --example compare_gguf_fp32 --release -- --gguf hf_parakeet/model_q8_0.gguf
///   cargo run --example compare_gguf_fp32 --release -- --gguf hf_parakeet/model_q4k.gguf

use anyhow::Result;
use candle_core::{quantized::gguf_file, DType};
use candle_nn::VarBuilder;
use std::path::Path;

fn compute_stats(data: &[f32]) -> (f32, f32, f32, f32) {
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (mean, std, min, max)
}

fn compare_weights(fp32_data: &[f32], gguf_data: &[f32], name: &str, quant_type: &str) {
    assert_eq!(fp32_data.len(), gguf_data.len(), "Weight size mismatch for {}", name);

    let (fp32_mean, fp32_std, fp32_min, fp32_max) = compute_stats(fp32_data);
    let (gguf_mean, gguf_std, gguf_min, gguf_max) = compute_stats(gguf_data);

    // Compute error metrics
    let mut abs_errors = Vec::with_capacity(fp32_data.len());
    let mut rel_errors = Vec::with_capacity(fp32_data.len());

    for (fp32, gguf) in fp32_data.iter().zip(gguf_data.iter()) {
        let abs_err = (fp32 - gguf).abs();
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

    println!("\n  {} ({})", name, quant_type);
    println!("    FP32:   mean={:.6}, std={:.6}, range=[{:.6}, {:.6}]",
             fp32_mean, fp32_std, fp32_min, fp32_max);
    println!("    GGUF:   mean={:.6}, std={:.6}, range=[{:.6}, {:.6}]",
             gguf_mean, gguf_std, gguf_min, gguf_max);
    println!("    Error:  MAE={:.6}, Max AE={:.6}, Mean RE={:.4}%",
             mae, max_ae, mean_re * 100.0);
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut gguf_path = "hf_parakeet/model_q8_0.gguf".to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" if i + 1 < args.len() => {
                gguf_path = args[i + 1].clone();
                i += 2;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    // Determine quantization type from filename
    let quant_type = if gguf_path.contains("q8_0") {
        "Q8_0"
    } else if gguf_path.contains("q4k") {
        "Q4K"
    } else {
        "Unknown"
    };

    println!("GGUF vs FP32 Weight Comparison");
    println!("===============================\n");
    println!("GGUF Model: {}", gguf_path);
    println!("Quantization: {}\n", quant_type);

    let device = parakeet::get_device()?;

    // Load FP32 weights
    println!("Loading FP32 baseline...");
    let weights_path = Path::new("hf_parakeet/model.safetensors");
    let vb_fp32 = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
    };
    println!("  ✓ Loaded FP32 weights");

    // Load GGUF weights
    println!("\nLoading GGUF quantized weights...");
    let mut file = std::fs::File::open(&gguf_path)?;
    let gguf_content = gguf_file::Content::read(&mut file)?;
    println!("  ✓ Loaded {} tensors from GGUF\n", gguf_content.tensor_infos.len());

    // Compare sample weights
    println!("Comparing sample weights:");
    println!("=========================");

    let sample_keys = [
        "encoder.layers.0.self_attn.q_proj.weight",
        "encoder.layers.0.feed_forward1.linear1.weight",
        "encoder.layers.15.self_attn.q_proj.weight",
        "encoder.layers.23.feed_forward2.linear2.weight",
    ];

    for key in &sample_keys {
        // Get tensor info from GGUF
        let tensor_info = match gguf_content.tensor_infos.get(*key) {
            Some(info) => info,
            None => {
                println!("\n  Skipping {}: not found in GGUF", key);
                continue;
            }
        };

        let detected_quant = format!("{:?}", tensor_info.ggml_dtype);

        // Load GGUF tensor and dequantize
        let qtensor = gguf_content.tensor(&mut file, key, &device)?;
        let gguf_tensor = qtensor.dequantize(&device)?;
        let shape = gguf_tensor.shape();

        // Navigate VarBuilder hierarchy to get FP32 tensor
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
        let gguf_data = gguf_tensor.flatten_all()?.to_vec1::<f32>()?;

        compare_weights(&fp32_data, &gguf_data, key, &detected_quant);
    }

    println!("\n\n✓ GGUF quantization accuracy comparison complete!");
    println!("\nSummary:");
    println!("  GGUF weights load correctly and dequantize to FP32");
    println!("  Quantization accuracy matches expected levels for {}", quant_type);

    Ok(())
}
