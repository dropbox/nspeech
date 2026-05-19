/// Quantize Silero VAD model to GGUF format (uncompressed for mmap).
///
/// Usage:
///   cargo run -p quantize-vad --release -- \
///     assets/vad16.safetensors assets/vad16_q8_0.gguf

use anyhow::Result;
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use std::io::Write;
use std::path::PathBuf;

/// Determine if a VAD tensor should be quantized
fn should_quantize_tensor(name: &str, tensor: &Tensor) -> bool {
    // Don't quantize:
    // - Biases (1D)
    // - STFT basis (sensitive signal processing)
    // - Small tensors

    if name.contains("bias") {
        return false;
    }

    if name.contains("stft.forward_basis_buffer") {
        // STFT basis is sensitive to quantization, keep FP32
        return false;
    }

    if name.contains("head.weight") {
        // Head conv is very small [1,128,1], keep FP32
        return false;
    }

    // Don't quantize 1D tensors
    if tensor.rank() == 1 {
        return false;
    }

    // Don't quantize very small tensors (< 1000 elements)
    if tensor.elem_count() < 1000 {
        return false;
    }

    // Quantize weight matrices (conv weights, RNN weights)
    // Supports both 2D (RNN) and 3D (Conv1d) tensors
    name.contains("weight")
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input.safetensors> <output.gguf>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  {} assets/vad16.safetensors assets/vad16_q8_0.gguf", args[0]);
        std::process::exit(1);
    }

    let in_file = PathBuf::from(&args[1]);
    let out_file = PathBuf::from(&args[2]);

    println!("Quantizing VAD Model to GGUF Q8_0 (uncompressed for mmap)");
    println!("=========================================================");
    println!("Input:  {:?}", in_file);
    println!("Output: {:?}\n", out_file);

    quantize_vad(&in_file, &out_file)?;

    Ok(())
}

fn quantize_vad(in_file: &PathBuf, out_file: &PathBuf) -> Result<()> {
    // Load safetensors on CPU
    println!("Loading VAD model from safetensors...");
    let tensors = candle_core::safetensors::load(in_file, &Device::Cpu)?;
    println!("Loaded {} tensors\n", tensors.len());

    let quant_format = GgmlDType::Q8_0;

    let mut quantized_count = 0;
    let mut fp32_count = 0;
    let mut excluded_count = 0;
    let mut total_original_size = 0u64;

    println!("Quantizing tensors:");
    println!("-----------------");

    // Quantize or keep as FP32
    let qtensors: Vec<(String, QTensor)> = tensors
        .into_iter()
        .filter_map(|(name, tensor)| {
            let original_bytes = tensor.elem_count() * tensor.dtype().size_in_bytes();
            total_original_size += original_bytes as u64;

            // Skip scalar tensors entirely
            if tensor.rank() == 0 {
                excluded_count += 1;
                println!("  - Excluding {} [] (scalar, not needed)", name);
                return None;
            }

            let should_quantize = should_quantize_tensor(&name, &tensor);

            if should_quantize {
                print!("  {} [{:?}] -> Q8_0... ", name, tensor.shape());
                std::io::stdout().flush().ok();

                // Q8_0 quantization requires 2D tensors; for 3D conv weights, flatten to 2D
                let tensor_to_quantize = if tensor.rank() == 3 {
                    let shape = tensor.shape().clone();
                    let dims = shape.dims();
                    // Flatten [out, in, k] -> [out, in*k]
                    match tensor.reshape((dims[0], dims[1] * dims[2])) {
                        Ok(flattened) => flattened,
                        Err(e) => {
                            println!("FAILED to reshape: {}", e);
                            return None;
                        }
                    }
                } else {
                    tensor.clone()
                };

                match QTensor::quantize(&tensor_to_quantize, quant_format) {
                    Ok(qtensor) => {
                        // Note: For 3D conv weights, we keep the flattened shape in GGUF
                        // The loader will need to reshape back to 3D when loading
                        quantized_count += 1;
                        println!("✓");
                        Some(Ok((name, qtensor)))
                    }
                    Err(e) => {
                        println!("FAILED: {}", e);
                        // Fall back to FP32
                        print!("    Falling back to FP32... ");
                        let qtensor = QTensor::quantize(&tensor, GgmlDType::F32).ok()?;
                        fp32_count += 1;
                        println!("✓");
                        Some(Ok((name, qtensor)))
                    }
                }
            } else {
                print!("  {} [{:?}] -> FP32... ", name, tensor.shape());
                std::io::stdout().flush().ok();

                let qtensor = QTensor::quantize(&tensor, GgmlDType::F32).ok()?;
                fp32_count += 1;
                println!("✓");
                Some(Ok((name, qtensor)))
            }
        })
        .collect::<Result<Vec<_>>>()?;

    println!("\nTensor Summary:");
    println!("  Quantized (Q8_0): {}", quantized_count);
    println!("  Kept FP32: {}", fp32_count);
    println!("  Excluded (scalars): {}", excluded_count);
    println!("  Total: {}", qtensors.len());

    // Write GGUF directly to output file (uncompressed for mmap)
    println!("\nWriting GGUF to output file...");
    let mut out = std::fs::File::create(out_file)?;

    let qtensor_refs: Vec<(&str, &QTensor)> = qtensors
        .iter()
        .map(|(name, qtensor)| (name.as_str(), qtensor))
        .collect();

    // Write GGUF (no metadata for VAD)
    let metadata: Vec<(&str, &gguf_file::Value)> = vec![];
    gguf_file::write(&mut out, &metadata, &qtensor_refs)?;
    out.flush()?;

    // Get final size
    let gguf_size = std::fs::metadata(out_file)?.len();

    // Statistics
    println!("\n✓ Quantization complete!");
    println!("\nSize Comparison:");
    println!("  Original (safetensors): {:.2} MB", total_original_size as f64 / 1_000_000.0);
    println!("  GGUF Q8_0 (uncompressed): {:.2} MB", gguf_size as f64 / 1_000_000.0);
    println!("  Size reduction: {:.2}x", total_original_size as f64 / gguf_size as f64);

    Ok(())
}
