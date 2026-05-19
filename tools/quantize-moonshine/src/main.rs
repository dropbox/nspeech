/// Quantize Moonshine V2 model to GGUF format (uncompressed for mmap).
///
/// Usage:
///   cargo run -p quantize-moonshine --release -- \
///     hf_moonshine/model.safetensors assets/moonshine_q8_0.gguf

use anyhow::Result;
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use std::io::Write;
use std::path::PathBuf;

/// Determine if a Moonshine tensor should be quantized
fn should_quantize_tensor(name: &str, tensor: &Tensor) -> bool {
    // Don't quantize biases (1D)
    if name.contains("bias") {
        return false;
    }

    // Don't quantize scalars (rank 0), e.g. comp.log_k
    if tensor.rank() == 0 {
        return false;
    }

    // Don't quantize 1D tensors (norms, etc.)
    if tensor.rank() == 1 {
        return false;
    }

    // Don't quantize very small tensors (< 1000 elements)
    if tensor.elem_count() < 1000 {
        return false;
    }

    true
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input.safetensors> <output.gguf>", args[0]);
        eprintln!("\nExample:");
        eprintln!(
            "  {} hf_moonshine/model.safetensors assets/moonshine_q8_0.gguf",
            args[0]
        );
        std::process::exit(1);
    }

    let in_file = PathBuf::from(&args[1]);
    let out_file = PathBuf::from(&args[2]);

    println!("Quantizing Moonshine V2 Model to GGUF Q8_0 (uncompressed for mmap)");
    println!("===================================================================");
    println!("Input:  {:?}", in_file);
    println!("Output: {:?}\n", out_file);

    quantize_moonshine(&in_file, &out_file)?;

    Ok(())
}

fn quantize_moonshine(in_file: &PathBuf, out_file: &PathBuf) -> Result<()> {
    // Load safetensors on CPU
    println!("Loading Moonshine model from safetensors...");
    let tensors = candle_core::safetensors::load(in_file, &Device::Cpu)?;
    println!("Loaded {} tensors\n", tensors.len());

    let quant_format = GgmlDType::Q8_0;

    let mut quantized_count = 0;
    let mut fp32_count = 0;
    let excluded_count = 0;
    let mut total_original_size = 0u64;

    println!("Quantizing tensors:");
    println!("-----------------");

    // Quantize or keep as FP32
    let qtensors: Vec<(String, QTensor)> = tensors
        .into_iter()
        .filter_map(|(name, tensor)| {
            let original_bytes = tensor.elem_count() * tensor.dtype().size_in_bytes();
            total_original_size += original_bytes as u64;

            // Reshape scalar tensors to [1] so GGUF can store them (e.g. comp.log_k)
            let tensor = if tensor.rank() == 0 {
                print!("  {} [] -> FP32 [1]... ", name);
                std::io::stdout().flush().ok();
                let reshaped = tensor.reshape((1,)).ok()?;
                let qtensor = QTensor::quantize(&reshaped, GgmlDType::F32).ok()?;
                fp32_count += 1;
                println!("OK");
                return Some(Ok((name, qtensor)));
            } else {
                tensor
            };

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
                        quantized_count += 1;
                        println!("OK");
                        Some(Ok((name, qtensor)))
                    }
                    Err(e) => {
                        println!("FAILED: {}", e);
                        // Fall back to FP32
                        print!("    Falling back to FP32... ");
                        let qtensor = QTensor::quantize(&tensor, GgmlDType::F32).ok()?;
                        fp32_count += 1;
                        println!("OK");
                        Some(Ok((name, qtensor)))
                    }
                }
            } else {
                print!("  {} [{:?}] -> FP32... ", name, tensor.shape());
                std::io::stdout().flush().ok();

                let qtensor = QTensor::quantize(&tensor, GgmlDType::F32).ok()?;
                fp32_count += 1;
                println!("OK");
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

    // Write GGUF (no metadata)
    let metadata: Vec<(&str, &gguf_file::Value)> = vec![];
    gguf_file::write(&mut out, &metadata, &qtensor_refs)?;
    out.flush()?;

    // Get final size
    let gguf_size = std::fs::metadata(out_file)?.len();

    // Statistics
    println!("\nQuantization complete!");
    println!("\nSize Comparison:");
    println!(
        "  Original (safetensors): {:.2} MB",
        total_original_size as f64 / 1_000_000.0
    );
    println!(
        "  GGUF Q8_0 (uncompressed): {:.2} MB",
        gguf_size as f64 / 1_000_000.0
    );
    println!(
        "  Size reduction: {:.2}x",
        total_original_size as f64 / gguf_size as f64
    );

    Ok(())
}
