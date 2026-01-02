/// Quantize Parakeet model to GGUF format with zstd compression
///
/// This tool quantizes a safetensors model to GGUF format and compresses with zstd.
/// Output files are automatically named with .zst extension.
///
/// Usage:
///   cargo run --example quantize_gguf --release -- \
///     assets/model.safetensors \
///     assets/model_q8_0.gguf.zst \
///     --format q8_0
///
///   cargo run --example quantize_gguf --release -- \
///     assets/model.safetensors \
///     assets/model_q4k.gguf.zst \
///     --format q4k

use anyhow::{anyhow, Result};
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;

/// Determine if a layer should be quantized
/// Quantize: Weight matrices (2D tensors that are multiples of block size)
/// Don't quantize: Biases, norms, embeddings, small tensors
fn should_quantize_tensor(name: &str, tensor: &Tensor, block_size: usize) -> bool {
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

    // Check if dimensions are compatible with block size
    let dim1 = tensor.dim(1).unwrap_or(0);
    if dim1 % block_size != 0 {
        return false;
    }

    // Quantize if it's a weight matrix
    name.contains("weight")
}

/// Parse quantization format string to GgmlDType
fn parse_quant_format(format: &str) -> Result<GgmlDType> {
    match format.to_lowercase().as_str() {
        "f32" => Ok(GgmlDType::F32),
        "f16" => Ok(GgmlDType::F16),
        "q4_0" => Ok(GgmlDType::Q4_0),
        "q4_1" => Ok(GgmlDType::Q4_1),
        "q5_0" => Ok(GgmlDType::Q5_0),
        "q5_1" => Ok(GgmlDType::Q5_1),
        "q8_0" => Ok(GgmlDType::Q8_0),
        "q8_1" => Ok(GgmlDType::Q8_1),
        "q2k" => Ok(GgmlDType::Q2K),
        "q3k" => Ok(GgmlDType::Q3K),
        "q4k" => Ok(GgmlDType::Q4K),
        "q5k" => Ok(GgmlDType::Q5K),
        "q6k" => Ok(GgmlDType::Q6K),
        "q8k" => Ok(GgmlDType::Q8K),
        _ => Err(anyhow!(
            "Unknown quantization format: {}. Supported: f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q8_1, q2k, q3k, q4k, q5k, q6k, q8k",
            format
        )),
    }
}

fn quantize_model(
    in_file: PathBuf,
    mut out_file: PathBuf,
    quant_format: GgmlDType,
) -> Result<()> {
    // Ensure output filename ends with .zst
    if out_file.extension().and_then(|s| s.to_str()) != Some("zst") {
        let mut new_name = out_file.clone();
        if let Some(name) = out_file.file_name().and_then(|s| s.to_str()) {
            new_name.set_file_name(format!("{}.zst", name));
            out_file = new_name;
        }
    }

    println!("Quantizing Parakeet Model to GGUF with zstd compression");
    println!("=========================================================");
    println!("Input:  {:?}", in_file);
    println!("Output: {:?}", out_file);
    println!("Format: {:?}\n", quant_format);

    // Load safetensors on CPU (quantization happens on CPU)
    println!("Loading model from safetensors...");
    let tensors = candle_core::safetensors::load(&in_file, &Device::Cpu)?;
    println!("Loaded {} tensors\n", tensors.len());

    let block_size = quant_format.block_size();
    println!("Block size for {:?}: {}\n", quant_format, block_size);

    // Quantize tensors
    println!("Quantizing tensors...");
    let mut quantized_count = 0;
    let mut skipped_count = 0;
    let mut excluded_count = 0;

    let qtensors: Vec<(String, QTensor)> = tensors
        .into_iter()
        .filter_map(|(name, tensor)| {
            // Skip scalar tensors entirely (cannot be quantized, not needed for inference)
            if tensor.rank() == 0 {
                excluded_count += 1;
                println!("  - Excluding {} [] (scalar, not needed for inference)", name);
                return None;
            }

            let should_quantize = should_quantize_tensor(&name, &tensor, block_size);

            if should_quantize {
                quantized_count += 1;
                let shape = tensor.shape();
                print!("  ✓ Quantizing {} {:?} ... ", name, shape.dims());
                std::io::stdout().flush().ok();

                let qtensor = QTensor::quantize(&tensor, quant_format).ok()?;
                println!("done");
                Some(Ok((name, qtensor)))
            } else {
                skipped_count += 1;
                let shape = tensor.shape();

                // Handle very small tensors
                if tensor.elem_count() < block_size {
                    println!("  - Skipping {} {:?} (too small, keeping as F32)", name, shape.dims());
                } else {
                    println!("  - Skipping {} {:?} (non-weight, keeping as F32)", name, shape.dims());
                }
                let qtensor = QTensor::quantize(&tensor, GgmlDType::F32).ok()?;
                Some(Ok((name, qtensor)))
            }
        })
        .collect::<Result<Vec<_>>>()?;

    println!(
        "\nQuantization summary: {} quantized, {} kept as F32, {} excluded (scalars)\n",
        quantized_count, skipped_count, excluded_count
    );

    // Write to GGUF in tempfile, then compress to output
    println!("Writing GGUF to tempfile...");
    let mut tmp = tempfile::tempfile()?;

    let qtensor_refs: Vec<(&str, &QTensor)> = qtensors
        .iter()
        .map(|(name, qtensor)| (name.as_str(), qtensor))
        .collect();

    // Write GGUF (no metadata for now)
    let metadata: Vec<(&str, &gguf_file::Value)> = vec![];
    gguf_file::write(&mut tmp, &metadata, &qtensor_refs)?;
    tmp.flush()?;
    tmp.seek(SeekFrom::Start(0))?;

    // Compress to output file
    println!("Compressing with zstd (level 19)...");
    let raw_out = std::fs::File::create(&out_file)?;
    let mut encoder = zstd::Encoder::new(raw_out, 19)?;
    std::io::copy(&mut tmp, &mut encoder)?;
    encoder.finish()?.sync_all()?;

    // Report file size
    let file_size = std::fs::metadata(&out_file)?.len();
    let size_mb = file_size as f64 / (1024.0 * 1024.0);
    println!("\n✓ Quantization complete!");
    println!("Output file size: {:.2} MB", size_mb);

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <input.safetensors> <output.gguf.zst> [--format FORMAT]", args[0]);
        eprintln!("\nFormats: f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q8_1, q2k, q3k, q4k, q5k, q6k, q8k");
        eprintln!("         (default: q8_0)");
        eprintln!("\nOutput files are automatically compressed with zstd level 19.");
        eprintln!("If output filename doesn't end with .zst, it will be added automatically.");
        eprintln!("\nExamples:");
        eprintln!("  {} model.safetensors model_q8_0.gguf.zst --format q8_0", args[0]);
        eprintln!("  {} model.safetensors model_q4k.gguf.zst --format q4k", args[0]);
        return Err(anyhow!("Invalid arguments"));
    }

    let in_file: PathBuf = args[1].clone().into();
    let out_file: PathBuf = args[2].clone().into();

    // Parse optional arguments
    let mut quant_format = GgmlDType::Q8_0; // Default

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--format" if i + 1 < args.len() => {
                quant_format = parse_quant_format(&args[i + 1])?;
                i += 2;
            }
            _ => {
                eprintln!("Warning: Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    quantize_model(in_file, out_file, quant_format)
}
