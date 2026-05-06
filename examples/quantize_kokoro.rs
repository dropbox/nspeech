//! Quantize Kokoro TTS model to GGUF format.
//!
//! Usage:
//!   cargo run --example quantize_kokoro --release -- [--format q8_0|q4k]

use anyhow::{anyhow, Result};
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use std::io::Write;

fn should_quantize(name: &str, tensor: &Tensor, block_size: usize) -> bool {
    if name.contains("bias")
        || name.contains("gamma")
        || name.contains("beta")
        || name.contains("alpha")
        || name.contains("embedding")
    {
        return false;
    }

    let elem = tensor.elem_count();
    if elem < 256 {
        return false;
    }

    // For 2D: last dim must be divisible by block_size
    // For 3D conv: flatten last two dims, check divisibility
    match tensor.rank() {
        2 => tensor.dim(1).unwrap_or(0) % block_size == 0,
        3 => {
            let in_ch = tensor.dim(1).unwrap_or(0);
            let kernel = tensor.dim(2).unwrap_or(0);
            (in_ch * kernel) % block_size == 0
        }
        _ => false,
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let format_str = args.iter().position(|a| a == "--format")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("q8_0");

    let quant_format = match format_str {
        "q8_0" => GgmlDType::Q8_0,
        "q4k" => GgmlDType::Q4K,
        "q4_0" => GgmlDType::Q4_0,
        "f16" => GgmlDType::F16,
        _ => return Err(anyhow!("Unsupported format: {}", format_str)),
    };

    let in_path = "hf_kokoro/kokoro-v1_0.safetensors";
    let out_path = format!("hf_kokoro/kokoro_{}.gguf", format_str);

    eprintln!("Quantizing Kokoro to {:?}", quant_format);
    eprintln!("  Input:  {}", in_path);
    eprintln!("  Output: {}", out_path);

    let tensors = candle_core::safetensors::load(in_path, &Device::Cpu)?;
    eprintln!("Loaded {} tensors", tensors.len());

    let block_size = quant_format.block_size();
    let mut quantized = 0usize;
    let mut kept_f32 = 0usize;

    let qtensors: Vec<(String, QTensor)> = tensors
        .into_iter()
        .filter_map(|(name, tensor)| {
            if tensor.rank() == 0 {
                return None;
            }

            let quantize = should_quantize(&name, &tensor, block_size);

            if quantize {
                // Reshape 3D to 2D for quantization
                let t_for_quant = if tensor.rank() == 3 {
                    let (d0, d1, d2) = tensor.dims3().ok()?;
                    tensor.reshape((d0, d1 * d2)).ok()?
                } else {
                    tensor
                };
                let qt = QTensor::quantize(&t_for_quant, quant_format).ok()?;
                quantized += 1;
                Some(Ok((name, qt)))
            } else {
                let qt = QTensor::quantize(&tensor, GgmlDType::F32).ok()?;
                kept_f32 += 1;
                Some(Ok((name, qt)))
            }
        })
        .collect::<Result<Vec<_>>>()?;

    eprintln!("Quantized: {}, kept F32: {}", quantized, kept_f32);

    let mut out = std::fs::File::create(&out_path)?;
    let qtensor_refs: Vec<(&str, &QTensor)> = qtensors.iter()
        .map(|(n, q)| (n.as_str(), q))
        .collect();
    gguf_file::write(&mut out, &[], &qtensor_refs)?;
    out.flush()?;

    let size = std::fs::metadata(&out_path)?.len();
    eprintln!("Output: {:.1} MB (was {:.1} MB)", size as f64 / 1e6, 327.0);

    Ok(())
}
