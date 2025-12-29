/// Test loading GGUF quantized weights
///
/// Usage:
///   cargo run --example test_gguf_load --release -- hf_parakeet/model_q8_0.gguf
///   cargo run --example test_gguf_load --release -- hf_parakeet/model_q4k.gguf

use anyhow::Result;
use candle_core::quantized::gguf_file;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        return Ok(());
    }

    let gguf_path = PathBuf::from(&args[1]);
    println!("Loading GGUF file: {:?}\n", gguf_path);

    // Load GGUF file
    let mut file = std::fs::File::open(&gguf_path)?;
    let gguf_content = gguf_file::Content::read(&mut file)?;

    println!("GGUF file loaded successfully!");
    println!("==============================\n");

    // Print metadata
    println!("Metadata:");
    for (key, value) in gguf_content.metadata.iter() {
        println!("  {}: {:?}", key, value);
    }
    println!();

    // Print tensor information
    println!("Tensors: {}", gguf_content.tensor_infos.len());
    println!("\nSample tensors:");

    let sample_keys = [
        "encoder.layers.0.self_attn.q_proj.weight",
        "encoder.layers.0.feed_forward1.linear1.weight",
        "ctc_head.weight",
    ];

    for key in &sample_keys {
        if let Some(tensor_info) = gguf_content.tensor_infos.get(*key) {
            println!("  {}", key);
            println!("    Type: {:?}", tensor_info.ggml_dtype);
            println!("    Shape: {:?}", tensor_info.shape.dims());
        }
    }

    // Check file size
    let metadata = std::fs::metadata(&gguf_path)?;
    let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
    println!("\nFile size: {:.2} MB", size_mb);

    println!("\n✓ GGUF file loaded and verified successfully!");

    Ok(())
}
