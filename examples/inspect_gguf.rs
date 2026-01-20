/// Inspect GGUF file to see what tensors it contains
///
/// Usage:
///   cargo run --example inspect_gguf --release -- assets/vad16_q8_0.gguf.zst

use anyhow::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <gguf.zst>", args[0]);
        std::process::exit(1);
    }

    let in_file = PathBuf::from(&args[1]);

    println!("Inspecting GGUF file: {:?}\n", in_file);

    // Decompress zstd
    let compressed = std::fs::read(&in_file)?;
    let decompressed = zstd::bulk::decompress(&compressed, 10_000_000)?;

    println!("Decompressed size: {:.2} MB\n", decompressed.len() as f64 / 1_000_000.0);

    // Parse GGUF
    use candle_core::quantized::gguf_file;
    let gguf = gguf_file::Content::read(&mut std::io::Cursor::new(&decompressed))?;

    println!("Tensors in GGUF file:");
    println!("-------------------");
    for (name, info) in gguf.tensor_infos.iter() {
        println!("  {} - {:?} {:?}", name, info.ggml_dtype, info.shape);
    }

    println!("\nTotal tensors: {}", gguf.tensor_infos.len());

    Ok(())
}
