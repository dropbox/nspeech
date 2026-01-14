use anyhow::Result;
use candle_core::quantized::gguf_file;
use speech::parakeet::STREAMING_TDT_MODEL_Q8_0_GGUF;
use std::io::{Cursor, Error, ErrorKind};

fn main() -> Result<()> {
    let assets = std::path::PathBuf::from("assets");
    let gguf_bytes = STREAMING_TDT_MODEL_Q8_0_GGUF.bytes(&assets)
        .map_err(|_| Error::new(ErrorKind::Other, "Failed to load GGUF bytes"))?;

    let gguf_file = gguf_file::Content::read(&mut Cursor::new(gguf_bytes))?;

    println!("Joint network tensors:");
    for (name, info) in gguf_file.tensor_infos.iter() {
        if name.contains("joint") {
            println!("  {} -> {:?}", name, info.shape);
        }
    }

    println!("\nPredictor tensors:");
    for (name, info) in gguf_file.tensor_infos.iter() {
        if name.contains("predictor") || name.contains("decoder.prediction") {
            println!("  {} -> {:?}", name, info.shape);
        }
    }

    Ok(())
}
