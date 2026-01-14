use anyhow::Result;
use candle_core::quantized::gguf_file;
use std::io::{Cursor, Error, ErrorKind};

fn main() -> Result<()> {
    // Load the streaming TDT GGUF
    use speech::parakeet::STREAMING_TDT_MODEL_Q8_0_GGUF;
    let assets = std::path::PathBuf::from("assets");
    let gguf_bytes = STREAMING_TDT_MODEL_Q8_0_GGUF.bytes(&assets)
        .map_err(|_| Error::new(ErrorKind::Other, "Failed to load GGUF bytes"))?;

    let gguf_file = gguf_file::Content::read(&mut Cursor::new(gguf_bytes))?;

    println!("Tensors in GGUF file:");
    println!("=====================\n");

    // Look for conv tensors in layer 0
    println!("Encoder layer 0 conv tensors:");
    for (name, info) in gguf_file.tensor_infos.iter() {
        if name.contains("encoder.layers.0.conv") {
            println!("  {} -> shape: {:?}", name, info.shape);
        }
    }

    // Look for subsampling related tensors
    let mut subsampling_tensors = Vec::new();
    for (name, info) in gguf_file.tensor_infos.iter() {
        if name.contains("subsampling") || name.contains("pre_encode") {
            subsampling_tensors.push((name, info));
        }
    }

    println!("\nSubsampling tensors:");
    for (name, info) in &subsampling_tensors {
        println!("  {} -> shape: {:?}", name, info.shape);
    }

    // Check the specific tensor that's causing the error
    if let Some(info) = gguf_file.tensor_infos.get("encoder.subsampling.linear.weight") {
        println!("\nFound encoder.subsampling.linear.weight:");
        println!("  shape: {:?}", info.shape);
        println!("  Expected: [1024, 4096], Got: {:?}", info.shape);

        // Calculate what feat_in would need to be
        let dims = info.shape.dims();
        if dims.len() == 2 {
            let output_dim = dims[0];
            let input_dim = dims[1];
            println!("\n  output_dim (d_model): {}", output_dim);
            println!("  input_dim (flatten_dim): {}", input_dim);

            // Assuming subsampling_channels = 256, subsampling_factor = 8
            let channels = 256;
            let factor = 8;
            let required_feat_in = (input_dim / channels) * factor;
            println!("  If channels={}, factor={}, then feat_in should be: {}",
                     channels, factor, required_feat_in);
        }
    } else {
        println!("\nencoder.subsampling.linear.weight not found!");
        println!("Checking alternative names...");
        for (name, info) in gguf_file.tensor_infos.iter() {
            if name.contains("linear") && (name.contains("subsampling") || name.contains("pre_encode") || name.contains("out")) {
                println!("  {} -> shape: {:?}", name, info.shape);
            }
        }
    }

    Ok(())
}
