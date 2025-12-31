/// End-to-end test of quantized Parakeet model inference
///
/// This example:
/// 1. Loads quantized weights (Q8_0 or Q4_0)
/// 2. Builds a complete Parakeet CTC model with mixed precision
/// 3. Runs inference on a test audio file
/// 4. Compares transcription accuracy with FP32 baseline
///
/// Usage:
///   cargo run --example test_quantized_inference --release -- --quant q8_0 --audio dots.wav
///   cargo run --example test_quantized_inference --release -- --quant q4_0 --audio dots.wav
///   cargo run --example test_quantized_inference --release -- --quant fp32 --audio dots.wav

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use parakeet::{get_device, load_wav_as_features};
use std::path::Path;

// NOTE: This example uses old code from src/old/
#[path = "../src/old/parakeet_ctc.rs"]
mod parakeet_ctc;
#[path = "../src/old/quantized_loader.rs"]
mod quantized_loader;
use parakeet_ctc::{ParakeetConfig, ParakeetCTC};
use quantized_loader::QuantizedLoader;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut quant_type = "q8_0";
    let mut audio_path = "dots.wav";

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--quant" if i + 1 < args.len() => {
                quant_type = &args[i + 1];
                i += 2;
            }
            "--audio" if i + 1 < args.len() => {
                audio_path = &args[i + 1];
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    println!("Testing Quantized Parakeet Model");
    println!("============================================");
    println!("Quantization: {}", quant_type.to_uppercase());
    println!("Audio file: {}\n", audio_path);

    let device = get_device()?;

    // Load config
    println!("Loading model configuration...");
    let config = ParakeetConfig::from_file("hf_parakeet/config.json")?;
    println!("  Vocab size: {}", config.vocab_size);
    println!("  Model dim: {}", config.d_model);
    println!("  Layers: {}", config.num_layers);

    // Load model weights (quantized or FP32)
    println!("\nLoading model weights...");
    let model = if quant_type == "fp32" {
        // Load standard FP32 model
        println!("  Format: FP32 (baseline)");
        let weights_path = Path::new("hf_parakeet/model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
        };

        // Load tokenizer
        let tokenizer_path = Path::new("hf_parakeet/tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("tokenizer load error: {e}"))?;

        ParakeetFastConformerCtc::new_with_tokenizer(config, vb, tokenizer)?
    } else {
        // Load quantized model
        println!("  Format: {} (quantized)", quant_type.to_uppercase());
        let npz_path = format!("hf_parakeet/model_{}.npz", quant_type);
        let qloader = QuantizedLoader::from_npz(&npz_path, device.clone())?;

        println!("  Loaded {} weight tensors", qloader.weight_names().len());

        // Create VarBuilder from quantized loader
        // TODO: This requires modifying ParakeetFastConformerCtc to accept
        // a custom weight loader instead of VarBuilder

        return Err(anyhow!(
            "Full quantized model construction not yet implemented.\n\
             Need to modify ParakeetFastConformerCtc to support QuantizedLoader.\n\
             Current test verified quantized weights load correctly - see test_quantized.rs"
        ));
    };

    // Load audio features
    println!("\nLoading audio file...");
    let feats = load_wav_as_features(audio_path, config.feat_in, &device)?;
    let (_, n_frames, _) = feats.dims3()?;
    println!("  Audio frames: {}", n_frames);

    // Run inference
    println!("\nRunning inference...");
    let logits = model.forward(&feats, false)?;
    let (_, t, v) = logits.dims3()?;
    println!("  Output shape: [{}, {}]", t, v);

    // Decode transcript
    println!("\nDecoding transcript...");
    let transcripts = model.greedy_decode(&logits)?;

    println!("\n============================================");
    println!("Transcription Results:");
    println!("============================================");
    for (i, text) in transcripts.iter().enumerate() {
        println!("  [{}]: {}", i, text);
    }

    println!("\n✓ Inference completed successfully");

    Ok(())
}
