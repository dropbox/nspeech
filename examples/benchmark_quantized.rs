/// Benchmark quantized vs FP32 inference performance
///
/// This compares:
/// 1. Quantized Q8_0 inference (current default)
/// 2. FP32 safetensors inference
///
/// Usage:
///   cargo run --example benchmark_quantized --release -- dots.wav
///   PARAKEET_DEVICE=cpu cargo run --example benchmark_quantized --release -- dots.wav

use anyhow::Result;
use speech::parakeet;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        return Ok(());
    }

    let audio_path = &args[1];
    println!("Benchmarking Parakeet Performance");
    println!("==================================\n");
    println!("Audio: {}\n", audio_path);

    // Get device
    let device = parakeet::get_device()?;
    println!("Device: {:?}\n", device);

    let assets = PathBuf::from("assets");

    // Check if CANDLE_DEQUANTIZE_ALL is set
    match std::env::var("CANDLE_DEQUANTIZE_ALL") {
        Ok(val) if !val.is_empty() && val != "0" => {
            println!("⚠️  WARNING: CANDLE_DEQUANTIZE_ALL={}", val);
            println!("    Quantized weights will be dequantized to FP32!");
            println!("    This defeats the purpose of quantization.\n");
        }
        _ => {
            println!("✓ CANDLE_DEQUANTIZE_ALL not set (good)\n");
        }
    }

    // Load audio
    println!("Loading audio...");
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    if spec.channels != 1 || spec.sample_rate != 16000 {
        return Err(anyhow::anyhow!("Expected mono 16kHz audio"));
    }

    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => {
            reader.samples::<i16>()
                .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
                .collect::<Result<Vec<_>, _>>()?
        }
        (hound::SampleFormat::Float, 32) => {
            reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err(anyhow::anyhow!("Unsupported audio format")),
    };

    let duration_sec = samples.len() as f32 / 16000.0;
    println!("✓ Audio loaded: {:.2}s ({} samples)\n", duration_sec, samples.len());

    // ========================================================================
    // Benchmark 1: Quantized Q8_0 (current default)
    // ========================================================================
    #[cfg(feature = "quantized")]
    {
        println!("=== Benchmark 1: Quantized Q8_0 ===");
        println!("Loading quantized model...");
        let load_start = Instant::now();
        let model = parakeet::load_parakeet_ctc_from_gguf_local(&assets, &device)?;
        let load_time = load_start.elapsed();
        println!("✓ Model loaded in {:.2}s\n", load_time.as_secs_f64());

        // Extract features
        println!("Extracting features...");
        let feat_start = Instant::now();
        let features = parakeet::extract_features_from_samples(
            &samples,
            model.cfg.feat_in,
            &device,
        )?;
        let feat_time = feat_start.elapsed();
        println!("✓ Features extracted in {:.2}ms\n", feat_time.as_secs_f64() * 1000.0);

        // Warm-up run
        println!("Warm-up inference...");
        let _ = model.forward(&features, false)?;

        // Timed inference runs
        println!("Running 5 timed inferences...");
        let mut times = Vec::new();
        for i in 1..=5 {
            let inf_start = Instant::now();
            let logits = model.forward(&features, false)?;
            let inf_time = inf_start.elapsed();
            times.push(inf_time.as_secs_f64() * 1000.0);
            println!("  Run {}: {:.2}ms", i, inf_time.as_secs_f64() * 1000.0);

            // Decode on last run
            if i == 5 {
                let dec_start = Instant::now();
                let transcripts = model.greedy_decode(&logits)?;
                let dec_time = dec_start.elapsed();
                println!("\nTranscript: \"{}\"", transcripts[0]);
                println!("Decode time: {:.2}ms", dec_time.as_secs_f64() * 1000.0);
            }
        }

        let avg_time = times.iter().sum::<f64>() / times.len() as f64;
        let min_time = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_time = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("\nQuantized Q8_0 Results:");
        println!("  Average: {:.2}ms ({:.1}x realtime)", avg_time, duration_sec as f64 * 1000.0 / avg_time);
        println!("  Min: {:.2}ms", min_time);
        println!("  Max: {:.2}ms", max_time);
        println!("  Throughput: {:.2}s audio / second\n", duration_sec as f64 * 1000.0 / avg_time);
    }

    // ========================================================================
    // Benchmark 2: FP32 safetensors (if available)
    // ========================================================================
    #[cfg(not(feature = "quantized"))]
    {
        println!("=== Benchmark 2: FP32 Safetensors ===");
        println!("Loading FP32 model...");
        let load_start = Instant::now();
        let model = parakeet::load_parakeet_ctc_from_local(&assets, &device)?;
        let load_time = load_start.elapsed();
        println!("✓ Model loaded in {:.2}s\n", load_time.as_secs_f64());

        // Extract features
        println!("Extracting features...");
        let feat_start = Instant::now();
        let features = parakeet::extract_features_from_samples(
            &samples,
            model.cfg.feat_in,
            &device,
        )?;
        let feat_time = feat_start.elapsed();
        println!("✓ Features extracted in {:.2}ms\n", feat_time.as_secs_f64() * 1000.0);

        // Warm-up run
        println!("Warm-up inference...");
        let _ = model.forward(&features, false)?;

        // Timed inference runs
        println!("Running 5 timed inferences...");
        let mut times = Vec::new();
        for i in 1..=5 {
            let inf_start = Instant::now();
            let logits = model.forward(&features, false)?;
            let inf_time = inf_start.elapsed();
            times.push(inf_time.as_secs_f64() * 1000.0);
            println!("  Run {}: {:.2}ms", i, inf_time.as_secs_f64() * 1000.0);

            // Decode on last run
            if i == 5 {
                let dec_start = Instant::now();
                let transcripts = model.greedy_decode(&logits)?;
                let dec_time = dec_start.elapsed();
                println!("\nTranscript: \"{}\"", transcripts[0]);
                println!("Decode time: {:.2}ms", dec_time.as_secs_f64() * 1000.0);
            }
        }

        let avg_time = times.iter().sum::<f64>() / times.len() as f64;
        let min_time = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_time = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("\nFP32 Safetensors Results:");
        println!("  Average: {:.2}ms ({:.1}x realtime)", avg_time, duration_sec as f64 * 1000.0 / avg_time);
        println!("  Min: {:.2}ms", min_time);
        println!("  Max: {:.2}ms", max_time);
        println!("  Throughput: {:.2}s audio / second\n", duration_sec as f64 * 1000.0 / avg_time);
    }

    Ok(())
}
