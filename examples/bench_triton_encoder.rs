/// Benchmark Triton-compiled encoder vs Candle's built-in encoder.
///
/// Loads the Moonshine model twice: once with the standard Candle encoder and once
/// with the Triton encoder. Runs encoder forward on the same audio and compares
/// performance and output correctness.
///
/// Usage:
///   cargo run --example bench_triton_encoder --release --features triton-metal -- dots.wav
///   cargo run --example bench_triton_encoder --release --features triton-metal -- dots.wav assets

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use speech::parakeet::get_device;
use std::path::PathBuf;
use std::time::Instant;

fn load_wav_samples(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        anyhow::bail!("expected mono wav, got {} channels", spec.channels);
    }
    if spec.sample_rate != 16000 {
        anyhow::bail!("expected 16kHz audio, got {} Hz", spec.sample_rate);
    }
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()?,
        _ => anyhow::bail!(
            "unsupported WAV format: {:?} {}bit",
            spec.sample_format,
            spec.bits_per_sample
        ),
    };
    Ok(samples)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let wav_path = args.get(1).map(|s| s.as_str()).unwrap_or("dots.wav");
    let model_dir = args.get(2).map(|s| s.as_str()).unwrap_or("assets");

    println!("=== Triton Encoder Benchmark ===\n");

    // Load audio
    let samples = load_wav_samples(wav_path)?;
    let duration_sec = samples.len() as f64 / 16000.0;
    println!("Audio: {} ({:.2}s, {} samples)\n", wav_path, duration_sec, samples.len());

    let device = get_device()?;
    println!("Device: {:?}\n", device);

    // ── Load model config and frontend ──
    use speech::moonshine::config::MoonshineConfig;

    let cfg_bytes = std::fs::read(PathBuf::from(model_dir).join("moonshine-config.json"))?;
    let cfg: MoonshineConfig = serde_json::from_slice(&cfg_bytes)?;
    println!(
        "Config: encoder_dim={}, depth={}, heads={}\n",
        cfg.encoder_dim, cfg.encoder_num_layers, cfg.encoder_num_heads,
    );

    // Memory-map GGUF and create var builder
    let gguf_path = PathBuf::from(model_dir).join("moonshine_q8_0.gguf");
    let gguf_bytes = unsafe { memmap2::Mmap::map(&std::fs::File::open(&gguf_path)?)? };
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
        &gguf_bytes, &device,
    )?;

    // Build frontend (shared between both encoders)
    use speech::moonshine::frontend::MoonshineFrontend;
    let frontend = MoonshineFrontend::new(&cfg, vb.pp("model.encoder.embedder"))?;

    // Prepare audio tensor and run frontend
    let frame_len = cfg.frame_len;
    let pad_len = (frame_len - samples.len() % frame_len) % frame_len;
    let mut padded = samples.clone();
    padded.extend(std::iter::repeat(0.0f32).take(pad_len));
    let audio = Tensor::from_vec(padded, (1, samples.len() + pad_len), &device)?;

    let features = frontend.forward(&audio)?;
    let (_, seq_len, enc_dim) = features.dims3()?;
    println!("Frontend output: [1, {seq_len}, {enc_dim}]\n");

    // ── Candle encoder ──
    let skip_candle = std::env::var_os("TRITON_ONLY").is_some();
    let ref_flat;
    if skip_candle {
        println!("--- Candle Encoder: SKIPPED (TRITON_ONLY=1) ---");
        ref_flat = Vec::new();
    } else {
    println!("--- Candle Encoder (standard) ---");
    {
        use speech::moonshine::encoder::MoonshineEncoder;
        let t0 = Instant::now();
        let encoder = MoonshineEncoder::new(&cfg, vb.pp("model.encoder"))?;
        println!("  Load: {:.0}ms", t0.elapsed().as_millis());

        // Warmup
        let _ = encoder.forward(&features)?;
        if let Device::Metal(md) = &device {
            md.wait_until_completed()?;
        }

        // Benchmark
        let n_iters = 5;
        let t1 = Instant::now();
        for _ in 0..n_iters {
            let _ = encoder.forward(&features)?;
        }
        if let Device::Metal(md) = &device {
            md.wait_until_completed()?;
        }
        let elapsed = t1.elapsed().as_secs_f64() / n_iters as f64;
        println!("  Encoder: {:.1}ms avg ({n_iters} iters)", elapsed * 1000.0);
        println!("  {:.2}x realtime", elapsed / duration_sec);

        // Get reference output for comparison
        let ref_output = encoder.forward(&features)?;
        if let Device::Metal(md) = &device {
            md.wait_until_completed()?;
        }
        ref_flat = ref_output.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        println!("  Output: {} elements, first 5: {:?}", ref_flat.len(), &ref_flat[..5.min(ref_flat.len())]);
    }
    } // skip_candle
    // Drop Candle encoder before Triton encoder to free GPU memory

    // ── Triton encoder ──
    #[cfg(feature = "triton-metal")]
    {
        println!("\n--- Triton Encoder ---");
        use speech::moonshine::triton_encoder::TritonEncoder;

        let t0 = Instant::now();
        let triton_encoder = TritonEncoder::new(&cfg, vb.pp("model.encoder"), &device)?;
        println!("  Load: {:.0}ms", t0.elapsed().as_millis());

        // Warmup
        let _ = triton_encoder.forward(&features)?;
        if let Device::Metal(md) = &device {
            md.wait_until_completed()?;
        }

        // Benchmark
        let n_iters = 5;
        let t1 = Instant::now();
        for _ in 0..n_iters {
            let _ = triton_encoder.forward(&features)?;
        }
        if let Device::Metal(md) = &device {
            md.wait_until_completed()?;
        }
        let elapsed = t1.elapsed().as_secs_f64() / n_iters as f64;
        println!("  Encoder: {:.1}ms avg ({n_iters} iters)", elapsed * 1000.0);
        println!("  {:.2}x realtime", elapsed / duration_sec);

        // Compare output against saved Candle reference
        let triton_output = triton_encoder.forward(&features)?;
        if let Device::Metal(md) = &device {
            md.wait_until_completed()?;
        }
        let tri_flat = triton_output.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        println!("  Output: {} elements, first 5: {:?}", tri_flat.len(), &tri_flat[..5.min(tri_flat.len())]);

        if !ref_flat.is_empty() {
            let max_err = ref_flat.iter().zip(tri_flat.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mean_err = ref_flat.iter().zip(tri_flat.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>() / ref_flat.len() as f32;

            println!("\n--- Comparison ---");
            println!("  max_err:  {:.6}", max_err);
            println!("  mean_err: {:.6}", mean_err);
        }
    }

    #[cfg(not(feature = "triton-metal"))]
    println!("\nTriton encoder not available. Build with --features triton-metal");

    Ok(())
}
