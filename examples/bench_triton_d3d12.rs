/// Benchmark Triton-compiled D3D12 encoder for Moonshine V2.
///
/// Usage:
///   bench_triton_d3d12.exe dots.wav [assets_dir]

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
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

    println!("=== Triton D3D12 Encoder Benchmark ===\n");

    // Load audio
    let samples = load_wav_samples(wav_path)?;
    let duration_sec = samples.len() as f64 / 16000.0;
    println!("Audio: {} ({:.2}s, {} samples)\n", wav_path, duration_sec, samples.len());

    // Load config
    use speech::moonshine::config::MoonshineConfig;
    let cfg_bytes = std::fs::read(PathBuf::from(model_dir).join("moonshine-config.json"))?;
    let cfg: MoonshineConfig = serde_json::from_slice(&cfg_bytes)?;
    println!(
        "Config: dim={}, layers={}, heads={}\n",
        cfg.encoder_dim, cfg.encoder_num_layers, cfg.encoder_num_heads,
    );

    // Use CPU for frontend (conv layers)
    let device = Device::Cpu;

    // Memory-map GGUF
    let gguf_path = PathBuf::from(model_dir).join("moonshine_q8_0.gguf");
    let gguf_bytes = unsafe { memmap2::Mmap::map(&std::fs::File::open(&gguf_path)?)? };
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
        &gguf_bytes, &device,
    )?;

    // Run frontend on CPU
    use speech::moonshine::frontend::MoonshineFrontend;
    let frontend = MoonshineFrontend::new(&cfg, vb.pp("model.encoder.embedder"))?;

    let frame_len = cfg.frame_len;
    let pad_len = (frame_len - samples.len() % frame_len) % frame_len;
    let mut padded = samples.clone();
    padded.extend(std::iter::repeat(0.0f32).take(pad_len));
    let audio = Tensor::from_vec(padded, (1, samples.len() + pad_len), &device)?;

    let t0 = Instant::now();
    let features = frontend.forward(&audio)?;
    let (_, seq_len, enc_dim) = features.dims3()?;
    println!("Frontend: [1, {seq_len}, {enc_dim}] in {:.0}ms\n", t0.elapsed().as_millis());

    // Helper to compare two flat vectors
    fn compare_vecs(name: &str, cpu: &[f32], gpu: &[f32]) {
        let n = cpu.len().min(gpu.len());
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        let mut max_diff_idx = 0;
        for i in 0..n {
            let diff = (cpu[i] - gpu[i]).abs();
            if diff > max_diff { max_diff = diff; max_diff_idx = i; }
            sum_diff += diff as f64;
        }
        println!("{name}:");
        println!("  CPU first 5: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]", cpu[0], cpu[1], cpu[2], cpu[3], cpu[4]);
        println!("  GPU first 5: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]", gpu[0], gpu[1], gpu[2], gpu[3], gpu[4]);
        println!("  Max abs diff: {:.6} at idx {max_diff_idx}", max_diff);
        println!("  Mean abs diff: {:.6}", sum_diff / n as f64);
    }

    #[cfg(feature = "triton-d3d12")]
    {
        use speech::moonshine::encoder::MoonshineEncoder;
        use speech::moonshine::gpu_encoder::GpuEncoder;
        use speech::moonshine::gpu_encoder_d3d12::D3D12EncoderBackend;
        use std::sync::Arc;

        let gpu = Arc::new(candle_d3d12_kernels::Gpu::new(0)?);
        let use_fp16_acc = std::env::var("USE_FP16_ACC").map_or(false, |v| v == "1");
        println!("D3D12 GPU context created (fp16_acc={use_fp16_acc})\n");

        // ── Per-layer comparison ──
        for n_layers in [0, 1] {
            let mut cfg_n = cfg.clone();
            cfg_n.encoder_num_layers = n_layers;

            let cpu_encoder = MoonshineEncoder::new(&cfg_n, vb.pp("model.encoder"))?;
            let cpu_out = cpu_encoder.forward(&features)?;
            let cpu_flat = cpu_out.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;

            let backend = D3D12EncoderBackend::new(&gpu, use_fp16_acc, cfg_n.encoder_dim)?;
            let d3d12_encoder = GpuEncoder::new(backend, &cfg_n, vb.pp("model.encoder"), 2048)?;
            let d3d12_out = d3d12_encoder.forward(&features)?;
            let d3d12_flat = d3d12_out.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;

            compare_vecs(&format!("{n_layers}-layer encoder"), &cpu_flat, &d3d12_flat);
            println!();
        }

        // ── Full benchmark ──
        let backend = D3D12EncoderBackend::new(&gpu, use_fp16_acc, cfg.encoder_dim)?;
        let encoder = GpuEncoder::new(backend, &cfg, vb.pp("model.encoder"), 2048)?;
        let n_iters = 3;
        println!("Benchmark ({n_iters} iters)...");
        let t1 = Instant::now();
        for _ in 0..n_iters {
            let _ = encoder.forward(&features)?;
        }
        let elapsed = t1.elapsed().as_secs_f64() / n_iters as f64;
        println!("  Encoder: {:.1}ms avg", elapsed * 1000.0);
        println!("  {:.2}x realtime", elapsed / duration_sec);
    }

    #[cfg(not(feature = "triton-d3d12"))]
    println!("Triton D3D12 encoder not available. Build with --features triton-d3d12");

    Ok(())
}
