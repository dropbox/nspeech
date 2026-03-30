/// Moonshine V2 inference benchmark.
///
/// Measures encode, decode, and full transcribe times separately.
/// Reports median/min/max over N iterations.
///
/// Usage:
///   cargo run --example bench_moonshine --release -- dots.wav
///   cargo run --example bench_moonshine --release -- dots.wav assets 10

use anyhow::Result;
use candle_core::Tensor;
use speech::moonshine::MoonshineModel;
use speech::parakeet::get_device;
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


fn median(values: &mut Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}

fn report(label: &str, times_ms: &mut Vec<f64>) {
    if times_ms.is_empty() {
        return;
    }
    let med = median(times_ms);
    let min = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  {:<20} median={:>8.1}ms  min={:>8.1}ms  max={:>8.1}ms  (n={})",
        label,
        med,
        min,
        max,
        times_ms.len()
    );
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let wav_path = args.get(1).map(|s| s.as_str()).unwrap_or("dots.wav");
    let model_dir = args.get(2).map(|s| s.as_str()).unwrap_or("assets");
    let iterations: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    println!("=== Moonshine V2 Benchmark ===\n");

    // Load audio
    let samples = load_wav_samples(wav_path)?;
    let duration_sec = samples.len() as f64 / 16000.0;
    println!(
        "Audio: {} ({:.2}s, {} samples)",
        wav_path,
        duration_sec,
        samples.len()
    );

    // Get device
    let device = get_device()?;

    // Load model
    let t0 = Instant::now();
    let model = MoonshineModel::load_from_gguf_mmap(model_dir, &device)?;
    println!("Model loaded in {:.0}ms\n", t0.elapsed().as_millis());

    // Prepare audio tensor (reuse across iterations)
    let frame_len = model.cfg.frame_len;
    let pad_len = (frame_len - samples.len() % frame_len) % frame_len;
    let mut padded = samples.clone();
    padded.extend(std::iter::repeat(0.0f32).take(pad_len));
    let audio = Tensor::from_vec(padded, (1, samples.len() + pad_len), &device)?;

    // Warmup
    print!("Warmup... ");
    let t = Instant::now();
    let text = model.transcribe(&samples, &device)?;
    println!("done in {:.0}ms", t.elapsed().as_millis());
    println!("  Text: \"{}\"\n", text);

    // Benchmark encode
    println!("Benchmarking ({} iterations):", iterations);

    let mut encode_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t = Instant::now();
        let _enc = model.encode(&audio)?;
        encode_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    report("encode", &mut encode_times);

    // Benchmark decode (reuse encoder output)
    let encoder_hidden = model.encode(&audio)?;
    let enc_frames = encoder_hidden.dim(1)?;
    let max_tokens = (enc_frames as f64 * 0.02 * 6.5).ceil() as usize + 10;

    let mut decode_times = Vec::with_capacity(iterations);
    let mut token_counts = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t = Instant::now();
        let tokens = model.greedy_decode(&encoder_hidden, max_tokens)?;
        decode_times.push(t.elapsed().as_secs_f64() * 1000.0);
        token_counts.push(tokens.len());
    }
    report("decode", &mut decode_times);
    let avg_tokens = token_counts.iter().sum::<usize>() as f64 / token_counts.len() as f64;
    let decode_median = median(&mut decode_times.clone());
    println!(
        "  {:<20} {:.1} tokens, {:.1}ms/token",
        "",
        avg_tokens,
        decode_median / avg_tokens
    );

    // Benchmark full transcribe (encode + decode)
    let mut transcribe_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t = Instant::now();
        let _text = model.transcribe(&samples, &device)?;
        transcribe_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    report("transcribe", &mut transcribe_times);

    let transcribe_median = median(&mut transcribe_times.clone());
    let rtf = (transcribe_median / 1000.0) / duration_sec;
    println!(
        "\n  RTF (real-time factor): {:.3}x  ({:.2}s audio in {:.0}ms = {:.1}x realtime)",
        rtf,
        duration_sec,
        transcribe_median,
        1.0 / rtf
    );

    Ok(())
}
