/// Moonshine V2 end-to-end transcription example.
///
/// Loads the Moonshine V2 medium streaming model from GGUF assets and transcribes a WAV file.
/// Encoder/decoder weights stay quantized (Q8_0) for reduced memory and faster inference.
///
/// Usage:
///   cargo run --example transcribe_moonshine --release -- dots.wav
///   cargo run --example transcribe_moonshine --release -- dots.wav assets

use anyhow::Result;
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
        _ => anyhow::bail!("unsupported WAV format: {:?} {}bit", spec.sample_format, spec.bits_per_sample),
    };
    Ok(samples)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let wav_path = args.get(1).map(|s| s.as_str()).unwrap_or("dots.wav");
    let model_dir = args.get(2).map(|s| s.as_str()).unwrap_or("assets");

    println!("=== Moonshine V2 Transcription (Quantized) ===\n");

    // Load audio
    let t0 = Instant::now();
    let samples = load_wav_samples(wav_path)?;
    let duration_sec = samples.len() as f64 / 16000.0;
    println!("Audio: {} ({:.2}s, {} samples)", wav_path, duration_sec, samples.len());
    println!("  Loaded in {:.0}ms", t0.elapsed().as_millis());

    // Get device
    let device = get_device()?;
    println!("Device: {:?}", device);

    // Load model from GGUF assets
    let t1 = Instant::now();
    println!("Loading from GGUF assets ({})...", model_dir);
    let model = MoonshineModel::load_from_gguf_mmap(model_dir, &device)?;
    println!("Model loaded in {:.0}ms\n", t1.elapsed().as_millis());

    // Transcribe
    let t2 = Instant::now();
    let text = model.transcribe(&samples, &device)?;
    let transcribe_ms = t2.elapsed().as_millis();

    println!("Transcription: {}", text);
    println!("\nTiming: {:.0}ms RTF (real-time factor): {:.3}x)",
        transcribe_ms, (transcribe_ms as f64 / 1000.0) / duration_sec);

    Ok(())
}
