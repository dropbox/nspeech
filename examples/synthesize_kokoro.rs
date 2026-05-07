//! Example: synthesize speech from text using Kokoro TTS.
//!
//! Usage:
//!   cargo run --example synthesize_kokoro --release -- "Hello, world!" output.wav

use anyhow::Result;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <text-or-ipa> <output.wav> [voice]", args[0]);
        std::process::exit(1);
    }

    let text = &args[1];
    let output_path = &args[2];
    let voice_name = args.get(3).map(|s| s.as_str()).unwrap_or("af_heart");

    let assets = Path::new("assets");
    let config_path = assets.join("kokoro-config.json");
    let gguf_path = assets.join("kokoro_q8_0.gguf");
    let voice_path = assets.join(format!("kokoro-{}.safetensors", voice_name));

    if !config_path.exists() || !gguf_path.exists() || !voice_path.exists() {
        eprintln!("Missing model files in assets/. Need: kokoro-config.json, kokoro_q8_0.gguf, kokoro-{voice_name}.safetensors");
        std::process::exit(1);
    }

    let config_json = std::fs::read_to_string(&config_path)?;
    let config = speech::kokoro::KokoroConfig::from_json(&config_json)?;

    eprintln!("Loading model...");
    let device = speech::parakeet::get_device()?;
    let model = speech::kokoro::KokoroModel::load_gguf(&gguf_path, config.clone(), &device)?;

    eprintln!("Loading phonemizer dictionaries...");
    let gold_json = std::fs::read_to_string(assets.join("us_gold.json"))
        .unwrap_or_else(|_| "{}".to_string());
    let silver_json = std::fs::read_to_string(assets.join("us_silver.json"))
        .unwrap_or_else(|_| "{}".to_string());
    let phonemizer = speech::kokoro::Phonemizer::new(&gold_json, &silver_json, &config.vocab)?;

    eprintln!("Phonemizing: \"{}\"", text);
    let ipa = phonemizer.to_ipa(text);
    eprintln!("  IPA: {}", ipa);
    let tokens = phonemizer.phonemize(text);
    eprintln!("  {} tokens", tokens.len());

    if tokens.is_empty() {
        eprintln!("No valid tokens produced. Input might not contain recognized phonemes.");
        std::process::exit(1);
    }

    eprintln!("Loading voice: {}", voice_name);
    let style = speech::kokoro::KokoroModel::load_voice(&voice_path, tokens.len() + 2, &device)?;

    // Synthesize
    eprintln!("Synthesizing...");
    let start = std::time::Instant::now();
    let audio = model.synthesize(&tokens, &style, 1.0)?;
    let elapsed = start.elapsed();
    eprintln!("  Generated {} samples ({:.2}s audio) in {:.2}ms",
        audio.len(),
        audio.len() as f64 / 24000.0,
        elapsed.as_secs_f64() * 1000.0,
    );

    // Write WAV
    write_wav(output_path, &audio, 24000)?;
    eprintln!("Saved to {}", output_path);

    Ok(())
}

fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let s16 = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(s16)?;
    }
    writer.finalize()?;
    Ok(())
}
