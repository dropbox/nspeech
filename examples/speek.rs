//! speek — like macOS `say`, but uses Kokoro TTS on GPU.
//!
//! All model assets are embedded in the binary for standalone use.
//!
//! Usage:
//!   speek "Hello, world!"
//!   echo "piped text" | speek

use anyhow::Result;
use std::io::Read;
use std::process::Command;

static GGUF_DATA: &[u8] = include_bytes!("../assets/kokoro_q8_0.gguf");
static CONFIG_JSON: &str = include_str!("../assets/kokoro-config.json");
static VOICE_DATA: &[u8] = include_bytes!("../assets/kokoro-af_heart.safetensors");
static GOLD_JSON: &str = include_str!("../assets/us_gold.json");
static SILVER_JSON: &str = include_str!("../assets/us_silver.json");

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let text = if args.is_empty() {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf.trim().to_string()
    } else {
        args.join(" ")
    };

    if text.is_empty() {
        eprintln!("Usage: speek <text>");
        eprintln!("       echo \"text\" | speek");
        std::process::exit(1);
    }

    let config = speech::kokoro::KokoroConfig::from_json(CONFIG_JSON)?;
    let device = speech::parakeet::get_device()?;

    eprintln!("Loading model...");
    let model = speech::kokoro::KokoroModel::load_gguf_bytes(GGUF_DATA, config.clone(), &device)?;

    let phonemizer = speech::kokoro::Phonemizer::new(GOLD_JSON, SILVER_JSON, &config.vocab)?;

    let tokens = phonemizer.phonemize(&text);
    if tokens.is_empty() {
        eprintln!("Could not phonemize input.");
        std::process::exit(1);
    }

    let style = speech::kokoro::KokoroModel::load_voice_bytes(VOICE_DATA, tokens.len() + 2, &device)?;
    let audio = model.synthesize(&tokens, &style, 1.0)?;

    let tmp = std::env::temp_dir().join("speek_out.wav");
    write_wav(tmp.to_str().unwrap(), &audio, 24000)?;
    Command::new("afplay").arg(&tmp).status()?;

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
