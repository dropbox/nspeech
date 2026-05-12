//! speek — like macOS `say`, but uses Kokoro TTS on GPU.
//!
//! Usage:
//!   speek "Hello, world!"
//!   echo "piped text" | speek
//!   speek -v af_bella "Different voice"

use anyhow::Result;
use std::io::Read;
use std::path::Path;
use std::process::Command;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut voice_name = "af_heart";
    let mut text_args: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "-v" {
            i += 1;
            if i < args.len() {
                voice_name = args[i].as_str();
            }
        } else {
            text_args.push(&args[i]);
        }
        i += 1;
    }

    let text = if text_args.is_empty() {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf.trim().to_string()
    } else {
        text_args.join(" ")
    };

    if text.is_empty() {
        eprintln!("Usage: speek [-v voice] <text>");
        eprintln!("       echo \"text\" | speek");
        std::process::exit(1);
    }

    let assets = Path::new("assets");
    let config_path = assets.join("kokoro-config.json");
    let gguf_path = assets.join("kokoro_q8_0.gguf");
    let voice_path = assets.join(format!("kokoro-{}.safetensors", voice_name));

    if !config_path.exists() || !gguf_path.exists() {
        eprintln!("Missing model files in assets/");
        std::process::exit(1);
    }
    if !voice_path.exists() {
        eprintln!("Unknown voice: {voice_name}");
        std::process::exit(1);
    }

    // Suppress library println! during model load
    let _suppress = SuppressStdout::new();

    let config_json = std::fs::read_to_string(&config_path)?;
    let config = speech::kokoro::KokoroConfig::from_json(&config_json)?;
    let device = speech::parakeet::get_device()?;
    let model = speech::kokoro::KokoroModel::load_gguf(&gguf_path, config.clone(), &device)?;

    let gold_json = std::fs::read_to_string(assets.join("us_gold.json"))
        .unwrap_or_else(|_| "{}".to_string());
    let silver_json = std::fs::read_to_string(assets.join("us_silver.json"))
        .unwrap_or_else(|_| "{}".to_string());
    let phonemizer = speech::kokoro::Phonemizer::new(&gold_json, &silver_json, &config.vocab)?;

    let tokens = phonemizer.phonemize(&text);
    if tokens.is_empty() {
        eprintln!("Could not phonemize input.");
        std::process::exit(1);
    }

    let style = speech::kokoro::KokoroModel::load_voice(&voice_path, tokens.len() + 2, &device)?;
    let audio = model.synthesize(&tokens, &style, 1.0)?;
    drop(_suppress);

    // Write to temp file and play with afplay
    let tmp = std::env::temp_dir().join("speek_out.wav");
    write_wav(tmp.to_str().unwrap(), &audio, 24000)?;
    Command::new("afplay").arg(&tmp).status()?;

    Ok(())
}

struct SuppressStdout {
    saved: std::fs::File,
}

impl SuppressStdout {
    fn new() -> Self {
        use std::os::unix::io::{AsRawFd, FromRawFd};
        unsafe extern "C" { fn dup(fd: i32) -> i32; fn dup2(fd: i32, fd2: i32) -> i32; }
        unsafe {
            let saved = std::fs::File::from_raw_fd(dup(1));
            let devnull = std::fs::File::open("/dev/null").unwrap();
            dup2(devnull.as_raw_fd(), 1);
            Self { saved }
        }
    }
}

impl Drop for SuppressStdout {
    fn drop(&mut self) {
        use std::os::unix::io::AsRawFd;
        unsafe extern "C" { fn dup2(fd: i32, fd2: i32) -> i32; }
        unsafe { dup2(self.saved.as_raw_fd(), 1); }
    }
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
