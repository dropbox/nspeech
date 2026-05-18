//! speek — like macOS `say`, but uses Kokoro TTS on GPU.
//!
//! All model assets are embedded in the binary for standalone use.
//! Works on macOS (afplay) and Windows (native WASAPI via cpal).
//!
//! Usage:
//!   speek "Hello, world!"
//!   echo "piped text" | speek

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::io::Read;
use std::sync::{Arc, Condvar, Mutex};

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

    play_audio(&audio, 24000)?;

    Ok(())
}

fn play_audio(samples: &[f32], sample_rate: u32) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("No audio output device available"))?;

    let default_config = device.default_output_config()?;
    let out_rate = default_config.sample_rate().0;
    let out_channels = default_config.channels() as usize;

    // Resample from source rate to device rate if needed
    let resampled: Vec<f32> = if out_rate != sample_rate {
        let ratio = out_rate as f64 / sample_rate as f64;
        let out_len = (samples.len() as f64 * ratio).ceil() as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src_pos = i as f64 / ratio;
            let idx = src_pos as usize;
            let frac = src_pos - idx as f64;
            let s0 = samples[idx.min(samples.len() - 1)];
            let s1 = samples[(idx + 1).min(samples.len() - 1)];
            out.push(s0 + (s1 - s0) * frac as f32);
        }
        out
    } else {
        samples.to_vec()
    };

    // Expand mono to match device channel count
    let output_samples: Vec<f32> = if out_channels > 1 {
        resampled.iter().flat_map(|&s| std::iter::repeat_n(s, out_channels)).collect()
    } else {
        resampled
    };

    let total_frames = output_samples.len();

    let config = cpal::StreamConfig {
        channels: out_channels as u16,
        sample_rate: cpal::SampleRate(out_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let output_samples = Arc::new(output_samples);
    let pos = Arc::new(Mutex::new(0usize));
    let done = Arc::new((Mutex::new(false), Condvar::new()));

    let samples_cl = Arc::clone(&output_samples);
    let pos_cl = Arc::clone(&pos);
    let done_cl = Arc::clone(&done);

    let stream = device.build_output_stream(
        &config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let mut p = pos_cl.lock().unwrap();
            for sample in data.iter_mut() {
                if *p < samples_cl.len() {
                    *sample = samples_cl[*p];
                    *p += 1;
                } else {
                    *sample = 0.0;
                }
            }
            if *p >= samples_cl.len() {
                let (lock, cvar) = &*done_cl;
                *lock.lock().unwrap() = true;
                cvar.notify_one();
            }
        },
        move |err| {
            eprintln!("Audio stream error: {}", err);
        },
        None,
    )?;

    stream.play()?;

    // Wait for playback to finish
    let (lock, cvar) = &*done;
    let mut finished = lock.lock().unwrap();
    while !*finished {
        finished = cvar.wait(finished).unwrap();
    }

    // Brief drain to ensure the last buffer is flushed to hardware
    let drain_ms = (total_frames as u64 * 1000 / out_rate as u64).min(100).max(50);
    std::thread::sleep(std::time::Duration::from_millis(drain_ms));

    Ok(())
}
