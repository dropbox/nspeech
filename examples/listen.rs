//! listen — live speech-to-text with interactive buffer control.
//!
//! Streams transcription to the terminal. Press ENTER to emit the buffer
//! to stdout and exit, ESC to clear it, or ESC on an empty buffer to quit.
//!
//! Usage:
//!   listen
//!   listen assets

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal;
use speech::moonshine::MoonshineModel;
use speech::parakeet::get_device;
use speech::silero::{SileroVad, VadStream};
use speech::streaming::{StreamingConfig, StreamingTranscriber};
use std::io::Write;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

fn main() {
    if let Err(e) = run() {
        eprintln!("listen: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = args.get(1).map(|s| s.as_str()).unwrap_or("assets");

    // Probe audio device first — fail fast before loading models
    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no microphone found — grant access in System Settings > Privacy > Microphone"))?;

    let default_config = input_device.default_input_config()
        .map_err(|e| anyhow::anyhow!("cannot access microphone ({e}) — grant access in System Settings > Privacy > Microphone"))?;
    let native_rate = default_config.sample_rate().0;
    let native_channels = default_config.channels();

    // Load models
    let device = get_device()?;
    let assets = PathBuf::from(model_dir);
    let vad = SileroVad::load_from_gguf_mmap(&assets, &device)?;
    let vad_stream = VadStream::new(vad, &device)?;
    let model = MoonshineModel::load_from_gguf_mmap(model_dir, &device)?;

    let mut transcriber = StreamingTranscriber::new(
        model,
        vad_stream,
        device,
        StreamingConfig::default(),
    );

    // Start audio capture
    let config = cpal::StreamConfig {
        channels: native_channels,
        sample_rate: cpal::SampleRate(native_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let need_resample = native_rate != 16000;
    let need_downmix = native_channels > 1;

    let (tx, rx) = mpsc::channel::<Vec<f32>>();

    let stream = input_device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let _ = tx.send(data.to_vec());
        },
        |err| eprintln!("Audio error: {}", err),
        None,
    ).map_err(|e| anyhow::anyhow!("cannot open microphone ({e}) — grant access in System Settings > Privacy > Microphone"))?;
    stream.play()
        .map_err(|e| anyhow::anyhow!("cannot start audio capture: {e}"))?;

    let resample_ratio = if need_resample {
        16000.0 / native_rate as f64
    } else {
        1.0
    };
    let mut resample_accum: f64 = 0.0;

    // Enter raw mode for key detection
    terminal::enable_raw_mode()?;
    let _raw_guard = RawGuard;

    eprintln!("Listening... (ENTER=emit, ESC=clear/quit)\r");

    let mut audio_buf: Vec<f32> = Vec::new();
    let mut text_buf = String::new();
    let mut partial = String::new();

    let output = loop {
        // Check for key events
        if event::poll(Duration::from_millis(10))? {
            if let Event::Key(KeyEvent { code, modifiers, .. }) = event::read()? {
                match code {
                    KeyCode::Enter => {
                        if !text_buf.is_empty() || !partial.is_empty() {
                            let mut out = text_buf.clone();
                            if !partial.is_empty() {
                                if !out.is_empty() {
                                    out.push(' ');
                                }
                                out.push_str(&partial);
                            }
                            eprint!("\r\x1b[K\x1b[1m{}\x1b[0m\r\n", out);
                            let _ = std::io::stderr().flush();
                            break Some(out);
                        }
                    }
                    KeyCode::Esc => {
                        if text_buf.is_empty() && partial.is_empty() {
                            eprint!("\r\x1b[K");
                            break None;
                        }
                        text_buf.clear();
                        partial.clear();
                        transcriber.reset()?;
                        render_line(&text_buf, &partial);
                    }
                    KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                        eprint!("\r\x1b[K");
                        break None;
                    }
                    _ => {}
                }
            }
        }

        // Drain audio
        while let Ok(raw) = rx.try_recv() {
            let mono: Vec<f32> = if need_downmix {
                raw.chunks(native_channels as usize)
                    .map(|frame| frame.iter().sum::<f32>() / native_channels as f32)
                    .collect()
            } else {
                raw
            };

            if need_resample {
                for &s in &mono {
                    resample_accum += resample_ratio;
                    while resample_accum >= 1.0 {
                        audio_buf.push(s);
                        resample_accum -= 1.0;
                    }
                }
            } else {
                audio_buf.extend_from_slice(&mono);
            }
        }

        if audio_buf.is_empty() {
            continue;
        }

        // Feed audio to transcriber
        let events = transcriber.push_audio(&audio_buf)?;
        audio_buf.clear();

        for evt in events {
            if evt.is_partial {
                partial = evt.text;
            } else {
                if !text_buf.is_empty() {
                    text_buf.push(' ');
                }
                text_buf.push_str(&evt.text);
                partial.clear();
            }
            render_line(&text_buf, &partial);
        }
    };

    // Raw mode is restored by RawGuard drop
    drop(_raw_guard);

    if let Some(text) = output {
        println!("{}", text);
    }

    Ok(())
}

fn render_line(committed: &str, partial: &str) {
    if partial.is_empty() {
        eprint!("\r\x1b[K\x1b[1;93m{}\x1b[0m", committed);
    } else if committed.is_empty() {
        eprint!("\r\x1b[K\x1b[93m{}\x1b[0m", partial);
    } else {
        eprint!("\r\x1b[K\x1b[1;93m{}\x1b[0m \x1b[93m{}\x1b[0m", committed, partial);
    }
    let _ = std::io::stderr().flush();
}

struct RawGuard;
impl Drop for RawGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
    }
}
