/// Live microphone streaming transcription using Moonshine V2.
///
/// Captures audio from the default input device, runs VAD to detect speech,
/// and streams partial + final transcriptions to the terminal.
///
/// Usage:
///   cargo run --example mic_streaming --release
///   cargo run --example mic_streaming --release -- assets

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use speech::moonshine::MoonshineModel;
use speech::parakeet::get_device;
use speech::silero::{SileroVad, VadStream};
use speech::streaming::{StreamingConfig, StreamingTranscriber};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = args.get(1).map(|s| s.as_str()).unwrap_or("assets");

    println!("=== Moonshine V2 Live Microphone Streaming ===\n");

    // Get device
    let device = get_device()?;

    // Load VAD
    let assets = PathBuf::from(model_dir);
    let vad = SileroVad::load_from_gguf_mmap(&assets, &device)?;
    let vad_stream = VadStream::new(vad, &device)?;

    // Load Moonshine model
    let t1 = Instant::now();
    println!("Loading Moonshine from GGUF assets ({})...", model_dir);
    let model = MoonshineModel::load_from_gguf_mmap(model_dir, &device)?;
    println!("Model loaded in {:.0}ms\n", t1.elapsed().as_millis());

    // Create streaming transcriber with default config
    let mut transcriber = StreamingTranscriber::new(
        model,
        vad_stream,
        device,
        StreamingConfig::default(),
    );

    // Set up audio input
    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no input device found"))?;
    println!("Input device: {}", input_device.name()?);

    let default_config = input_device.default_input_config()?;
    let native_rate = default_config.sample_rate().0;
    let native_channels = default_config.channels();
    println!("Native format: {}Hz, {} channel(s)", native_rate, native_channels);

    let config = cpal::StreamConfig {
        channels: native_channels,
        sample_rate: cpal::SampleRate(native_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let need_resample = native_rate != 16000;
    let need_downmix = native_channels > 1;
    if need_resample {
        println!("Will resample {}Hz -> 16000Hz", native_rate);
    }

    // Channel to send audio from the cpal callback to the main thread
    let (tx, rx) = mpsc::channel::<Vec<f32>>();

    let stream = input_device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let _ = tx.send(data.to_vec());
        },
        |err| eprintln!("Audio input error: {}", err),
        None,
    )?;
    stream.play()?;

    // Simple linear resampler state
    let resample_ratio = if need_resample {
        16000.0 / native_rate as f64
    } else {
        1.0
    };
    let mut resample_accum: f64 = 0.0;

    println!("Listening... (Ctrl+C to stop)\n");

    // Accumulate 16kHz mono audio
    let mut audio_buf: Vec<f32> = Vec::new();
    let mut last_segment = 0u32;

    loop {
        // Drain all available audio from the channel
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

        // If no audio yet, block briefly for the next chunk
        if audio_buf.is_empty() {
            if let Ok(raw) = rx.recv() {
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
            continue;
        }

        // Feed all accumulated audio to the transcriber
        let events = transcriber.push_audio(&audio_buf)?;
        audio_buf.clear();

        for evt in events {
            if evt.segment_index != last_segment {
                last_segment = evt.segment_index;
                eprintln!("\n--- Segment {} ---", last_segment);
            }

            if evt.is_partial {
                eprint!("\r\x1b[K  [partial] {}", evt.text);
            } else {
                eprintln!("\r\x1b[K  [final]   {}", evt.text);
                println!("{}", evt.text);
            }
        }
    }
}
