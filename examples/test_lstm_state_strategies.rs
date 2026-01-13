/// Compare LSTM state management strategies for streaming transcription
///
/// This example tests three approaches to understand quality differences:
/// 1. Reset LSTM after every chunk (current approach) - expect ~75 tokens
/// 2. Never reset LSTM (continuous state) - expect 110-140 tokens
/// 3. Reset only after silence (NeMo-style) - expect 130-140 tokens
///
/// Usage:
///   cargo run --example test_lstm_state_strategies --release -- dots.wav

use anyhow::Result;
use speech::parakeet::{
    get_device, load_parakeet_tdt_from_local,
    ParakeetFeatureExtractor, StreamingConfig, StreamingTransducer,
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        return Ok(());
    }

    let audio_path = &args[1];

    println!("LSTM State Management Strategy Comparison");
    println!("==========================================\n");
    println!("Audio: {}\n", audio_path);

    let device = get_device()?;

    // Load audio
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    let samples = if spec.channels == 2 {
        samples.chunks(2).map(|c| (c[0] + c[1]) / 2.0).collect()
    } else {
        samples
    };

    println!("Audio: {} samples, {:.2}s\n", samples.len(), samples.len() as f32 / 16000.0);

    let feat_extractor = ParakeetFeatureExtractor::new(128);

    // Configuration
    const CHUNK_SECONDS: f32 = 3.0;
    const OVERLAP_SECONDS: f32 = 0.5;
    const SAMPLES_PER_CHUNK: usize = (16000.0 * CHUNK_SECONDS) as usize;
    const OVERLAP_SAMPLES: usize = (16000.0 * OVERLAP_SECONDS) as usize;
    let stride = SAMPLES_PER_CHUNK - OVERLAP_SAMPLES;
    let total_chunks = (samples.len() + stride - 1) / stride;

    println!("Chunk configuration:");
    println!("  Chunk size: {:.1}s ({} samples)", CHUNK_SECONDS, SAMPLES_PER_CHUNK);
    println!("  Overlap: {:.1}s ({} samples)", OVERLAP_SECONDS, OVERLAP_SAMPLES);
    println!("  Total chunks: {}\n", total_chunks);

    // Strategy 1: Reset after every chunk (current)
    println!("=== STRATEGY 1: Reset After Every Chunk (Current) ===\n");
    {
        println!("Loading model...");
        let mut model = load_parakeet_tdt_from_local(".cache/parakeet-tdt", &device)?;
        model.load_tokenizer(".cache/parakeet-tdt")?;

        let streaming_config = StreamingConfig {
            chunk_samples: SAMPLES_PER_CHUNK,
            overlap_samples: OVERLAP_SAMPLES,
            emit_partial: true,
        };
        let mut transcriber = StreamingTransducer::new(model, streaming_config);

        for chunk_idx in 0..total_chunks {
            let chunk_start = chunk_idx * stride;
            let chunk_end = (chunk_start + SAMPLES_PER_CHUNK).min(samples.len());
            let chunk_samples = &samples[chunk_start..chunk_end];

            let features = feat_extractor.extract_to_tensor(chunk_samples, &device)?;
            let features = if !device.is_cpu() {
                features.to_dtype(candle_core::DType::BF16)?
            } else {
                features
            };

            let new_tokens = transcriber.process_features(&features)?;
            println!("[Chunk {}] {} new tokens, {} total",
                     chunk_idx + 1, new_tokens.len(), transcriber.tokens().len());
        }

        let text = transcriber.decode_text()?;
        println!("\nTranscript ({} tokens): {}\n", transcriber.tokens().len(), text.trim());
    }

    // Strategy 2: Never reset (continuous state)
    println!("=== STRATEGY 2: Never Reset LSTM (Continuous State) ===\n");
    {
        println!("Loading model...");
        let mut model = load_parakeet_tdt_from_local(".cache/parakeet-tdt", &device)?;
        model.load_tokenizer(".cache/parakeet-tdt")?;

        unsafe { std::env::set_var("NO_LSTM_RESET", "1"); }

        let streaming_config = StreamingConfig {
            chunk_samples: SAMPLES_PER_CHUNK,
            overlap_samples: OVERLAP_SAMPLES,
            emit_partial: true,
        };
        let mut transcriber = StreamingTransducer::new(model, streaming_config);

        for chunk_idx in 0..total_chunks {
            let chunk_start = chunk_idx * stride;
            let chunk_end = (chunk_start + SAMPLES_PER_CHUNK).min(samples.len());
            let chunk_samples = &samples[chunk_start..chunk_end];

            let features = feat_extractor.extract_to_tensor(chunk_samples, &device)?;
            let features = if !device.is_cpu() {
                features.to_dtype(candle_core::DType::BF16)?
            } else {
                features
            };

            let new_tokens = transcriber.process_features(&features)?;
            println!("[Chunk {}] {} new tokens, {} total",
                     chunk_idx + 1, new_tokens.len(), transcriber.tokens().len());
        }

        let text = transcriber.decode_text()?;
        println!("\nTranscript ({} tokens): {}\n", transcriber.tokens().len(), text.trim());

        unsafe { std::env::remove_var("NO_LSTM_RESET"); }
    }

    // Strategy 3: Reset only after silence
    println!("=== STRATEGY 3: Reset After Silence (NeMo-Style) ===\n");
    {
        println!("Loading model...");
        let mut model = load_parakeet_tdt_from_local(".cache/parakeet-tdt", &device)?;
        model.load_tokenizer(".cache/parakeet-tdt")?;

        unsafe { std::env::set_var("SILENCE_RESET", "1"); }

        let streaming_config = StreamingConfig {
            chunk_samples: SAMPLES_PER_CHUNK,
            overlap_samples: OVERLAP_SAMPLES,
            emit_partial: true,
        };
        let mut transcriber = StreamingTransducer::new(model, streaming_config);

        for chunk_idx in 0..total_chunks {
            let chunk_start = chunk_idx * stride;
            let chunk_end = (chunk_start + SAMPLES_PER_CHUNK).min(samples.len());
            let chunk_samples = &samples[chunk_start..chunk_end];

            let features = feat_extractor.extract_to_tensor(chunk_samples, &device)?;
            let features = if !device.is_cpu() {
                features.to_dtype(candle_core::DType::BF16)?
            } else {
                features
            };

            let new_tokens = transcriber.process_features(&features)?;
            println!("[Chunk {}] {} new tokens, {} total",
                     chunk_idx + 1, new_tokens.len(), transcriber.tokens().len());
        }

        let text = transcriber.decode_text()?;
        println!("\nTranscript ({} tokens): {}\n", transcriber.tokens().len(), text.trim());

        unsafe { std::env::remove_var("SILENCE_RESET"); }
    }

    println!("===========================================");
    println!("\nComparison Summary:");
    println!("  Strategy 1 (reset every chunk): Expected ~75 tokens");
    println!("  Strategy 2 (never reset): Expected 110-140 tokens");
    println!("  Strategy 3 (silence-based): Expected 130-140 tokens");
    println!("\nThe token count difference directly shows the impact of");
    println!("LSTM state management on transcription quality.");

    Ok(())
}
