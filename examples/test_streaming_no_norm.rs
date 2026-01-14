/// Test streaming model with NO normalization (matching NeMo config)
///
/// This directly tests if the blank domination issue is caused by normalization mismatch.
/// Streaming model config has normalize='NA' (no normalization).

use anyhow::Result;
use speech::parakeet::{
    get_device, load_parakeet_streaming_tdt_from_local,
    ParakeetFeatureExtractor,
    streaming_encoder::StreamingEncoderCache,
};
use candle_core::DType;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        return Ok(());
    }

    let device = get_device()?;
    println!("Device: {:?}\n", device);

    // Load Streaming TDT model
    println!("Loading Streaming TDT model...");
    let mut model = load_parakeet_streaming_tdt_from_local(".cache/parakeet-streaming-tdt", &device)?;
    println!("✓ Model loaded");
    println!("  Vocab size: {}", model.config.vocab_size);
    println!("  Blank ID: {}", model.config.blank_id);
    println!("  Mel bins: {}\n", model.encoder.cfg.feat_in);

    // Load tokenizer
    model.load_tokenizer(".cache/parakeet-streaming-tdt")?;

    // Load audio
    let audio_path = &args[1];
    let mut reader = hound::WavReader::open(&audio_path)?;
    let audio_samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;

    let duration = audio_samples.len() as f64 / 16000.0;
    println!("Audio: {:.2}s ({} samples)\n", duration, audio_samples.len());

    // Streaming configuration
    let chunk_size_samples = 72000;  // 4.5s chunks
    let max_cache_frames = 70;

    println!("=== Cache-Aware Streaming with NO NORMALIZATION ===\n");

    // Feature extractor WITHOUT normalization (normalize=false)
    let num_mel_bins = model.encoder.cfg.feat_in;
    let feat_extractor = ParakeetFeatureExtractor::new_with_config(num_mel_bins, false);

    // Initialize cache
    let batch_size = 1;
    let num_heads = model.encoder.cfg.num_heads;
    let head_dim = model.encoder.cfg.d_model / num_heads;
    let d_model = model.encoder.cfg.d_model;
    let conv_kernel_size = model.encoder.cfg.conv_kernel_size;
    let num_layers = model.encoder.cfg.num_layers;

    let dtype = if !device.is_cpu() {
        DType::BF16
    } else {
        DType::F32
    };

    let mut encoder_cache = StreamingEncoderCache::with_capacity(
        num_layers,
        batch_size,
        num_heads,
        max_cache_frames,
        head_dim,
        d_model,
        conv_kernel_size,
        &device,
        dtype,
    )?;

    // Streaming state
    let mut pred_states = None;
    let mut last_token = model.config.blank_id as u32;
    let mut all_tokens = Vec::new();
    let mut decoded_count = 0;

    // Process chunks
    let mut offset = 0;
    let mut chunk_num = 0;

    while offset < audio_samples.len() {
        let chunk_end = (offset + chunk_size_samples).min(audio_samples.len());
        let chunk = &audio_samples[offset..chunk_end];

        chunk_num += 1;
        print!("[Chunk {}] ", chunk_num);
        std::io::Write::flush(&mut std::io::stdout())?;

        // Extract features WITHOUT normalization
        let features = feat_extractor.extract_to_tensor(chunk, &device)?;
        let features = if !device.is_cpu() {
            features.to_dtype(DType::BF16)?
        } else {
            features
        };

        // Run encoder with cache
        let encoder_out = model.encoder.forward_with_cache(&features, false, Some(&mut encoder_cache))?;

        // Streaming greedy decode
        let (new_tokens, new_states, new_last_token) = model.greedy_decode_streaming(
            &encoder_out,
            pred_states,
            last_token,
        )?;

        pred_states = new_states;
        last_token = new_last_token;
        all_tokens.extend_from_slice(&new_tokens);

        // Decode incrementally
        if !new_tokens.is_empty() {
            let new_text = model.decode_tokens_incremental(&all_tokens, decoded_count)?;
            if !new_text.is_empty() {
                print!("{}", new_text);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            decoded_count = all_tokens.len();
        }
        println!();

        offset += chunk_size_samples;
    }

    // Results
    println!("\n=== RESULTS ===");
    println!("  Total tokens: {}", all_tokens.len());
    println!("  NeMo reference: 225 tokens");
    println!("  Quality: {:.1}%", all_tokens.len() as f32 / 225.0 * 100.0);

    let full_text = model.decode_tokens(&all_tokens)?;
    println!("\n=== Full Transcription ===");
    println!("{}", full_text);

    if all_tokens.len() > 150 {
        println!("\n✅ SUCCESS! Removing normalization fixed the blank domination issue!");
        println!("   The model achieves {:.1}% quality (expected ~84% for cache-aware streaming)",
                 all_tokens.len() as f32 / 225.0 * 100.0);
    } else {
        println!("\n❌ Still broken - only {} tokens produced", all_tokens.len());
    }

    Ok(())
}
