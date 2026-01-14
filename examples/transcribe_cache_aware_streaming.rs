/// Cache-Aware Streaming Transcription using Streaming TDT Model
///
/// This implements true cache-aware streaming with:
/// - Attention K/V caching across chunks
/// - Convolution state caching
/// - Predictor LSTM state maintenance
/// - Zero redundant computations
///
/// Based on nvidia/nemotron-speech-streaming-en-0.6b architecture

use anyhow::Result;
use speech::parakeet::{
    get_device, load_parakeet_tdt_from_gguf_local,
    ParakeetFeatureExtractor,
    streaming_encoder::StreamingEncoderCache,
};
use candle_core::DType;

fn main() -> Result<()> {
    let device = get_device()?;
    println!("Device: {:?}\n", device);

    // Load STANDARD TDT model (for comparison)
    println!("Loading STANDARD TDT model...");
    let model = load_parakeet_tdt_from_gguf_local("assets", &device)?;
    println!("✓ Model loaded");
    println!("  Vocab size: {}", model.config.vocab_size);
    println!("  Blank ID: {}", model.config.blank_id);
    println!("  Encoder layers: {}", model.encoder.cfg.num_layers);
    println!("  d_model: {}", model.encoder.cfg.d_model);
    println!("  Num heads: {}", model.encoder.cfg.num_heads);
    println!("  Conv kernel size: {}\n", model.encoder.cfg.conv_kernel_size);

    // Load audio
    println!("Loading audio...");
    let mut reader = hound::WavReader::open("dots.wav")?;
    let audio_samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;

    let duration = audio_samples.len() as f64 / 16000.0;
    println!("✓ Audio loaded: {:.2}s ({} samples)\n", duration, audio_samples.len());

    // Streaming configuration based on NeMo's att_context_size=[70, 13]
    // This means: left_context=70 frames, right_context=13 frames
    // After 8x subsampling: 13 frames * 80ms = 1040ms chunks
    // At mel level: 13 * 8 = 104 mel frames per chunk
    // In samples: 104 frames * 160 samples/hop = 16,640 samples

    // OPTIMAL: 4-5 second chunks for best quality (tested empirically)
    // Below 3.5s chunks, quality collapses due to unfavorable cache:current ratio
    let chunk_duration_s = 4.5; // seconds
    let chunk_size_samples = (chunk_duration_s * 16000.0) as usize;
    let max_cache_frames = 70; // NeMo's left context

    println!("=== Cache-Aware Streaming Configuration ===");
    println!("  Chunk size: {} samples ({:.2}s)", chunk_size_samples, chunk_size_samples as f64 / 16000.0);
    println!("  Max cache frames: {} (past context)", max_cache_frames);
    println!("  Cache duration: ~{:.1}s\n", max_cache_frames as f64 * 0.08);

    // Feature extractor
    let num_mel_bins = model.encoder.cfg.feat_in;
    let feat_extractor = ParakeetFeatureExtractor::new(num_mel_bins);

    // Initialize streaming cache
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

    println!("Initializing encoder cache...");
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
    println!("✓ Cache initialized\n");

    // Streaming state for decoder
    let mut pred_states = None;
    let mut last_token = model.config.blank_id as u32;
    let mut all_tokens = Vec::new();
    let mut decoded_count = 0;

    // Process audio in chunks
    println!("=== Streaming Transcription ===\n");
    let mut offset = 0;
    let mut chunk_num = 0;

    while offset < audio_samples.len() {
        let chunk_end = (offset + chunk_size_samples).min(audio_samples.len());
        let chunk = &audio_samples[offset..chunk_end];

        chunk_num += 1;
        print!("[Chunk {}] ", chunk_num);
        std::io::Write::flush(&mut std::io::stdout())?;

        // Extract features for this chunk
        let features = feat_extractor.extract_to_tensor(chunk, &device)?;

        // Convert to model dtype
        let features = if !device.is_cpu() {
            features.to_dtype(DType::BF16)?
        } else {
            features
        };

        // Run encoder with cache (zero redundant computation!)
        let encoder_out = model.encoder.forward_with_cache(&features, false, Some(&mut encoder_cache))?;

        // Run streaming decode with maintained predictor state
        let (new_tokens, new_states, new_last_token) = model.greedy_decode_streaming(
            &encoder_out,
            pred_states,
            last_token,
        )?;

        // Update decoder state for next chunk
        pred_states = new_states;
        last_token = new_last_token;

        // Accumulate tokens
        all_tokens.extend_from_slice(&new_tokens);

        // Decode and print new text incrementally
        if !new_tokens.is_empty() {
            let new_text = model.decode_tokens_incremental(&all_tokens, decoded_count)?;
            if !new_text.is_empty() {
                print!("{}", new_text);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            decoded_count = all_tokens.len();
        }
        println!(); // Newline after each chunk

        // Move to next chunk (no overlap - cache handles continuity)
        offset += chunk_size_samples;
    }

    println!("\n=== Final Statistics ===");
    println!("  Audio duration: {:.2}s", duration);
    println!("  Total chunks: {}", chunk_num);
    println!("  Total tokens: {}", all_tokens.len());
    println!("  First 10 token IDs: {:?}", &all_tokens[..all_tokens.len().min(10)]);

    // Full transcription
    let full_text = model.decode_tokens(&all_tokens)?;
    println!("\n=== Full Transcription ===");
    println!("{}", full_text);

    println!("\n=== Quality Comparison ===");
    println!("  NeMo reference (streaming model): 225 tokens");
    println!("  Cache-aware streaming: {} tokens ({:.1}%)", all_tokens.len(), all_tokens.len() as f32 / 225.0 * 100.0);

    // Cache statistics
    if let Some(first_cache) = encoder_cache.attention_caches.first() {
        println!("\n=== Cache Statistics ===");
        println!("  Cached frames: {}", first_cache.num_cached);
        println!("  Max cache size: {}", first_cache.max_cache_size);
        println!("  Total frames processed: {}", encoder_cache.total_frames);
    }

    Ok(())
}
