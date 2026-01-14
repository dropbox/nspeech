/// Test streaming model with correct chunk configuration
///
/// Uses nvidia/nemotron-speech-streaming-en-0.6b with:
/// - 1.04s chunks (as designed, from att_context_size [70, 13])
/// - 136 mel bins
/// - vocab_size=1024, blank_id=1024

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

    // Load Streaming TDT model (BF16 safetensors)
    println!("Loading Streaming TDT model (BF16 safetensors)...");
    let mut model = load_parakeet_streaming_tdt_from_local(".cache/parakeet-streaming-tdt", &device)?;
    println!("✓ Model loaded");
    println!("  Vocab size: {}", model.config.vocab_size);
    println!("  Blank ID: {}", model.config.blank_id);
    println!("  Encoder d_model: {}", model.encoder.cfg.d_model);
    println!("  Feature dimension: {}\n", model.encoder.cfg.feat_in);

    // Load tokenizer
    println!("Loading tokenizer...");
    model.load_tokenizer(".cache/parakeet-streaming-tdt")?;
    println!("✓ Tokenizer loaded\n");

    // Load audio
    println!("Loading audio...");
    let audio_path = &args[1];
    let mut reader = hound::WavReader::open(&audio_path)?;
    let audio_samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;

    let duration = audio_samples.len() as f64 / 16000.0;
    println!("✓ Audio loaded: {:.2}s ({} samples)\n", duration, audio_samples.len());

    // Streaming configuration based on att_context_size=[70, 13]
    // 13 encoder frames * 8 subsampling * 160 samples/hop = 16,640 samples ≈ 1.04s
    let chunk_size_samples = 16640;  // 1.04s chunks
    let max_cache_frames = 70;  // Left context from att_context_size

    println!("=== Streaming Configuration ===");
    println!("  Chunk size: {} samples ({:.2}s)", chunk_size_samples, chunk_size_samples as f64 / 16000.0);
    println!("  Max cache frames: {} (past context)", max_cache_frames);
    println!("  Expected encoder frames per chunk: ~13\n");

    // Feature extractor (136 mel bins for streaming model)
    // CRITICAL: Streaming model uses normalize='NA' (no normalization)
    let num_mel_bins = model.encoder.cfg.feat_in;
    println!("Using {} mel bins for feature extraction (NO NORMALIZATION)", num_mel_bins);
    let feat_extractor = ParakeetFeatureExtractor::new_with_config(num_mel_bins, false);

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

    // Streaming state for greedy decoder
    let mut pred_states = None;
    let mut last_token = model.config.blank_id as u32;
    let mut all_tokens = Vec::new();
    let mut decoded_count = 0;

    // Process audio in chunks
    println!("\n=== Streaming Transcription ===\n");
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

        // Run encoder with cache
        let encoder_out = model.encoder.forward_with_cache(&features, false, Some(&mut encoder_cache))?;

        // Check encoder output dimensions
        if chunk_num == 1 {
            let (_, enc_frames, _) = encoder_out.dims3()?;
            println!("(encoder produced {} frames) ", enc_frames);
        }

        // Run streaming greedy decode
        let (new_tokens, new_states, new_last_token) = model.greedy_decode_streaming(
            &encoder_out,
            pred_states,
            last_token,
        )?;

        // Update decoder state
        pred_states = new_states;
        last_token = new_last_token;

        // Accumulate tokens
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

        // Move to next chunk
        offset += chunk_size_samples;
    }

    println!("\n=== Final Statistics ===");
    println!("  Audio duration: {:.2}s", duration);
    println!("  Total chunks: {}", chunk_num);
    println!("  Total tokens: {}", all_tokens.len());
    println!("  First 20 token IDs: {:?}", &all_tokens[..all_tokens.len().min(20)]);

    // Full transcription
    let full_text = model.decode_tokens(&all_tokens)?;
    println!("\n=== Full Transcription ===");
    println!("{}", full_text);

    println!("\n=== Quality Comparison ===");
    println!("  NeMo reference: 225 tokens");
    println!("  Our streaming: {} tokens ({:.1}%)", all_tokens.len(), all_tokens.len() as f32 / 225.0 * 100.0);

    // Cache statistics
    if let Some(first_cache) = encoder_cache.attention_caches.first() {
        println!("\n=== Cache Statistics ===");
        println!("  Cached frames: {}", first_cache.num_cached);
        println!("  Max cache size: {}", first_cache.max_cache_size);
        println!("  Total frames processed: {}", encoder_cache.total_frames);
    }

    Ok(())
}
