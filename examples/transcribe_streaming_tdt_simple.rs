/// Simple streaming transcription using Streaming TDT model
///
/// This approach:
/// - Processes audio in chunks
/// - Maintains predictor LSTM state across chunks (critical for transducer)
/// - Uses overlapping encoder chunks (not true cache-aware, but functional)
/// - Provides real-time transcription output

use anyhow::Result;
use speech::parakeet::{
    get_device, load_parakeet_streaming_tdt_from_local,
    ParakeetFeatureExtractor,
};
use candle_core::DType;

fn main() -> Result<()> {
    let device = get_device()?;
    println!("Device: {:?}\n", device);

    // Load streaming TDT model
    println!("Loading Streaming TDT model...");
    let model = load_parakeet_streaming_tdt_from_local(".cache/parakeet-streaming-tdt", &device)?;
    println!("✓ Model loaded");
    println!("  Vocab size: {}", model.config.vocab_size);
    println!("  Blank ID: {}\n", model.config.blank_id);

    // Load audio
    println!("Loading audio...");
    let mut reader = hound::WavReader::open("dots.wav")?;
    let audio_samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;

    let duration = audio_samples.len() as f64 / 16000.0;
    println!("✓ Audio loaded: {:.2}s ({} samples)\n", duration, audio_samples.len());

    // Streaming configuration
    // NeMo streaming model uses att_context_size=[70, 13] = 1190ms chunks
    // After 8x subsampling: 13 encoder frames * 80ms = 1040ms
    // At mel level (10ms frames): 1040ms / 10ms = 104 frames
    // But we want the mel input size, so: 13 frames * 8 (subsampling) = 104 mel frames
    // 104 mel frames * 160 samples/frame (hop) = 16,640 samples per chunk

    let chunk_size_samples = 17920; // 1.12s chunks (matches NeMo's [70,13] config)
    let overlap_samples = 4480;     // 280ms overlap for encoder continuity

    println!("=== Streaming Configuration ===");
    println!("  Chunk size: {} samples ({:.2}s)", chunk_size_samples, chunk_size_samples as f64 / 16000.0);
    println!("  Overlap: {} samples ({:.2}s)", overlap_samples, overlap_samples as f64 / 16000.0);
    println!();

    // Feature extractor
    let num_mel_bins = model.encoder.cfg.feat_in;
    let feat_extractor = ParakeetFeatureExtractor::new(num_mel_bins);

    // Streaming state
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

        // Extract features for this chunk
        let features = feat_extractor.extract_to_tensor(chunk, &device)?;

        // Convert to model dtype
        let features = if !device.is_cpu() {
            features.to_dtype(DType::BF16)?
        } else {
            features
        };

        // Run encoder on chunk
        let encoder_out = model.encoder.forward(&features, false)?;

        // Run streaming decode with maintained predictor state
        let (new_tokens, new_states, new_last_token) = model.greedy_decode_streaming(
            &encoder_out,
            pred_states,
            last_token,
        )?;

        // Update state for next chunk
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

        // Move to next chunk (with overlap for continuity)
        offset += chunk_size_samples - overlap_samples;
    }

    println!("\n\n=== Final Statistics ===");
    println!("  Audio duration: {:.2}s", duration);
    println!("  Total chunks: {}", chunk_num);
    println!("  Total tokens: {}", all_tokens.len());
    println!("  First 10 token IDs: {:?}", &all_tokens[..all_tokens.len().min(10)]);

    // Full transcription
    let full_text = model.decode_tokens(&all_tokens)?;
    println!("\n=== Full Transcription ===");
    println!("{}", full_text);

    Ok(())
}
