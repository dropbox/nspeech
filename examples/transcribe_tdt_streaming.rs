/// Streaming transcription example using Parakeet TDT (Transducer) model
///
/// This example demonstrates pseudo-streaming transcription by:
/// 1. Processing audio in overlapping chunks
/// 2. Emitting results as chunks complete
/// 3. Showing progressive transcription output
///
/// Note: True streaming with FastConformer requires attention caching, which is
/// not yet implemented. This example shows a practical approach for incremental
/// transcription of buffered audio.
///
/// Usage:
///   cargo run --example transcribe_tdt_streaming --release -- dots.wav
///   cargo run --example transcribe_tdt_streaming --release -- MLKDream_16k.wav

use anyhow::Result;
use speech::parakeet::{
    get_device, load_parakeet_tdt_from_local,
    ParakeetFeatureExtractor,
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        eprintln!("\nThis example demonstrates streaming-style transcription with Parakeet TDT.");
        eprintln!("Audio is processed in chunks, with results emitted incrementally.");
        return Ok(());
    }

    let audio_path = &args[1];

    println!("Parakeet TDT Streaming Transcription");
    println!("=====================================\n");
    println!("Audio: {}\n", audio_path);

    // Get device
    let device = get_device()?;

    // Load TDT model
    println!("Loading TDT model...");
    let mut model = load_parakeet_tdt_from_local(".cache/parakeet-tdt", &device)?;
    println!("  Model: nvidia/parakeet-tdt-0.6b-v3");
    println!("  Architecture: Transducer (RNN-T)\n");

    // Load tokenizer
    println!("Loading tokenizer...");
    model.load_tokenizer(".cache/parakeet-tdt")?;
    println!("  Tokenizer loaded\n");

    // Load audio
    println!("Loading audio...");
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    println!("  Sample rate: {} Hz", spec.sample_rate);
    println!("  Channels: {}", spec.channels);
    println!("  Bits per sample: {}\n", spec.bits_per_sample);

    // Read all samples
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    // Convert to mono if stereo
    let samples = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect()
    } else {
        samples
    };

    println!("  Audio samples: {}", samples.len());
    println!("  Duration: {:.2}s\n", samples.len() as f32 / spec.sample_rate as f32);

    // Feature extractor
    let feat_extractor = ParakeetFeatureExtractor::new(128);  // TDT uses 128 mel bins

    // Process audio in chunks of ~3 seconds
    // This is large enough for the encoder to have sufficient context
    const CHUNK_SECONDS: f32 = 3.0;
    const SAMPLES_PER_CHUNK: usize = (16000.0 * CHUNK_SECONDS) as usize;  // 3s at 16kHz
    let total_chunks = (samples.len() + SAMPLES_PER_CHUNK - 1) / SAMPLES_PER_CHUNK;

    println!("Streaming configuration:");
    println!("  Chunk size: {:.1}s ({} samples)", CHUNK_SECONDS, SAMPLES_PER_CHUNK);
    println!("  Total chunks: {}\n", total_chunks);
    println!("=== STREAMING TRANSCRIPTION ===\n");

    let start_time = std::time::Instant::now();
    let mut all_text = Vec::new();

    for (chunk_idx, chunk_samples) in samples.chunks(SAMPLES_PER_CHUNK).enumerate() {
        let chunk_start = std::time::Instant::now();

        // Extract features for this chunk
        let features = feat_extractor.extract_to_tensor(chunk_samples, &device)?;

        // Convert to BF16 for GPU
        let features = if !device.is_cpu() {
            features.to_dtype(candle_core::DType::BF16)?
        } else {
            features
        };

        // Run encoder
        let encoder_out = model.encoder.forward(&features, false)?;

        // Greedy decode
        let tokens = model.greedy_decode(&encoder_out)?;

        // Decode to text
        let text = model.decode_tokens(&tokens)?;

        let chunk_time = chunk_start.elapsed();

        // Print progress
        print!("[Chunk {}/{}] ", chunk_idx + 1, total_chunks);
        print!("{:.1}s processed in {:.2}s ", chunk_samples.len() as f32 / 16000.0, chunk_time.as_secs_f32());
        println!("({} tokens)", tokens.len());
        println!("  → {}\n", text.trim());

        all_text.push(text);
    }

    let total_time = start_time.elapsed();

    println!("================================\n");

    // Combine all chunks
    let final_text = all_text.join(" ");

    println!("=== FINAL TRANSCRIPTION ===");
    println!("{}", final_text.trim());
    println!("===========================\n");

    println!("✓ Streaming transcription complete!");
    println!("\nPerformance:");
    println!("  Total time: {:.2}s", total_time.as_secs_f32());
    println!("  Audio duration: {:.2}s", samples.len() as f32 / spec.sample_rate as f32);
    println!("  Real-time factor: {:.2}x", total_time.as_secs_f32() / (samples.len() as f32 / spec.sample_rate as f32));
    println!("\nStreaming approach:");
    println!("  - Processes audio in {:.1}s chunks", CHUNK_SECONDS);
    println!("  - Emits results as each chunk completes");
    println!("  - Suitable for buffered/near-realtime applications");
    println!("\nFor true streaming with frame-level latency:");
    println!("  - Requires attention caching (not yet implemented)");
    println!("  - Would process ~40ms chunks with cached attention");

    Ok(())
}
