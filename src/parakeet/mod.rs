use anyhow::Result;
use candle_core::Device;

// Submodules
pub mod assets;
pub mod fast_conformer;
pub mod features;

// Re-export commonly used types and functions
pub use fast_conformer::{
    FastConformerConfig, ParakeetFastConformerCtc,
    load_parakeet_ctc_from_hf, load_parakeet_ctc_from_local,
    load_parakeet_ctc_from_gguf_hf, load_parakeet_ctc_from_gguf_local,
    VAD_CONFIG, VAD_MODEL, // Re-export VAD assets for use in silero module
};
pub use features::{
    ParakeetFeatureExtractor, extract_features_from_samples, load_wav_as_features,
    load_python_encoder_input,
};

/// Transcribe audio with optional left/right context for streaming
///
/// This implements context-based streaming for smooth transcription boundaries.
/// When context is provided, the full audio window is processed but only the
/// middle chunk (excluding context) is decoded and returned.
///
/// # Arguments
/// * `chunk_samples` - The audio samples to transcribe (middle portion)
/// * `left_context` - Optional left context from previous chunk (for stability)
/// * `right_context` - Optional right context / look-ahead (for stability)
/// * `model` - The Parakeet model
/// * `device` - The device to run on
///
/// # Returns
/// The transcription text for the chunk (not including context portions)
pub fn transcribe_streaming_chunk(
    chunk_samples: &[f32],
    left_context: Option<&[f32]>,
    right_context: Option<&[f32]>,
    model: &ParakeetFastConformerCtc,
    device: &Device,
) -> Result<String> {
    // Build full input with context
    let left_ctx = left_context.unwrap_or(&[]);
    let right_ctx = right_context.unwrap_or(&[]);

    let mut full_input = Vec::with_capacity(left_ctx.len() + chunk_samples.len() + right_ctx.len());
    full_input.extend_from_slice(left_ctx);
    full_input.extend_from_slice(chunk_samples);
    full_input.extend_from_slice(right_ctx);

    // Extract features from full input (with context)
    let features = extract_features_from_samples(
        &full_input,
        model.cfg.feat_in,
        device,
    )?;

    // Run inference on full input
    let logits = model.forward(&features, false)?;

    // If we have context, slice logits to get only the middle chunk portion
    let chunk_logits = if left_ctx.is_empty() && right_ctx.is_empty() {
        // No context, use all logits
        logits
    } else {
        use candle_core::IndexOp;

        // Calculate frame indices for the chunk portion
        // Parakeet uses 8x subsampling, with ~10ms frames -> 80 samples per output frame
        let samples_per_frame = 80; // Approximate (8x subsampling * 10ms hop)
        let (_, total_frames, _) = logits.dims3()?;

        let left_context_frames = (left_ctx.len() + samples_per_frame - 1) / samples_per_frame;
        let chunk_frames = (chunk_samples.len() + samples_per_frame - 1) / samples_per_frame;
        let end_frame = (left_context_frames + chunk_frames).min(total_frames);

        if end_frame > left_context_frames {
            logits.i((.., left_context_frames..end_frame, ..))?
        } else {
            // Not enough frames, return empty
            return Ok(String::new());
        }
    };

    // Decode the chunk (or full input if no context)
    let transcriptions = model.greedy_decode(&chunk_logits)?;

    Ok(transcriptions.first().cloned().unwrap_or_default())
}

/// Select the best available device for inference
/// Prefers Metal on macOS if PARAKEET_DEVICE env var is not set to "cpu"
/// Falls back to CPU with Accelerate framework
pub fn get_device() -> Result<Device> {
    // Allow forcing CPU mode via environment variable
    if std::env::var("PARAKEET_DEVICE").as_deref() == Ok("cpu") {
        println!("Using CPU (forced by PARAKEET_DEVICE=cpu)");
        return Ok(Device::Cpu);
    }

    #[cfg(target_os = "macos")]
    {
        // Note: Metal acceleration has some known issues with certain tensor operations
        // in Candle. If you encounter errors, set PARAKEET_DEVICE=cpu
        match Device::new_metal(0) {
            Ok(device) => {
                println!("Using Metal GPU acceleration");
                println!("  (If you encounter errors, try: PARAKEET_DEVICE=cpu)");
                return Ok(device);
            }
            Err(e) => {
                println!("Metal not available ({}), using CPU with Accelerate", e);
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    println!("Using CPU");

    #[cfg(target_os = "macos")]
    println!("Using CPU with Accelerate framework");

    Ok(Device::Cpu)
}
