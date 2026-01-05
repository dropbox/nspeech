use anyhow::Result;
use candle_core::Device;

// Submodules
pub mod assets;
pub mod fast_conformer;
pub mod features;

// Re-export commonly used types and functions
pub use fast_conformer::{
    FastConformerConfig, ParakeetFastConformerCtc, QParakeetFastConformerCtc,
    load_parakeet_ctc_from_hf, load_parakeet_ctc_from_local,
    load_parakeet_ctc_from_gguf_hf, load_parakeet_ctc_from_gguf_local,
    VAD_CONFIG, VAD_MODEL, // Re-export VAD assets for use in silero module
};

// Conditional type alias based on quantized feature
#[cfg(feature = "quantized")]
pub type ParakeetCtc = QParakeetFastConformerCtc;

#[cfg(not(feature = "quantized"))]
pub type ParakeetCtc = ParakeetFastConformerCtc;
pub use features::{
    ParakeetFeatureExtractor, extract_features_from_samples, load_wav_as_features,
    load_python_encoder_input,
};

/// Trait for Parakeet CTC models (both quantized and regular)
pub trait ParakeetCtcModel {
    fn forward(&self, xs: &candle_core::Tensor, train: bool) -> Result<candle_core::Tensor>;
    fn greedy_decode(&self, logits: &candle_core::Tensor) -> Result<Vec<String>>;
    fn config(&self) -> &FastConformerConfig;
}

// Implement trait for regular model
impl ParakeetCtcModel for ParakeetFastConformerCtc {
    fn forward(&self, xs: &candle_core::Tensor, train: bool) -> Result<candle_core::Tensor> {
        self.forward(xs, train)
    }
    fn greedy_decode(&self, logits: &candle_core::Tensor) -> Result<Vec<String>> {
        self.greedy_decode(logits)
    }
    fn config(&self) -> &FastConformerConfig {
        &self.cfg
    }
}

// Implement trait for quantized model
impl ParakeetCtcModel for QParakeetFastConformerCtc {
    fn forward(&self, xs: &candle_core::Tensor, train: bool) -> Result<candle_core::Tensor> {
        self.forward(xs, train)
    }
    fn greedy_decode(&self, logits: &candle_core::Tensor) -> Result<Vec<String>> {
        self.greedy_decode(logits)
    }
    fn config(&self) -> &FastConformerConfig {
        &self.cfg
    }
}

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
pub fn transcribe_streaming_chunk<M: ParakeetCtcModel>(
    chunk_samples: &[f32],
    left_context: Option<&[f32]>,
    right_context: Option<&[f32]>,
    model: &M,
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
        model.config().feat_in,
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

/// Add punctuation and capitalization to raw CTC output with comma support
///
/// Parakeet CTC outputs lowercase text without punctuation. This function
/// applies basic rule-based punctuation restoration:
/// - Capitalizes first word and after periods
/// - Capitalizes common proper nouns (I, Americans, etc.)
/// - Adds commas between phrases (if comma_separated is true)
/// - Adds periods at sentence boundaries
///
/// When comma_separated=true, expects input like ["phrase one", "phrase two"]
/// joined with " , " to insert commas between natural pauses.
///
/// For production use, consider using a dedicated punctuation restoration model
/// like oliverguhr/fullstop-punctuation-multilang-large
pub fn add_punctuation_internal(text: &str, comma_separated: bool) -> String {
    if text.is_empty() {
        return String::new();
    }

    // Common proper nouns to capitalize (expandable)
    let proper_nouns = [
        "i", "americans", "america", "american", "god", "jesus", "christ",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
    ];

    // If comma_separated, split on " , " marker and process phrases
    if comma_separated {
        let phrases: Vec<&str> = text.split(" , ").collect();
        let mut result = String::new();
        let mut capitalize_next = true;

        for (phrase_idx, phrase) in phrases.iter().enumerate() {
            let words: Vec<&str> = phrase.split_whitespace().collect();

            for (i, word) in words.iter().enumerate() {
                if !result.is_empty() && (i > 0 || phrase_idx > 0) {
                    result.push(' ');
                }

                let processed_word = if capitalize_next || proper_nouns.contains(&word.to_lowercase().as_str()) {
                    // Capitalize first letter
                    let mut chars = word.chars();
                    match chars.next() {
                        None => String::new(),
                        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                    }
                } else {
                    word.to_string()
                };

                result.push_str(&processed_word);
                capitalize_next = false;
            }

            // Add comma after phrase (except last one)
            if phrase_idx < phrases.len() - 1 {
                result.push(',');
            }
        }

        // Ensure final period
        if !result.ends_with('.') && !result.ends_with('?') && !result.ends_with('!') {
            result.push('.');
        }

        return result;
    }

    // Original single-phrase logic
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut result = String::new();
    let mut capitalize_next = true;

    for (i, word) in words.iter().enumerate() {
        if i > 0 {
            result.push(' ');
        }

        let mut processed_word = if capitalize_next || proper_nouns.contains(&word.to_lowercase().as_str()) {
            // Capitalize first letter
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        } else {
            word.to_string()
        };

        // Check if this word should end a sentence
        // Look for sentence-ending patterns (simple heuristics)
        let next_word = words.get(i + 1).map(|s| *s);
        let should_add_period = match next_word {
            None => true, // End of text
            Some(next) => {
                // Add period before transition words that typically start new sentences
                matches!(
                    next.to_lowercase().as_str(),
                    "but" | "and" | "so" | "because" | "again" | "however" | "therefore"
                ) && processed_word.len() > 3 // Avoid short words
            }
        };

        if should_add_period && !processed_word.ends_with('.') {
            processed_word.push('.');
            capitalize_next = true;
        } else {
            capitalize_next = false;
        }

        result.push_str(&processed_word);
    }

    // Ensure final period
    if !result.ends_with('.') && !result.ends_with('?') && !result.ends_with('!') {
        result.push('.');
    }

    result
}

/// Add punctuation and capitalization to raw CTC output (backward compatibility)
pub fn add_punctuation(text: &str) -> String {
    add_punctuation_internal(text, false)
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
