use anyhow::Result;
use candle_core::Device;

// Submodules
pub mod assets;
pub mod fast_conformer;
pub mod features;
pub mod transducer;
pub mod streaming_transducer;
pub mod streaming_encoder;
#[cfg(feature = "triton-metal")]
pub mod triton_encoder;

// Re-export commonly used types and functions
pub use fast_conformer::{
    FastConformerConfig, ParakeetFastConformerCtc, QParakeetFastConformerCtc,
    load_parakeet_ctc_from_local, load_parakeet_ctc_from_gguf_local,
    VAD_CONFIG, VAD_MODEL, // Re-export VAD assets for use in silero module
};

// Re-export Transducer (TDT) types and functions
pub use transducer::{
    TransducerConfig, TransducerModel, PredictionNetwork, JointNetwork,
    TokenWithTimestamp, BeamStreamingState, StreamingBeamHypothesis,
    load_parakeet_tdt_from_local,
    load_parakeet_tdt_from_gguf_local, load_parakeet_tdt_from_gguf_mmap_local,
    load_parakeet_streaming_tdt_from_local,  // BF16 safetensors
    //load_parakeet_streaming_tdt_from_gguf_local,  // Cache-aware streaming variant (quantized)
    TDT_CONFIG, TDT_MODEL, TDT_MODEL_Q8_0_GGUF, TDT_MODEL_Q8_0_GGUF_MMAP,
    TDT_TOKENIZER, TDT_TOKENIZER_JSON, // Re-export TDT assets
    STREAMING_TDT_CONFIG, STREAMING_TDT_MODEL, // Streaming TDT assets
    //STREAMING_TDT_CONFIG, STREAMING_TDT_MODEL, STREAMING_TDT_MODEL_Q8_0_GGUF,  // Streaming TDT assets
    STREAMING_TDT_TOKENIZER, STREAMING_TDT_TOKENIZER_JSON,
};

// Re-export Streaming Transducer types
pub use streaming_transducer::{
    StreamingTransducer, StreamingConfig, StreamingState,
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
    fn dtype(&self) -> candle_core::DType;
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
    fn dtype(&self) -> candle_core::DType {
        self.dtype
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
    fn dtype(&self) -> candle_core::DType {
        // Quantized models internally compute in F32
        candle_core::DType::F32
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

    // Convert features to match model dtype (models can be F16 or F32)
    // F16 models need F16 inputs, quantized models handle their own conversion
    let features = features.to_dtype(model.dtype())?;

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

    // Common proper nouns to capitalize (very limited - not a comprehensive solution)
    // TODO: Replace with proper capitalization model like:
    //   - truecaser/recaser neural model
    //   - language model based capitalization
    //   - or use a different ASR model that outputs capitalization
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

/// Select the best available device for inference.
///
/// Auto-detects the fastest available backend:
/// - Apple Silicon (aarch64): Metal GPU (Candle + Triton kernels)
/// - Intel Mac (x86_64): CPU with fbgemm (Triton encoder uses its own Metal device)
/// - Windows: CPU with fbgemm (Triton D3D12 encoder uses its own GPU context)
/// - Linux: CPU with fbgemm
///
/// Set PARAKEET_DEVICE=cpu to force CPU (useful for debugging).
pub fn get_device() -> Result<Device> {
    if std::env::var("PARAKEET_DEVICE").as_deref() == Ok("cpu") {
        return Ok(Device::Cpu);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if metal_devices_available() {
            let result = std::panic::catch_unwind(|| Device::new_metal(0));
            if let Ok(Ok(device)) = result {
                return Ok(device);
            }
        }
    }

    Ok(Device::Cpu)
}

/// Check if any Metal GPU devices are accessible (false inside sandboxes that deny iokit-open).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn metal_devices_available() -> bool {
    #[link(name = "Metal", kind = "framework")]
    unsafe extern "C" {
        fn MTLCopyAllDevices() -> *const std::ffi::c_void;
    }
    #[link(name = "Foundation", kind = "framework")]
    unsafe extern "C" {
        fn CFRelease(cf: *const std::ffi::c_void);
    }
    // CFArray count
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFArrayGetCount(arr: *const std::ffi::c_void) -> isize;
    }
    unsafe {
        let arr = MTLCopyAllDevices();
        if arr.is_null() {
            return false;
        }
        let count = CFArrayGetCount(arr);
        CFRelease(arr);
        count > 0
    }
}
