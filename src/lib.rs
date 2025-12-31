// Parakeet CTC speech recognition module
pub mod parakeet;

// Re-export commonly used functions for examples
pub use parakeet::{
    extract_features_from_samples, get_device, load_parakeet_ctc_from_gguf_local,
    load_parakeet_ctc_from_gguf_hf, load_parakeet_ctc_from_hf, load_parakeet_ctc_from_local,
    load_wav_as_features, FastConformerConfig, ParakeetFastConformerCtc,
};

// Node.js NAPI bindings (only compiled when 'napi' feature is enabled)
#[cfg(feature = "napi")]
pub mod napi_bindings;
