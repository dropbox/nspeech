//! Moonshine V2 Streaming ASR model implementation.
//!
//! A 265M parameter encoder-decoder transformer for speech recognition.
//! Processes raw audio waveforms directly (no mel spectrograms needed).
//!
//! Architecture:
//! - Frontend: Conv layers that process raw audio → features
//! - Encoder: 14-layer transformer with sliding window attention
//! - Decoder: 14-layer transformer with RoPE, cross-attention, GLU MLP
//!
//! Based on the HuggingFace implementation:
//! `UsefulSensors/moonshine-streaming-medium`

use crate::embed_asset;

embed_asset!(pub MOONSHINE_CONFIG,              "moonshine-config.json");
embed_asset!(pub MOONSHINE_TOKENIZER,           "moonshine-tokenizer.json");
embed_asset!(pub MOONSHINE_MODEL_Q8_0_GGUF_MMAP,"moonshine_q8_0.gguf");

pub mod config;
pub mod decoder;
pub mod encoder;
pub mod frontend;
pub mod model;
#[cfg(feature = "triton-metal")]
pub mod triton_encoder;
#[cfg(feature = "triton-d3d12")]
pub mod triton_d3d12_encoder;

pub use config::MoonshineConfig;
pub use model::MoonshineModel;
pub use model::MoonshineStream;
