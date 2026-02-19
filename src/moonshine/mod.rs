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

pub mod config;
pub mod decoder;
pub mod encoder;
pub mod frontend;
pub mod model;

pub use config::MoonshineConfig;
pub use model::MoonshineModel;
