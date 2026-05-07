//! Kokoro TTS — 82M parameter text-to-speech synthesis.
//!
//! Architecture: StyleTTS 2 (decoder-only) + ISTFTNet vocoder.
//! - CustomALBERT for phoneme context encoding
//! - ProsodyPredictor for duration, F0, and energy
//! - ISTFTNet generator for direct waveform synthesis at 24kHz
//!
//! Input: IPA phoneme tokens + style vector (256-dim)
//! Output: 24kHz audio waveform

pub mod config;
pub mod model;
pub mod albert;
pub mod text_encoder;
pub mod prosody;
pub mod decoder;
pub mod phonemizer;
#[cfg(feature = "triton-metal")]
pub mod gpu_decoder;

pub use config::KokoroConfig;
pub use model::KokoroModel;
pub use phonemizer::{phonemize, Phonemizer};
