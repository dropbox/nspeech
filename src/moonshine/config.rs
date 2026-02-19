//! Moonshine V2 Streaming model configuration.
//!
//! Corresponds to the `streaming_config.json` produced by `scripts/download_moonshine.py`.

use serde::Deserialize;

/// Frontend (audio embedder) configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct FrontendConfig {
    /// Model dimension (output of linear projection).
    pub d_model: usize,
    /// Conv1 output channels (2 * d_model).
    pub c1: usize,
    /// Conv2 output channels (d_model).
    pub c2: usize,
    /// Conv kernel size (5).
    pub kernel_size: usize,
    /// Conv stride (2).
    pub stride: usize,
}

/// Full Moonshine V2 streaming model configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct MoonshineConfig {
    // Encoder
    pub encoder_dim: usize,
    pub encoder_intermediate_size: usize,
    pub encoder_num_heads: usize,
    pub encoder_num_kv_heads: usize,
    pub encoder_head_dim: usize,
    pub encoder_hidden_act: String,
    pub encoder_num_layers: usize,

    // Decoder
    pub decoder_dim: usize,
    pub decoder_intermediate_size: usize,
    pub decoder_num_heads: usize,
    pub decoder_num_kv_heads: usize,
    pub decoder_head_dim: usize,
    pub decoder_hidden_act: String,
    pub decoder_num_layers: usize,

    // Vocabulary and tokens
    pub vocab_size: usize,
    pub bos_id: usize,
    pub eos_id: usize,
    pub pad_id: usize,
    pub max_position_embeddings: usize,

    // Audio
    pub frame_len: usize,
    pub sample_rate: usize,

    // RoPE
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,

    // Sliding window attention [left, right] per layer
    pub sliding_windows: Vec<[usize; 2]>,

    // Frontend
    pub frontend: FrontendConfig,

    // Weight tying
    pub tie_word_embeddings: bool,
}

impl MoonshineConfig {
    /// Rotary embedding dimension = head_dim * partial_rotary_factor.
    pub fn rotary_dim(&self) -> usize {
        (self.decoder_head_dim as f64 * self.partial_rotary_factor) as usize
    }
}
