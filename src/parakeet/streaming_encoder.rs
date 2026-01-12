/// Streaming-aware FastConformer encoder with attention and convolution caching
///
/// Enables true streaming ASR by caching:
/// 1. Keys/Values (K/V) from past chunks for self-attention
/// 2. Convolution padding state for depthwise convolution
/// 3. Accumulated position encodings for relative attention
///
/// This allows processing audio in small chunks (~40-80ms) while maintaining
/// the full context needed for accurate transcription.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::collections::VecDeque;

use super::fast_conformer::FastConformerEncoder;

/// Cache for a single attention layer
#[derive(Clone)]
pub struct AttentionCache {
    /// Cached keys: [B, past_frames, num_heads, head_dim]
    pub keys: Option<Tensor>,
    /// Cached values: [B, past_frames, num_heads, head_dim]
    pub values: Option<Tensor>,
    /// Number of past frames cached
    pub num_cached: usize,
}

impl AttentionCache {
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            num_cached: 0,
        }
    }

    /// Append new keys/values to cache
    pub fn append(&mut self, new_keys: Tensor, new_values: Tensor) -> Result<()> {
        let (_, new_frames, _, _) = new_keys.dims4()?;

        self.keys = Some(if let Some(ref cached_keys) = self.keys {
            Tensor::cat(&[cached_keys, &new_keys], 1)?
        } else {
            new_keys
        });

        self.values = Some(if let Some(ref cached_values) = self.values {
            Tensor::cat(&[cached_values, &new_values], 1)?
        } else {
            new_values
        });

        self.num_cached += new_frames;
        Ok(())
    }

    /// Get all cached keys (past + current)
    pub fn get_keys(&self) -> Option<&Tensor> {
        self.keys.as_ref()
    }

    /// Get all cached values (past + current)
    pub fn get_values(&self) -> Option<&Tensor> {
        self.values.as_ref()
    }

    /// Clear the cache
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.num_cached = 0;
    }
}

/// Cache for convolution module padding
#[derive(Clone)]
pub struct ConvCache {
    /// Cached padding: [B, d_model, padding_size]
    /// Stores the last (kernel_size - 1) frames for causal convolution
    pub padding: Option<Tensor>,
}

impl ConvCache {
    pub fn new() -> Self {
        Self { padding: None }
    }

    pub fn reset(&mut self) {
        self.padding = None;
    }
}

/// Complete streaming cache for encoder
pub struct StreamingEncoderCache {
    /// Attention caches for each layer
    pub attention_caches: Vec<AttentionCache>,
    /// Convolution caches for each layer
    pub conv_caches: Vec<ConvCache>,
    /// Total frames processed so far
    pub total_frames: usize,
}

impl StreamingEncoderCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            attention_caches: (0..num_layers).map(|_| AttentionCache::new()).collect(),
            conv_caches: (0..num_layers).map(|_| ConvCache::new()).collect(),
            total_frames: 0,
        }
    }

    pub fn reset(&mut self) {
        for cache in &mut self.attention_caches {
            cache.reset();
        }
        for cache in &mut self.conv_caches {
            cache.reset();
        }
        self.total_frames = 0;
    }
}

/// Streaming encoder wrapper
pub struct StreamingEncoder {
    encoder: FastConformerEncoder,
    cache: StreamingEncoderCache,
    /// Maximum frames to cache (prevents unbounded memory growth)
    max_cache_frames: usize,
}

impl StreamingEncoder {
    pub fn new(encoder: FastConformerEncoder, max_cache_frames: usize) -> Self {
        let num_layers = encoder.cfg.num_layers;
        Self {
            encoder,
            cache: StreamingEncoderCache::new(num_layers),
            max_cache_frames,
        }
    }

    /// Process a chunk of audio features with caching
    ///
    /// # Arguments
    /// * `features` - Audio features [B, chunk_frames, feat_dim]
    /// * `is_final` - Whether this is the final chunk (affects padding)
    ///
    /// # Returns
    /// Encoder output [B, chunk_output_frames, d_model]
    pub fn process_chunk(&mut self, features: &Tensor, is_final: bool) -> Result<Tensor> {
        let (batch_size, chunk_frames, _feat_dim) = features.dims3()?;

        // For now, we'll use the encoder directly without caching
        // True streaming requires modifying the encoder's forward pass to:
        // 1. Accept and update caches
        // 2. Compute attention over cached + new frames
        // 3. Handle convolution state

        // This is a placeholder that processes the full chunk
        // TODO: Implement incremental processing with cache updates
        let encoder_out = self.encoder.forward(features, false)?;

        // Update total frames
        self.cache.total_frames += chunk_frames;

        // If cache exceeds max size, trim oldest frames (sliding window)
        if self.cache.total_frames > self.max_cache_frames {
            self.trim_cache()?;
        }

        Ok(encoder_out)
    }

    /// Trim cache to keep only recent frames (sliding window)
    fn trim_cache(&mut self) -> Result<()> {
        let frames_to_remove = self.cache.total_frames - self.max_cache_frames;

        for attn_cache in &mut self.cache.attention_caches {
            if let Some(ref keys) = attn_cache.keys {
                let (_, total_frames, _, _) = keys.dims4()?;
                if total_frames > frames_to_remove {
                    attn_cache.keys = Some(keys.narrow(1, frames_to_remove, total_frames - frames_to_remove)?);
                }
            }
            if let Some(ref values) = attn_cache.values {
                let (_, total_frames, _, _) = values.dims4()?;
                if total_frames > frames_to_remove {
                    attn_cache.values = Some(values.narrow(1, frames_to_remove, total_frames - frames_to_remove)?);
                }
            }
            attn_cache.num_cached = (attn_cache.num_cached as isize - frames_to_remove as isize).max(0) as usize;
        }

        self.cache.total_frames = self.max_cache_frames;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.cache.reset();
    }

    pub fn get_cache(&self) -> &StreamingEncoderCache {
        &self.cache
    }
}

/// Configuration for streaming with look-ahead
#[derive(Debug, Clone)]
pub struct StreamingAttentionConfig {
    /// Number of past frames to attend to (for causal attention)
    pub left_context_frames: usize,

    /// Number of future frames to attend to (look-ahead)
    /// 0 = strictly causal, >0 = allows some latency for better accuracy
    pub right_context_frames: usize,

    /// Chunk size for processing (in frames, after subsampling)
    pub chunk_size: usize,
}

impl Default for StreamingAttentionConfig {
    fn default() -> Self {
        Self {
            left_context_frames: 256,  // ~2s of context at 125fps (after 8x subsampling)
            right_context_frames: 16,  // ~128ms look-ahead
            chunk_size: 50,            // ~400ms chunks
        }
    }
}

// Note: To complete the streaming implementation, we need to:
// 1. Modify MultiHeadSelfAttention to accept and use cached K/V
// 2. Modify ConvModule to maintain padding state
// 3. Handle position encodings across chunks
// 4. Update FastConformerBlock forward pass to use caches
//
// This requires significant changes to the core encoder. For now, this
// module provides the infrastructure and can be gradually implemented.
