# Cache-Aware Streaming Implementation Status

## Summary

We implemented cache-aware streaming for the nvidia/nemotron-speech-streaming-en-0.6b model with K/V attention caching and convolution state caching. The implementation compiles and runs, but quality is currently at 29.3% of the reference (66 tokens vs 225 expected).

## What Was Implemented

### 1. Attention K/V Caching (`MultiHeadSelfAttention::forward_with_cache`)
- Concatenates cached K/V from previous chunks with new K/V projections
- Queries attend to full sequence (cached + current)
- Cache is updated after each chunk
- **Status**: ✓ Working, but relative position encoding needs fixing

### 2. Convolution State Caching (`ConvModule::forward_with_cache`)
- Prepends cached padding to input before convolution
- Extracts output corresponding to current chunk
- Updates cache with rightmost frames for next chunk
- **Status**: ✓ Working correctly

### 3. Encoder Streaming (`FastConformerEncoder::forward_with_cache`)
- Processes audio chunks with per-layer caches
- Computes position encodings for full sequence (cached + current)
- Passes caches through all transformer blocks
- **Status**: ✓ Working, position encoding calculation fixed

### 4. Quantized Variants
- Implemented `QMultiHeadSelfAttention::forward_with_cache`
- Implemented `QFastConformerBlock::forward_with_cache`
- Implemented `QFastConformerEncoder::forward_with_cache`
- **Status**: ✓ Complete (mirrors regular implementations)

### 5. Streaming Example (`transcribe_cache_aware_streaming.rs`)
- Initializes caches with proper dimensions
- Processes audio in ~1s chunks (16640 samples)
- Maintains predictor LSTM state across chunks
- Uses max 70 frames of cached context
- **Status**: ✓ Running, but output quality poor

## Performance Results

| Approach | Tokens | Quality | Notes |
|----------|--------|---------|-------|
| **NeMo Reference** | 225 | 100% | Target baseline |
| **Standard TDT (non-streaming)** | 150 | 66.7% | Good quality, full context |
| **Cache-aware streaming** | 66 | 29.3% | Caching works, quality issue |
| **Simple streaming (overlap)** | 23 | 10.2% | Overlap-based, poor quality |

## Known Issues

### 1. Relative Position Encoding (Critical)
**Problem**: The `rel_shift` operation is designed for self-attention (same length Q/K) but doesn't work correctly for cached attention (short Q, long K).

**Current behavior**:
- rel_shift outputs `[B, H, t, t]` for t queries
- For cached attention, we need `[B, H, t, total_frames]` for total_frames keys
- Currently padding with zeros, meaning cached frames get zero relative position bias

**Impact**: Major quality degradation. Without proper relative position encoding, the model can't properly attend to cached frames.

**Attempted fixes**:
1. ✗ Adjust narrow operation to take k_len instead of t - still pads with zeros
2. ✗ Use zeros_like for all position bias - quality drops to 5.3%
3. ✗ Using broken position bias (zeros for cached frames) - 29.3% quality

**Correct solution needed**: Implement proper relative position indexing for cached attention. Options:
- Compute position bias matrix differently for cached vs current frames
- Use absolute position embeddings instead of relative for streaming
- Investigate how NeMo handles this in their streaming implementation

### 2. Position Encoding Calculation
**Fixed**: Encoder now correctly computes position encodings for `total_frames` (cached + current) instead of just current chunk.

### 3. Tensor Contiguity
**Fixed**: K/V concatenation now explicitly calls `.contiguous()` to avoid Metal matmul errors.

## Configuration

### Chunk Size
- **Samples**: 16640 (1.04s at 16kHz)
- **Mel frames**: 104 (after feature extraction)
- **Encoder frames**: 13 (after 8x subsampling)
- **Rationale**: Matches NeMo's att_context_size=[70,13] where 13 is chunk size

### Cache Size
- **Max frames**: 70 encoder frames (5.6s of context)
- **Rationale**: Matches NeMo's att_context_size=[70,13] where 70 is left context
- **Behavior**: Cache maintains most recent 70 frames, trimming older frames when exceeded

## Next Steps

### Priority 1: Fix Relative Position Encoding
This is the blocking issue for quality. Options to investigate:
1. Study NeMo's streaming attention implementation
2. Try absolute position embeddings instead of relative
3. Implement proper rel_shift for cross-attention-style setup
4. Compute position bias directly without rel_shift

### Priority 2: Validate Other Components
- Check if predictor LSTM state is maintained correctly across chunks
- Verify blank token handling (blank_id=1024)
- Test with different chunk sizes to see if quality changes

### Priority 3: Optimize
- Profile encoder_cache overhead
- Measure memory usage vs non-streaming
- Test Real-Time Factor (RTF) for real-time viability

## Code Locations

### Cache Structures
- `src/parakeet/streaming_encoder.rs`: AttentionCache, ConvCache, StreamingEncoderCache

### Implementation Files
- `src/parakeet/fast_conformer.rs`: MultiHeadSelfAttention, ConvModule, FastConformerEncoder
  - Lines ~334-446: MultiHeadSelfAttention::forward_with_cache
  - Lines ~517-587: ConvModule::forward_with_cache
  - Lines ~1437-1510: FastConformerEncoder::forward_with_cache
  - Lines ~743-854: QMultiHeadSelfAttention::forward_with_cache
  - Lines ~969-1014: QFastConformerBlock::forward_with_cache
  - Lines ~1053-1085: QFastConformerEncoder::forward_with_cache

### Example
- `examples/transcribe_cache_aware_streaming.rs`: Full streaming demo with cache initialization

## Git Status
**NOTE**: fast_conformer.rs changes were accidentally reverted. Need to re-implement:
- MultiHeadSelfAttention::forward_with_cache
- QMultiHeadSelfAttention::forward_with_cache
- ConvModule::forward_with_cache
- FastConformerBlock::forward_with_cache
- QFastConformerBlock::forward_with_cache
- FastConformerEncoder::forward_with_cache (with fixed position encoding)
- QFastConformerEncoder::forward_with_cache

## References

- NeMo model: `nvidia/nemotron-speech-streaming-en-0.6b`
- Architecture: FastConformer-RNNT with streaming support
- Config: att_context_size=[70,13], subsampling_factor=8
- Paper: "FastConformer: Local Augmentation and Lookahead Bias for Faster and Better Streaming ASR"
