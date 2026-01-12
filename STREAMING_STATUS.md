# Streaming Transcription Status

## What We've Built

### ✅ Completed

1. **Native Rust Tokenizer** (DONE)
   - Removed Python dependency for token decoding
   - Supports both tokenizer.json and tokenizer.model (SentencePiece)
   - Integrated directly into TransducerModel

2. **Streaming Infrastructure** (DONE)
   - `StreamingTransducer` with state management
   - `StreamingEncoderCache` with attention/conv caching types
   - Overlapping chunk processing
   - LSTM predictor state maintenance across chunks

3. **Practical Streaming Example** (WORKING)
   - Processes audio in configurable chunks (default: 2s with 0.5s overlap)
   - Emits incremental results
   - Achieves 0.54x real-time factor (faster than real-time!)
   - `cargo run --example transcribe_tdt_streaming --release -- audio.wav`

### ⚠️ In Progress

**Chunked Processing Quality**
- Overlapping chunks process successfully
- LSTM state carries across chunks
- Output quality degraded due to state management issues
- **Root cause**: LSTM state from overlapping regions creates confusion

## Architecture

### Current Implementation

```
Audio Stream
  ↓
Overlapping Chunks (2s chunks, 0.5s overlap)
  ↓
Feature Extraction (per chunk)
  ↓
Encoder (FastConformer - processes full chunk)
  ↓
Transducer Decoder (maintains LSTM state)
  ↓
Incremental Text Output
```

### Key Components

1. **StreamingTransducer** (`src/parakeet/streaming_transducer.rs`)
   - Manages audio buffering and overlap
   - Maintains LSTM predictor states
   - Tracks decoded tokens
   - `process_features()` - Processes feature chunks
   - `decode_text()` - Converts accumulated tokens to text

2. **StreamingEncoderCache** (`src/parakeet/streaming_encoder.rs`)
   - `AttentionCache` - K/V cache for self-attention (infrastructure ready)
   - `ConvCache` - Padding state for convolution (infrastructure ready)
   - Sliding window support for bounded memory usage

3. **Configuration**
   ```rust
   StreamingConfig {
       chunk_samples: 32000,      // 2s at 16kHz
       overlap_samples: 8000,     // 0.5s overlap
       emit_partial: true,
   }
   ```

## Performance

Current streaming example on dots.wav (35s audio):
- **Total time**: 18.92s
- **Real-time factor**: 0.54x (faster than real-time!)
- **Chunk processing**: ~0.77s per 2s chunk
- **Latency**: ~2.5s (buffering + processing)

## What's Needed for Production

### 1. State Management Fixes (High Priority)

**Problem**: LSTM state from overlapping regions causes token repetition and quality issues.

**Solution Options**:
a. **Reset LSTM on overlaps** - Clear state at overlap boundaries
b. **Skip overlapping tokens** - Only emit tokens from non-overlapping regions
c. **Larger chunks, less overlap** - Reduce overlap percentage

**Implementation**:
```rust
// Option B: Track overlap and skip redundant tokens
if in_overlap_region {
    // Process but don't emit tokens
    let _ = self.decode_chunk(&encoder_out, overlap_frames)?;
} else {
    // Emit tokens from non-overlapping region
    let tokens = self.decode_chunk(&encoder_out, new_frames)?;
    self.state.tokens.extend(&tokens);
}
```

### 2. True Frame-Level Streaming (Lower Priority)

For <100ms latency streaming with 40-80ms chunks:

**Requirements**:
1. **Attention Caching**
   - Modify `MultiHeadSelfAttention::forward()` to accept K/V cache
   - Cache keys/values from past chunks
   - New queries attend to cached + current keys

2. **Convolution State**
   - Maintain padding buffer for depthwise convolution
   - Kernel size = 9, need last 8 frames of context

3. **Position Encoding**
   - Handle relative positions across chunks
   - Accumulated position offset per chunk

**Implementation Complexity**: High
- Requires modifying core FastConformer encoder
- Changes to ~5-6 functions in fast_conformer.rs
- Risk of breaking non-streaming inference

**Alternative**: Use current overlapping approach with optimized parameters

### 3. Recommended Next Steps

**For Production Use (Easier Path)**:
1. Fix LSTM state handling in overlapping regions
2. Tune chunk size and overlap:
   - Try 3s chunks with 1s overlap
   - Or 4s chunks with 0.5s overlap
3. Add confidence thresholding for token emission
4. Test on various audio lengths and conditions

**For True Streaming (Harder Path)**:
1. Implement attention caching in `MultiHeadSelfAttention`
2. Add convolution state management in `ConvModule`
3. Handle position encodings across chunks
4. Create streaming-specific encoder forward pass
5. Extensive testing to ensure quality matches non-streaming

## Usage Examples

### Current Streaming

```bash
# Overlapping chunks (2s with 0.5s overlap)
cargo run --example transcribe_tdt_streaming --release -- audio.wav
```

### Non-Streaming (Reference)

```bash
# Full audio processing (best quality)
cargo run --example transcribe_tdt --release -- audio.wav
```

## Comparison

| Approach | Latency | Quality | Complexity |
|----------|---------|---------|------------|
| Non-streaming | Full audio | 100% | Low |
| Overlapping chunks (current) | ~2.5s | ~60-70% | Medium |
| Frame-level streaming (future) | <100ms | ~95% | High |

## Key Insights

1. **FastConformer uses global attention** - Each frame attends to ALL other frames in the sequence. This makes true streaming challenging without caching.

2. **RNN-T decoder is naturally streaming** - The transducer with LSTM predictor is designed for incremental processing. This part works well!

3. **Overlapping chunks work** - Processing overlapping chunks is a practical approach that achieves good performance, just needs quality tuning.

4. **Trade-offs are inevitable** - Streaming ASR always trades some quality for latency. The key is finding the right balance for your use case.

## Code Locations

- Streaming transducer: `src/parakeet/streaming_transducer.rs`
- Encoder caching infrastructure: `src/parakeet/streaming_encoder.rs`
- Streaming example: `examples/transcribe_tdt_streaming.rs`
- Core encoder: `src/parakeet/fast_conformer.rs`
- Transducer decoder: `src/parakeet/transducer.rs`

## References

- NVIDIA Parakeet: https://github.com/NVIDIA/NeMo
- RNN-T paper: https://arxiv.org/abs/1211.3711
- FastConformer: https://arxiv.org/abs/2305.05084
