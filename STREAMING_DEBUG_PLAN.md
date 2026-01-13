# Streaming Transcription Debug & Fix Plan

## Problem Summary

**Observation**: Streaming transcription starts well but degrades:
- First ~1s of each chunk: Good quality
- After silence (chunks 4-7): State corrupts
- Remaining chunks: Garbage tokens (Cyrillic, `<unk>`, blanks)
- Total: 179 tokens with garbage vs 140 clean tokens (non-streaming)

**Root Cause Hypothesis**: LSTM predictor state degrades within 2-second chunks, especially after processing silence.

## Phase 1: Verification & Diagnosis

### 1.1 Direct Comparison Tool

Create `examples/compare_streaming_quality.rs`:

```rust
// Compare streaming vs non-streaming token-by-token
// Output:
// - Divergence point (which chunk/token)
// - Token-level diff
// - WER (Word Error Rate) calculation
```

**What to measure**:
- Where does divergence start? (chunk number, token position)
- How many tokens match?
- What types of errors? (substitution, insertion, deletion)

### 1.2 Instrument the Decoder

Add debug logging to `decode_chunk()`:

```rust
// Log per-timestep:
// - Blank probability
// - Top-3 token predictions with probabilities
// - LSTM hidden state statistics (mean, std, max, min)
// - Whether token was emitted or blank

// Log per-chunk:
// - Total blanks vs non-blanks
// - Average token confidence
// - LSTM state drift metric (compare to fresh state)
```

**Create**: `examples/transcribe_tdt_streaming_debug.rs`

### 1.3 Chunk Isolation Test

Test each chunk independently (no state carry):

```rust
// For each chunk:
//   1. Reset LSTM state to blank
//   2. Process chunk alone
//   3. Record output
//
// Compare:
//   - Independent chunks vs continuous streaming
//   - Identify which chunks produce garbage even in isolation
```

### 1.4 Chunk Size Experiment

Test different chunk sizes systematically:

```rust
// Test: 0.5s, 1.0s, 1.5s, 2.0s, 3.0s chunks
// For each size, measure:
//   - Quality (WER)
//   - Latency
//   - Token count match to reference
//   - Garbage token percentage
```

## Phase 2: Fix Strategies (Priority Order)

### Strategy A: Adaptive State Reset (RECOMMENDED - Try First)

**Approach**: Reset LSTM when quality degrades, detected by low confidence.

```rust
// In decode_chunk():
let mut low_confidence_count = 0;

// After getting token:
let token_confidence = log_probs.max()?.to_scalar::<f32>()?;

if token_confidence < -5.0 {  // Low confidence threshold
    low_confidence_count += 1;

    if low_confidence_count > 10 {  // 10 consecutive low-confidence predictions
        // Reset LSTM state
        self.state.predictor_states = None;
        self.state.last_token = self.state.blank_id as u32;
        low_confidence_count = 0;
    }
} else {
    low_confidence_count = 0;
}
```

**Benefits**:
- Adaptive, not blind reset
- Resets only when model is uncertain
- Maintains state during good quality regions

**Test**:
```bash
cargo run --example transcribe_tdt_streaming --release -- dots.wav
# Should see fewer garbage tokens, better quality
```

### Strategy B: Silence Detection & State Reset

**Approach**: Detect silence chunks and reset state after them.

```rust
// After decode_chunk():
let blank_ratio = (enc_frames - chunk_tokens.len()) as f32 / enc_frames as f32;

if blank_ratio > 0.9 {  // 90% blanks = silence
    // Reset LSTM for next chunk
    self.state.predictor_states = None;
    self.state.last_token = self.state.blank_id as u32;
}
```

**Benefits**:
- Prevents state corruption during silence
- Fresh state when speech resumes
- Simple to implement

### Strategy C: Smaller Chunks

**Approach**: Reduce chunk size to 1.0s or 0.5s.

```rust
const CHUNK_SECONDS: f32 = 1.0;  // Instead of 2.0
const OVERLAP_SECONDS: f32 = 0.25; // Instead of 0.5
```

**Benefits**:
- Less time for LSTM to drift within chunk
- Lower latency
- More frequent opportunities for state management

**Trade-offs**:
- More chunks to process
- Less encoder context per chunk
- May reduce accuracy slightly

### Strategy D: Token Confidence Filtering

**Approach**: Only emit tokens above confidence threshold.

```rust
// In decode_chunk(), after getting token:
let token_confidence = log_probs.get(token as usize)?.to_scalar::<f32>()?;

if token_confidence > -3.0 {  // Confidence threshold
    decoded.push(token);
    self.state.last_token = token;
} else {
    // Low confidence: treat as blank
    self.state.last_token = token; // Update state
    break;
}
```

**Benefits**:
- Filters out low-confidence garbage
- Maintains state properly
- Simple heuristic

### Strategy E: Hybrid Approach

**Combine multiple strategies**:

```rust
// 1. Use smaller chunks (1.0s)
// 2. Reset on silence (>90% blanks)
// 3. Apply confidence filtering (>-3.0 threshold)
// 4. Reset on consecutive low confidence (>10 frames)
```

## Phase 3: Testing Protocol

### 3.1 Automated Quality Metrics

```rust
struct TranscriptionQuality {
    total_tokens: usize,
    garbage_tokens: usize,      // <unk>, Cyrillic, etc.
    blank_tokens: usize,
    word_error_rate: f32,       // vs reference
    character_error_rate: f32,
    chunk_quality: Vec<f32>,    // per-chunk quality
}
```

### 3.2 Test Suite

```bash
# Test on multiple audio files
cargo run --example test_streaming_quality -- dots.wav
cargo run --example test_streaming_quality -- jfk.wav
cargo run --example test_streaming_quality -- MLKDream_16k.wav

# Generate report:
# - Quality metrics per file
# - Chunk-by-chunk breakdown
# - Problem areas identified
```

### 3.3 Regression Tests

```rust
#[test]
fn test_streaming_no_cyrillic() {
    // Ensure no Cyrillic characters in English audio
    let result = transcribe_streaming("dots.wav")?;
    assert!(!result.contains(|c: char| matches!(c, '\u{0400}'..='\u{04FF}')));
}

#[test]
fn test_streaming_no_unk_tokens() {
    // Ensure no <unk> tokens
    let result = transcribe_streaming("dots.wav")?;
    assert!(!result.contains("<unk>"));
}

#[test]
fn test_streaming_quality_threshold() {
    // At least 70% quality vs non-streaming
    let streaming = transcribe_streaming("dots.wav")?;
    let reference = transcribe_non_streaming("dots.wav")?;
    let quality = calculate_quality(&streaming, &reference);
    assert!(quality > 0.7);
}
```

## Phase 4: Expected Outcomes

### Success Criteria

1. **No garbage tokens**: No Cyrillic, no `<unk>` markers
2. **>80% token count**: At least 112 tokens (vs 140 reference)
3. **Coherent text**: Readable, makes sense
4. **Low latency maintained**: <0.6x RTF
5. **Progressive output**: Text appears incrementally

### Fallback Position

If streaming quality can't reach 80%:

1. **Document limitations**: Clearly state streaming is experimental
2. **Recommend non-streaming**: For production/pre-recorded audio
3. **Use streaming selectively**: Only for true real-time use cases
4. **Consider alternatives**:
   - VAD-based segmentation (transcribe speech segments)
   - Sentence-level buffering (accumulate complete sentences)
   - Different model (FastConformer-CTC instead of Transducer)

## Implementation Priority

1. **Start with Strategy A** (Adaptive State Reset) - Easiest, most likely to help
2. **Add instrumentation** (Phase 1.2) - Understand what's happening
3. **Try Strategy B** (Silence Detection) - If A doesn't work
4. **Experiment with Strategy C** (Smaller Chunks) - If B doesn't work
5. **Implement testing suite** (Phase 3) - Validate improvements

## Next Steps

1. Create `examples/transcribe_tdt_streaming_debug.rs` with logging
2. Implement Strategy A (adaptive state reset)
3. Test on `dots.wav` and compare results
4. Iterate based on findings
