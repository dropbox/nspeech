# Cache-Aware Streaming Debug Session

## Summary

Attempted to fix cache-aware streaming implementation, but quality remains stuck at ~10% (same as simple overlap streaming).

## Bugs Fixed

### 1. Non-Contiguous Cache Bug (CRITICAL)
**Location**: `src/parakeet/streaming_encoder.rs:86-90`

**Problem**: The `narrow()` operation in cache trimming creates a non-contiguous tensor view. This caused Metal matmul errors and crashes on chunk 7.

**Fix**: Added `.contiguous()` after narrowing:
```rust
self.keys = Some(keys.narrow(2, frames_to_remove, total_frames - frames_to_remove)?.contiguous()?);
```

**Result**: No more crashes! But quality didn't improve.

### 2. rel_shift Narrowing Bug
**Location**: `src/parakeet/fast_conformer.rs:434` (old line 431)

**Problem**: Original rel_shift was narrowing to query length internally, but for streaming we need the full position matrix.

**Fix**: Removed the final `.narrow(D::Minus1, 0, t)?` from rel_shift, letting the caller narrow to key length.

**Result**: No crash, but quality didn't improve.

## Quality Results

| Approach | Tokens | Quality | Status |
|----------|--------|---------|--------|
| **NeMo Reference** | 225 | 100% | Target baseline |
| **Standard TDT (non-streaming)** | 150 | 66.7% | Model baseline |
| **Cache-aware (reported earlier)** | 66 | 29.3% | Lost - wasn't committed |
| **Simple overlap streaming** | 23 | 10.2% | Comparison |
| **Current cache-aware (with fixes)** | 22 | 9.8% | **SAME AS OVERLAP!** |

## What We Tried

### Approach 1: Update Cache First, Use Trimmed K/V
- Update cache with `cache.append()` (concatenates + trims)
- Get trimmed K/V from cache
- Compute position for min(cached + current, max_cache)
- **Result**: 22 tokens (9.8%)

### Approach 2: Concatenate First, Trim After Attention (NeMo-style)
- Concatenate cached K/V with new K/V
- Attend over FULL sequence
- Store concatenated tensors, trim for next chunk
- Compute position for full cached + current
- **Result**: 23 tokens (10.2%) - actually WORSE!

### Approach 3: Various Cloning Strategies
- Clone k/v when getting from cache
- Clone k/v when storing to cache
- **Result**: No improvement

## Root Cause Analysis

The fact that cache-aware streaming gives THE SAME quality as simple overlap streaming (both ~10%) suggests:

1. **Cache isn't helping**: Either position encoding is wrong, or attention isn't benefiting from the cache
2. **Position encoding might be the issue**: Even with fixes to match NeMo's approach, quality didn't improve
3. **Predictor state might be the issue**: The LSTM predictor state needs proper handling across chunks
4. **The "29% quality" might have been a measurement error**: We never committed that code, so we can't verify

## Key Differences from NeMo

### What We Matched
- ✓ Position encoding computed for `2*total_frames-1`
- ✓ rel_shift implementation (identical to NeMo)
- ✓ Narrowing after rel_shift to key length
- ✓ Cache concatenation and trimming logic

### What Might Be Different
- ? Predictor (LSTM) state handling across chunks
- ? Decoder token accumulation strategy
- ? Attention mask usage (we pass None)
- ? Dropout behavior in streaming mode

## Next Steps to Debug

### Option 1: Compare Intermediate Outputs
- Add debug logging to save encoder outputs per chunk
- Compare with NeMo's outputs for the same audio
- Find WHERE the outputs diverge

### Option 2: Test Without Caching
- Run attention with cache disabled (fresh context every chunk)
- If quality is still ~10%, the issue is elsewhere (predictor, decoder)
- If quality improves, the cache is definitely the problem

### Option 3: Test Different Cache Sizes
- Try cache_size=28 (2 chunks) instead of 70
- Try cache_size=140 (10 chunks)
- See if quality changes at all

### Option 4: Check Predictor State
- The predictor LSTM state might not be maintained correctly
- Add logging to verify state is actually being passed between chunks
- Compare with non-streaming predictor behavior

## Code Status

### What Works
- ✓ Cache structures defined and initialized
- ✓ forward_with_cache methods implemented for all layers
- ✓ Encoder processes chunks without crashing
- ✓ Predictor state maintained across chunks (according to example code)
- ✓ Contiguous tensors after trimming

### What Doesn't Work
- ✗ Quality stuck at 10% (same as no cache)
- ✗ Can't reproduce the reported 29% quality
- ✗ No clear path forward without more investigation

## Hypothesis

The most likely issue is **position encoding doesn't work correctly for streaming attention**. Even though we match NeMo's formula, there might be a subtle difference in how relative positions are computed when the query and key lengths differ (14 queries, 70 keys).

Alternative hypothesis: The predictor or decoder is the bottleneck, not the encoder. Even if the encoder produces perfect representations, the decoder might not be able to use them effectively in streaming mode.

## Files Modified

1. `src/parakeet/fast_conformer.rs`:
   - Added `forward_with_cache` to MultiHeadSelfAttention
   - Added `forward_with_cache` to ConvModule
   - Added `forward_with_cache` to FastConformerBlock
   - Added `forward_with_cache` to FastConformerEncoder
   - Modified `rel_shift` to not narrow internally

2. `src/parakeet/streaming_encoder.rs`:
   - Added `.contiguous()` to cache trimming (lines 86, 90)

## Recommendation

Stop debugging the cache implementation and instead:
1. Add comprehensive debug logging
2. Compare outputs with NeMo step-by-step
3. Or try a completely different approach (e.g., attention masking instead of explicit caching)

The cache-aware approach is theoretically sound and matches NeMo's code, but something fundamental isn't working. Without better debugging tools, we're just guessing.
