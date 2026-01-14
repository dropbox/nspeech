# Cache-Aware Streaming: Root Cause Analysis

## TL;DR

**The cache-aware encoder implementation is fundamentally broken.** Even though it doesn't crash, the cached attention produces garbage encoder representations that destroy transcription quality.

## Quality Comparison

### Non-Streaming (Standard TDT Model - nvidia/parakeet-tdt-0.6b-v3)
| Decoder | Tokens | Quality | Transcription |
|---------|--------|---------|---------------|
| Beam search (size=2) | 187 | **100%** | Perfect, readable |
| Greedy decode | 150 | 80.2% | Readable, some words missing |

### Streaming (Standard TDT Model)
| Approach | Tokens | vs Greedy Non-Streaming | Transcription |
|----------|--------|------------------------|---------------|
| **Cache-aware greedy** | **17** | **88.7% loss** | Only dots |
| Overlap greedy | 15 | 90.0% loss | Only dots |

### Streaming (Streaming-Specific TDT Model - nvidia/nemotron-speech-streaming-en-0.6b)
| Approach | Tokens | vs Greedy Non-Streaming | Transcription |
|----------|--------|------------------------|---------------|
| **Cache-aware greedy** | **22** | **85.3% loss** | Fragments |
| Overlap greedy | 15 | 90.0% loss | Fragments |

## Key Findings

### 1. Greedy vs Beam Search: 19.8% Quality Loss (Acceptable)
- **Beam search**: 187 tokens, perfect transcription
- **Greedy**: 150 tokens, readable transcription
- **Verdict**: Greedy is acceptable for testing. The real issue is streaming.

### 2. Cache-Aware Streaming: 85-90% Quality Loss (BROKEN)
- **Non-streaming greedy**: 150 tokens, readable
- **Cache-aware streaming greedy**: 17-22 tokens, garbage
- **Verdict**: Cache implementation destroys encoder quality

### 3. Standard vs Streaming-Specific Model
- **Standard TDT + cache**: 17 tokens (worse)
- **Streaming TDT + cache**: 22 tokens (slightly better, but still broken)
- **Verdict**: Streaming-specific model is marginally better, but cache is still fundamentally broken

## Evidence of Encoder Corruption

### Token 7883 Repetition (Dots)
The standard TDT + cache-aware produces **only token 7883** (dots) for all 17 output tokens. This indicates:
- Encoder representations are near-zero or constant
- No temporal variation in encoder output
- Cache is causing representations to collapse

### Token 8192 (Out of Vocab)
Earlier test showed "Token 8192" being selected, which is beyond `vocab_size=8192`. This is impossible with correct softmax, indicating numerical instability or garbage logits from the encoder.

## What Works

1. ✓ **Cache structures**: Properly defined, initialized, no crashes
2. ✓ **forward_with_cache methods**: Implemented for all layers
3. ✓ **Contiguous tensors**: Fixed narrow() non-contiguous bug
4. ✓ **Position encoding formula**: Matches NeMo (2*total_frames-1)
5. ✓ **rel_shift implementation**: Identical to NeMo
6. ✓ **Predictor state**: Maintained across chunks
7. ✓ **Decoder logic**: greedy_decode_streaming works (proven by non-streaming greedy getting 150 tokens)

## What's Broken

### The Cache-Aware Attention Implementation

Despite matching NeMo's formulas and structure, the cache implementation produces encoder outputs that are unusable for decoding. Possible causes:

1. **Position encoding for cached attention**:
   - Formula matches NeMo: `2*total_frames-1`
   - But something about how positions interact with cached K/V is wrong
   - Queries at positions [56, 57, ..., 69] attending to keys at [0, 1, ..., 69]
   - Relative position bias might not be computed correctly for this setup

2. **Cache trimming side effects**:
   - When cache exceeds 70 frames, we trim oldest frames
   - This changes the absolute positions of cached keys
   - Position encodings might not account for this shift

3. **Attention mask missing**:
   - We pass `attn_mask=None` in all forward_with_cache calls
   - NeMo might use attention masks for streaming
   - Without masks, attention might attend to invalid positions

4. **Cache initialization**:
   - First chunk has no cache (cold start)
   - Later chunks use cache from previous chunks
   - There might be a discontinuity in how first vs later chunks are handled

## The Smoking Gun: Token 7883

The fact that the standard TDT outputs **only dots** with cache-aware streaming proves the encoder is producing degenerate representations. With proper encoder outputs, even greedy decode should produce varied tokens (as proven by non-streaming greedy getting 150 tokens).

This means the problem is NOT:
- ❌ Greedy vs beam decode
- ❌ Predictor state handling
- ❌ Decoder logic
- ❌ Model choice (standard vs streaming)

The problem IS:
- ✅ **Encoder representations with cache are garbage**

## Why Cache-Aware Is Slightly Better Than Overlap

Cache-aware (22 tokens) > Overlap (15 tokens), but both are terrible. This suggests:
- Cache provides SOME benefit over no context
- But the benefit is minimal because cache corrupts representations
- Without cache (overlap), encoder has no context → bad
- With corrupt cache, encoder has wrong context → still bad, slightly less bad

## Next Steps

### Option 1: Debug Encoder Outputs (Recommended)
Add logging to save:
1. Non-streaming encoder output for full audio → baseline
2. Cache-aware encoder output for each chunk → compare
3. Find the FIRST chunk where outputs diverge
4. Inspect attention weights, position encodings, K/V cache at that chunk

### Option 2: Test Without Position Encoding
Temporarily disable relative position encoding in cache-aware attention:
- Set position bias to zeros
- If quality improves, position encoding is the culprit
- If quality stays bad, something else is wrong

### Option 3: Test Cache with Single Chunk
Process entire audio as ONE chunk (no cache trimming):
- This eliminates cache trimming as a variable
- If quality improves, trimming logic is broken
- If quality stays bad, concatenation or position encoding is broken

### Option 4: Compare Intermediate Tensors with NeMo
Run same audio through NeMo's streaming model:
- Save encoder outputs per chunk
- Save attention weights
- Save K/V cache contents
- Compare byte-by-byte with our implementation

## Conclusion

The cache-aware streaming implementation has all the right pieces but produces catastrophically wrong results. The encoder with cache generates representations that cause the decoder to output mostly blank tokens (dots).

**Root cause**: Something in the cached attention mechanism (likely position encoding or cache state management) causes encoder representations to collapse or become constant across time.

**Proof**: Standard TDT works perfectly in non-streaming (187 tokens) but outputs only dots in cache-aware streaming (17 tokens), with identical decoder logic.

**Recommendation**: Either fix the cache-aware attention through detailed debugging, or pivot to a simpler streaming approach (e.g., fixed-context windows without explicit caching).
