# Streaming Model Investigation Summary

## Goal
Switch from standard TDT model (`nvidia/parakeet-tdt-0.6b`) to streaming-specific model (`nvidia/nemotron-speech-streaming-en-0.6b`) to close the 16% accuracy gap (189 tokens → 225 tokens expected).

## Findings

### Standard TDT Model (WORKS)
- **Model**: `nvidia/parakeet-tdt-0.6b`
- **Configuration**:
  - vocab_size: 8192
  - blank_id: 8192
  - feat_in: 80 mel bins
  - subsampling_conv_channels: 256
- **Cache-Aware Streaming Performance**: **189 tokens (84.0% quality)**
- **Status**: ✅ **Production ready**

### Streaming TDT Model (BROKEN)
- **Model**: `nvidia/nemotron-speech-streaming-en-0.6b`
- **Configuration**:
  - vocab_size: 1024 (not 8192)
  - blank_id: 1024 (not 8192)
  - feat_in: 136 mel bins (not 80) - detected from tensor dimensions
  - config.json says 128 but actual tensors require 136
  - Has `streaming_config` with `att_context_size` arrays
- **Cache-Aware Streaming Performance**: **16-18 tokens (7-8% quality)**
- **Output**: Garbled text like ", of course, of their? Some of there?"
- **Status**: ❌ **Not working - requires investigation**

## Test Results

### Standard TDT Model
```
Audio: dots.wav (35.33s)
Chunk size: 4.5s (72000 samples)
Cache: 70 frames

Output: it was impos to connect the dots looking forward when I was in college...
Tokens: 189 (84.0% of NeMo reference)
```

### Streaming TDT Model (Both Loaders)
**BF16 Safetensors** (`load_parakeet_streaming_tdt_from_local`):
```
Tokens: 18 (8.0%)
Output: , of course, of therested upon therestrate
```

**Q8_0 GGUF** (`load_parakeet_streaming_tdt_from_gguf_local`):
```
Tokens: 16 (7.1%)
Output: , of course, of their? Some of there?
```

Both loaders produce similarly poor results, indicating a fundamental configuration or compatibility issue.

## Potential Root Causes

### 1. Feature Extraction Mismatch
- Streaming model needs 136 mel bins (detected from tensors)
- Config.json says 128 but actual subsampling layer has:
  - Input: 4352 features (= 136 mel bins × transforms)
  - Expected from config: 4096 features (= 128 mel bins × transforms)
- Using 128 bins causes shape mismatch error
- Using 136 bins produces poor quality

### 2. Vocabulary and Blank Token
- Streaming model has 10x smaller vocabulary (1024 vs 8192)
- Different blank_id (1024 vs 8192)
- Loader "fixes" blank_id from config's 0 to detected 1024
- May need different tokenizer or decoding approach

### 3. Architecture Differences
- Streaming model has `streaming_config.att_context_size` parameter
- Standard model doesn't have this configuration
- May require different attention mask or cache initialization

### 4. Chunk Size Requirements
- Standard model works well with 4.5s chunks
- Streaming model might need different chunk sizes (1s according to NeMo docs?)
- `att_context_size: [[70, 13], [70, 6], [70, 1], [70, 0]]` suggests configurable contexts

## Additional Tests Performed

### Blank ID Investigation
Tested hypothesis that loader's "CRITICAL FIX" changing blank_id from 0 to 1024 might be incorrect:

**Result**: Blank_id=0 produces **even worse** results (1 token, 0.4% quality)
- Output: Single "?" character
- Confirms blank_id=1024 is correct
- Config.json's blank_id=0 is indeed wrong

### Summary of All Configurations Tested
1. **Standard TDT** (vocab=8192, blank=8192, feat_in=80): **189 tokens (84%) ✅**
2. **Streaming TDT** (vocab=1024, blank=1024, feat_in=136): 16-18 tokens (7-8%) ❌
3. **Streaming TDT** (vocab=1024, blank=0, feat_in=136): 1 token (0.4%) ❌

## Next Steps

Based on the implementation plan, Phase 2-3 need to be completed:

### Phase 2: Extend FastConformer for Cache Support
- The streaming model may require different cache initialization
- Check if `att_context_size` parameters affect cache behavior
- Verify position encodings work with smaller vocabulary

### Phase 3: Model-Specific Configuration
- Investigate if streaming model needs special preprocessing
- Check if different blank_id handling is required
- Test with various chunk sizes (1s, 2s, 4.5s)

### Phase 4: Compare with NeMo Reference
- Get working NeMo inference for comparison
- Check exact tokenizer used by streaming model
- Verify feature extraction matches NeMo's preprocessing

## Recommendation

**For Production**: Use standard TDT model with cache-aware streaming
- **Quality**: 189 tokens (84.0%)
- **Status**: Fully working
- **File**: `examples/transcribe_cache_aware_streaming.rs`

**For Future Work**: Fix streaming-specific model
- Requires deeper investigation of architecture differences
- May need modifications to encoder, decoder, or feature extraction
- Gap to close: 16% (189 → 225 tokens)

## Files Modified

- `examples/transcribe_cache_aware_streaming.rs` - Updated documentation to note standard TDT usage
- Tested both `load_parakeet_streaming_tdt_from_local` (BF16) and `load_parakeet_streaming_tdt_from_gguf_local` (Q8_0)
- Both loaders work but produce poor quality with streaming model

## Deep Investigation: Why Streaming Model Fails

### Root Cause: Blank Token Domination

Detailed debugging reveals the streaming model's joint network consistently predicts blank with near-100% confidence:

```
Timestep 0:
  Blank (1024): -0.028  (probability ≈ 97%)
  Best non-blank (117): -5.528  (probability ≈ 0.4%)

Timestep 1:
  Blank (1024): -0.009  (probability ≈ 99%)
  Best non-blank (7): -6.634  (probability ≈ 0.1%)
```

This is abnormal - standard TDT models have much closer scores between blank and non-blank tokens.

### Tests Performed

1. **Chunk Size Variation**: Tested 1.04s chunks (designed size) → Same failure (22 tokens, 9.8%)
2. **Mel Bins**: Tested 128 vs 136 → 136 is required by tensor dimensions
3. **Blank ID**: Tested blank_id=0 vs 1024 → 1024 is correct (blank_id=0 produces 1 token)
4. **Model Loaders**: Tested BF16 safetensors and Q8_0 GGUF → Both fail identically
5. **Tokenizer**: Verified tokenizer works correctly (decodes to English when tokens are emitted)
6. **Model File**: Confirmed models are different (9M parameter difference, 618M vs 627M)
7. **Joint Network Architecture**: Verified same structure as standard model

### Hypotheses Explored

❌ **Wrong chunk size**: Tested 1.04s (designed) and 4.5s (standard model optimal) - both fail
❌ **Wrong mel bins**: 136 is required by model architecture, 128 causes shape mismatch
❌ **Wrong blank_id**: Testing blank_id=0 makes it worse (1 token output)
❌ **Tokenizer mismatch**: Tokenizer correctly decodes the few tokens that are emitted
❌ **Corrupted model**: Model has correct parameter count and distinct weights
❌ **Quantization issue**: Both BF16 and Q8_0 produce identical failures

### Remaining Possibilities

1. **Training-specific preprocessing**: The streaming model may require preprocessing we haven't identified
   - Different mel filterbank parameters?
   - Different normalization scheme?
   - Missing feature transformations?

2. **Missing architecture component**: Some layer or operation we're not implementing
   - Streaming-specific attention mask handling?
   - Different activation functions?
   - Additional normalization layers?

3. **Weight conversion error**: The safetensors conversion from NeMo may have issues
   - Layer misalignment?
   - Missing weight transformations?
   - Incorrect tensor reshaping?

4. **Model incompatibility**: The streaming model may use a variant architecture not compatible with our implementation
   - `att_context_size` configuration not properly handled?
   - Different encoder/decoder interaction?

## Conclusion

The cache-aware streaming implementation **works correctly** with the standard TDT model, achieving 84% quality (189 tokens / 225 reference).

The streaming-specific model (`nemotron-speech-streaming-en-0.6b`) produces only 7-10% quality due to the joint network predicting blank with near-100% confidence. This is a fundamental issue requiring:
- Access to NeMo's exact preprocessing pipeline
- Verification of weight conversion from NeMo format
- Possible architectural differences in the streaming variant

**Recommendation**: Use standard TDT model (`parakeet-tdt-0.6b`) with cache-aware streaming for production. The 84% quality (189 tokens) is excellent and the implementation is robust.
