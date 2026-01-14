# Normalization Investigation: Streaming Model Blank Domination

## Investigation Summary

Investigated whether the blank domination issue (streaming model producing only 7-10% quality) was caused by feature normalization mismatch.

## Key Findings

### 1. NeMo Config Has Multiple Errors

**From model_config.yaml in .nemo file:**
```yaml
preprocessor:
  features: 128           # ❌ WRONG - model actually uses 136
  normalize: NA           # No normalization
  window_size: 0.025
  window_stride: 0.01
  n_fft: 512
  preemph: 0.97
  dither: 1.0e-05

decoder:
  vocab_size: 1024

# Config also has blank_id: 0 which is wrong (should be 1024)
```

**Actual Model Requirements (from weight dimensions):**
- `encoder.pre_encode.out.weight`: [1024, 4352]
- 4352 / 32 = **136 mel bins** (not 128!)
- Blank ID: **1024** (last position in vocab, not 0)

### 2. Normalization Test Results

Tested streaming model with BOTH normalization approaches:

| Approach | Tokens | Quality | Output |
|----------|--------|---------|--------|
| WITHOUT normalization (config='NA') | 48 | 21.3% | Gibberish: ", I want to say, here, of course..." |
| WITH per-feature normalization | 65 | 28.9% | Gibberish: ", therana's oftening theories..." |
| **NeMo Reference** | **225** | **100%** | Clean transcription |

**Conclusion**: Per-feature normalization is slightly BETTER (65 vs 48 tokens), but BOTH produce gibberish far below the 225-token reference.

### 3. Normalization in NeMo Code

From `/Users/jhansen/src/NeMo/nemo/collections/asr/parts/preprocessing/features.py`:

```python
def normalize_batch(x, seq_len, normalize_type):
    ...
    if normalize_type == "per_feature":
        # Apply per-feature (per mel bin) normalization
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2), x_mean, x_std
    elif normalize_type == "all_features":
        # Normalize entire spectrogram
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1), x_mean, x_std
    else:
        # normalize='NA' or unrecognized -> NO normalization
        return x, x_mean, x_std
```

So `normalize: NA` means **skip normalization entirely**.

### 4. Implementation Changes Made

Modified `ParakeetFeatureExtractor` to support configurable normalization:

```rust
pub struct ParakeetFeatureExtractor {
    pub normalize: bool,  // NEW: true = per-feature norm, false = none
    ...
}

impl ParakeetFeatureExtractor {
    pub fn new(feature_size: usize) -> Self {
        Self::new_with_config(feature_size, true)  // Default: WITH normalization
    }

    pub fn new_with_config(feature_size: usize, normalize: bool) -> Self {
        // Can now control normalization
    }
}
```

### 5. Comparison with Standard TDT Model

| Model | Config Mel Bins | Actual Mel Bins | Works? |
|-------|----------------|-----------------|--------|
| **Standard TDT** | 128 | 128 (4096/32) | ✅ YES (189 tokens, 84%) |
| **Streaming TDT** | 128 | 136 (4352/32) | ❌ NO (48-65 tokens, 21-29%) |

The standard TDT model's config is CORRECT and the model works fine. The streaming model's config has errors AND doesn't work.

## Root Cause: NOT Normalization

The normalization mismatch is **NOT** the root cause of blank domination:
- Without normalization: 48 tokens (21.3%)
- With normalization: 65 tokens (28.9%)
- Expected: 225 tokens (100%)

Both produce incomprehensible output, suggesting a more fundamental issue.

## Remaining Hypotheses

### 1. Mel Bin Configuration Error
- Model expects 136 bins (unusual number)
- Why 136? Could be:
  - Training bug/accident
  - Special padding (128 + 8 = 136)
  - Different mel filterbank configuration
  - **We correctly detect and use 136, but still get poor results**

### 2. Weight Conversion Error
- Model converted from .nemo to safetensors
- Conversion might have:
  - Misaligned layers
  - Incorrect tensor reshaping
  - Missing transformations

### 3. Preprocessing Pipeline Mismatch
- Dither: NeMo adds 1e-5 noise during training only (not inference)
- Mel filterbank: Both use Slaney normalization
- Log scale: Both use log10
- Pre-emphasis: Both use 0.97
- **All major settings match**

### 4. Architecture Differences
- Streaming model has `att_context_size` configuration
- May require special handling we're not implementing
- Cache-aware attention might need different initialization

### 5. Tokenizer Mismatch
- Streaming: 1024 vocab, 48K tokenizer.json
- Standard: 8192 vocab, 403K tokenizer.json
- **But tokenizer mismatch wouldn't explain too few tokens, just wrong decoding**

## Next Steps

1. **Compare with NeMo directly**: Run streaming model in NeMo (if dependencies can be fixed) to get exact preprocessing pipeline
2. **Weight verification**: Check if safetensors conversion from .nemo is correct
3. **Architecture review**: Examine if `att_context_size` requires special encoder modifications
4. **Try different model**: Test if a freshly downloaded streaming model works better
5. **Mel filterbank deep dive**: Investigate if 136 bins requires special frequency range or mel scale

## Files Modified

- `src/parakeet/features.rs`: Added `normalize: bool` parameter
- `examples/test_streaming_no_norm.rs`: Test without normalization
- `examples/test_streaming_both_norm.rs`: Compare both approaches

## Conclusion

Normalization configuration mismatch (config says 'NA', we used 'per_feature') was investigated but is **NOT the root cause**. The streaming model produces gibberish regardless of normalization approach, at only 21-29% quality vs 100% expected.

The issue remains unsolved and likely requires deeper investigation into:
- Weight conversion process
- Architectural differences in streaming variant
- Exact preprocessing pipeline used during training
