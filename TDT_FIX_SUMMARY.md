# TDT Model Fix Summary

## Problem

The Parakeet TDT (Transducer) model was failing on jfk.wav, producing 0 tokens (all blanks) despite working on dots.wav. NeMo baseline transcribed both files perfectly, indicating our implementation had a bug.

## Root Cause

**Feature extraction mismatch between Rust and NeMo preprocessor.**

### Issue 1: Wrong Normalization

**Our implementation:** Per-utterance mean normalization
```rust
// WRONG: Subtract global mean from all features
let mean = feats.iter().sum::<f32>() / feats.len() as f32;
for val in feats.iter_mut() {
    *val -= mean;
}
```

**NeMo uses:** Per-feature normalization (`normalize='per_feature'`)
```rust
// CORRECT: Normalize each mel bin independently to mean=0, std=1
for each feature dimension f:
    mean[f] = average across all time frames
    std[f] = std dev across all time frames
    for each time frame t:
        features[t][f] = (features[t][f] - mean[f]) / std[f]
```

### Issue 2: Wrong Window Function

**Our implementation:** Periodic Hann window
```rust
// WRONG: Periodic window used by PyTorch default
w[n] = 0.5 - 0.5 * cos(2π * n / N)
```

**NeMo uses:** Symmetric Hann window
```rust
// CORRECT: Symmetric window (NeMo default)
w[n] = 0.5 - 0.5 * cos(2π * n / (N-1))
```

## Impact on Features

### Before Fix (jfk.wav)
- Features: mean=-0.000014, std=**1.588869** (59% too high!), range=[-3.125, 5.219]
- Encoder: mean=-0.000113, std=**0.009697** (53% too low!), range=[-0.051, 0.100]
- Result: Encoder output had wrong variance → model produced all blanks

### After Fix (jfk.wav)
- Features: mean=-0.000005, std=**1.000038** ✅, range=[-5.656, 10.125] ✅
- Encoder: mean=-0.000032, std=**0.020089** ✅, range=[-0.149, 0.152] ✅
- Result: Perfect transcription!

### NeMo Baseline (for comparison)
- Features: mean=0.000000, std=**0.999087**, range=[-5.603, 10.102]
- Encoder: mean=-0.000037, std=**0.020613**, range=[-0.152, 0.153]

## Fix

**File:** `src/parakeet/features.rs`

### Change 1: Per-Feature Normalization
```rust
// Apply per-feature normalization (NeMo's normalize='per_feature')
// Normalize each mel bin (feature dimension) independently to mean=0, std=1
let num_features = self.feature_size;

if frames > 0 {
    // Calculate mean and std for each feature dimension
    let mut means = vec![0.0f32; num_features];
    let mut stds = vec![0.0f32; num_features];

    // Calculate means
    for t in 0..frames {
        for f in 0..num_features {
            means[f] += feats[t * num_features + f];
        }
    }
    for mean in means.iter_mut() {
        *mean /= frames as f32;
    }

    // Calculate standard deviations
    for t in 0..frames {
        for f in 0..num_features {
            let diff = feats[t * num_features + f] - means[f];
            stds[f] += diff * diff;
        }
    }
    for std in stds.iter_mut() {
        *std = (*std / frames as f32).sqrt();
        if *std < 1e-10 {
            *std = 1.0; // Avoid division by zero
        }
    }

    // Normalize: (x - mean) / std for each feature
    for t in 0..frames {
        for f in 0..num_features {
            let idx = t * num_features + f;
            feats[idx] = (feats[idx] - means[f]) / stds[f];
        }
    }
}
```

### Change 2: Symmetric Hann Window
```rust
/// Hann window, symmetric: w[n]=0.5-0.5*cos(2*pi*n/(N-1))
/// This matches NeMo's default (symmetric Hann window)
fn hann_window(n: usize) -> Vec<f32> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = (n - 1) as f32;  // Changed from n to (n-1)
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * (i as f32) / denom).cos())
        .collect()
}
```

## Results

### jfk.wav Transcription

**Before Fix:** 0 tokens (all blanks)

**After Fix (Beam Search, beam_size=2):**
```
And so, my fellow Americans, ask not what your country can do for you,
ask what you can do for your country.
```
✅ **Perfect transcription!** (38 tokens)

**Expected:**
```
And so, my fellow Americans: ask not what your country can do for you—
ask what you can do for your country
```

**Comparison:** Words are 100% correct. Only minor punctuation differences (comma vs colon/dash).

### dots.wav Transcription

**Before Fix:** 186 tokens (worked, but slightly off)

**After Fix:** 187 tokens ✅ Perfect transcription of Steve Jobs' "connecting the dots" speech

## NeMo Preprocessor Configuration

Discovered via `inspect_nemo_preprocessor.py`:

```
Preprocessor type: AudioToMelSpectrogramPreprocessor

Key settings:
  sample_rate: 16000
  n_fft: 512
  win_length: 400
  hop_length: 160
  nfilt: 128 (mel bins)
  preemph: 0.97
  normalize: per_feature  ← KEY!
  log: True
  log_zero_guard_type: add
  log_zero_guard_value: 5.960464477539063e-08 (2^-24)
  mag_power: 2.0
  dither: 1e-05 (not yet implemented in Rust)
  window: symmetric Hann ← KEY!
```

## Remaining Differences

1. **Dithering:** NeMo adds `1e-05` random noise to waveform (not implemented in Rust yet)
2. **Small numerical differences** due to BF16 precision on GPU and minor implementation details

These remaining differences are minor and don't affect transcription quality.

## Lessons Learned

1. **Always compare with reference implementation** - Running NeMo baseline revealed the bug
2. **Feature extraction is critical** - Small differences in preprocessing cascade through the model
3. **Normalization matters** - Per-feature vs per-utterance normalization had huge impact
4. **Tools like jrpython are invaluable** - Being able to run NeMo in a container made debugging possible

## Commands

### Test Transcription
```bash
# jfk.wav (now works!)
cargo run --example transcribe_tdt --release -- jfk.wav

# dots.wav (still works!)
cargo run --example transcribe_tdt --release -- dots.wav
```

### Compare with NeMo Baseline
```bash
# NeMo transcription
~/bin/jrpython nemo_baseline_tdt.py jfk.wav

# Compare encoder outputs
~/bin/jrpython compare_encoder_outputs.py jfk.wav
cargo run --example extract_encoder_output --release -- jfk.wav
```

## Files Modified

- `src/parakeet/features.rs` - Fixed normalization and window function
- `examples/extract_encoder_output.rs` - Created for debugging (needs `use candle_core::IndexOp;`)
- `compare_encoder_outputs.py` - Created to extract NeMo encoder outputs
- `inspect_nemo_preprocessor.py` - Created to examine NeMo preprocessor config

## Related Documents

- `BEAM_SEARCH_STATUS.md` - Documents that beam search didn't fix jfk.wav (issue was in features)
- `NEMO_VS_RUST_TDT.md` - Documents NeMo vs Rust comparison that led to this investigation
