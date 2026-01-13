# Beam Search Implementation Status

## Summary

Implemented basic beam search for TDT decoding, but jfk.wav still fails while dots.wav works.

## Root Cause Identified

Using `jrpython`, we discovered that **NeMo uses beam_size=2** (greedy_batch strategy), while our implementation used pure greedy (beam_size=1).

**NeMo Configuration:**
```
Strategy: greedy_batch
Beam size: 2
Max symbols: 10
```

## Implementation Progress

### What Works ✅
- **dots.wav**: 186 tokens with beam search (vs 140 greedy, vs 105 words NeMo)
- **NeMo baseline**: Perfect on both files (jrpython tool works!)

### What Doesn't Work ❌
- **jfk.wav**: 0 tokens with beam search (same issue as greedy)

## Key Issue

The jfk.wav problem is **not fixed by beam search**. The model produces blank tokens for all timesteps on jfk.wav, suggesting a deeper issue than just decoding strategy.

## NeMo vs Rust Comparison

| File | NeMo (jrpython) | Rust Beam Search | Rust Greedy |
|------|-----------------|------------------|-------------|
| dots.wav | 105 words ✅ | 186 tokens ✅ | 140 tokens ✅ |
| jfk.wav | 22 words ✅ | 0 tokens ❌ | 44 hallucinated tokens ❌ |

## Next Steps

The issue with jfk.wav is likely one of:
1. **Feature extraction mismatch** - Our mel features differ from NeMo's
2. **Model dtype issues** - BF16 vs F32 handling
3. **Encoder output differences** - Need to compare encoder outputs
4. **Blank bias or logits issue** - Model favoring blank incorrectly

**Recommended approach:**
1. Compare encoder outputs between NeMo and Rust on jfk.wav
2. Compare logits at first timestep
3. Verify feature extraction matches NeMo's exactly

## Files Modified

- `src/parakeet/transducer.rs` - Added `BeamHypothesis` struct and `beam_decode()` method
- `examples/transcribe_tdt.rs` - Updated to use beam search
- `nemo_baseline_tdt.py` - Added decoding config printing

## Useful Commands

```bash
# Test with beam search
cargo run --example transcribe_tdt --release -- jfk.wav
cargo run --example transcribe_tdt --release -- dots.wav

# Test with NeMo (baseline)
~/bin/jrpython nemo_baseline_tdt.py jfk.wav
~/bin/jrpython nemo_baseline_tdt.py dots.wav
```

## Resolution (FIXED!)

**Root cause identified:** Feature extraction mismatch!

The issue was NOT in beam search or decoding. The problem was in `src/parakeet/features.rs`:

1. **Wrong normalization:** We were doing per-utterance mean normalization, but NeMo uses **per-feature normalization** (each mel bin normalized to mean=0, std=1)
2. **Wrong window:** We were using periodic Hann window, but NeMo uses **symmetric Hann window**

### Fix Applied

- Changed to per-feature normalization (normalize each of 128 mel bins independently)
- Changed to symmetric Hann window: `w[n] = 0.5 - 0.5 * cos(2π*n/(N-1))`

### Results After Fix

| File | Before | After | Status |
|------|--------|-------|--------|
| dots.wav | 186 tokens | 187 tokens | ✅ Still works |
| jfk.wav | 0 tokens | 38 tokens | ✅ **FIXED! Perfect transcription!** |

**jfk.wav transcription:**
```
And so, my fellow Americans, ask not what your country can do for you,
ask what you can do for your country.
```

**Feature statistics now match NeMo:**
- Rust: mean=-0.000005, std=1.000, range=[-5.7, 10.1] ✅
- NeMo: mean=0.000000, std=0.999, range=[-5.6, 10.1] ✅

**Encoder statistics now match NeMo:**
- Rust: mean=-0.000032, std=0.0201, range=[-0.149, 0.152] ✅
- NeMo: mean=-0.000037, std=0.0206, range=[-0.152, 0.153] ✅

See `TDT_FIX_SUMMARY.md` for complete details.
