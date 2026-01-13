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

## Conclusion

Beam search implementation is functional but incomplete. The real issue is that our TDT model produces incorrect outputs on jfk.wav regardless of decoding strategy, while NeMo works perfectly.

This suggests the problem is in **feature extraction, encoder, or model loading**, not in decoding.
