# Cache-Aware Streaming: Accuracy Gap Analysis

## Current Performance

| Method | Tokens | vs NeMo Reference | Notes |
|--------|--------|------------------|-------|
| **NeMo streaming reference** | **225** | **100%** | Beam search, streaming model |
| **Our cache-aware streaming** | **189** | **84.0%** | Greedy decode, standard TDT, 4.5s chunks |
| Non-streaming baseline (greedy) | 150 | 66.7% | Standard TDT, greedy decode |
| Non-streaming baseline (beam=2) | 187 | 83.1% | Standard TDT, beam search |

**Gap to close: 36 tokens (16% improvement needed)**

## Missing Pieces

### 1. Beam Search in Streaming Mode (HIGH IMPACT)

**Current state**: We use greedy decode in streaming (`greedy_decode_streaming`)
**NeMo approach**: Uses beam search (beam_size=2 typically)

**Expected improvement**: 20-30% token increase
- Non-streaming: greedy=150 → beam=187 (+24.7%)
- Applying same ratio: 189 → ~236 tokens (+47 tokens)
- **This alone would exceed NeMo's 225 tokens!**

**Implementation needed**:
```rust
pub fn beam_decode_streaming(
    &self,
    encoder_out: &Tensor,
    beam_size: usize,
    beam_state: Option<BeamStreamingState>,
) -> Result<(Vec<u32>, BeamStreamingState)>
```

**Key differences from non-streaming beam search**:
- Must maintain beam hypotheses across chunks (not just predictor state)
- Each hypothesis has its own predictor LSTM state
- Beam pruning happens at chunk boundaries
- Need to track multiple hypothesis states, not just the best one

**Complexity**: Moderate - adapt existing `beam_decode()` to accept/return beam state

---

### 2. Streaming-Specific Model (MEDIUM IMPACT)

**Current state**: Using `nvidia/parakeet-tdt-0.6b-v3` (standard TDT model)
**NeMo reference**: Uses `nvidia/nemotron-speech-streaming-en-0.6b` (streaming-specific)

**Potential improvement**: 5-10%
- Streaming model may have different training optimizations
- Weights may be tuned for cache-aware attention
- May handle small chunks better

**Trade-offs**:
- Streaming model is optimized for 1.04s chunks (but we found those don't work well)
- Standard model works well with our 4.5s chunk approach
- **Recommendation**: Test after beam search, may not be necessary

---

### 3. Cache Size Optimization (LOW-MEDIUM IMPACT)

**Current state**: Fixed 70 frames (5.6s of context)
**Analysis needed**: Test different cache sizes

**Experiments to run**:
```bash
# Smaller cache (less memory, faster, but less context)
--cache-size 50  # 4.0s context

# Larger cache (more context, may improve quality)
--cache-size 100 # 8.0s context

# Match encoder frames (more balanced ratio)
--cache-size 56  # 4.5s context (matches chunk size)
```

**Expected improvement**: 3-7%
- Current ratio: 70 cached : 45 current (1.56:1)
- Balanced ratio: 56 cached : 45 current (1.24:1)
- More context: 100 cached : 45 current (2.22:1)

**Hypothesis**: Balanced ratio (cache ≈ chunk) might work better

---

### 4. Chunk Size Fine-Tuning (LOW IMPACT)

**Current state**: 4.5s chunks (~45 encoder frames)
**Tested alternatives**:
- 4.4s: 167 tokens (88.4%)
- 3.5s: 105 tokens (70%)

**Analysis**:
- 4.5s is near-optimal for greedy decode
- With beam search, might be able to use smaller chunks
- **Recommendation**: Test after beam search implementation

---

### 5. Incremental Decoding Optimization (LOW IMPACT)

**Current state**: We call `decode_tokens_incremental()` but still decode from scratch
**Potential improvement**: ~2-5% speedup (not quality)

**Details**:
- `decode_tokens_incremental()` decodes only new tokens
- But we're already doing this efficiently
- Minimal quality impact

---

## Implementation Priority

### Phase 1: Beam Search for Streaming (CRITICAL)
**Expected gain**: +37-47 tokens (20-25%)
**Estimated effort**: 4-6 hours
**Risk**: Low (adapt existing beam_decode)

**Steps**:
1. Define `BeamStreamingState` struct to hold beam hypotheses
2. Implement `beam_decode_streaming()` method
3. Update `transcribe_cache_aware_streaming.rs` to use beam search
4. Test on dots.wav

**Success criteria**: ≥220 tokens (97.8% of NeMo reference)

---

### Phase 2: Cache Size Tuning (QUICK WIN)
**Expected gain**: +7-14 tokens (3-7%)
**Estimated effort**: 1-2 hours
**Risk**: Very low (just parameter tuning)

**Steps**:
1. Add `--cache-size` CLI argument
2. Test cache sizes: 50, 56, 70, 100
3. Find optimal for beam search
4. Document results

**Success criteria**: Match or exceed 225 tokens

---

### Phase 3: Streaming Model Testing (OPTIONAL)
**Expected gain**: +5-10 tokens (2-5%)
**Estimated effort**: 2-3 hours
**Risk**: Medium (model download, compatibility)

**Steps**:
1. Download streaming-specific model
2. Test with same cache-aware approach
3. Compare quality

**Decision point**: Only pursue if beam search + cache tuning doesn't reach 225 tokens

---

## Technical Details: Beam Search for Streaming

### Key Insight
Standard beam search maintains K hypotheses and expands them at each timestep. For streaming, we need to:
1. **Preserve beam across chunks**: Don't collapse to single best hypothesis
2. **Maintain K predictor states**: One LSTM state per hypothesis
3. **Prune at chunk boundaries**: Keep top-K after each chunk
4. **Final selection**: Return best hypothesis after all chunks

### Data Structure
```rust
#[derive(Clone)]
pub struct StreamingBeamHypothesis {
    tokens: Vec<u32>,
    score: f32,
    pred_state: Vec<rnn::LSTMState>,
    last_token: u32,
}

pub struct BeamStreamingState {
    hypotheses: Vec<StreamingBeamHypothesis>,
    beam_size: usize,
}
```

### Algorithm Sketch
```rust
pub fn beam_decode_streaming(
    &self,
    encoder_out: &Tensor,           // Current chunk: [1, T_chunk, D]
    beam_size: usize,
    beam_state: Option<BeamStreamingState>,
) -> Result<(Vec<u32>, BeamStreamingState)> {
    // Initialize or restore beam
    let mut beam = match beam_state {
        Some(state) => state.hypotheses,
        None => vec![initial_hypothesis()],
    };

    let (_, time_steps, _) = encoder_out.dims3()?;

    // Decode chunk with beam search
    for t in 0..time_steps {
        let mut candidates = Vec::new();

        // Expand each hypothesis
        for hyp in &beam {
            // Inner loop: predict until blank (like greedy_decode_streaming)
            let mut current_hyp = hyp.clone();

            loop {
                // Predictor + Joint
                let (pred_out, new_state) = self.predictor.forward(..., Some(&current_hyp.pred_state))?;
                let logits = self.joint.forward(&enc_t, &pred_out)?;
                let log_probs = log_softmax(&logits)?;

                // Get top-K tokens (beam expansion)
                let top_k = get_top_k(&log_probs, beam_size)?;

                for (token, score) in top_k {
                    if token == blank {
                        // Add to candidates with updated timestep
                        let mut blank_hyp = current_hyp.clone();
                        blank_hyp.score += score;
                        candidates.push(blank_hyp);
                        break;
                    } else {
                        // Extend hypothesis
                        let mut new_hyp = current_hyp.clone();
                        new_hyp.tokens.push(token);
                        new_hyp.score += score;
                        new_hyp.pred_state = new_state;
                        new_hyp.last_token = token;
                        current_hyp = new_hyp;
                    }
                }
            }
        }

        // Prune to beam_size
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        beam = candidates.into_iter().take(beam_size).collect();
    }

    // Return tokens from best hypothesis + beam state for next chunk
    let best = &beam[0];
    Ok((best.tokens.clone(), BeamStreamingState { hypotheses: beam, beam_size }))
}
```

### Key Challenges
1. **Inner loop with beam**: Each hypothesis can emit multiple tokens before blank
   - Solution: Run inner loop per hypothesis, track all branches
2. **Beam size explosion**: K hypotheses × K tokens = K² candidates
   - Solution: Prune aggressively after each timestep
3. **LSTM state management**: Each hypothesis has independent state
   - Solution: Clone states properly (already working in greedy)

---

## Expected Final Results

With beam search (Phase 1) + cache tuning (Phase 2):

| Configuration | Tokens | vs NeMo | Status |
|---------------|--------|---------|--------|
| NeMo reference | 225 | 100% | Baseline |
| **Our streaming (beam=2, cache=70)** | **~236** | **~105%** | **Target** |
| Our streaming (beam=2, cache=56) | ~230 | ~102% | Alternative |
| Our streaming (greedy, cache=70) | 189 | 84% | Current |

**Recommendation**: Implement Phase 1 first. It's likely sufficient to match or exceed NeMo quality.

---

## Open Questions

1. **Why does NeMo use 1.04s chunks if they cause quality collapse?**
   - Answer: They likely use beam search + streaming-specific model optimizations
   - Our finding: 4.5s chunks work better with standard TDT + greedy

2. **Can we use smaller chunks with beam search?**
   - Test after Phase 1: Try 3.5s chunks with beam=2
   - May enable lower latency while maintaining quality

3. **Is streaming model necessary?**
   - Probably not - standard model + beam search should be sufficient
   - Only test if beam search doesn't reach target

---

## Success Metrics

**Minimum viable**: 220 tokens (97.8% of NeMo)
**Target**: 225 tokens (100% of NeMo)
**Stretch goal**: >225 tokens (exceed NeMo)

**Hypothesis**: Phase 1 alone (beam search) should achieve stretch goal.
