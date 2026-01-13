# LSTM State Management Comparison Results

## Test Setup

**Audio**: dots.wav (35.33s)
**Configuration**: 3s chunks with 0.5s overlap
**MIN_LCS_LENGTH**: Changed from 3 → 1 (NeMo-style)

## Results

| Strategy | Tokens | Quality vs Baseline | Transcript Quality |
|----------|--------|---------------------|-------------------|
| **Baseline (non-streaming)** | **140** | **100%** | **Excellent** |
| Strategy 1: Reset every chunk | 99 | 71% | Good - coherent |
| Strategy 2: Never reset (continuous) | 30 | 21% | Poor - incomplete |
| Strategy 3: Reset after silence | 50 | 36% | Fair - fragmented |

## Detailed Transcripts

### Strategy 1: Reset After Every Chunk (99 tokens)
> "But it was very, very clear looking backwards ten years ago. Again, you can't do that. You can't connect the dots looking forward. You can only connect So you have to do that. the dots will somehow connect in your future. Your gut, destiny, life, karma, whatever. And that will make all the difference."

**Quality**: Good - Most content is recognizable

### Strategy 2: Never Reset LSTM (30 tokens) ⚠️ UNEXPECTED
> "But it was very, very clear looking backwards ten years ago. You can only connect the yeah."

**Quality**: Poor - LSTM got stuck after 2 chunks, produces mostly blanks

### Strategy 3: Reset After Silence (50 tokens)
> "But it was very, very clear looking backwards ten years ago. You can't connect the dots looking forward. You can only connect And that will make all the difference."

**Quality**: Fair - Missing chunks between silence periods

## Analysis

### Unexpected Finding #1: Never Reset Performs WORST

**Expected** (based on NeMo analysis):
- Maintaining LSTM state should give ~110-140 tokens (best quality)
- Language model continuity should help predictions

**Actual**:
- Only 30 tokens (21% of baseline)
- LSTM gets "stuck" predicting blanks after chunk 2
- Significantly worse than resetting every chunk

**Why This Happens**:
Our LSTM predictor enters a "stuck" state where it predicts blanks continuously:
1. Chunk 1: Processes initial audio, builds state ✓
2. Chunk 2: Uses state from Chunk 1, produces some tokens ✓
3. Chunk 3+: State becomes "poisoned", predicts only blanks ✗

This suggests a **fundamental issue with how we pass LSTM state between chunks**.

### Unexpected Finding #2: MIN_LCS_LENGTH Improves Quality

**Before** (MIN_LCS_LENGTH = 3): 75-85 tokens
**After** (MIN_LCS_LENGTH = 1): 99 tokens

**Improvement**: +17-32% quality

This confirms NeMo's approach of very permissive LCS matching (threshold=1) is correct.

### Why NeMo Can Maintain State But We Can't

Based on the evidence, there are likely issues in our implementation:

#### Hypothesis 1: State Format/Shape Mismatch
```rust
// Our state: Option<Vec<rnn::LSTMState>>
// Each LSTMState: (h, c) tensors

// When we save: self.state.predictor_states = Some(new_states);
// When we restore: self.model.predictor.forward(&pred_input, self.state.predictor_states.as_ref())?

// Potential issues:
// - Tensor device mismatch (CPU vs GPU)
// - Batch dimension handling
// - State concatenation/selection
```

#### Hypothesis 2: Blank State Rollback Interaction
```rust
// During decode, we rollback on blanks:
if token == blank_id {
    self.state.predictor_states = saved_states;  // Rollback
}

// At chunk boundary, we preserve:
// self.state.predictor_states = Some(new_states);

// If last prediction was blank, new_states is already rolled back
// This might create cumulative corruption
```

#### Hypothesis 3: State Stale Detection
```rust
// We don't track how "old" state is
// After 2-3 chunks, state might become stale/incompatible
// NeMo might have freshness checks or partial resets
```

## Deep Dive Investigation Needed

To match NeMo's quality, we need to debug:

### 1. State Shape/Format Verification
Create diagnostic to print:
- LSTM state tensor shapes after each chunk
- Device placement
- Numerical ranges (detect NaN/Inf)

### 2. Per-Chunk Token Production
Track exactly when LSTM gets stuck:
```
Chunk 1: 25 tokens ✓
Chunk 2: 15 tokens ✓
Chunk 3: 0 tokens ✗ ← LSTM stuck here
Chunk 4: 0 tokens ✗
```

### 3. State Corruption Detection
Compare state statistics:
- Before saving (end of chunk N)
- After restoring (start of chunk N+1)
- Detect if values diverge unexpectedly

### 4. NeMo Code Review
Study NeMo's exact state handling:
- `batch_select_state()` - How they extract state
- `batch_concat_states()` - How they prepare state
- State device management

## Recommended Next Steps

### Step 1: Add State Diagnostics
```rust
// In process_features, after decode:
eprintln!("  [STATE] Predictor state: {:?}",
    self.state.predictor_states.as_ref().map(|states| {
        states.iter().map(|s| {
            format!("h: {:?}, c: {:?}", s.h().dims(), s.c().dims())
        }).collect::<Vec<_>>()
    })
);
```

### Step 2: Test Smaller Audio
Use 10-15s audio to isolate when/where LSTM gets stuck with fewer chunks.

### Step 3: Compare with Non-Streaming
Run both modes on same audio, compare per-chunk:
- Non-streaming: Process 3s, get tokens
- Streaming chunk 1: Process same 3s, compare tokens

Should be identical for first chunk. If not, state initialization differs.

### Step 4: Review Predictor Forward Pass
Check if our `predictor.forward()` properly handles `Option<&Vec<LSTMState>>`:
- None case: Initialize fresh
- Some case: Use provided state

Might have bugs in state unpacking/repacking.

## Current Best Strategy

**For Production** (until state passing is fixed):
- ✅ Use Strategy 1: Reset after every chunk
- ✅ Set MIN_LCS_LENGTH = 1
- ✅ Use 3s chunks with 0.5s overlap
- ✅ Expect ~99 tokens (71% of baseline)

**Quality Profile**:
- Good for most content
- Some phrases slightly off
- Missing ~30% of tokens
- Acceptable for many use cases

## Path to 100% Quality

To match non-streaming (140 tokens):

1. **Fix LSTM state passing** (required)
   - Debug why state causes stuck predictions
   - Implement proper state save/restore
   - Expected gain: +40-50 tokens (99 → 140+)

2. **Improve LCS deduplication** (optional)
   - Add diagonal expansion for partial matches
   - Better handling of repeated tokens
   - Expected gain: +5-10 tokens

3. **Utterance boundary detection** (optional)
   - Reset at natural pause points
   - Use VAD or energy-based detection
   - Expected gain: Robustness, not token count

## Conclusion

**Key Insight**: Our LSTM state passing has a bug that causes stuck predictions. Resetting every chunk works around this bug but sacrifices language model continuity.

**Quality ranking** (current implementation):
1. Reset every chunk: **99 tokens** (71%) ← Best we can do now
2. Reset after silence: **50 tokens** (36%)
3. Never reset: **30 tokens** (21%) ← Exposes the bug

**To match NeMo**: Must fix state passing, not just change reset strategy.

The investigation continues in the LSTM predictor forward pass implementation...
