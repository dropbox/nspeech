# Critical Differences Between Our Implementation and NeMo

## Summary of Findings

After analyzing NeMo's streaming RNN-T implementation, I've identified several key differences that explain our lower quality (54-61% vs baseline).

## Key Difference #1: LSTM State Management ⚠️ CRITICAL

### NeMo's Approach
```python
# NeMo maintains LSTM state across ALL chunks in an utterance
def _greedy_decode(x, out_len, partial_hypotheses=None):
    if partial_hypotheses is not None:
        # RESTORE state from previous chunk
        hypothesis.dec_state = partial_hypotheses.dec_state
        hypothesis.last_token = partial_hypotheses.last_token

    # Process current chunk with existing state
    for time_idx in range(out_len):
        g, hidden_prime = self._pred_step(last_label, hypothesis.dec_state)
        hypothesis.dec_state = hidden_prime  # Update state

    # Return hypothesis WITH state for next chunk
    return hypothesis
```

**State is only reset when:**
- End-of-sentence (EOS token detected)
- Between unrelated audio samples
- Explicit reset requested

### Our Current Approach
```rust
// We RESET LSTM after EVERY chunk
self.state.predictor_states = None;
self.state.last_token = self.state.blank_id as u32;
```

**Impact:** This destroys language model continuity across chunks, causing:
- Inability to predict tokens that depend on context from previous chunks
- Lower confidence predictions
- Missing content between chunks

---

## Key Difference #2: Frame Processing Strategy

### NeMo's Approach
```python
# NeMo decodes ALL frames including overlap
# The LCS handles deduplication at the TOKEN level
for time_idx in range(out_len):  # ALL encoder frames
    # Decode frame
    # State carries forward naturally
```

### Our Current Approach
```rust
// We also decode all frames (CORRECT)
let chunk_tokens = self.decode_chunk_masked(&encoder_out, 0, enc_frames)?;
```

**Status:** ✅ We do this correctly

---

## Key Difference #3: LCS Token Deduplication

### NeMo's Approach
```python
def longest_common_subsequence_merge(X, Y):
    """
    X: Last N tokens from previous buffer (tail search)
    Y: All tokens from current chunk

    Returns: (i, j, length) - where to slice Y to remove overlap
    """
    # Build DP table to find longest matching subsequence
    # Handle partial matches with diagonal expansion
    # Minimum length threshold: MIN_MERGE_SUBSEQUENCE_LEN = 1

    # Critical: LCS works on TOKENS, not frames
    # Can handle misalignment due to LSTM state changes
```

**Key Parameters:**
- Search size: `delay * max_steps_per_timestep`
- Delay: `(total_buffer - chunk_len) / model_stride`
- Min LCS length: 1 token (very permissive)

### Our Current Approach
```rust
fn find_lcs_slice_point(&self, buffer: &[u32], new_tokens: &[u32]) -> usize {
    const MAX_SEARCH_LEN: usize = 50;
    const MIN_LCS_LENGTH: usize = 3;  // More restrictive

    // Search in buffer tail
    // Find longest matching subsequence
    // Return slice point
}
```

**Issues:**
- MIN_LCS_LENGTH = 3 might be too restrictive (NeMo uses 1)
- Our LCS implementation is simpler, might miss some matches
- No diagonal expansion for partial matches

---

## Key Difference #4: State Handling on Blanks

### NeMo's Approach
```python
# When blank is predicted
if token == blank_id:
    hidden_prime = self.decoder.batch_copy_states(
        hidden_prime, hidden, blank_indices
    )
    k[blank_indices] = last_label[blank_indices]
```

### Our Approach
```rust
if token == self.state.blank_id as u32 {
    self.state.predictor_states = saved_states;  // Rollback
    break;
}
```

**Status:** ✅ We do this correctly

---

## Key Difference #5: Utterance Boundary Detection

### NeMo's Approach
```python
# Reset state on EOS token (end of sentence)
if len(pred) > 0 and pred[-1] == self.eos_id:
    self.previous_hypotheses[idx].dec_state = reset_states
```

### Our Approach
```rust
// We don't detect utterance boundaries
// We reset after EVERY chunk regardless
```

**Impact:** We can't distinguish between:
- Mid-sentence chunk boundary (should maintain state)
- End-of-sentence boundary (should reset state)

---

## Root Cause Analysis

### Primary Issue: Excessive LSTM Resets

Our implementation resets LSTM after every 3s chunk, which means:

1. **Lost Context:** Each chunk starts with blank predictor state
2. **Poor Token Predictions:** LSTM can't use language model context from earlier in the sentence
3. **Missing Tokens:** Some tokens require context to be predicted correctly
4. **Lower Confidence:** Without context, model is less certain

### Example Impact

**Sentence:** "But it was very, very clear looking backwards ten years ago."

**Our chunks (3s each):**
- Chunk 1: "But it was very, very clear" [RESET]
- Chunk 2: "looking backwards" [RESET - lost context from "clear"]
- Chunk 3: "ten years ago" [RESET - lost context from "backwards"]

**NeMo's approach:**
- Process entire sentence with continuous LSTM state
- Only reset at sentence end or EOS token

---

## Secondary Issue: LCS Parameters

Our `MIN_LCS_LENGTH = 3` is more restrictive than NeMo's `= 1`. This means:
- We might keep duplicate tokens that should be removed
- We fail to deduplicate short overlaps

---

## Proposed Fixes

### Fix 1: Maintain LSTM State Across Chunks (CRITICAL)

```rust
// Remove this:
// self.state.predictor_states = None;
// self.state.last_token = self.state.blank_id as u32;

// Only reset on explicit signal (EOS token, utterance boundary, etc.)
```

### Fix 2: Implement Utterance Boundary Detection

Options:
1. **Detect long silence** (>500ms of blanks)
2. **Use VAD** (external voice activity detection)
3. **Energy-based** (audio power drops below threshold)
4. **Confidence-based** (low token confidence indicates boundary)

### Fix 3: Adjust LCS Parameters

```rust
const MIN_LCS_LENGTH: usize = 1;  // Match NeMo
```

### Fix 4: Improve LCS Algorithm

Consider implementing NeMo's diagonal expansion for partial matches.

---

## Expected Improvements

### After Fix 1 (Maintain LSTM State)
- **Quality:** 54-61% → 80-95%
- **Tokens:** 75-85 → 110-130+
- **Coherence:** Significantly improved sentence continuity

### After Fix 2 (Utterance Boundaries)
- **Quality:** 80-95% → 95-100%
- **State management:** More intelligent, context-aware

### After Fix 3+4 (Better LCS)
- **Deduplication:** Fewer duplicate tokens
- **Quality:** Marginal improvement (1-2%)

---

## Testing Strategy

### Test 1: No LSTM Reset
```rust
// Comment out reset
// self.state.predictor_states = None;

// Run on dots.wav
// Expected: 110+ tokens (vs 75-85 current)
```

### Test 2: Silence-Based Reset
```rust
// Reset only after >500ms silence
if consecutive_blank_frames > 30 {  // 30 frames = ~600ms
    self.state.predictor_states = None;
}
```

### Test 3: LCS Tuning
```rust
const MIN_LCS_LENGTH: usize = 1;  // More permissive
```

### Test 4: Full NeMo-Style Implementation
- Maintain state across utterance
- Detect boundaries with silence
- Use MIN_LCS_LENGTH = 1
- Expected: 130-140 tokens (match baseline)

---

## Conclusion

The primary issue is **over-aggressive LSTM state resetting**. NeMo maintains state throughout an utterance, only resetting at sentence boundaries or explicit signals. Our current approach resets every 3s regardless of context, destroying language model continuity.

**Next Steps:**
1. Implement Fix 1 (remove blanket reset)
2. Implement Fix 2 (silence-based utterance detection)
3. Test on dots.wav and compare with baseline
4. Iterate on boundary detection parameters
5. Fine-tune LCS parameters if needed

This should bring our streaming quality from 54-61% to 90-95%+ of baseline.
