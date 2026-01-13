# LSTM State Bug Found!

## The Bug

**Symptom**: Maintaining LSTM state produces only 30 tokens vs 99 tokens with reset.

**Root Cause**: LSTM state remains `None` for chunks that produce 0 tokens, and once state exists, it causes subsequent chunks to get stuck.

## Diagnostic Output

```
Chunk 1: State is None → 0 tokens produced
Chunk 2: State is None → 0 tokens produced
Chunk 3: State exists (2 layers) → 20 tokens produced ✓
Chunk 4: State exists → 0 tokens (STUCK)
Chunk 5: State exists → 6 tokens
Chunk 6: State exists → 4 tokens
Chunk 7: State exists → 0 tokens (STUCK)
Chunk 8+: State exists → 0 tokens (STUCK)
```

## Why This Happens

### Part 1: State Stays None

In `decode_chunk_masked()`:

```rust
// Save state BEFORE prediction
let saved_states = self.state.predictor_states.clone();  // Clone None

// Run predictor
let (pred_out, new_states) = self.model.predictor.forward(
    &pred_input,
    self.state.predictor_states.as_ref()  // Pass None
)?;

if token == blank_id {
    // All blanks → rollback to saved_states (which is None!)
    self.state.predictor_states = saved_states;  // Stays None
    break;
}
```

**Issue**: When a chunk produces only blanks:
1. We start with `predictor_states = None`
2. We clone it to `saved_states = None`
3. All predictions are blank → rollback to `None`
4. End result: **State never gets created!**

### Part 2: Once State Exists, It Gets "Poisoned"

After Chunk 3 produces tokens and creates state:

```rust
// Chunk 4 starts with state from Chunk 3
self.state.predictor_states = Some([...])  // From previous chunk

// Decode Chunk 4
// All predictions are blank
// Rollback to saved_states (which was the Chunk 3 state)
// State doesn't change, but something about using it again causes problems
```

The state from Chunk 3 becomes "stale" or incompatible with Chunk 4's acoustic content, causing the model to predict only blanks.

## NeMo's Approach (What We're Missing)

### 1. State Initialization

NeMo ensures state is ALWAYS initialized, even on first chunk:

```python
if partial_hypotheses is None or partial_hypotheses.dec_state is None:
    # Initialize fresh state
    hypothesis.dec_state = self.decoder.initialize_state(encoder_output)
else:
    # Use provided state
    hypothesis.dec_state = partial_hypotheses.dec_state
```

**Key**: State is initialized from encoder output, not just left as None.

### 2. State Refresh Strategy

NeMo likely has logic to:
- Detect when state becomes stale
- Refresh or reset state at appropriate boundaries
- Use encoder context to re-initialize

### 3. Different State Handling on Blanks

NeMo's batch operations handle blanks differently:

```python
# They use batch operations that can handle mixed blank/non-blank
hidden_prime = self.decoder.batch_copy_states(
    hidden_prime, hidden, blank_indices
)
```

This works differently than our simple rollback - it's designed for batched decoding where some samples predict blank and others don't.

## Proposed Fixes

### Fix 1: Initialize State from Encoder (Recommended)

```rust
// In decode_chunk_masked, at the START:
if self.state.predictor_states.is_none() {
    // Initialize state from encoder context
    // This ensures we have a valid starting point
    let init_input = Tensor::new(&[self.state.blank_id as u32], encoder_out.device())?
        .unsqueeze(0)?;
    let (_init_out, init_states) = self.model.predictor.forward(
        &init_input,
        None  // Start fresh
    )?;
    self.state.predictor_states = Some(init_states);
}
```

This ensures:
- State is never None after first initialization
- Even 0-token chunks maintain a valid state
- State is tied to encoder context

### Fix 2: Detect and Reset Stale State

```rust
// Track how many consecutive blanks we've seen
if chunk_tokens.is_empty() {
    self.state.consecutive_blank_chunks += 1;

    if self.state.consecutive_blank_chunks >= 2 {
        // State is likely stale/stuck - reset it
        self.state.predictor_states = None;
        self.state.consecutive_blank_chunks = 0;
    }
} else {
    self.state.consecutive_blank_chunks = 0;
}
```

This provides a safety valve for stuck states.

### Fix 3: Don't Rollback to None

```rust
if token == blank_id {
    if saved_states.is_some() {
        // Only rollback if we have a valid state
        self.state.predictor_states = saved_states;
    }
    // If saved_states is None, keep new_states (at least we have something)
    break;
}
```

Ensures we maintain SOME state rather than None.

## Testing Plan

### Test Fix 1: State Initialization

```bash
# Implement Fix 1
# Run: NO_LSTM_RESET=1 cargo run --example diagnose_lstm_state --release -- dots.wav

# Expected:
# - All chunks show "State exists"
# - No "State is None" after chunk 1
# - 110-140 tokens (match baseline)
```

### Test Fix 2: Stale State Detection

```bash
# Implement Fix 2
# Run with NO_LSTM_RESET=1

# Expected:
# - Automatic resets after 2 consecutive blank chunks
# - Better recovery from stuck states
# - 90-120 tokens
```

## Expected Outcome

**Before Fix**:
- Strategy 1 (reset every chunk): 99 tokens ✓ (works around bug)
- Strategy 2 (never reset): 30 tokens ✗ (exposes bug)

**After Fix**:
- Strategy 1 (reset every chunk): 99 tokens (same)
- Strategy 2 (never reset): **130-140 tokens** ✓ (matches baseline!)

## Implementation Priority

1. **Fix 1** (state initialization) - CRITICAL, likely solves the issue
2. **Fix 2** (stale detection) - SAFETY NET, prevents getting stuck
3. **Fix 3** (don't rollback to None) - OPTIONAL, minor improvement

## Next Steps

1. Implement Fix 1 (state initialization from encoder)
2. Test with `NO_LSTM_RESET=1` on dots.wav
3. Verify we get 110-140 tokens
4. If still issues, add Fix 2 (stale detection)
5. Run full comparison again
6. Update production code to use never-reset strategy

This should bring our streaming quality from 71% → 95-100% of baseline!
