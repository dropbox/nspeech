# Critical Bug Fix: Sample Counter Mismatch in VAD Processing

## Summary

Fixed a critical timestamp bug in `src/lib.rs` where `total_samples_processed` was being incremented incorrectly, causing transcript timestamps to be inaccurate.

## The Bug

### Location
`src/lib.rs` lines 327-453 in `SpeechInner::process_samples()`

### Problem
The sample counter was incremented by 512 for each VAD probability returned, but only 160 samples were actually processed per iteration:

```rust
const CHUNK_SIZE: usize = 160;  // Feed 160 samples to VAD

while idx < samples.len() {
    let chunk = &samples[idx..end];  // ← 160 samples
    let probs = self.vad_stream.push(chunk)?;

    for prob in probs {
        // ... process probability ...
        self.total_samples_processed += 512;  // ← BUG: Wrong increment!
    }

    idx = end;  // ← Only advanced by 160
}
```

### Root Cause

**How Silero VAD Works** (from `src/silero.rs`):
1. VAD internally buffers incoming samples until it accumulates 512 samples (chunk_size)
2. When buffer reaches 512 samples, it processes them and returns probabilities
3. Each probability represents VAD state for a ~32ms frame

**The Mismatch**:
- Code feeds 160-sample chunks to VAD
- VAD buffers until it has 512 samples
- When VAD returns probabilities, code incremented counter by 512*N
- But only 160 actual samples were processed in that iteration!

**Example**:
```
Iteration 1: Feed 160 samples → VAD buffers (160 total) → 0 probs → Counter += 0
Iteration 2: Feed 160 samples → VAD buffers (320 total) → 0 probs → Counter += 0
Iteration 3: Feed 160 samples → VAD buffers (480 total) → 0 probs → Counter += 0
Iteration 4: Feed 160 samples → VAD buffers (640 total) → Process 512 → 1-2 probs → Counter += 512-1024! 🔥

Total samples fed: 640
Counter value: 512-1024  ← WRONG!
```

### Impact

**Incorrect Timestamps**:
- `start_time` and `end_time` for transcriptions were calculated from `total_samples_processed`
- Counter advanced too quickly → timestamps ahead of actual audio position
- Transcripts appeared earlier than they should

**Example**:
- Real audio position: 2.5 seconds
- Counter shows: 3.2 seconds
- Transcript shows: "2.8s - 3.2s" when it should be "2.0s - 2.5s"

## The Fix

Move the counter increment outside the probability loop and use actual chunk size:

```rust
while idx < samples.len() {
    let chunk = &samples[idx..end];
    let probs = self.vad_stream.push(chunk)?;

    // Process VAD probabilities (removed counter increment from here)
    for prob in probs {
        // ... process probability ...
        // NO counter increment here!
    }

    // Accumulate samples
    if self.current_segment_start.is_some() {
        self.current_segment.extend_from_slice(chunk);
    }

    // ✅ FIX: Increment by actual samples processed
    self.total_samples_processed += chunk.len();

    idx = end;
}
```

### Why This is Correct

1. **Tracks Actual Input**: Counter now tracks actual samples fed to the system, not VAD output frames
2. **Monotonic Growth**: Counter increases by exactly `chunk.len()` (typically 160) per iteration
3. **Accurate Timestamps**: `start_time` and `end_time` now reflect real audio position
4. **VAD Independent**: Counter no longer depends on when VAD decides to emit probabilities

## Verification

**Before Fix**:
- Counter advanced in bursts (0, 0, 0, 512, 512, ...)
- Timestamps ahead of actual audio
- Unpredictable based on VAD buffering behavior

**After Fix**:
- Counter advances smoothly (160, 160, 160, 160, ...)
- Timestamps match actual audio position
- Consistent regardless of VAD internal buffering

## Testing

To test this fix:

```bash
# Build with fix
cargo build --release

# Test with Node.js module
node test-flush-audio.cjs

# Verify timestamps in logs:
# - start_time/end_time should match audio duration
# - No gaps or overlaps in transcript segments
```

## Related Files

- `src/lib.rs` - Main fix applied here
- `src/silero.rs` - VAD implementation (explains buffering behavior)
- `examples/transcribe_with_vad.rs` - Similar pattern used correctly in CLI example

## Notes

The CLI example (`examples/transcribe_with_vad.rs`) had a similar pattern but used a different approach where it manually tracked samples, which is why it didn't exhibit this bug. The Node.js bindings relied on the counter being accurate for timestamp generation.
