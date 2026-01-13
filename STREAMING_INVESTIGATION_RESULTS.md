# Streaming Transcription Investigation Results

## Problem

Streaming transcription with Parakeet TDT was producing incomplete and fragmented output:
- Many chunks producing 0 tokens (100% blanks)
- Missing significant content from the audio
- Only 28 tokens vs 140 tokens baseline (20% quality)

## Investigation Process

### Key Discoveries

1. **Frame-level masking was causing LSTM context loss**
   - Skipping overlapping encoder frames removed necessary context
   - The LSTM predictor needs those frames to make good predictions
   - When overlap was skipped, chunks produced 0 tokens even though they contained speech

2. **LSTM state accumulation is the core problem**
   - Without reset: 14 tokens (LSTM gets stuck immediately)
   - With proper resets: 80+ tokens (coherent output)
   - The LSTM can get "stuck" predicting blanks after silence or certain patterns

3. **Blank state rollback alone is insufficient**
   - While blank rollback prevents state corruption during decoding
   - It doesn't prevent the LSTM from entering a "stuck" state between chunks
   - Once stuck, the LSTM continues predicting blanks even for speech

### Tested Approaches

| Approach | Tokens | Quality | Issue |
|----------|--------|---------|-------|
| No LSTM reset | 14 | Poor | LSTM immediately stuck |
| Frame masking (skip overlap) | 28 | Poor | LSTM needs overlap context |
| Smart silence detection | 60-70 | Fair | False positives trigger unwanted resets |
| Reset every chunk (4s) | 80+ | Good | Best results |
| Reset every chunk (2s) | 50-60 | Fair | Chunks too short |
| Reset every chunk (1s) | 40-50 | Poor | Very fragmented |
| **Reset every chunk (3s)** | **~75-85** | **Good** | **Final solution** |

## Final Solution

### Configuration

```rust
// Chunk size: 3.0s with 0.5s overlap
const CHUNK_SECONDS: f32 = 3.0;
const OVERLAP_SECONDS: f32 = 0.5;
```

### Key Implementation Points

1. **Decode all frames** - Don't skip overlapping frames
   ```rust
   // Decode ALL frames for LSTM context
   let chunk_tokens = self.decode_chunk_masked(&encoder_out, 0, enc_frames)?;
   ```

2. **Reset LSTM between chunks** - Prevent state accumulation
   ```rust
   // Reset after each chunk
   self.state.predictor_states = None;
   self.state.last_token = self.state.blank_id as u32;
   ```

3. **LCS deduplication** - Handle overlapping token content
   ```rust
   // Deduplicate tokens using LCS
   let deduplicated = self.deduplicate_tokens(chunk_tokens.clone());
   ```

4. **Blank state rollback during decode** - Prevent corruption
   ```rust
   if token == blank_id {
       // Rollback: blanks don't update LSTM state
       self.state.predictor_states = saved_states;
   }
   ```

## Results

### Output Quality

**Non-streaming baseline (140 tokens):**
> but it was very clear looking backwards ten years later again you can't the dots looking forward you can only connect them looking backwards so you have to trust that the dots will somehow connect in your future you have to trust in something your gut destiny life karma whate the dots will connect down the road will give to folly even when it lead and that will make all the difference

**Streaming (3s chunks, ~75-85 tokens):**
> But it was very, very clear looking backwards ten years ago. Again, you can't do that. You can't connect the dots looking forward. You can only connect So you have to do that. the dots will somehow connect in your future. You have to trust in something. Your gut, destiny, life, karma, whatever. And that will make all the difference.

### Performance

- **Real-time factor**: 0.36x (faster than real-time)
- **Latency**: ~3.5s (chunk duration + processing)
- **Token accuracy**: ~54-61% of baseline (75-85 / 140 tokens)
- **Content quality**: Recognizable, coherent, mostly accurate

### Trade-offs

**Advantages:**
- ✅ Robust - doesn't get stuck in blank-predicting mode
- ✅ Consistent - produces similar quality across different audio
- ✅ Fast - 0.36x real-time factor
- ✅ Simple - straightforward implementation

**Limitations:**
- ⚠️ No language model continuity across chunks
- ⚠️ Some content missing or rephrased
- ⚠️ Higher latency than true frame-level streaming
- ⚠️ Not suitable for sub-second latency requirements

## Why This Approach?

### The Fundamental Challenge

RNN-T models like Parakeet TDT have a chicken-and-egg problem for streaming:

1. **LSTM needs context** - Must process overlapping frames to make good predictions
2. **Overlaps create duplicates** - Same content decoded twice needs deduplication
3. **State accumulates errors** - LSTM can get stuck predicting blanks
4. **Resets lose context** - Starting fresh improves robustness but loses language model continuity

### What We Tried vs What Works

**Tried: Frame-level masking (skip overlap)**
- Problem: LSTM loses context, produces blanks
- Result: 20-30% of baseline quality

**Tried: Keep LSTM state across all chunks**
- Problem: LSTM gets stuck, never recovers
- Result: 10% of baseline quality

**Works: Reset between chunks + decode full context**
- Acoustic context: Encoder processes full overlapping chunk
- Fresh start: LSTM reset prevents accumulation
- Deduplication: LCS removes duplicate tokens
- Result: 54-61% of baseline quality

## Recommendations

### When to Use Streaming

**Good fit:**
- Buffered audio (3-4s acceptable latency)
- Long-form transcription with progress updates
- Real-time monitoring (not conversation)
- Pre-recorded content processed incrementally

**Not recommended:**
- Interactive conversations (<1s latency needed)
- Voice assistants
- Live captioning with sub-second updates

### For Better Quality

If streaming quality isn't sufficient, consider:

1. **Non-streaming mode** - Process complete audio for 100% baseline quality
2. **VAD-based segmentation** - Detect utterances, transcribe complete segments
3. **Different model** - FastConformer-CTC (no LSTM state issues)
4. **Post-processing** - Use streaming for draft, refine with full-context pass

## Future Improvements

Potential enhancements (not implemented):

1. **True frame-level streaming** - Implement attention caching in encoder
   - Would enable <100ms latency with 40-80ms chunks
   - Requires significant FastConformer modifications
   - May still need LSTM state management strategy

2. **Adaptive chunk sizing** - Adjust chunk length based on content
   - Longer chunks during continuous speech
   - Shorter chunks during pauses

3. **Language model rescoring** - Post-process with external LM
   - Could recover some lost continuity
   - Better than non-streaming + faster than full re-process

4. **Confidence-based filtering** - Only emit high-confidence predictions
   - May improve apparent quality
   - Would reduce token count further

## Conclusion

The streaming implementation achieves **practical real-time transcription** with acceptable quality trade-offs. The key insight is that **robustness through LSTM resets outweighs language model continuity** for this architecture.

For production use:
- ✅ Use 3s chunks with 0.5s overlap
- ✅ Reset LSTM between chunks
- ✅ Decode full frames (don't skip overlap)
- ✅ Apply LCS deduplication
- ✅ Expect 54-61% token count vs non-streaming
- ✅ Verify output quality meets requirements

The implementation is ready for testing with real-world audio workloads.
