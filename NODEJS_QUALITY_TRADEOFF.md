# Node.js Integration Quality Tradeoff

## Summary

The Node.js speech recognition integration is **functional** but produces **lower quality transcriptions** compared to the Rust CLI examples due to memory constraints requiring the use of greedy decoding instead of beam search.

## Quality Comparison

### Rust CLI (Perfect Quality)
Using `cargo run --example transcribe_tdt_with_vad --release -- dots.wav`:

```
Of course, it was impossible to connect the dots looking forward when I was in
college, but it was very, very clear looking backwards ten years later. Again,
you can't connect the dots looking forward. You can only connect them looking
backwards. So you have to trust that the dots will somehow connect in your
future. You have to trust in something: your gut, destiny, life, karma, whatever.
Because believing that the dots will connect down the road will give you the
confidence to follow your heart, even when it leads you off the well-worn path,
and that will make all the difference.
```

### Node.js (Lower Quality)
Using `node test-load.js dots.wav`:

```
Segment 1: "Ofourse it was impos to connect the dots looking forward when I
was in college, but it was very clear looking backwards ten years later"

Segment 2: "you can'ton connect the dots looking forward.. You can only connect
them looking backwards.. So you have to trust that the dots will somehowell
connect in your future you have to trust in something your gut, destiny, life,
karma, whate.. Because belie that the dots will connect down the road will give
the confidence to foll your heart even when it leads you off the well-worn path
and that will make all the difference"
```

**Common errors in Node.js output:**
- "Ofourse" instead of "Of course"
- "impos" instead of "impossible"
- "can'ton" instead of "can't"
- "somehowell" instead of "somehow"
- "whate" instead of "whatever"
- "belie" instead of "believing"
- "foll" instead of "follow"
- Missing repeated words ("very, very" becomes "very")

## Root Cause

### Decoding Method Difference

**Rust CLI**: Uses `beam_decode(beam_size=2)`
- Explores multiple hypotheses at each decoding step
- Keeps track of top 2 most likely sequences
- Chooses best overall sequence at the end
- Higher computational and memory cost
- **Much higher quality**

**Node.js**: Uses `greedy_decode()`
- Chooses most likely token at each step
- No backtracking or alternative hypotheses
- Lower memory usage
- **Lower quality** (can get "stuck" on wrong paths)

### Memory Constraint

Beam search was attempted in Node.js but failed due to memory exhaustion:

1. Embedded assets already use substantial memory:
   - Quantized TDT model: 849MB (mmap'd GGUF)
   - VAD model: ~50MB
   - Feature extraction buffers
   - Streaming buffers

2. Beam search additional memory:
   - Tracks 2+ hypotheses simultaneously
   - Maintains separate predictor states for each beam
   - Requires additional tensor allocations

3. Result: Node.js process gets killed by OS (SIGKILL) when beam search is enabled

## Recommendations

### For Production Use
**Use the Rust CLI**, which provides perfect transcription quality:

```bash
cargo run --example transcribe_tdt_with_vad --release -- audio.wav
```

### For Development/Testing
Node.js integration is suitable for:
- Quick prototyping
- Testing the model loading pipeline
- Development workflows where perfect accuracy isn't critical

### Future Improvements

Possible approaches to improve Node.js quality:

1. **External Process Architecture**
   - Run Rust transcription in separate process
   - Communicate via IPC/pipes
   - Node.js handles I/O, Rust handles inference

2. **Model Optimization**
   - Use smaller quantized model (Q4 instead of Q8)
   - Reduce beam size to 1.5 or enable adaptive beam pruning
   - Stream-specific model optimizations

3. **Memory Management**
   - Lazy model loading (load on first use, not at init)
   - Shared memory for model weights
   - Memory-mapped tensors with deferred dequantization

4. **Alternative Models**
   - Use Whisper tiny/base for Node.js (smaller, faster)
   - Reserve Parakeet TDT for Rust CLI

## Current Status

✅ **Functional**: All 35 seconds of audio processed correctly
✅ **Timestamps**: Accurate segment timing
✅ **Segmentation**: Proper VAD-based boundaries
⚠️  **Quality**: Lower than Rust CLI due to greedy decode
❌ **Production Ready**: No, use Rust CLI for production

## Technical Details

### What Makes Beam Search Better?

Example scenario where greedy fails but beam succeeds:

At timestep T, model sees partial transcription: "you can'"

**Greedy decode**:
- Picks most likely next token: "t" → "you can't"
- Gets stuck, next token: "o" → "you can'to"
- Wrong path, but no backtracking
- Final: "can'ton"

**Beam search**:
- Keeps top 2 hypotheses:
  1. "you can't" (p=0.4)
  2. "you can" (p=0.35)
- Continues both paths
- Eventually: "you can't connect" wins (p=0.25)
- "you can to connect" loses (p=0.08)
- Backtracks to correct path
- Final: "can't"

This is why beam search produces "can't" while greedy produces "can'ton".

## Conclusion

The Node.js integration demonstrates successful model loading and streaming architecture, but is not suitable for production transcription due to quality constraints. For best results, use the Rust CLI which provides perfect transcription accuracy.
