# Testing TDT Streaming Transcription

This guide shows how to test the Parakeet TDT model with streaming functionality after the feature extraction fix.

## Prerequisites

1. **TDT model downloaded:**
   ```bash
   python scripts/download_parakeet.py --cache .cache/parakeet-tdt
   ```

2. **Audio files:** Test with 16kHz mono WAV files (jfk.wav, dots.wav, etc.)

## Test 1: Non-Streaming (Batch) Transcription

This is the standard batch mode - processes entire audio at once.

```bash
# Test on jfk.wav (11 seconds)
cargo run --example transcribe_tdt --release -- jfk.wav

# Test on dots.wav (35 seconds)
cargo run --example transcribe_tdt --release -- dots.wav
```

**Expected Results:**

**jfk.wav:**
```
And so, my fellow Americans, ask not what your country can do for you,
ask what you can do for your country.
```
✅ Perfect transcription (38 tokens)

**dots.wav:**
```
Of course, it was impossible to connect the dots looking forward when I was
in college, but it was very, very clear looking backwards ten years later...
```
✅ Perfect transcription (187 tokens)

## Test 2: Streaming Transcription (Overlapping Chunks)

This processes audio in overlapping chunks, emitting results incrementally.

```bash
# Test streaming on jfk.wav
cargo run --example transcribe_tdt_streaming --release -- jfk.wav

# Test streaming on dots.wav
cargo run --example transcribe_tdt_streaming --release -- dots.wav
```

### How Streaming Works

The `transcribe_tdt_streaming` example demonstrates:

1. **Chunked Processing:**
   - Chunk size: 3.0s (48,000 samples)
   - Overlap: 0.5s (8,000 samples) for acoustic context
   - Stride: 2.5s (advance by chunk_size - overlap)

2. **State Management:**
   - Maintains LSTM predictor state across chunks
   - Uses LCS (Longest Common Subsequence) deduplication for overlap regions
   - Accumulates tokens incrementally

3. **Output:**
   - Shows progress for each chunk
   - Displays NEW text as it's decoded
   - Shows accumulated full text
   - Final complete transcription at end

### Streaming Configuration

You can adjust streaming parameters in the example:

```rust
const CHUNK_SECONDS: f32 = 3.0;     // Chunk size
const OVERLAP_SECONDS: f32 = 0.5;   // Overlap for context
```

**Trade-offs:**
- **Larger chunks (5-10s):**
  - ✅ Better accuracy (more encoder context)
  - ❌ Higher latency

- **Smaller chunks (1-2s):**
  - ✅ Lower latency
  - ❌ Reduced accuracy at chunk boundaries

### Performance Characteristics

**Real-time factor:** ~0.35-0.38x (faster than real-time)

- jfk.wav: 11s audio → 4.2s processing (0.38x RTF)
- dots.wav: 35s audio → 12.5s processing (0.35x RTF)

**Latency:**
- Chunk latency: ~3.5s (buffering + processing)
- Suitable for near-realtime/buffered applications
- NOT suitable for ultra-low-latency live transcription

## Test 3: Comparing Streaming vs Batch

Run both modes and compare:

```bash
# Batch mode (full context)
echo "=== BATCH MODE ==="
cargo run --example transcribe_tdt --release -- jfk.wav 2>&1 | grep -A 5 "TRANSCRIPTION"

# Streaming mode (chunked)
echo "=== STREAMING MODE ==="
cargo run --example transcribe_tdt_streaming --release -- jfk.wav 2>&1 | grep -A 5 "FINAL TRANSCRIPTION"
```

**Expected Differences:**

- **Batch mode:** Best accuracy, full encoder context, higher latency
- **Streaming mode:** Slightly lower accuracy due to chunking, but incremental results

## Understanding the Output

### Streaming Output Format

```
[Chunk 1/5] 3.0s processed in 0.95s (11 new tokens, 11 total)
  + "And so, my fellow Americans."
  → Full: And so, my fellow Americans.
```

- **Chunk info:** Which chunk, processing time
- **Token counts:** New tokens this chunk, total accumulated
- **+ "text":** NEW text decoded from this chunk
- **→ Full:** Accumulated text so far

### LCS Deduplication

You may see messages like:
```
[LCS] Dedup: 13 raw tokens → 0 deduplicated (13 removed)
```

This means overlap deduplication detected and removed 13 duplicate tokens from the overlapping region.

## Advanced Testing

### Test with Different Audio

```bash
# Convert your audio to correct format
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# Test streaming
cargo run --example transcribe_tdt_streaming --release -- output.wav
```

### Adjust Chunk Size

Edit `examples/transcribe_tdt_streaming.rs`:

```rust
// For lower latency (1s chunks)
const CHUNK_SECONDS: f32 = 1.0;
const OVERLAP_SECONDS: f32 = 0.2;

// For better accuracy (5s chunks)
const CHUNK_SECONDS: f32 = 5.0;
const OVERLAP_SECONDS: f32 = 1.0;
```

Then rebuild:
```bash
cargo build --example transcribe_tdt_streaming --release
```

### CPU Testing

Force CPU mode if GPU issues:

```bash
PARAKEET_DEVICE=cpu cargo run --example transcribe_tdt_streaming --release -- jfk.wav
```

## Streaming Implementation Details

The streaming uses `StreamingTransducer` and `StreamingState` from `src/parakeet/streaming_transducer.rs`:

**Key Features:**
- Maintains predictor (LSTM) state across chunks
- Uses overlapping windows for acoustic context
- LCS-based deduplication to remove duplicate tokens in overlap regions
- Incremental token decoding
- Frame-level masking to skip already-processed encoder frames

**Limitations:**
- Not true frame-level streaming (requires attention caching)
- Chunk-level latency (not suitable for live conversation)
- Accuracy may vary at chunk boundaries
- Large chunks recommended (2-5s) for best quality

## Current Feature Extraction

After the fix (2026-01-13), feature extraction now matches NeMo:

✅ **Per-feature normalization** (each mel bin normalized to mean=0, std=1)
✅ **Symmetric Hann window** (not periodic)
✅ **128 mel bins** for TDT model
✅ **Preemphasis 0.97**
✅ **Log10 scaling** (not natural log)

This ensures encoder receives correct inputs and produces accurate transcriptions.

## Troubleshooting

### Low Quality Transcription

Try:
1. Increase chunk size for more context
2. Increase overlap for better boundary handling
3. Check audio is 16kHz mono
4. Use batch mode for comparison

### Performance Issues

Try:
1. Use smaller chunks (but may reduce accuracy)
2. Force CPU mode if GPU issues
3. Process shorter audio files

### Missing Words at Boundaries

Try:
1. Increase overlap (e.g., 1.0s instead of 0.5s)
2. Increase chunk size for more context
3. LCS dedup may be too aggressive - check logs

## Next Steps

- **For production:** Use batch mode (`transcribe_tdt`) for best accuracy
- **For near-realtime:** Use streaming with 3-5s chunks
- **For live transcription:** Wait for attention caching implementation (coming soon)

## Related Files

- `examples/transcribe_tdt.rs` - Batch mode transcription
- `examples/transcribe_tdt_streaming.rs` - Streaming mode transcription
- `src/parakeet/streaming_transducer.rs` - Streaming implementation
- `src/parakeet/transducer.rs` - TDT model and decoding
- `src/parakeet/features.rs` - Feature extraction (fixed!)
- `TDT_FIX_SUMMARY.md` - Details on the feature extraction fix
