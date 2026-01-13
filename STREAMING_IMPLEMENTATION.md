# Streaming Transcription Implementation

This document explains the context-based streaming implementation for Parakeet CTC transcription.

## Overview

The streaming implementation uses a common library function that handles context-based streaming. The system uses Silero VAD to detect speech segments and transcribes them using Parakeet CTC.

## Core Function: `transcribe_streaming_chunk`

**Location:** `src/parakeet/mod.rs`

```rust
pub fn transcribe_streaming_chunk(
    chunk_samples: &[f32],
    left_context: Option<&[f32]>,
    right_context: Option<&[f32]>,
    model: &ParakeetFastConformerCtc,
    device: &Device,
) -> Result<String>
```

### How It Works

1. **Build Full Input**: Concatenates `[left_context | chunk | right_context]`
2. **Extract Features**: Converts audio samples to mel-spectrogram features
3. **Run Inference**: Processes full window through the model
4. **Slice Logits**: Extracts only the middle chunk frames (if context provided)
5. **Decode**: Applies CTC greedy decoding to get text

### Frame Calculation

Parakeet uses 8x subsampling, approximately 80 audio samples per output frame:
- Left context frames = `(left_context.len() + 79) / 80`
- Chunk frames = `(chunk_samples.len() + 79) / 80`
- Extract logits: `logits[:, left_context_frames..left_context_frames+chunk_frames, :]`

### Benefits

✅ **Reusable** - Single implementation for both VAD and bypass-VAD modes
✅ **Context-aware** - Optional left/right context for stable predictions
✅ **Efficient** - Each audio frame processed exactly once
✅ **Flexible** - Works with or without context

## Usage

**Location:** `src/lib.rs` - `SpeechInner::transcribe_segment()`

The system transcribes complete speech segments detected by Silero VAD:

```rust
let text = parakeet::transcribe_streaming_chunk(
    &self.current_segment,  // Full speech segment
    None,                    // No left context (segment is complete)
    None,                    // No right context
    &self.parakeet_model,
    &self.device,
)?;
```

**Why no context?** VAD segments are already bounded by natural pauses, so we have complete utterances without needing additional context. The streaming function supports optional context for future use cases where fixed-time chunking might be needed.

## Testing

```bash
cargo run --example transcribe_with_vad --release -- dots.wav
```

Expected: 10 speech segments, accurate transcription of Steve Jobs "connecting the dots" speech.

## Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│          parakeet::transcribe_streaming_chunk   │
│                                                 │
│  1. Build: [left_ctx | chunk | right_ctx]      │
│  2. Extract features                            │
│  3. Run model inference                         │
│  4. Slice logits (if context)                   │
│  5. CTC decode                                  │
└─────────────────────────────────────────────────┘
                        ▲
                        │
                   ┌────┴────┐
                   │  VAD    │
                   │  Mode   │
                   │         │
                   │ No ctx  │
                   └─────────┘
```

## Key Implementation Details

### VAD Mode
- Uses Silero VAD to detect speech segments
- Accumulates samples when speech detected
- Transcribes on silence threshold (300ms)
- No context needed (natural boundaries)
- Variable latency based on speech patterns

### Processing Flow

```
Time:    0s      3s      6s      9s      12s
        ─────────────────────────────────────
Audio:  [──speech──][silence][───speech───]
VAD:         ✓                     ✓
Output:      "text1"               "text2"
```

## Files Modified

1. **`src/parakeet/mod.rs`**
   - Added `transcribe_streaming_chunk()` function
   - Supports optional left/right context for flexible streaming

2. **`src/lib.rs`**
   - VAD mode: Uses streaming function without context
   - Simplified transcription implementation

3. **Examples:**
   - `examples/transcribe_with_vad.rs` - Tests VAD-based transcription
   - `examples/transcribe_quantized.rs` - Tests basic inference

## Performance Characteristics

- **Latency**: Variable (depends on speech patterns, typically 300ms after pause)
- **Accuracy**: High (complete utterances with natural boundaries)
- **CPU/GPU**: Only processes actual speech (efficient)
- **Memory**: Low (accumulates only current segment)

## Future Optimizations

If you need lower latency or different streaming behavior:

1. **Encoder caching** - Cache encoder states between segments
2. **Configurable thresholds** - Make VAD parameters adjustable via API
3. **Fixed-time chunking** - Use the context parameter for time-based chunks
4. **Beam search** - Replace greedy decode with beam search for better accuracy
