# Flush API Implementation

## Summary

Added a `flush()` method to the Speech Node.js API that forces immediate transcription of any accumulated speech segment, without waiting for a natural pause.

## Changes Made

### 1. Rust Implementation (`src/lib.rs`)

**Added `SpeechInner::flush()` method:**
- Forces transcription of any active speech segment
- Clears the segment state after transcription
- Returns `Option<Transcription>` (Some if transcribed, None if nothing to flush)
- Respects `min_speech_duration_ms` threshold

**Added `FlushTask` struct:**
- Async task for executing flush on background thread
- Follows same pattern as `TranscribeTask`
- Calls callback with transcription result if present

**Added `Speech::flush()` method:**
- Public Node.js API method
- Takes `Env` parameter for spawning async task
- Returns immediately, transcription happens in background

### 2. Node.js API

**New method available:**
```javascript
speech.flush()
```

**Use cases:**
- End of audio stream without natural pause
- Forcing intermediate transcriptions
- Real-time streaming scenarios

### 3. Documentation (`README.md`)

Updated Node.js usage examples to show:
- How to call `flush()` after streaming audio
- Proper timing with `setTimeout()` for async completion
- Corrected field names (`startTime`, `endTime` in camelCase)

## Testing

**Test script:** `test-flush-audio.cjs`

**Results:**
- ✅ `flush()` method exposed correctly
- ✅ Forces transcription of accumulated segment
- ✅ Works with real audio input
- ✅ Transcription callback receives results

**Example output:**
```
[Transcription 1] "Of course it was impossible to connect the dots..."
  Time: 0.18s - 8.93s

[Transcription 2] "You can't connect the dots looking forward..."
  Time: 8.98s - 23.68s

[Transcription 3] "Because thoughts will connect down the road..."  ← From flush()
  Time: 23.51s - 29.02s
```

## API Usage Example

```javascript
const { Speech } = require('./index.node');

const speech = new Speech('assets', (transcription) => {
  console.log(`"${transcription.text}"`);
  console.log(`Time: ${transcription.startTime}s - ${transcription.endTime}s`);
});

// Stream audio chunks
for (const chunk of audioChunks) {
  speech.input(chunk); // Array of Float64
}

// Wait for processing
await new Promise(resolve => setTimeout(resolve, 1000));

// Force final transcription
speech.flush();

// Wait for flush to complete
await new Promise(resolve => setTimeout(resolve, 1000));

speech.shutdown();
```

## Implementation Details

### Flush Behavior

1. **Checks for active segment**: Returns `None` if no speech is being accumulated
2. **Validates duration**: Only transcribes if segment meets `min_speech_duration_ms` (250ms default)
3. **Transcribes segment**: Uses same `transcribe_segment()` logic as normal flow
4. **Handles phrase boundaries**: Respects comma pauses within the segment
5. **Clears state**: Resets `current_segment`, timestamps, and phrase boundaries

### Async Execution

- Flush runs on background thread via `FlushTask`
- Returns immediately to JavaScript
- Callback invoked on main thread when complete
- Safe concurrent access via `Arc<Mutex<SpeechInner>>`

### Error Handling

- Lock errors propagated as NAPI errors
- Transcription errors logged and returned
- Empty segments handled gracefully (no callback invocation)

## Build Instructions

```bash
# Build native module
cargo build --release --lib

# Copy to Node.js
cp target/release/libspeech.dylib index.node  # macOS
# or
cp target/release/libspeech.so index.node     # Linux

# Test
node test-flush-audio.cjs
```

## Notes

- Flush is **optional** - VAD will naturally transcribe at pauses
- Use flush when audio stream ends abruptly
- Add delays between `input()` batches and `flush()` for proper async handling
- Multiple flushes are safe (no-op if no active segment)
