# Streaming Transcription Approaches

This codebase implements two different streaming approaches for real-time transcription:

## 1. VAD-Based Streaming (transcribe_with_vad)

**How it works:**
- Uses Silero VAD to detect speech segments
- Accumulates audio only during speech
- Transcribes complete utterances at natural pauses
- Adds punctuation based on pause duration

**Configuration:**
```rust
speech_threshold: 0.5            // VAD probability threshold
min_speech_duration_ms: 250.0    // Minimum segment length
comma_pause_duration_ms: 150.0   // Short pause → comma
period_pause_duration_ms: 500.0  // Long pause → transcribe
pre_buffer_ms: 300.0             // Capture before speech starts
```

**Pros:**
- ✅ High quality transcription (segments at natural boundaries)
- ✅ Automatic punctuation based on pauses
- ✅ Only processes speech (ignores silence)
- ✅ Natural sentence structure

**Cons:**
- ❌ Requires pauses to trigger transcription
- ❌ Latency = pause duration (500ms default)
- ❌ Fast continuous speech without pauses can cause issues

**Best for:**
- Conversations with natural pauses
- Dictation / voice commands
- Applications where accuracy > latency
- Turn-based dialogue

**Example:**
```bash
cargo run --example transcribe_with_vad --release -- audio.wav
```

**Output:**
```
[Segment 1] 0.21s - 8.93s (8.72s, 3 phrases) - "Of course it was impossible to connect the dots looking forward when I was in college, but it was very very clear looking backwards ten years later."
[Segment 2] 8.94s - 14.85s (5.91s, 4 phrases) - "Again, you can't connect the dots looking forward, you can only connect them looking backwards."
```

## 2. Rolling Buffer Streaming (transcribe_streaming)

**How it works:**
- Maintains a rolling window of last N seconds
- Transcribes entire buffer every M seconds
- Commits lines when buffer fills
- Keeps overlap for context continuity

**Configuration:**
```rust
BLOCK_DURATION_SECS: 0.75   // How often to transcribe
MAX_BUFFER_SECS: 10.0       // Rolling window size
OVERLAP_SECS: 0.25          // Context kept after commit
```

**Pros:**
- ✅ Low latency (block duration ~750ms)
- ✅ Continuous updates (no waiting for pauses)
- ✅ Works with continuous speech
- ✅ Matches index.html streaming behavior

**Cons:**
- ❌ Lower quality (cuts at arbitrary boundaries)
- ❌ Repetitive transcriptions as buffer rolls
- ❌ Lines may split mid-sentence when committing
- ❌ Higher compute (transcribes every block)

**Best for:**
- Live captions / subtitles
- Real-time feedback (low latency critical)
- Continuous speech without clear pauses
- Matching web UI behavior

**Example:**
```bash
cargo run --example transcribe_streaming --release -- audio.wav
```

**Output:**
```
[0.75s] Buffer: 0.75s → "Of course it was impossible."
[1.50s] Buffer: 1.50s → "Of course it was impossible to connect the dots."
[2.25s] Buffer: 2.25s → "Of course it was impossible to connect the dots looking forward."
...
[9.75s] Buffer: 10.00s → "...ten years later. Again you can't."
  → Committing line (buffer full)
[10.50s] Buffer: 1.75s → "You can't connect the dots."
```

## Implementation Details

### Shared Code

The `streaming_buffer` module (`src/streaming_buffer.rs`) provides reusable buffer management:

```rust
use speech::streaming_buffer::StreamingBuffer;

let mut buffer = StreamingBuffer::new(
    10.0,  // max_buffer_secs
    0.25,  // overlap_secs
    16000  // sample_rate
);

// Add samples
let should_commit = buffer.push_samples(chunk);

// Transcribe
let audio = buffer.get_buffer();
let text = transcribe(&audio);
buffer.update_current_line(text);

// Commit when full
if should_commit {
    buffer.commit_and_trim(chunk.len());
}
```

This module can be used in:
- `examples/transcribe_streaming.rs` (CLI)
- `src/lib.rs` Node.js module (future integration)
- Custom applications

### index.html Compatibility

The `transcribe_streaming` example implements the exact same buffering logic as `index.html`:

**JavaScript (index.html):**
```javascript
function bufferAppendKeepLast(buf, chunk, maxLen) { ... }
function commitTrim(buf, lastChunkLen) { ... }

// Every BLOCK_DURATION (750ms):
buffer16k = bufferAppendKeepLast(buffer16k, chunk16k_i16, MAX_BUFFER);
samples_since_commit += chunk16k_i16.length;

if (samples_since_commit >= MAX_BUFFER) {
    samples_since_commit = 0;
    buffer16k = commitTrim(buffer16k, chunk16k_i16.length);
    if (currentLine.trim()) committedLines.push(currentLine.trim());
    currentLine = "";
}
```

**Rust (streaming_buffer.rs):**
```rust
pub fn push_samples(&mut self, samples: &[f32]) -> bool {
    // Add samples, maintain rolling window
    for &sample in samples {
        if self.buffer.len() >= self.max_buffer_samples {
            self.buffer.pop_front();
        }
        self.buffer.push_back(sample);
    }

    self.samples_since_commit += samples.len();
    self.samples_since_commit >= self.max_buffer_samples
}

pub fn commit_and_trim(&mut self, last_chunk_len: usize) {
    // Commit line
    if !self.current_line.trim().is_empty() {
        self.committed_lines.push(self.current_line.trim().to_string());
    }

    // Trim to (last_chunk_len + overlap)
    let keep_samples = (last_chunk_len + self.overlap_samples).min(self.buffer.len());
    // ...drop old samples...

    self.samples_since_commit = 0;
}
```

## Choosing an Approach

### Use VAD-Based Streaming when:
- ✅ You need high-quality, accurate transcriptions
- ✅ Audio has natural pauses (conversations, dictation)
- ✅ Sentence structure and punctuation matter
- ✅ 500ms latency is acceptable
- ✅ You want automatic punctuation

### Use Rolling Buffer Streaming when:
- ✅ You need low latency (~750ms)
- ✅ You need continuous updates without waiting
- ✅ You want to match index.html web UI behavior
- ✅ Audio is continuous without clear pauses
- ✅ You prioritize responsiveness over accuracy

### Hybrid Approach (Future)
Combining both approaches could provide:
- VAD to segment at natural boundaries
- Rolling buffer for continuous updates between pauses
- Best of both worlds: quality + responsiveness

## Performance Comparison

**dots.wav (35 seconds, Steve Jobs speech):**

| Approach | Latency | Transcriptions | Quality | Output Lines |
|----------|---------|----------------|---------|--------------|
| VAD-Based | 500ms (pause) | 5 segments | High | 5 complete sentences |
| Rolling Buffer | 750ms (block) | 48 updates | Medium | 3 partial + 1 final |

**Transcription Quality:**

VAD-Based:
```
"Of course it was impossible to connect the dots looking forward when I was in college, but it was very very clear looking backwards ten years later."
"Again, you can't connect the dots looking forward, you can only connect them looking backwards."
```

Rolling Buffer:
```
"...ten years later. Again you can't."
"Dots looking forward you can only connect them looking backwards..."
```

The rolling buffer cuts at arbitrary 10-second boundaries, resulting in fragmented sentences at commit points.

## Node.js Integration

### Current (VAD-Based)
```javascript
const speech = new Speech((transcription) => {
  console.log(transcription.text);
  // Called when 500ms pause detected
});

speech.input(audioChunk); // 4096 samples (256ms)
```

### Future (Rolling Buffer)
```javascript
const speech = new StreamingSpeech({
  onUpdate: (currentLine) => {
    // Called every 750ms with current buffer transcription
    updateSubtitle(currentLine);
  },
  onCommit: (committedLine) => {
    // Called when buffer fills (10 seconds)
    addToTranscript(committedLine);
  }
});

speech.input(audioChunk);
```

## Files

- `src/streaming_buffer.rs` - Shared buffer management
- `examples/transcribe_with_vad.rs` - VAD-based approach
- `examples/transcribe_streaming.rs` - Rolling buffer approach
- `index.html` - Web UI with rolling buffer (JavaScript)
- `src/lib.rs` - Node.js module (currently VAD-based)
