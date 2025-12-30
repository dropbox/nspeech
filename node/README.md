# Parakeet Node.js Module

Native Node.js bindings for streaming speech transcription using Silero VAD + Parakeet CTC.

## Features

- **Streaming transcription**: Process audio in real-time with async `input()` method
- **Voice Activity Detection**: Only transcribes actual speech (using Silero VAD)
- **Automatic pause detection**: Emits transcriptions at natural speech boundaries
- **Quantized models**: Uses Q8_0 GGUF quantization for efficient inference
- **Pure Rust/Native**: No Python dependencies, fast native performance

## Installation

```bash
cd node
npm install
npm run build
```

## Usage

### Basic Example

```javascript
const { TranscriptionStream, init_logging } = require('./parakeet-node');

// Optional: Enable logging
init_logging();

// Create stream with callback
const stream = new TranscriptionStream('./assets', (transcription) => {
  console.log(`[${transcription.start_time}s - ${transcription.end_time}s]`);
  console.log(`"${transcription.text}"`);
});

// Stream audio samples (16kHz mono, normalized to [-1, 1])
const samples = new Float64Array([...]); // Your audio data
stream.input(samples);

// Flush any remaining audio
stream.flush();
```

### Complete Example

See `example.js` for a complete implementation that loads and transcribes WAV files:

```bash
node example.js ../path/to/assets audio.wav
```

## API

### `TranscriptionStream`

Main class for streaming transcription.

#### Constructor

```typescript
new TranscriptionStream(
  assetsPath: string,
  callback: (transcription: Transcription) => void
)
```

**Parameters:**
- `assetsPath`: Path to directory containing model files
- `callback`: Function called for each transcription result

**Assets Directory Structure:**
```
assets/
├── vad16.safetensors
├── vad16.config.json
└── hf_parakeet/
    ├── config.json
    ├── model_q8_0.gguf  (or model_q4k.gguf)
    └── tokenizer.json
```

#### Methods

##### `input(samples: Float64Array): void`

Process audio samples and emit transcriptions via callback.

- `samples`: Audio samples (16kHz mono, normalized to [-1, 1])
- Returns: void (synchronous)
- Side effect: Calls callback for each detected speech segment

##### `flush(): Transcription | null`

Flush any remaining audio and get final transcription.

- Returns: Final transcription if any speech remains, or null

### `Transcription`

Transcription result object.

```typescript
interface Transcription {
  text: string;        // Transcribed text
  start_time: number;  // Start time in seconds
  end_time: number;    // End time in seconds
}
```

### `init_logging(): void`

Initialize Rust logging to console. Call once at startup.

## Configuration

The following parameters can be tuned by modifying `src/lib.rs`:

- `speech_threshold`: 0.5 (VAD probability threshold)
- `min_speech_duration_ms`: 250ms (minimum segment length)
- `min_silence_duration_ms`: 300ms (silence duration to trigger transcription)

## Model Files

### Required Files

1. **Silero VAD** (1.2 MB):
   - `vad16.safetensors`
   - `vad16.config.json`
   - Download from: https://github.com/snakers4/silero-vad

2. **Parakeet CTC** (~835 MB for Q8_0):
   - `hf_parakeet/config.json`
   - `hf_parakeet/model_q8_0.gguf` (recommended) or `model_q4k.gguf`
   - `hf_parakeet/tokenizer.json`
   - Create from: https://huggingface.co/nvidia/parakeet-ctc-0.6b

### Creating Quantized Models

From the parent directory:

```bash
# Download model
python scripts/download_parakeet_safetensors.py \
  --repo nvidia/parakeet-ctc-0.6b \
  --output ./hf_parakeet

# Create Q8_0 quantized version (recommended)
cargo run --example quantize_gguf --release -- \
  hf_parakeet/model.safetensors \
  hf_parakeet/model_q8_0.gguf \
  --format q8_0
```

## Performance

- **Latency**: Real-time processing with ~300ms pause detection
- **Memory**: ~1 GB (models + inference overhead)
- **CPU**: Optimized for Apple Silicon (Metal) and x86_64
- **Accuracy**: Q8_0 quantization provides 2-4% error vs FP32

## Audio Format

Input audio must be:
- **Sample rate**: 16kHz
- **Channels**: Mono (1 channel)
- **Format**: Normalized float samples in range [-1, 1]

## Error Handling

All methods return Promises. Errors are thrown for:
- Missing model files
- Invalid audio format
- Inference failures

```javascript
try {
  stream.input(samples);
} catch (err) {
  console.error('Transcription error:', err);
}
```

## Limitations

- Single-threaded processing (no concurrent streams)
- No streaming audio output (only completed segments)
- English language only (Parakeet model limitation)
- No beam search (greedy decoding only)

## License

Same as parent project.
