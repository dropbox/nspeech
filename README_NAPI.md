# NAPI Bindings for Node.js

The NAPI interface has been integrated into the main library at `src/napi_bindings.rs`.

## Building for Node.js

To build the Node.js bindings:

```bash
cargo build --release --features napi
```

This will create a `.dylib` (macOS) / `.so` (Linux) / `.dll` (Windows) file that can be loaded from Node.js.

## Usage from Node.js

```javascript
const { TranscriptionStream, init_logging } = require('./target/release/libparakeet.node');

// Optional: enable logging
init_logging();

// Create a transcription stream
const stream = new TranscriptionStream('./assets', (transcription) => {
  console.log(`[${transcription.start_time.toFixed(2)}s - ${transcription.end_time.toFixed(2)}s]`);
  console.log(`  "${transcription.text}"`);
});

// Stream audio samples (16kHz mono, normalized to [-1, 1])
// Float64Array of audio samples
await stream.input(audioSamples);
```

## Required Assets Structure

Your `assets` directory should contain:

```
assets/
├── vad16.safetensors          # Silero VAD model
├── vad16.config.json          # VAD config
└── hf_parakeet/               # Parakeet model directory
    ├── config.json
    ├── model_q8_0.gguf        # Quantized model
    └── tokenizer.json
```

## Location in Codebase

- **NAPI bindings**: `src/napi_bindings.rs`
- **Feature flag**: Enable with `--features napi`
- **Build configuration**: `Cargo.toml` (lib crate-type includes "cdylib")
