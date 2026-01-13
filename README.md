# Parakeet Speech Recognition - Rust Implementation

Pure Rust implementation of NVIDIA Parakeet CTC ASR using Candle, with GGUF quantization support and Silero VAD integration.

## Features

- ✅ **Pure Rust** - No Python dependencies for inference
- ✅ **GGUF Quantization** - Q8_0 format for 2.65x compression with minimal accuracy loss
- ✅ **Silero VAD** - Intelligent speech detection with automatic punctuation
- ✅ **Streaming Ready** - Process audio chunks in real-time
- ✅ **Node.js Module** - Use from JavaScript/TypeScript applications
- ✅ **Metal/GPU Support** - Hardware acceleration on macOS

## Prerequisites

### System Requirements
- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))
- Python 3.8+ with venv (only needed for downloading models)
- ~3GB disk space for models

### Install Dependencies

**macOS:**
```bash
brew install zstd  # For model compression
```

**Ubuntu/Debian:**
```bash
sudo apt install zstd python3-venv
```

**Cargo dependencies are handled automatically**

## Quick Start from Fresh Checkout

**Complete setup in 3 commands:**

```bash
# 1. Install Python dependencies
pip install huggingface_hub safetensors torch

# 2. Download and setup both models (~800MB total)
python scripts/download_vad.py  # No PyTorch required
python scripts/download_parakeet.py --repo nvidia/parakeet-ctc-0.6b

# 3. Run transcription
cargo run --release --features quantized --example transcribe_with_vad -- audio.wav
```

That's it! The scripts automatically download, quantize, compress, and place all files in the correct locations.

## Detailed Setup

### 1. Download and Setup Models

Download both the Parakeet and VAD models:

```bash
# Install Python dependencies (one-time)
# Note: PyTorch only needed for Parakeet quantization, not VAD
pip install huggingface_hub safetensors torch

# Step 1: Download and setup Silero VAD model
python scripts/download_vad.py

# Step 2: Download and quantize Parakeet CTC model
python scripts/download_parakeet.py --repo nvidia/parakeet-ctc-0.6b
```

**What these scripts do:**

`download_vad.py`:
- Downloads Silero VAD v4.0 from GitHub in safetensors format
- No PyTorch conversion needed
- Compresses and places files in `assets/` directory

`download_parakeet.py`:
- Downloads Parakeet model from Hugging Face
- Quantizes to Q8_0 GGUF format (recommended)
- Compresses all files and places in `assets/` directory

After running both scripts, your `assets/` directory will contain:
- `vad16.safetensors.zst` - VAD model (~948KB)
- `vad16.config.json.zst` - VAD config
- `model_q8_0.gguf.zst` - Quantized Parakeet (~790MB)
- `config.json.zst` - Parakeet config
- `tokenizer.json.zst` - Tokenizer

All files are automatically decompressed at runtime.

### 2. Run Transcription

#### Option A: VAD-based Transcription (Recommended)

Uses Silero VAD to detect speech and transcribe at natural pauses:

```bash
# Build
cargo build --release --features quantized

# Transcribe with automatic pause detection
cargo run --release --features quantized --example transcribe_with_vad -- audio.wav

# Force CPU mode (if GPU has issues)
PARAKEET_DEVICE=cpu cargo run --release --features quantized \
  --example transcribe_with_vad -- audio.wav
```

**Output:**
```
[Segment 1] 0.21s - 8.93s (8.72s, 3 phrases) - "Of course it was impossible..."
[Segment 2] 8.94s - 14.85s (5.91s, 4 phrases) - "Again, you can't connect..."
```

#### Option B: Direct Transcription (No VAD)

Transcribe entire audio file without speech detection:

```bash
cargo run --release --features quantized \
  --example transcribe_quantized -- audio.wav
```

#### Using FP32 (Full Precision) Instead of Quantized

```bash
# Build without quantized feature
cargo build --release --no-default-features

# Run with FP32 weights
cargo run --release --no-default-features \
  --example transcribe_with_vad -- audio.wav
```

## Audio Format Requirements

The model expects audio in this format:
- **Sample rate**: 16kHz
- **Channels**: Mono (1 channel)
- **Format**: WAV (PCM16)

### Convert Audio with FFmpeg

```bash
# Convert any audio to correct format
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# From video
ffmpeg -i video.mp4 -ar 16000 -ac 1 audio.wav

# Resample existing WAV
ffmpeg -i input.wav -ar 16000 -ac 1 output.wav
```

## Node.js Module Usage

### Build Node Module

```bash
# Install Node.js dependencies
npm install

# Build native module
npm run build
```

### Use in JavaScript/TypeScript

```javascript
const { Speech } = require('./index.node');

// Create speech instance with callback
const speech = new Speech('assets', (transcription) => {
  console.log('Transcription:', transcription.text);
  console.log('Time:', transcription.startTime, '-', transcription.endTime);
});

// Stream audio chunks (Array of Float64, 16kHz mono)
// Chunks should be ~256ms (4096 samples at 16kHz)
speech.input(audioChunk);

// Force transcription of current segment (optional)
// Useful when audio stream ends without a natural pause
speech.flush();

// When done streaming
speech.shutdown();
```

The module uses VAD to automatically detect speech segments and transcribe at natural pauses. Call `flush()` to force transcription of any accumulated audio without waiting for a pause.

## Advanced: Manual Quantization

The `download_parakeet.py` script automatically quantizes to Q8_0 format. For more control or different quantization formats:

```bash
# Build quantization tool
cargo build --release --example quantize_gguf

# Quantize to Q8_0 (recommended - 2.65x smaller, 2-4% error)
./target/release/examples/quantize_gguf \
  .cache/parakeet/model.safetensors \
  hf_parakeet/model_q8_0.gguf \
  --format q8_0

# Quantize to Q4K (3.8x smaller, higher error, memory-constrained only)
./target/release/examples/quantize_gguf \
  .cache/parakeet/model.safetensors \
  hf_parakeet/model_q4k.gguf \
  --format q4k

# Then compress manually
zstd -19 hf_parakeet/model_q8_0.gguf -o assets/model_q8_0.gguf.zst
```

**Quantization Results:**
- **Q8_0**: 835MB (2.65x compression), 2-4% max error, **recommended**
- **Q4K**: 582MB (3.8x compression), 70-130% max error, **not recommended**

## Configuration

### VAD Parameters

Edit `examples/transcribe_with_vad.rs` or `src/lib.rs`:

```rust
const SPEECH_THRESHOLD: f32 = 0.5;           // VAD probability (0.0-1.0)
const MIN_SPEECH_DURATION_MS: f32 = 250.0;   // Minimum segment length
const PRE_BUFFER_MS: f32 = 300.0;            // Capture before speech starts
const COMMA_PAUSE_DURATION_MS: f32 = 150.0;  // Short pause → comma
const PERIOD_PAUSE_DURATION_MS: f32 = 500.0; // Long pause → period
```

**Tuning Tips:**
- Increase `SPEECH_THRESHOLD` (e.g., 0.6) to reduce false positives
- Increase `PERIOD_PAUSE_DURATION_MS` (e.g., 800ms) if speech is being cut off
- Decrease `PRE_BUFFER_MS` (e.g., 200ms) if you don't need to capture hesitations

### Device Selection

```bash
# Use CPU
PARAKEET_DEVICE=cpu cargo run --example transcribe_with_vad -- audio.wav

# Use GPU (default on macOS Metal)
cargo run --example transcribe_with_vad -- audio.wav
```

## Testing & Validation

### Validate Quantization Accuracy

```bash
# Compare quantized weights with FP32 original
cargo run --release --example compare_gguf_fp32 -- \
  hf_parakeet/model.safetensors \
  hf_parakeet/model_q8_0.gguf
```

### Test Mel Feature Extraction

```bash
# Compare Rust mel features with Python reference
cargo run --release --example test_rust_mel_v2 -- audio.wav
```

Expected output:
```
✓ Frame count matches
Max abs diff: ~0.00001
Mean abs diff: ~0.000001
```

### Test GGUF Loading

```bash
# Verify GGUF file can be loaded
cargo run --release --example test_gguf_load -- hf_parakeet/model_q8_0.gguf
```

## Project Structure

```
speech/
├── src/
│   ├── lib.rs                    # Node.js module API
│   ├── parakeet/
│   │   ├── mod.rs                # Module exports
│   │   ├── fast_conformer.rs    # Conformer model implementation
│   │   └── assets.rs             # Embedded asset management
│   ├── silero.rs                 # Silero VAD implementation
│   └── main.rs                   # Old VAD binary (not used)
├── examples/
│   ├── transcribe_with_vad.rs    # VAD + Parakeet (recommended)
│   ├── transcribe_quantized.rs   # Direct transcription
│   ├── quantize_gguf.rs          # Model quantization tool
│   ├── compare_gguf_fp32.rs      # Validate quantization
│   └── test_*.rs                 # Various tests
├── scripts/
│   └── download_parakeet_safetensors.py  # Download models
├── assets/                       # Compressed model assets
│   ├── *.json.zst               # Configs and tokenizer
│   ├── model_q8_0.gguf.zst      # Quantized model
│   └── vad16.*.zst              # VAD model
├── hf_parakeet/                 # Uncompressed models (gitignored)
│   ├── model.safetensors        # FP32 weights
│   ├── model_q8_0.gguf          # Quantized weights
│   └── *.json                   # Configs
└── CLAUDE.md                    # Development guide
```

## Troubleshooting

### "failed to load VAD config from assets"

The VAD model files are missing. Download them with:
```bash
python scripts/download_vad.py
```

This will download and compress the Silero VAD model to `assets/`. No PyTorch required.

### "No GGUF file found"

The Parakeet model files are missing. Download them with:
```bash
pip install huggingface_hub safetensors
python scripts/download_parakeet.py --repo nvidia/parakeet-ctc-0.6b
```

This will download, quantize, and compress the Parakeet model to `assets/`.

### Metal GPU Errors

Force CPU mode:
```bash
PARAKEET_DEVICE=cpu cargo run --example transcribe_with_vad -- audio.wav
```

### "Audio must be 16kHz"

Convert with FFmpeg:
```bash
ffmpeg -i input.wav -ar 16000 -ac 1 output_16k.wav
```

### Poor Transcription Quality

1. **Check audio format**: Must be 16kHz mono
2. **Try FP32 mode**: `--no-default-features` for full precision
3. **Adjust VAD threshold**: Increase `SPEECH_THRESHOLD` if detecting noise
4. **Check audio quality**: Clean audio works best

### Node Module Build Fails

```bash
# Ensure Rust and Node are installed
rustc --version
node --version

# Clean and rebuild
rm -rf target/ node_modules/
npm install
npm run build
```

### Transcriptions Cut Off at Start

This is fixed with the pre-buffer. If still occurring:
- Increase `PRE_BUFFER_MS` to 500ms
- Lower `SPEECH_THRESHOLD` to detect speech earlier

## Performance

**Model Sizes:**
- FP32 Safetensors: 2.3GB
- Q8_0 GGUF: 835MB (recommended)
- Q4K GGUF: 582MB (memory-constrained only)

**Inference Speed (M2 MacBook):**
- Real-time factor: ~0.1x (10x faster than real-time)
- 1 minute audio → ~6 seconds processing
- Quantized vs FP32: Similar speed (GPU bound)

**Memory Usage:**
- Quantized: ~1.2GB RAM
- FP32: ~3GB RAM

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Complete development guide for Claude Code
- **[FEATURE_FLAGS.md](FEATURE_FLAGS.md)** - Quantized vs FP32 feature flags
- **[PARAKEET_CTC_IMPLEMENTATION.md](PARAKEET_CTC_IMPLEMENTATION.md)** - Architecture details
- **[GGUF_QUANTIZATION.md](GGUF_QUANTIZATION.md)** - Quantization guide

## Examples

### Transcribe File

```bash
cargo run --release --features quantized --example transcribe_with_vad -- speech.wav
```

### Batch Process

```bash
for file in audio/*.wav; do
  echo "Processing: $file"
  PARAKEET_DEVICE=cpu cargo run --release --features quantized \
    --example transcribe_with_vad -- "$file" > "${file%.wav}.txt"
done
```

### Use in Node.js

```javascript
const { Speech } = require('./index.node');
const fs = require('fs');
const WavDecoder = require('wav-decoder');

async function transcribeFile(wavPath) {
  const buffer = fs.readFileSync(wavPath);
  const audioData = await WavDecoder.decode(buffer);

  const samples = audioData.channelData[0]; // Float32Array
  const float64Samples = new Float64Array(samples);

  const speech = new Speech('assets', (transcription) => {
    console.log(transcription.text);
  });

  // Send in chunks of 4096 samples (256ms at 16kHz)
  const chunkSize = 4096;
  for (let i = 0; i < float64Samples.length; i += chunkSize) {
    const chunk = float64Samples.slice(i, i + chunkSize);
    speech.input(Array.from(chunk));
  }

  // Wait for processing to complete
  await new Promise(resolve => setTimeout(resolve, 1000));

  // Force transcription of any remaining audio
  speech.flush();

  // Wait for flush to complete
  await new Promise(resolve => setTimeout(resolve, 1000));

  speech.shutdown();
}

transcribeFile('audio.wav');
```

## License

See the NVIDIA Parakeet model license at:
https://huggingface.co/nvidia/parakeet-ctc-0.6b

## Credits

- **NVIDIA** - Parakeet CTC model
- **Snakers4** - Silero VAD
- **Candle** - Rust ML framework
- **GGUF** - Quantization format from llama.cpp

## Support

For issues:
- Check [CLAUDE.md](CLAUDE.md) for detailed implementation notes
- Verify audio is 16kHz mono WAV
- Try CPU mode if GPU fails
- Check disk space (~3GB needed for models)
