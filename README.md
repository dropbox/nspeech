# Parakeet Speech Recognition - Rust Implementation

Pure Rust implementation of NVIDIA Parakeet TDT (Transducer) ASR using Candle, with GGUF quantization support and Silero VAD integration.

## Features

- ✅ **Pure Rust** - No Python dependencies for inference
- ✅ **GGUF Quantization** - Q8_0 format for smaller models and faster loading
- ✅ **Silero VAD** - Intelligent speech detection with automatic segmentation
- ✅ **Parakeet TDT v3** - State-of-the-art transducer model with punctuation
- ✅ **Node.js Module** - Use from JavaScript/TypeScript applications
- ✅ **Metal/GPU Support** - Hardware acceleration on macOS

## What's Included

**Models:**
- **Parakeet TDT v3** - 600M parameter transducer model (620MB quantized)
- **Silero VAD** - Voice activity detection (194KB quantized)

**Examples:**
- `transcribe_tdt_with_vad.rs` - VAD-based segmentation + TDT transcription

**Tools:**
- `quantize_vad_gguf.rs` - Quantize VAD model to GGUF Q8_0
- `quantize_gguf.rs` - Quantize TDT model to GGUF Q8_0
- `inspect_gguf.rs` - Inspect GGUF file contents

**Python Scripts:**
- `scripts/download_vad.py` - Download and quantize Silero VAD
- `scripts/download_parakeet_tdt.py` - Download and quantize Parakeet TDT
- `scripts/compress.py` - Compression utility

## Prerequisites

### System Requirements
- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))
- Python 3.8+ with pip (only needed for downloading models)
- ~800MB disk space for quantized models

### Install Dependencies

**macOS:**
```bash
brew install zstd  # For model compression
```

**Ubuntu/Debian:**
```bash
sudo apt install zstd python3-venv python3-pip
```

**Cargo dependencies are handled automatically**

## Quick Start

**Complete setup in 3 commands:**

```bash
# 1. Install Python dependencies
pip install huggingface_hub safetensors torch zstd

# 2. Download and setup both models (~800MB total)
python scripts/download_vad.py
python scripts/download_parakeet_tdt.py

# 3. Run transcription with VAD
cargo run --example transcribe_tdt_with_vad --release -- audio.wav
```

**Example output:**
```
Loading Silero VAD...
✓ VAD loaded

Loading Parakeet TDT model...
✓ TDT model loaded

=== STREAMING TRANSCRIPTION ===

[Segment 1] Transcribing 0.00s - 35.33s (final)
  Text: Of course, it was impossible to connect the dots looking forward...

✓ Quality matches baseline!
```

## Node.js Usage

The Rust library is exported as a Node.js native module using NAPI:

```javascript
const { Speech, setLogCallback } = require('./index.node');

// Optional: capture logs
setLogCallback((level, message) => {
  console.log(`[${level}] ${message}`);
});

// Create transcriber (loads models on first use)
const transcriber = new Speech('assets', (transcription) => {
  console.log('Transcription:', transcription.text);
  console.log('Timestamp:', transcription.start_time, '-', transcription.end_time);
});

// Feed audio samples (16kHz, mono, float32 in range [-1, 1])
const samples = new Float32Array(/* your audio data */);
transcriber.input(samples);

// Flush to get final transcription
transcriber.flush();

// Shutdown when done
transcriber.shutdown();
```

**Building the Node module:**
```bash
cargo build --lib --release
cp target/release/libspeech.dylib index.node  # macOS
# or
cp target/release/libspeech.so index.node  # Linux
```

## Architecture

### Quantization Approach

See [QUANTIZATION.md](QUANTIZATION.md) for detailed explanation of quantized storage vs. quantized inference.

**Summary:**
- **Parakeet TDT**: Partial quantized inference (QLinear for encoder/joint, FP32 for LSTM)
- **Silero VAD**: Quantized storage only (FP32 inference)

Both models use GGUF Q8_0 format for storage, providing significant disk/download savings while maintaining quality.

### Silero VAD Integration

The VAD (Voice Activity Detection) automatically:
1. Detects speech vs. silence in audio stream
2. Segments audio at natural pauses
3. Buffers context before speech starts
4. Triggers transcription after silence threshold

This approach:
- ✅ Processes only speech (saves compute)
- ✅ Natural segmentation at sentence boundaries
- ✅ Handles long audio files efficiently
- ✅ Provides timestamped output

## Development

### Project Structure

```
src/
├── lib.rs              # Node.js module and main API
├── silero.rs           # Silero VAD implementation
└── parakeet/           # Parakeet TDT model
    ├── mod.rs          # Module exports
    ├── transducer.rs   # TDT model implementation
    ├── fast_conformer.rs # Conformer encoder
    ├── features.rs     # Audio feature extraction
    ├── quantized_layers.rs # Quantized operations
    └── quantized_builder.rs # Model builder

examples/
├── transcribe_tdt_with_vad.rs  # Main transcription example
├── quantize_vad_gguf.rs        # VAD quantization tool
├── quantize_gguf.rs            # TDT quantization tool
└── inspect_gguf.rs             # GGUF inspection utility

scripts/
├── download_vad.py             # Download & quantize VAD
├── download_parakeet_tdt.py    # Download & quantize TDT
└── compress.py                 # Compression utility

assets/
├── vad16.config.json.zst       # VAD configuration
├── vad16_q8_0.gguf.zst         # VAD quantized model (194KB)
├── parakeet-tdt-config.json.zst
├── parakeet-tdt-tokenizer.json.zst
└── parakeet-tdt-model_q8_0.gguf.zst  # TDT quantized model (620MB)
```

### Building

```bash
# Library only (Node module)
cargo build --lib --release

# Example
cargo build --example transcribe_tdt_with_vad --release

# All (no warnings)
cargo build --all --release
```

### Testing

```bash
# Test transcription
cargo run --example transcribe_tdt_with_vad --release -- test.wav

# Test Node module
node test-load.js test.wav
```

## Model Details

### Parakeet TDT v3
- **Architecture**: FastConformer-RNNT transducer
- **Parameters**: ~600M
- **Quantized size**: 620MB (GGUF Q8_0)
- **Features**: Built-in punctuation and capitalization
- **Sample rate**: 16kHz
- **Source**: `nvidia/parakeet-tdt-1.1b` (v3 checkpoint)

### Silero VAD v4.0
- **Architecture**: Conv1d + LSTM
- **Parameters**: ~1M
- **Quantized size**: 194KB (GGUF Q8_0)
- **Sample rate**: 16kHz
- **Chunk size**: 512 samples (32ms)
- **Source**: `snakers4/silero-vad`

## Performance

### Quantization Results

| Model | Original | Quantized (Q8_0) | Compression | Quality |
|-------|----------|------------------|-------------|---------|
| Parakeet TDT | 2.3 GB | 620 MB | 3.7x | Identical |
| Silero VAD | 948 KB | 194 KB | 6.2x | Identical |

### Runtime Performance
- **RTF (Real-Time Factor)**: ~1.0 on Metal GPU (M1/M2)
- **Memory**: ~1.5GB peak (TDT + VAD loaded)
- **Latency**: Depends on VAD segmentation (typically 1-3s)

## Troubleshooting

### Metal GPU Errors
If you see Metal-related errors, force CPU:
```bash
PARAKEET_DEVICE=cpu cargo run --example transcribe_tdt_with_vad --release -- audio.wav
```

### Missing Models
Run the download scripts:
```bash
python scripts/download_vad.py
python scripts/download_parakeet_tdt.py
```

### Audio Format
Models expect:
- 16kHz sample rate
- Mono (single channel)
- 16-bit PCM or float32 in [-1, 1]

## License

This project uses:
- **Candle** (Apache 2.0 / MIT)
- **NVIDIA Parakeet** (CC-BY-4.0)
- **Silero VAD** (MIT)

## References

- [Parakeet Models](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Candle Framework](https://github.com/huggingface/candle)
