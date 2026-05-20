# Speech

Pure Rust speech processing: ASR (speech-to-text) and TTS (text-to-speech) using the Candle deep learning framework. No Python dependencies at runtime.

## Models

| Model | Task | Parameters | Quantized Size |
|-------|------|-----------|---------------|
| **Parakeet TDT v3** | ASR (transducer, punctuation) | 600M | 620 MB |
| **Moonshine V2** | ASR (streaming encoder-decoder) | 195M | 200 MB |
| **Kokoro** | TTS (text-to-speech) | 82M | 85 MB |
| **Silero VAD** | Voice activity detection | 1M | 194 KB |

All models use GGUF Q8_0 quantization for compact storage with memory-mapped loading.

## Features

- Pure Rust inference with no Python runtime dependency
- GGUF Q8_0 quantized models (memory-mapped, fast cold start)
- Silero VAD for intelligent speech segmentation
- GPU acceleration via Triton-compiled kernels (Metal on macOS, D3D12 on Windows)
- CPU-optimized path with fbgemm packed GEMM (Linux/x86)
- Node.js native module (NAPI bindings)
- Streaming ASR with Moonshine V2
- VAD-based chunking for arbitrarily long audio files

## Prerequisites

- Rust toolchain ([rustup.rs](https://rustup.rs))
- [uv](https://docs.astral.sh/uv/) (Python package manager)

## Quick Start

```bash
make build
```

This creates a Python venv via uv, installs dependencies, downloads model weights from HuggingFace, runs the Rust quantizers (GGUF Q8_0), and builds all examples. Each step is idempotent.

### Usage

```bash
# Transcribe with Parakeet TDT + VAD segmentation
cargo run --example transcribe_tdt_with_vad --release -- audio.wav

# Transcribe with Moonshine V2 + VAD
cargo run --example transcribe_moonshine_with_vad --release -- audio.wav

# Streaming transcription (Moonshine)
cargo run --example transcribe_moonshine_streaming --release -- audio.wav

# Text-to-speech (Kokoro)
cargo run --example synthesize_kokoro --release -- "Hello world" output.wav
```

### The `speek` CLI

A single binary combining TTS + audio playback:

```bash
make speek
echo "Hello from the GPU" | ./speek
speek "Ninety five point three percent accuracy"
```

Install to `~/bin` and register AI coding assistant skills:

```bash
make speek-install
```

This installs skills for Claude Code and Codex that cause them to proactively speak a one-sentence summary aloud at the end of each completed task.

## Building

The Makefile auto-detects platform and selects appropriate features:

```bash
make build          # Download models + build all examples (default)
make speek          # Build the speek CLI
make module         # Build Node.js native module
make bench          # Build and run encoder benchmark
make win            # Cross-compile for Windows (D3D12)
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `triton-metal` | Triton-compiled Metal GPU kernels (macOS) |
| `triton-d3d12` | Triton-compiled HLSL kernels (Windows D3D12) |
| `fbgemm-bf16` | CPU-optimized bf16 packed GEMM (Linux default) |
| `fast-cpu` | Pre-dequantize to F32 + BLAS with rayon parallelism |
| `use-moonshine` | NAPI binding uses Moonshine instead of Parakeet |
| `auto-transcribe-on-pause` | Auto-transcribe when silence detected in stream |
| `embed-assets` | Bake model assets into the binary |

Platform defaults (set by Makefile):
- **Apple Silicon**: `triton-metal`
- **Intel Mac**: `triton-metal`
- **Linux**: `fbgemm-bf16`
- **Windows**: `triton-d3d12`

### Manual Build

```bash
# macOS with Metal GPU
cargo build --release --features triton-metal --example transcribe_tdt_with_vad

# Linux CPU
cargo build --release --features fbgemm-bf16 --example transcribe_moonshine_with_vad

# Force CPU at runtime
PARAKEET_DEVICE=cpu cargo run --example transcribe_tdt_with_vad --release -- audio.wav
```

## Node.js Module

The library exports NAPI bindings for use from JavaScript/TypeScript:

```javascript
const { Speech, setLogCallback } = require('./index.node');

const transcriber = new Speech('assets', (transcription) => {
  console.log(transcription.text);
});

// Feed 16kHz mono float32 audio
transcriber.input(new Float32Array(samples));
transcriber.flush();
transcriber.shutdown();
```

Build the module:
```bash
make module
cp target/release/libspeech.dylib index.node  # macOS
```

## GPU Kernels

The `triton-metal` and `triton-d3d12` features use pre-compiled GPU kernels (Metal AIR / DXIL bytecode) that are checked into the repo under `kernels/out/*.tar.zst`. These are embedded into the binary at build time — no runtime kernel compilation occurs.

The kernels are compiled from Triton Python sources (`kernels/*.py`) using a custom Triton compiler fork that targets Metal and D3D12. This compiler is not publicly available. If you need to modify the kernels, you can:

1. Edit the Triton Python sources in `kernels/`
2. Compile with `cd kernels && python build.py` (requires the Triton venv with the custom fork)
3. The build system detects if the venv is missing and falls back to pre-built archives

For users without the custom compiler, the pre-built kernel archives work out of the box on all supported hardware. The `fbgemm-bf16` feature (Linux default) uses CPU-only code paths and does not require GPU kernels.

## Audio Requirements

All models expect:
- 16 kHz sample rate
- Mono (single channel)
- 16-bit PCM WAV or float32 samples in [-1, 1]

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

Model weights have their own licenses:
- Parakeet TDT: CC-BY-4.0 (NVIDIA)
- Moonshine V2: MIT (Useful Sensors)
- Kokoro: Apache-2.0 (Hexgrad)
- Silero VAD: MIT (Silero)
