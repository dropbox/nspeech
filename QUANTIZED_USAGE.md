# Using Quantized Models in Parakeet

## Overview

Parakeet now supports GGUF quantized models for faster inference and reduced memory usage. This guide shows how to use quantized models in your code.

## Quick Start

### 1. Quantize Your Model

First, quantize the model to GGUF format (Q8_0 recommended):

```bash
cargo run --example quantize_gguf --release -- \
  hf_parakeet/model.safetensors \
  hf_parakeet/model_q8_0.gguf \
  --format q8_0
```

### 2. Use the Quantized Model

#### Simple Transcription Example

```bash
# Run transcription with quantized model
cargo run --example transcribe_quantized --release -- your_audio.wav

# Force CPU inference (sometimes more stable)
PARAKEET_DEVICE=cpu cargo run --example transcribe_quantized --release -- your_audio.wav
```

#### In Your Code

```rust
use anyhow::Result;
use parakeet::{get_device, load_parakeet_ctc_from_gguf_local, load_wav_as_features};

fn main() -> Result<()> {
    // Get device (Metal, or CPU)
    let device = get_device()?;

    // Load quantized model from local directory
    // Automatically tries Q8_0 first, then Q4K
    let model = load_parakeet_ctc_from_gguf_local("hf_parakeet", &device)?;

    // Load audio and extract features
    let features = load_wav_as_features("audio.wav", model.cfg.feat_in, &device)?;

    // Run inference
    let logits = model.forward(&features, false)?;

    // Decode transcription
    let transcripts = model.greedy_decode(&logits)?;
    println!("Transcription: {}", transcripts[0]);

    Ok(())
}
```

## API Reference

### Loading Functions

#### `load_parakeet_ctc_from_gguf_local`

Load GGUF quantized model from local directory.

```rust
pub fn load_parakeet_ctc_from_gguf_local<P: AsRef<Path>>(
    dir: P,
    device: &Device,
) -> Result<ParakeetFastConformerCtc>
```

**Expected files in directory:**
- `config.json` - Model configuration
- `model_q8_0.gguf` or `model_q4k.gguf` - Quantized weights (tries Q8_0 first)
- `tokenizer.json` - Tokenizer

**Example:**
```rust
let device = get_device()?;
let model = load_parakeet_ctc_from_gguf_local("hf_parakeet", &device)?;
```

#### `load_parakeet_ctc_from_gguf_hf`

Load GGUF quantized model from Hugging Face Hub.

```rust
pub fn load_parakeet_ctc_from_gguf_hf(
    repo_id: &str,
    gguf_filename: &str,
    device: &Device,
) -> Result<ParakeetFastConformerCtc>
```

**Example:**
```rust
let device = get_device()?;
let model = load_parakeet_ctc_from_gguf_hf(
    "nvidia/parakeet-ctc-0.6b",
    "model_q8_0.gguf",
    &device
)?;
```

**Note:** You need to upload the GGUF file to Hugging Face first.

## Feature Flag Support

The codebase includes feature flags to toggle between quantized and full precision:

```toml
# Cargo.toml
[features]
default = []           # Use full precision
quantized = []         # Enable quantized code paths
```

### Current Status

Both full precision and quantized versions are available **without** feature flags. The functions have different names:

**Full Precision:**
- `load_parakeet_ctc_from_local()` - Loads safetensors
- `load_parakeet_ctc_from_hf()` - Loads safetensors from HF

**Quantized:**
- `load_parakeet_ctc_from_gguf_local()` - Loads GGUF
- `load_parakeet_ctc_from_gguf_hf()` - Loads GGUF from HF

### Future: Feature-Gated Default

In the future, you can make quantized the default by:

1. Setting `default = ["quantized"]` in Cargo.toml
2. Adding conditional compilation in the code:

```rust
#[cfg(feature = "quantized")]
pub use load_parakeet_ctc_from_gguf_local as load_parakeet_ctc_from_local;

#[cfg(not(feature = "quantized"))]
pub use original_load_parakeet_ctc_from_local as load_parakeet_ctc_from_local;
```

This would allow code to transparently use quantized models without changes.

## Performance Comparison

### Model Size

| Format | File Size | Compression | Memory |
|--------|-----------|-------------|--------|
| FP32 Safetensors | 2.21 GB | 1.0x | High |
| Q8_0 GGUF | 835 MB | 2.65x | Medium |
| Q4K GGUF | 582 MB | 3.8x | Low |

### Accuracy

| Format | Mean Relative Error | Recommendation |
|--------|-------------------|----------------|
| FP32 | Baseline | Development/reference |
| Q8_0 | 2-4% | ✅ **Production (recommended)** |
| Q4K | 70-130% | Memory-constrained only |

### Inference Speed

Tested on Apple M1 Max with `dots.wav` (3534 frames):

| Format | Device | Time | Notes |
|--------|--------|------|-------|
| FP32 | CPU | ~2.0s | Full precision |
| Q8_0 | CPU | ~1.6s | ✅ Faster + less memory |
| Q4K | CPU | ~1.5s | Fastest but lower accuracy |

**Note:** GGUF uses optimized CPU kernels from Candle which provides ~20% speedup.

## Recommended Workflow

### Development
```rust
// Use full precision for development and debugging
let model = load_parakeet_ctc_from_local("hf_parakeet", &device)?;
```

### Production
```rust
// Use Q8_0 quantized for production
let model = load_parakeet_ctc_from_gguf_local("hf_parakeet", &device)?;
```

### Memory-Constrained Environments
```rust
// Use Q4K only if memory is critical
// Ensure to test accuracy on your use case
let model = load_parakeet_ctc_from_gguf_local("hf_parakeet", &device)?;
```

## Troubleshooting

### Model File Not Found

**Error:** `No GGUF file found in directory`

**Solution:** Quantize your model first:
```bash
cargo run --example quantize_gguf --release -- \
  hf_parakeet/model.safetensors \
  hf_parakeet/model_q8_0.gguf \
  --format q8_0
```

### Out of Memory

**Error:** Out of memory during inference

**Solution 1:** Use CPU device:
```bash
PARAKEET_DEVICE=cpu your_program
```

**Solution 2:** Use Q4K for maximum compression:
```bash
cargo run --example quantize_gguf --release -- \
  hf_parakeet/model.safetensors \
  hf_parakeet/model_q4k.gguf \
  --format q4k
```

### Slow Inference

**Solution:** GGUF quantized models should be faster. Ensure:
1. Using `--release` build
2. Running on CPU (GGUF kernels optimized for CPU)
3. Using Q8_0 or Q4K format

## Examples

All examples in `examples/` directory:

### Quantization Tools
- **`quantize_gguf.rs`** - Quantize safetensors to GGUF
- **`test_gguf_load.rs`** - Verify GGUF file loads
- **`compare_gguf_fp32.rs`** - Compare accuracy vs FP32
- **`test_gguf_inference.rs`** - Test model inference

### Transcription
- **`transcribe_quantized.rs`** - Transcribe audio with quantized model

### Usage
```bash
# Quantize model
cargo run --example quantize_gguf --release -- \
  hf_parakeet/model.safetensors \
  hf_parakeet/model_q8_0.gguf \
  --format q8_0

# Verify GGUF file
cargo run --example test_gguf_load --release -- hf_parakeet/model_q8_0.gguf

# Compare accuracy
cargo run --example compare_gguf_fp32 --release -- --gguf hf_parakeet/model_q8_0.gguf

# Transcribe audio
cargo run --example transcribe_quantized --release -- your_audio.wav
```

## Migration Guide

### Migrating Existing Code

**Before (Full Precision):**
```rust
use parakeet::{load_parakeet_ctc_from_local, get_device};

let device = get_device()?;
let model = load_parakeet_ctc_from_local("hf_parakeet", &device)?;
```

**After (Quantized):**
```rust
use parakeet::{load_parakeet_ctc_from_gguf_local, get_device};

let device = get_device()?;
let model = load_parakeet_ctc_from_gguf_local("hf_parakeet", &device)?;
```

Only the function name changes! The rest of the API remains identical.

## Benefits of GGUF Quantization

✅ **2.65x smaller** model size (Q8_0)
✅ **~20% faster** inference on CPU
✅ **Less memory** usage
✅ **Excellent accuracy** (2-4% error with Q8_0)
✅ **Pure Rust** tooling (no Python)
✅ **Industry standard** format
✅ **Optimized kernels** in Candle

## See Also

- **GGUF_QUANTIZATION.md** - Detailed quantization guide
- **QUANTIZATION_RESULTS.md** - Accuracy comparison (NPZ format)
- **examples/quantize_gguf.rs** - Source code for quantization tool
