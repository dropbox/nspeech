# Feature Flags

## Quantized Inference

The `quantized` feature controls whether to use quantized inference (faster) or full-precision FP32 inference (potentially more accurate).

### Default: Quantized Inference (Recommended)

By default, the library uses **quantized inference** which provides:
- ✅ **Faster inference** - Uses QMatMul with Q8_0/Q4K quantized weights
- ✅ **Lower memory usage** - Weights stay in compressed format
- ✅ **Good accuracy** - Q8_0 quantization has minimal accuracy loss

```bash
# Build with default quantized inference
cargo build --release

# Run example with quantized inference
cargo run --example transcribe_with_vad --release -- audio.wav
```

**Output:**
```
Loading Q8_0 quantized model (recommended, compressed)
  Creating quantized VarBuilder (keeps weights in Q8_0/Q4K format)...
  ✓ Quantized VarBuilder created (weights stay quantized for speed)
  Building model...
✓ Quantized model loaded successfully
```

### FP32 Full-Precision Inference

To use full-precision FP32 inference, disable the `quantized` feature:

```bash
# Build without quantized feature (FP32)
cargo build --release --no-default-features

# Run example with FP32 inference
cargo run --example transcribe_with_vad --release --no-default-features -- audio.wav
```

**Output:**
```
Loading model with FP32 full-precision inference (from safetensors)
✓ Model loaded successfully (FP32 inference)
```

**Note:** FP32 mode loads the original `.safetensors` weights (true FP32), not dequantized GGUF weights. This ensures you get full-precision inference without any quantization artifacts.

### When to Use Each Mode

**Use Quantized (default):**
- ✅ Production deployments
- ✅ When speed is important
- ✅ When memory is limited
- ✅ Most use cases

**Use FP32:**
- 🔬 Research/experimentation
- 🎯 When maximum accuracy is critical
- 📊 Benchmarking quantization loss
- 🐛 Debugging numerical issues

### Performance Comparison

| Mode | Speed | Memory | Accuracy | Use Case |
|------|-------|--------|----------|----------|
| **Quantized (Q8_0)** | Fast ⚡ | Low 💾 | Very Good 📊 | Production (recommended) |
| **FP32** | Baseline ⏱️ | High 💾💾 | Best 🎯 | Research/debug |

### Implementation Details

The feature flag conditionally selects the model implementation:

```rust
// With "quantized" feature (default):
pub type ParakeetCtc = QParakeetFastConformerCtc;  // Uses QMatMul with GGUF weights

// Without "quantized" feature:
pub type ParakeetCtc = ParakeetFastConformerCtc;   // Uses Linear with safetensors
```

**Weight Sources:**
- **Quantized mode**: Loads from `.gguf` files (Q8_0/Q4K quantized weights)
- **FP32 mode**: Loads from `.safetensors` files (original FP32 weights)

Both implementations:
- Have identical APIs
- Produce equivalent transcriptions
- Support all features (VAD, streaming, punctuation)

The differences are:
1. **Inference speed**: Quantized is faster due to QMatMul operations
2. **Memory usage**: Quantized uses less memory (compressed weights)
3. **Weight source**: Quantized uses GGUF, FP32 uses safetensors
4. **Precision**: FP32 has true full-precision, quantized has minimal loss
