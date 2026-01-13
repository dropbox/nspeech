# Parakeet Performance Analysis

## Problem Statement
- Parakeet transcription is slow (~250-3500ms per segment)
- **Quantized inference is SLOWER than FP32** (unexpected!)

## Investigation Findings

### 1. Quantized Model Architecture

The current implementation uses QMatMul for linear layers:
- ✅ Feed-forward networks (2 per block, 48 total): **QMatMul**
- ✅ Attention projections (4 per block, 96 total): **QMatMul**
- ❌ Convolution layers (3 per block, 72 total): **Dequantized to FP32**
- ❌ LayerNorm (5 per block, 120 total): **Dequantized to FP32**
- ❌ Subsampling Conv2D layers: **Dequantized to FP32**

**Critical Discovery**: From `candle-core/src/quantized/mod.rs:723-737`:

```rust
pub fn from_arc(qtensor: std::sync::Arc<QTensor>) -> Result<Self> {
    let dequantize = match qtensor.dtype() {
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => true,
        _ => DEQUANTIZE_ALL.with(|b| *b),  // Check CANDLE_DEQUANTIZE_ALL env var
    };
    let t = if dequantize {
        Self::Tensor(tensor)  // FP32 matmul - SLOW!
    } else {
        Self::QTensor(qtensor)  // Quantized matmul - FAST!
    };
}
```

### 2. Possible Issues

#### A. Environment Variable Override
**Check**: Is `CANDLE_DEQUANTIZE_ALL` set?
```bash
echo $CANDLE_DEQUANTIZE_ALL
```

If set to "1", ALL quantized weights are dequantized to FP32 at load time, defeating the purpose of quantization.

#### B. Metal Quantized Operations
On macOS Metal GPU, quantized operations may have different performance characteristics:
- CPU quantized matmul: Uses optimized SIMD (AVX2/NEON)
- Metal quantized matmul: May lack optimized kernels for certain quant formats
- **Hypothesis**: Metal may be dequantizing internally or using slow fallback

#### C. Conv Layer Overhead
Since Conv layers are ALWAYS dequantized (lines 567-599 in fast_conformer.rs):
- 72 Conv1D operations per forward pass
- All weights converted from Q8_0 → FP32
- This adds significant overhead

#### D. Device Selection
From the test output: `Using CPU (forced by PARAKEET_DEVICE=cpu)`
- Running on CPU, not Metal GPU!
- CPU quantized ops should be fast (optimized with k-quants)
- But may not be as optimized as full FP32 BLAS operations

### 3. Benchmark Comparison Needed

We need to compare:
1. **Quantized on CPU** (current)
2. **FP32 on CPU**
3. **Quantized on Metal GPU**
4. **FP32 on Metal GPU**

### 4. Expected Performance

For reference, typical ASR model inference times:
- Whisper tiny (39M params): ~50-100ms per 30s segment on CPU
- Whisper base (74M params): ~100-200ms per 30s segment on CPU
- Parakeet (608M params): Should be ~500-1000ms per 30s segment on CPU

Our observed times of 3.5 seconds for 120K samples (7.5s audio) suggests:
- **~467ms per second of audio** on CPU
- This is actually reasonable for a 608M parameter model on CPU!

### 5. Why Quantized is Slower

**Hypothesis**:
- Quantized models use Q8_0 matmul kernels (8-bit integer operations)
- FP32 models use highly optimized BLAS (OpenBLAS, Intel MKL, Accelerate framework)
- On CPU, **optimized FP32 BLAS > Q8_0 kernels** for this model size
- Q8_0 kernels are optimized for memory bandwidth, not compute
- With 608M params, CPU cache misses dominate, so Q8_0 wins on memory but loses on compute

### 6. Solutions

#### Immediate: Profile to identify bottlenecks
```bash
# Run with CPU and capture detailed timings
RUST_LOG=info PARAKEET_DEVICE=cpu cargo run --example transcribe_with_vad --release -- dots.wav 2>&1 | grep "ms"

# Check if CANDLE_DEQUANTIZE_ALL is affecting us
CANDLE_DEQUANTIZE_ALL=1 PARAKEET_DEVICE=cpu cargo run --example transcribe_with_vad --release -- dots.wav
```

#### Short-term: Try Metal GPU
```bash
# Remove PARAKEET_DEVICE=cpu to use Metal
cargo run --example transcribe_with_vad --release -- dots.wav
```

Metal GPU should be MUCH faster due to:
- Massive parallelism (thousands of cores)
- High memory bandwidth
- Optimized matmul kernels

#### Medium-term: Keep Conv layers quantized
Modify `QFastConformerBlock::new()` to use quantized Conv ops instead of dequantizing.

#### Long-term: Consider different quantization formats
- Q4_K_M: Smaller but less accurate
- Mixed precision: Q4_K for FFN, FP16 for attention
- Metal-specific optimizations

### 7. Action Items

1. ✅ Add performance timers (DONE)
2. ⚠️ **Test on Metal GPU** (remove PARAKEET_DEVICE=cpu)
3. ⚠️ **Compare quantized vs FP32 on same device**
4. ⚠️ **Check CANDLE_DEQUANTIZE_ALL environment variable**
5. ⚠️ **Profile to find specific bottlenecks** (is it matmul, conv, or attention?)
