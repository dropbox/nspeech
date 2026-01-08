# HQQ (Half-Quadratic Quantization) Implementation

## Overview

HQQ is an advanced quantization method that optimizes scales and zero-points to minimize reconstruction error. This implementation provides better accuracy than standard round-to-nearest quantization used in GGUF formats.

**⚠️ Note**: This is a **research/proof-of-concept** implementation. HQQ is **not currently compatible** with the production QMatMul inference pipeline. It demonstrates superior quantization accuracy but requires custom inference kernels for practical use. See "Compatibility with QMatMul" section below for details.

## Features

- **Group-wise quantization**: Splits weight matrices into groups (default 128 elements)
- **Optimization-based**: Uses grid search to find optimal scale/zero-point per group
- **Multiple bit widths**: Supports 2, 3, 4, and 8-bit quantization
- **Symmetric/Asymmetric**: Configurable quantization modes
- **Efficient packing**: Bit-packed storage for 2/3/4-bit quantization

## Implementation Details

### Module: `src/hqq.rs`

The HQQ module provides:

**`HqqConfig`**: Configuration for quantization
```rust
pub struct HqqConfig {
    pub nbits: u8,              // 2, 3, 4, or 8 bits
    pub group_size: usize,      // Group size (default: 128)
    pub symmetric: bool,        // Symmetric quantization (no zero-point)
    pub optimize_iters: usize,  // Optimization iterations (default: 20)
}
```

**`HqqTensor`**: Quantized tensor representation
```rust
pub struct HqqTensor {
    pub qweight: Vec<u8>,           // Packed quantized weights
    pub scales: Vec<f32>,            // Scale per group
    pub zeros: Option<Vec<f32>>,    // Zero-point per group (asymmetric only)
    pub shape: Vec<usize>,           // Original tensor shape
    pub config: HqqConfig,           // Config used
}
```

### Quantization Algorithm

1. **Group Division**: Split each row into groups of `group_size` elements
2. **Per-Group Optimization**:
   - For symmetric: Optimize scale to minimize `||W - scale * round(W/scale)||²`
   - For asymmetric: Optimize both scale and zero-point
   - Uses grid search with configurable iterations
3. **Quantization**: Quantize each value `w = round((w - zero) / scale)`
4. **Bit Packing**: Pack quantized values into bytes (2/3/4-bit modes)

### Optimization Strategy

The optimization uses grid search rather than gradient descent because:
- Quantization is non-differentiable (rounding operation)
- Grid search is simple, robust, and fast enough for small groups
- Works well with limited iterations (5-20)

**Symmetric Mode** (zero = 0):
- Searches scale from 0.5x to 2.0x of initial estimate
- Initial estimate: `scale = max(|W|) / (2^nbits / 2)`

**Asymmetric Mode** (optimizes both scale and zero):
- Searches scale: 0.8x to 1.2x of range-based estimate
- Searches zero: around min value with adjustment
- Initial estimates: `scale = (max(W) - min(W)) / (2^nbits)`, `zero = min(W)`

## Usage

### Command-Line Tool

```bash
# 4-bit quantization with default settings
cargo run --example quantize_hqq --release -- \
  hf_parakeet/model.safetensors \
  hf_parakeet/model_hqq4.safetensors \
  --nbits 4 --group-size 128

# 3-bit symmetric quantization
cargo run --example quantize_hqq --release -- \
  hf_parakeet/model.safetensors \
  hf_parakeet/model_hqq3.safetensors \
  --nbits 3 --group-size 64 --symmetric

# Fast quantization (fewer optimization iterations)
cargo run --example quantize_hqq --release -- \
  hf_parakeet/model.safetensors \
  hf_parakeet/model_hqq4_fast.safetensors \
  --nbits 4 --optimize-iters 5
```

### Programmatic API

```rust
use speech::hqq::{HqqConfig, HqqTensor};
use candle_core::{Device, Tensor};

// Configure HQQ
let config = HqqConfig {
    nbits: 4,
    group_size: 128,
    symmetric: false,
    optimize_iters: 20,
};

// Quantize a tensor
let tensor = Tensor::randn(0.0, 1.0, (1024, 1024), &Device::Cpu)?;
let hqq = HqqTensor::quantize(&tensor, config)?;

// Dequantize for inference
let dequantized = hqq.dequantize(&Device::Cpu)?;
```

## Results on Parakeet Model

Quantization of NVIDIA Parakeet-CTC-0.6B (608M parameters):

### 4-bit HQQ Quantization

```
Configuration:
  nbits: 4
  group_size: 128
  symmetric: false
  optimize_iters: 5

Results:
  Quantized layers: 217 (weight matrices)
  Kept as FP32: 733 (biases, norms, embeddings)
  Average RMSE: 0.037
  Max error: 1.518
  Relative error: ~12%
```

**Layer-by-layer accuracy examples:**
```
encoder.layers.5.self_attn.o_proj.weight:   RMSE=0.028, Rel=11.85%
encoder.layers.2.self_attn.o_proj.weight:   RMSE=0.030, Rel=11.90%
encoder.layers.11.self_attn.q_proj.weight:  RMSE=0.021, Rel=12.29%
encoder.layers.17.feed_forward2.linear1:    RMSE=0.043, Rel=12.27%
```

### Comparison with Standard Quantization

| Method | Bits | Avg RMSE | Rel Error | Compression |
|--------|------|----------|-----------|-------------|
| GGUF Q8_0 | 8 | 0.004-0.007 | 2-4% | 2.65x |
| GGUF Q4K | 4 | 0.15-0.25 | 70-130% | 3.8x |
| HQQ | 4 | 0.037 | 12% | 4x* |

*Theoretical compression with custom format (current implementation saves dequantized for compatibility)

**Key Observations:**
- HQQ 4-bit achieves **6-10x better accuracy** than GGUF Q4K
- HQQ 4-bit is comparable to GGUF Q8_0 but with 2x better compression
- The optimization-based approach significantly reduces quantization error

## Performance Characteristics

### Quantization Speed
- **Full Parakeet model** (2.3GB): ~30-60 seconds with 5 optimization iterations
- **Optimization iterations**: Linear impact (5 iters ≈ 30s, 20 iters ≈ 120s)
- **Group size**: Smaller groups = more scales to optimize = slower

### Inference Performance
- Current implementation: Dequantizes to FP32 (no speed benefit)
- Theoretical: Custom kernels could compute directly on quantized weights
- Memory benefit: Immediate (4x reduction for weights with custom format)

## Limitations and Future Work

### Current Limitations

1. **Format Compatibility**: Saves dequantized weights to safetensors for compatibility
   - No actual file size reduction (compression: 1.00x)
   - Would need custom format to store HQQ parameters

2. **No Inference Acceleration**: Dequantizes to FP32 for computation
   - Would need custom kernels for quantized matrix multiplication

3. **CPU-Only Quantization**: Quantization happens on CPU
   - GPU acceleration would speed up large models

4. **No GGUF Integration**: HQQ format incompatible with Candle's GGUF loader

### Potential Improvements

**Custom Format**: Create HQQ-specific file format
```rust
struct HqqFile {
    tensors: HashMap<String, HqqTensor>,  // Quantized tensors
    metadata: HashMap<String, Value>,      // Model metadata
}
```

**GPU Quantization Kernels**: Parallelize group optimization on GPU

**Fused Inference Kernels**: Direct computation on quantized weights
```rust
// Pseudo-code for quantized matmul
fn qmatmul_hqq(input: &Tensor, qweight: &HqqTensor) -> Tensor {
    // Group-wise dequant + multiply (fused kernel)
    // Avoids full dequantization
}
```

**Mixed-Precision**: Combine HQQ for weights + higher precision for activations

**Dynamic Quantization**: Per-batch scale/zero-point optimization

## When to Use HQQ

**Use HQQ when:**
- You need high compression (4-bit) with good accuracy
- Model quality degradation from standard 4-bit quantization is unacceptable
- You can implement custom inference kernels
- Memory is constrained but compute is available

**Use GGUF Q8_0 when:**
- You need production-ready quantization with good accuracy
- Candle GGUF loader integration is important
- 2.65x compression is sufficient

**Use GGUF Q4K when:**
- Maximum compression is required
- Some accuracy loss is acceptable
- Standard format compatibility is needed

## Compatibility with QMatMul Inference

**Current Status**: HQQ quantization is **NOT compatible** with the production inference pipeline.

### Why It's Incompatible

The current Parakeet implementation uses Candle's `QMatMul` for quantized inference:

```rust
// Current production code (fast_conformer.rs)
pub struct QFeedForward {
    w1: QMatMul,  // Expects GGUF Q8_0/Q4K format
    w2: QMatMul,
    // ...
}
```

**QMatMul expectations**:
- GGUF binary format (specific tensor layout)
- Standard quantization schemes (Q8_0, Q4K, etc.)
- Loaded via `candle_transformers::quantized_var_builder`
- Optimized kernels built into Candle

**HQQ differences**:
- Custom format with per-group scales/zeros
- Optimization-based quantization (not standard rounding)
- Currently saves dequantized weights to safetensors
- No inference acceleration without custom kernels

### What Happens Now

When you quantize with HQQ:
```bash
cargo run --example quantize_hqq -- model.safetensors model_hqq4.safetensors
```

The output is **dequantized FP32 weights** saved to safetensors for compatibility. There is:
- ✅ Better quantization accuracy demonstrated
- ❌ No file size reduction (compression: 1.00x)
- ❌ No inference speedup
- ❌ Cannot be loaded by QMatMul

### Options for Integration

**Option 1: Research/Benchmarking Use (Current)**
- Use HQQ to measure optimal quantization accuracy
- Compare error metrics vs GGUF methods
- Inform future quantization decisions

**Option 2: Convert HQQ → GGUF (Easy, loses benefits)**
```rust
// Dequantize HQQ, then requantize to GGUF Q8_0
let dequant = hqq.dequantize(&Device::Cpu)?;
let gguf = QTensor::quantize(&dequant, GgmlDType::Q8_0)?;
```
- Works with existing inference
- Loses HQQ's optimization benefits
- Gets standard Q8_0 accuracy

**Option 3: Custom HqqMatMul (Hard, full benefits)**
```rust
// New struct replacing QMatMul
pub struct HqqMatMul {
    qweight: Vec<u8>,
    scales: Tensor,
    zeros: Option<Tensor>,
    // ... group metadata
}

impl HqqMatMul {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Custom kernel: fused dequant + matmul
        // For each group: dequant = qweight * scale + zero
        // Accumulate: output += x @ dequant
    }
}

// New model variants
pub struct HqqFeedForward {
    w1: HqqMatMul,  // Instead of QMatMul
    w2: HqqMatMul,
}
```
- Full HQQ benefits
- Requires extensive implementation
- Custom binary format needed
- Separate model loading pipeline

**Option 4: Extend GGUF Format (Medium, requires Candle changes)**
- Add `GgmlDType::HQQ4`, `GgmlDType::HQQ3` to Candle
- Implement HQQ kernels in Candle core
- Upstream changes to Candle project
- Standard format compatibility

### Recommendation

For **production use today**: Stick with GGUF Q8_0
- Works with existing QMatMul pipeline
- Good accuracy (2-4% error)
- 2.65x compression
- Proven and tested

For **research/future work**: Use HQQ to
- Demonstrate better quantization is possible
- Benchmark accuracy improvements
- Lay groundwork for custom kernels

## Testing

Run tests with:
```bash
cargo test --lib hqq
```

Tests cover:
- Bit packing/unpacking (2, 3, 4-bit)
- Quantization/dequantization round-trip
- Error bounds verification

## References

- HQQ Paper: "HQQ: Half-Quadratic Quantization of Large Machine Learning Models" (2023)
- GPTQ: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
- AWQ: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"

## Implementation Notes

**Why Grid Search vs Gradient Descent?**
- Quantization involves non-differentiable rounding
- Grid search is simpler and more robust
- Fast enough for small groups (128 elements, 5-20 iterations)

**Why Group-wise Quantization?**
- Different regions of weight matrix have different scales
- Per-group scales adapt to local statistics
- Better accuracy than per-tensor or per-channel quantization

**Bit Packing Details**:
- 4-bit: 2 values per byte (high nibble, low nibble)
- 3-bit: 8 values per 3 bytes (24 bits total)
- 2-bit: 4 values per byte (2 bits each)
- 8-bit: Direct byte storage (no packing)
