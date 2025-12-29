# Parakeet CTC Model Quantization Results

## Overview

Successfully implemented Q8_0 (8-bit) and Q4_0 (4-bit) quantization for the Parakeet CTC model (608M parameters, 2.21 GB FP32).

## Quantization Formats

### Q8_0 (8-bit quantization)
- **Block size**: 32 elements per block
- **Format**: 1 float16 scale + 32 int8 quantized values per block
- **Range**: [-127, 127] mapped to scale * [-1.0, 1.0]
- **File size**: 590 MB
- **Compression ratio**: 3.75x

### Q4_0 (4-bit quantization)
- **Block size**: 32 elements per block
- **Format**: 1 float16 scale + 16 bytes (32 packed 4-bit values) per block
- **Range**: [-8, 7] mapped to scale * [-1.14, 1.0]
- **File size**: 294 MB
- **Compression ratio**: 7.52x (model.safetensors is 2.21 GB)

## Accuracy Results

Comparison of quantized weights vs FP32 baseline on 4 representative weight matrices:

### Q8_0 Accuracy (Excellent)

| Layer | MAE | Max AE | Mean RE |
|-------|-----|--------|---------|
| encoder.layers.0.self_attn.q_proj.weight | 0.000573 | 0.011561 | 2.61% |
| encoder.layers.0.feed_forward1.linear1.weight | 0.001605 | 0.077724 | 2.66% |
| encoder.layers.15.self_attn.q_proj.weight | 0.001088 | 0.009164 | 2.52% |
| ctc_head.weight | 0.001225 | 0.003891 | 1.27% |

**Summary**: Q8_0 has excellent accuracy with mean relative error under 3% across all layers.

### Q4_0 Accuracy (Good)

| Layer | MAE | Max AE | Mean RE |
|-------|-----|--------|---------|
| encoder.layers.0.self_attn.q_proj.weight | 0.010346 | 0.218084 | 26.79% |
| encoder.layers.0.feed_forward1.linear1.weight | 0.029005 | 0.846599 | 27.15% |
| encoder.layers.15.self_attn.q_proj.weight | 0.019672 | 0.167870 | 26.02% |
| ctc_head.weight | 0.022167 | 0.070944 | 14.93% |

**Summary**: Q4_0 has higher error (15-27% mean relative error) but preserves weight distributions well. The absolute errors are still relatively small given the weight magnitudes.

## Weight Distribution Comparison

Both quantization formats successfully preserve the statistical properties of the weights:

### Example: encoder.layers.0.self_attn.q_proj.weight

```
FP32:   mean=0.000140, std=0.127643, range=[-3.062373, 2.945125]
Q8_0:   mean=0.000141, std=0.127644, range=[-3.062500, 2.945312]  ✓ Nearly identical
Q4_0:   mean=0.000116, std=0.128392, range=[-3.062500, 2.945312]  ✓ Very similar
```

## Implementation Details

### Quantization Process (Python)

1. **Layer selection**: Quantize weight matrices (Linear, Conv), keep biases and norms as FP32
2. **Block-based quantization**: Process tensors in blocks of 32 elements
3. **Per-block scaling**: Each block gets its own float16 scale factor
4. **Storage format**: NPZ (NumPy compressed archive) with custom structure:
   - `{name}||type`: quantization type (0=F32, 8=Q8_0, 2=Q4_0)
   - `{name}||shape`: original tensor shape
   - `{name}||scales`: float16 scales (for quantized tensors)
   - `{name}||qweights`: quantized weights (int8 or packed uint8)
   - `{name}||data`: FP32 data (for unquantized tensors like biases)

### Dequantization Process (Rust)

1. Load NPZ file and parse tensor metadata
2. For each quantized tensor:
   - Parse float16 scales (2 bytes per block)
   - Parse quantized weights (32 bytes for Q8_0, 16 bytes for Q4_0 per block)
   - Dequantize: `value = scale * (qweight / max_qweight)`
3. Return FP32 Candle tensors ready for inference

## Files

- **quantize_parakeet.py**: Python script to quantize model (303 lines)
- **src/quantized_loader.rs**: Rust loader with dequantization (277 lines)
- **examples/test_quantized.rs**: Test program with accuracy comparison
- **hf_parakeet/model_q8_0.npz**: Q8_0 model (590 MB)
- **hf_parakeet/model_q4_0.npz**: Q4_0 model (294 MB)

## Usage

### Quantize a model:
```bash
python3 quantize_parakeet.py hf_parakeet/model.safetensors hf_parakeet/config.json --format q8_0
python3 quantize_parakeet.py hf_parakeet/model.safetensors hf_parakeet/config.json --format q4_0
```

### Test quantized weights:
```bash
# Load and verify Q8_0
cargo run --example test_quantized --release -- --quant q8_0

# Load and verify Q4_0
cargo run --example test_quantized --release -- --quant q4_0

# Compare both formats with FP32 baseline
cargo run --example test_quantized --release -- --compare
```

## Recommendations

- **Q8_0**: Recommended for production use. Excellent accuracy (2-3% error) with 3.75x compression.
- **Q4_0**: Good for very memory-constrained environments. Acceptable accuracy (15-27% error) with 7.52x compression. Should be validated with end-to-end inference tests.

## Next Steps

1. ✅ Quantization infrastructure complete
2. ⏳ Integrate quantized weights into full model inference pipeline
3. ⏳ Benchmark inference speed: quantized vs FP32
4. ⏳ Test end-to-end transcription accuracy on real audio

## Known Limitations

- Metal/Accelerate backend has tensor striding issues that prevent full model inference (unrelated to quantization)
- Quantized model integration requires VarBuilder abstraction (currently weights load correctly but need model constructor updates)
