# Streaming TDT Model Analysis

## Model Verification

**Repository:** `nvidia/nemotron-speech-streaming-en-0.6b`
**Status:** ✅ Verified - Model exists and is accessible

## Available Files

| File | Size | Notes |
|------|------|-------|
| `nemotron-speech-streaming-en-0.6b.nemo` | 2.47 GB | Main model checkpoint |
| `README.md` | 16.2 KB | Documentation |
| Various `.md` files | — | Model cards (bias, safety, privacy, explainability) |

## Architecture Details

### Model Specifications
- **Type:** Cache-Aware FastConformer-RNNT (Transducer)
- **Parameters:** 600M
- **Encoder:** 24-layer Cache-Aware FastConformer
- **Decoder:** RNNT (Recurrent Neural Network Transducer)
- **Features:** Built-in punctuation and capitalization

### Input Requirements
- **Sample Rate:** 16kHz
- **Channels:** Mono (single-channel)
- **Minimum Duration:** ≥80ms chunks

### Performance (WER @ 1.12s chunks)
- **Average WER:** 7.16%
- **LibriSpeech test-clean:** 2.31%
- **LibriSpeech test-other:** 4.75%

## Streaming Configuration

### Frame Timing
- **Frame Duration:** 80ms per frame (8x subsampling from 10ms mel frames)
- **Configuration Parameter:** `att_context_size = [left_context, right_context]`

### Chunk Size Options

| att_context_size | Chunk Size | Latency | Use Case |
|------------------|------------|---------|----------|
| `[70, 0]` | 80ms | Ultra-low | Real-time voice commands |
| `[70, 1]` | 160ms | Very low | Live captioning |
| `[70, 6]` | 560ms | Low | Standard streaming |
| `[70, 13]` | 1120ms (1.12s) | Medium | High-quality streaming |

**Note:** Left context of 70 frames = 5.6s of past context

### Cache-Aware Design Features
1. **Non-overlapping Processing:** Each audio frame processed exactly once
2. **Zero Redundant Computations:** Cached attention and convolution states reused
3. **Dynamic Latency Control:** Choose chunk size at inference time
4. **Memory Efficient:** Fixed cache size regardless of audio length

## Configuration Structure

Based on NeMo YAML configs, expect the following structure in `model_config.yaml`:

```yaml
encoder:
  _target_: nemo.collections.asr.modules.conformer_encoder.ConformerEncoder
  d_model: 1024
  n_layers: 24
  n_heads: 8
  ff_expansion_factor: 4
  conv_kernel_size: 9
  feat_in: 128  # Mel bins
  subsampling_factor: 8

  # Cache-aware specific
  att_context_size: [70, 13]  # Default: 1.12s chunks
  cache_aware: true

decoder:
  prednet:
    pred_hidden: 640
    pred_rnn_layers: 2
  vocab_size: 8192
  blank_id: 0

joint:
  jointnet:
    joint_hidden: 640
    activation: relu
```

## Implementation Requirements

### Phase 1: Download & Conversion (Current)
- [x] Create download script (`scripts/download_parakeet_streaming_tdt.py`)
- [ ] Run download (requires: `huggingface_hub`, `torch`, `safetensors`, `pyyaml`)
- [ ] Convert to BF16 safetensors
- [ ] Compress with zstd
- [ ] Quantize to GGUF Q8_0
- [ ] Verify config contains cache parameters

### Phase 2: Code Modifications
Key changes needed:
1. **FastConformer encoder** - Accept and update attention/conv caches
2. **Streaming encoder** - Implement cache initialization and update methods
3. **Model loader** - Support streaming-specific config parameters
4. **Example** - Demonstrate cache-aware streaming transcription

## Cache Memory Estimates

Per-layer cache sizes (batch_size=1, dtype=BF16):

**Attention Cache (K+V):**
- Dimensions: `[batch, num_heads, cache_len, head_dim]`
- For 70 frames context: `[1, 8, 70*8, 128]` ≈ 0.7 MB per layer
- For 24 layers: ≈ 17 MB total

**Convolution Cache:**
- Dimensions: `[batch, channels, kernel_size-1]`
- For kernel_size=9: `[1, 1024, 8]` ≈ 16 KB per layer
- For 24 layers: ≈ 0.4 MB total

**Total Cache Size:** ~17.4 MB (very manageable!)

## Comparison with Existing Approaches

| Approach | Quality | Latency | Overlap | Cache Memory | Real-time |
|----------|---------|---------|---------|--------------|-----------|
| **Baseline (non-streaming)** | 100% | Full audio | N/A | Full context | No |
| **StreamingTransducer** | 54-61% | Medium | 0.5s overlap | Medium | Yes |
| **VAD-based** | 100% | High | None | Full segments | No |
| **Cache-aware (NEW)** | Target: 95%+ | 80-1120ms | Zero | ~17MB fixed | Yes |

## Next Steps

1. **Download Model:**
   ```bash
   python scripts/download_parakeet_streaming_tdt.py --cache .cache/parakeet-streaming-tdt --assets assets
   ```

2. **Verify Config:**
   - Check for `att_context_size` parameter
   - Confirm `cache_aware: true` setting
   - Document any additional streaming-specific fields

3. **Proceed to Phase 2:**
   - Implement cache support in FastConformer
   - Create initialization and update methods
   - Test cache behavior with unit tests

## Open Questions

1. **Default Chunk Size:** Should we default to `[70, 6]` (560ms) for balanced latency/quality, or `[70, 13]` (1120ms) for maximum quality?

2. **Cache Reset Policy:** When should caches be reset?
   - Between utterances?
   - After silence periods?
   - Explicit reset() calls only?

3. **Encoder Output Shape:** Does the streaming encoder output match the standard encoder output shape for compatibility with existing predictor/joint networks?

4. **Punctuation/Capitalization:** Is this built into the model weights or requires post-processing?

## References

- Model Card: https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- NeMo Toolkit: https://github.com/NVIDIA/NeMo
- Cache-Aware Streaming: NeMo's `asr_cache_aware_streaming` examples
