# NeMo Baseline Scripts for Parakeet TDT v3

This directory contains Python scripts using NVIDIA NeMo's official implementation of Parakeet TDT 0.6B v3 to establish reference baselines for comparison with our Rust implementation.

## Setup

### Install Dependencies

```bash
uv pip install -r nemo_requirements.txt
```

Or install directly:

```bash
uv pip install nemo_toolkit[asr]
```

The NeMo toolkit will automatically download the model on first use.

## Scripts

### 1. `nemo_baseline_tdt.py` - Standard Non-Streaming Transcription

Official NeMo baseline transcription (no streaming).

**Usage:**
```bash
python nemo_baseline_tdt.py dots.wav
python nemo_baseline_tdt.py MLKDream_16k.wav
```

**What it does:**
- Loads `nvidia/parakeet-tdt-0.6b-v3` from Hugging Face Hub
- Transcribes entire audio file
- Reports word count, timing, and RTF

**Expected output for dots.wav:**
- Transcription of Steve Jobs speech
- Word count comparison with Rust implementation
- Processing time and real-time factor

### 2. `nemo_streaming_tdt.py` - Streaming/Buffered Transcription

NeMo's streaming approach with configurable chunk sizes.

**Usage:**
```bash
# Default: 1.6s chunks with 0.4s overlap
python nemo_streaming_tdt.py dots.wav

# Custom chunk size
python nemo_streaming_tdt.py dots.wav --chunk-size 3.0 --overlap 0.5

# Force CPU
python nemo_streaming_tdt.py dots.wav --device cpu
```

**Options:**
- `--chunk-size`: Chunk duration in seconds (default: 1.6s)
- `--overlap`: Overlap between chunks (default: 0.4s)
- `--device`: Device to use (`cpu`, `cuda`, or `auto`)

**What it does:**
- Demonstrates NeMo's buffered/streaming inference
- Shows how NeMo maintains LSTM state across chunks
- Compares with our Rust implementations

## Comparison with Rust Implementation

### Our Rust Implementations

**1. Non-Streaming Baseline** (`examples/transcribe_tdt.rs`):
```bash
cargo run --example transcribe_tdt --release -- dots.wav
```
- Quality: 140 tokens (100%)
- Direct port of the model architecture

**2. VAD-Based Segmentation** (`examples/transcribe_tdt_with_vad.rs`):
```bash
cargo run --example transcribe_tdt_with_vad --release -- dots.wav
```
- Quality: 140 tokens (100%) ✓
- Uses natural utterance boundaries
- **Recommended for production**

**3. Chunked Streaming** (`examples/transcribe_tdt_streaming.rs`):
```bash
cargo run --example transcribe_tdt_streaming --release -- dots.wav
```
- Quality: 99 tokens (71%)
- Fixed 3s chunks with LSTM reset
- Lower quality but predictable latency

### Quality Comparison Table

| Approach | Quality | Latency | Use Case |
|----------|---------|---------|----------|
| NeMo baseline (Python) | Reference | Full audio | Development reference |
| Rust non-streaming | 100% (140 tokens) | Full audio | Batch transcription |
| Rust VAD-based | 100% (140 tokens) | ~0.5-2s | **Production (recommended)** |
| Rust chunked | 71% (99 tokens) | ~3.5s | Low-latency priority |
| NeMo streaming (Python) | TBD | Variable | Reference comparison |

## Purpose of These Scripts

These NeMo scripts serve as **reference baselines** to:

1. **Validate our Rust implementation** - Ensure our model loading and inference matches NeMo's official implementation
2. **Compare streaming approaches** - Understand how NeMo handles streaming vs our approach
3. **Debug discrepancies** - When quality differs, determine if it's a model issue or implementation issue
4. **Document expected behavior** - Establish ground truth for what the model should produce

## Model Details

**Model**: `nvidia/parakeet-tdt-0.6b-v3`

**Architecture**:
- **Encoder**: FastConformer (24 layers, 1024 hidden, 8 heads)
- **Predictor**: 2-layer LSTM (512 hidden)
- **Joint Network**: 512 hidden → 8193 vocab
- **Blank token**: 8192

**Training**: The model was trained by NVIDIA on large-scale ASR datasets.

## Investigation Summary

Our investigation revealed:

1. **Chunked streaming** with LSTM reset achieves 71% quality
2. **VAD-based segmentation** achieves 100% quality by using natural boundaries
3. **Continuous LSTM state** across chunks failed (28% quality) due to state corruption

The NeMo baseline scripts help validate that:
- Our non-streaming implementation matches NeMo (100% ✓)
- Our VAD-based approach achieves same quality as baseline (100% ✓)
- The quality gap in chunked streaming is architectural, not a bug

## Notes

### Token vs Word Count
- **NeMo**: Reports word count (split on whitespace)
- **Rust**: Reports SentencePiece token count
- These numbers differ but both are valid metrics

### Device Selection
- NeMo will use CUDA if available
- Force CPU with `--device cpu` if needed
- Rust uses Metal on macOS, CPU otherwise

### First Run
- NeMo downloads model on first use (~2.3GB)
- Subsequent runs use cached model
- Cache location: `~/.cache/huggingface/hub/`

## Troubleshooting

### Import Error: NeMo not installed
```bash
uv pip install nemo_toolkit[asr]
```

### CUDA/GPU Issues
```bash
python nemo_baseline_tdt.py dots.wav --device cpu
```

### Model Download Issues
If the model fails to download automatically:
```bash
huggingface-cli download nvidia/parakeet-tdt-0.6b-v3
```

## References

- **NeMo Documentation**: https://docs.nvidia.com/nemo-framework/
- **Parakeet TDT Model Card**: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- **NeMo GitHub**: https://github.com/NVIDIA/NeMo
- **Our Investigation**: See `FINAL_SOLUTION_SUMMARY.md`
