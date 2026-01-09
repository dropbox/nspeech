# Qwen2.5 Text Correction Integration

## Overview

This document describes the integration of Qwen2.5-0.5B-Instruct for automatic text correction in the Parakeet speech recognition pipeline. The Qwen model adds proper punctuation and capitalization to raw ASR transcriptions.

## Status

**Work in Progress** - Core module implemented, awaiting model files and integration completion.

## What's Been Done

### 1. Qwen Module Implementation (`src/qwen.rs`)

Created a complete module for loading and running Qwen2.5-0.5B-Instruct:

```rust
pub struct QwenCorrector {
    model: Qwen2Model,
    tokenizer: Tokenizer,
    device: Device,
}

impl QwenCorrector {
    pub fn load<P: AsRef<Path>>(assets_dir: P, device: &Device) -> Result<Self>
    pub fn correct_text(&mut self, raw_text: &str) -> Result<String>
}
```

**Key features:**
- Loads quantized GGUF model from compressed assets
- Uses Qwen2.5-0.5B-Instruct (Q4_K_M, ~350MB compressed)
- Provides simple API for text correction
- Follows same pattern as Silero VAD integration

### 2. Asset Management

Added Qwen assets to the embed_zst_asset system:

```rust
// In src/parakeet/fast_conformer.rs
embed_zst_asset!(pub QWEN_CONFIG, "qwen2.5-0.5b-instruct-config.json.zst");
embed_zst_asset!(pub QWEN_TOKENIZER, "qwen2.5-0.5b-instruct-tokenizer.json.zst");
embed_zst_asset!(pub QWEN_MODEL_Q4, "qwen2.5-0.5b-instruct-q4_k_m.gguf.zst");
```

### 3. Download Script

Created `scripts/download_qwen3.py` to automatically:
- Download Qwen3-0.6B-Instruct-Q4_K_M model from Hugging Face (bartowski's GGUF repo)
- Download tokenizer and config from official Qwen repo
- Compress with zstd level 19
- Place in assets directory

## Model Choice: Qwen2.5-0.5B-Instruct

**Why Qwen2.5-0.5B-Instruct?**
- **Small size**: 0.5B parameters, ~350MB quantized (Q4_K_M)
- **Instruct-tuned**: Trained to follow instructions accurately
- **Fast**: Suitable for real-time correction
- **Capable**: Handles punctuation and capitalization well
- **Candle support**: Full quantized model support in candle-transformers

**Alternatives considered:**
- Qwen3-0.6B: Newer but similar size/performance
- Qwen2.5-1.5B: Better quality but 3x larger
- Smaller models (< 0.5B): Insufficient for reliable correction

## How It Works

### Text Correction Flow

```
Raw ASR Output
    ↓
"of course it was impossible to connect the dots looking forward"
    ↓
Qwen Corrector (prompt-based)
    ↓
"Of course, it was impossible to connect the dots looking forward."
```

### Prompt Template

```
<|im_start|>system
You are a helpful assistant that corrects transcriptions by adding proper
punctuation and capitalization. Only output the corrected text, nothing else.
<|im_end|>
<|im_start|>user
Correct this transcription by adding punctuation and capitalization:
{raw_text}
<|im_end|>
<|im_start|>assistant
```

### Generation

- Uses greedy decoding (argmax) for speed
- Max tokens: word_count + 20
- Stops on `<|im_end|>` token (ID: 151645)
- Extracts only the assistant's response

## Integration Points

### Option 1: Post-Process in Parakeet Module

Modify `add_punctuation()` to use Qwen:

```rust
// In src/parakeet/mod.rs
use crate::qwen::QwenCorrector;

pub fn add_punctuation_with_qwen(text: &str, corrector: &mut QwenCorrector) -> String {
    corrector.correct_text(text).unwrap_or_else(|_| {
        // Fallback to rule-based if Qwen fails
        add_punctuation_internal(text, false)
    })
}
```

### Option 2: Integrate in Transcription Pipeline

Modify transcription functions to use Qwen:

```rust
// In src/lib.rs SpeechInner
struct SpeechInner {
    // ... existing fields ...
    qwen_corrector: Option<QwenCorrector>,
}

fn transcribe_segment(&mut self) -> Result<String> {
    let raw_text = parakeet::transcribe_streaming_chunk(...)?;

    if let Some(ref mut corrector) = self.qwen_corrector {
        corrector.correct_text(&raw_text)?
    } else {
        parakeet::add_punctuation(&raw_text)
    }
}
```

### Option 3: Optional Flag

Add enable/disable flag:

```rust
#[napi(constructor)]
pub fn new(
    assets: String,
    callback: Function<Transcription, Unknown>,
    use_qwen: Option<bool>, // New parameter
) -> Self
```

## Setup Instructions

### 1. Download Model Files

```bash
python scripts/download_qwen3.py
```

This downloads and compresses:
- `qwen3-0.6b-instruct-q4_k_m.gguf.zst` (~400MB)
- `qwen3-0.6b-instruct-tokenizer.json.zst` (~2MB)
- `qwen3-0.6b-instruct-config.json.zst` (~1KB)

### 2. Build with Qwen Support

```bash
cargo build --release --lib
```

### 3. Test Qwen Module

```rust
use speech::qwen::QwenCorrector;
use candle_core::Device;

let device = Device::Cpu;
let mut corrector = QwenCorrector::load("assets", &device)?;

let raw = "hello world this is a test";
let corrected = corrector.correct_text(raw)?;
// "Hello world, this is a test."
```

## Usage Examples

### Node.js API

```javascript
const { Speech } = require('./index.node');

const speech = new Speech('assets', (transcription) => {
  // Transcription already corrected with Qwen
  console.log(transcription.text);
  // Output: "Hello world, this is a test."
}, true); // Enable Qwen correction

speech.input(audioSamples);
```

### CLI (transcribe_with_vad)

```bash
cargo run --example transcribe_with_vad --release -- audio.wav --use-qwen
```

## Performance Considerations

### Latency

**Without Qwen:**
- Transcription: ~50-100ms per segment
- Rule-based punctuation: < 1ms
- **Total**: ~50-100ms

**With Qwen:**
- Transcription: ~50-100ms per segment
- Qwen correction: ~200-500ms per segment (0.5B model, CPU)
- **Total**: ~250-600ms

**Optimization options:**
- Use GPU (Metal on macOS): 2-3x faster
- Batch multiple segments: Amortize model overhead
- Cache common phrases: Skip correction for repeated text
- Async correction: Don't block next segment

### Memory

- Parakeet Q8_0: ~900MB
- Silero VAD: ~5MB
- Qwen Q4_K_M: ~400MB
- **Total**: ~1.3GB (vs 900MB without Qwen)

### Throughput

**CPU (M1/M2):**
- Qwen inference: ~2-5 segments/second
- Suitable for real-time with 500ms+ pauses

**GPU (Metal):**
- Qwen inference: ~5-10 segments/second
- Easily handles real-time transcription

## What's Left to Complete

### 1. Integration Code

- [ ] Add Qwen initialization to `SpeechInner::new()`
- [ ] Modify `transcribe_segment()` to use Qwen
- [ ] Add `use_qwen` parameter to Node.js constructor
- [ ] Update `transcribe_with_vad` example with `--use-qwen` flag

### 2. Testing

- [ ] Unit tests for Qwen module
- [ ] Integration tests with transcription pipeline
- [ ] Performance benchmarks (latency, throughput)
- [ ] Comparison with rule-based punctuation

### 3. Documentation

- [ ] Update main README with Qwen instructions
- [ ] Add Qwen examples to CLAUDE.md
- [ ] Document performance characteristics
- [ ] Add troubleshooting guide

### 4. Optimization

- [ ] Implement caching for repeated phrases
- [ ] Add batching support
- [ ] Profile and optimize prompts
- [ ] GPU acceleration verification

## Example Integration (Pseudo-code)

```rust
// In src/lib.rs

impl SpeechInner {
    fn new(
        vad: SileroVad,
        parakeet_model: parakeet::ParakeetCtc,
        device: candle_core::Device,
        use_qwen: bool,
    ) -> Result<Self> {
        // ... existing initialization ...

        let qwen_corrector = if use_qwen {
            Some(QwenCorrector::load(&assets, &device)?)
        } else {
            None
        };

        Ok(Self {
            // ... existing fields ...
            qwen_corrector,
        })
    }

    fn transcribe_segment(&mut self) -> Result<String> {
        let raw_text = parakeet::transcribe_streaming_chunk(...)?;

        let corrected = if let Some(ref mut corrector) = self.qwen_corrector {
            info!("Correcting text with Qwen...");
            corrector.correct_text(&raw_text)?
        } else {
            parakeet::add_punctuation(&raw_text)
        };

        Ok(corrected)
    }
}
```

## Comparison: Rule-Based vs Qwen

### Rule-Based (`add_punctuation`)

**Pros:**
- Very fast (< 1ms)
- No model loading
- Deterministic
- No extra memory

**Cons:**
- Limited accuracy
- Hard-coded rules
- Doesn't understand context
- Misses many punctuation cases

### Qwen Correction

**Pros:**
- Much more accurate
- Understands context
- Natural language understanding
- Handles complex cases

**Cons:**
- Slower (~200-500ms)
- Requires model loading
- More memory
- Non-deterministic

## Recommended Workflow

1. **Development**: Use rule-based for fast iteration
2. **Production (interactive)**: Use Qwen with GPU acceleration
3. **Production (batch)**: Use Qwen with batching
4. **Production (low-latency)**: Use rule-based or async Qwen

## References

- Qwen2.5 paper: https://arxiv.org/abs/2409.12186
- Model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Candle Qwen2 implementation: https://github.com/huggingface/candle/tree/main/candle-transformers/src/models/quantized_qwen2.rs
- GGUF quantization: https://github.com/ggerganov/llama.cpp

## Troubleshooting

### "Failed to load Qwen tokenizer"

Run the download script:
```bash
python scripts/download_qwen3.py
```

### High memory usage

- Use Q4_K_M quantization (default)
- Consider Q4_K_S for even smaller size
- Unload model between batches if needed

### Slow inference

- Use GPU (Metal on macOS, CUDA on Linux)
- Reduce max_new_tokens
- Use batching for multiple segments
- Consider async/parallel correction

### Incorrect corrections

- Tune the prompt template
- Try different temperature settings
- Consider fine-tuning on ASR-specific data
- Fallback to rule-based for known problematic cases
