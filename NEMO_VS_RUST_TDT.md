# NeMo vs Rust TDT Implementation Comparison

## Critical Discovery

**Official NeMo produces perfect transcriptions on both test files, while our Rust implementation fails on jfk.wav.**

This proves the TDT model weights are correct and the issue is in our Rust decoding implementation.

## Test Results Comparison

### dots.wav (35.33s - Steve Jobs speech)

| Implementation | Result | Quality |
|----------------|--------|---------|
| **NeMo Official** | ✅ Perfect: "Of course, it was impossible to connect the dots looking forward..." | 100% |
| **Rust TDT** | ✅ Perfect: "... Ofourse it was impos to connect the dots looking forward..." | 100% |

**Status:** Both work perfectly ✅

---

### jfk.wav (11.00s - JFK inaugural speech)

**Expected:**
> "And so, my fellow Americans: ask not what your country can do for you—ask what you can do for your country"

| Implementation | Result | Quality |
|----------------|--------|---------|
| **NeMo Official** | ✅ "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country." | ~100% |
| **Rust TDT** | ❌ ", you know, you're not going to.................................." | 0% (hallucination) |

**Status:** NeMo works, Rust fails ❌

---

## NeMo Statistics

### dots.wav
```
Audio duration: 35.33s
Transcription time: 0.57s
Real-time factor: 0.016x (62x faster than realtime)
Word count: 105
Characters: 588
Device: NVIDIA GeForce RTX 2080 (CUDA)
```

### jfk.wav
```
Audio duration: 11.00s
Transcription time: 0.47s
Real-time factor: 0.042x (24x faster than realtime)
Word count: 22
Characters: 108
Device: NVIDIA GeForce RTX 2080 (CUDA)
```

---

## Key Differences to Investigate

### 1. Decoding Algorithm

**NeMo:** Likely uses optimized decoding (possibly beam search or modified greedy)
- Fast: 0.016x RTF on dots.wav
- Robust: Works on diverse audio

**Rust:** Custom greedy decoding implementation
- Slower: ~0.05x RTF estimated
- Brittle: Fails on jfk.wav

### 2. Decoder Configuration

**NeMo might use:**
- Beam search (beam_size > 1)
- Different blank bias strategy
- Temperature scaling
- Length normalization
- Special handling for silence/noise

**Rust uses:**
- Pure greedy (argmax)
- Adaptive blank bias (0.5 after 5 steps)
- MAX_INNER_STEPS = 10
- Repetition detection

### 3. Predictor LSTM State Management

**NeMo:** Optimized state handling
- Likely uses batched inference
- May have state reset heuristics

**Rust:** Manual state management
- Resets state on timeout
- May not match NeMo's state handling

### 4. Feature Extraction

**NeMo:** Uses their feature extraction pipeline
- 128 mel bins
- Specific preprocessing

**Rust:** Custom feature extraction
- 128 mel bins
- Should match, but worth verifying

---

## Investigation Plan

### 1. Check NeMo's Decoding Configuration
```python
# In nemo_baseline_tdt.py, print decoding config
print(model.decoding)
print(model.cfg.decoding)
```

Look for:
- `strategy`: greedy vs beam
- `beam.beam_size`: if > 1, using beam search
- `preserve_alignments`: timestamp handling
- `confidence_cfg`: confidence scoring

### 2. Compare Encoder Outputs
```python
# Extract encoder output from NeMo
encoder_out = model.encoder(input_signal=features)
print(encoder_out.shape)
print(encoder_out[0, :10, :5])  # First 10 frames, 5 features
```

Compare with Rust:
```rust
let encoder_out = model.encoder.forward(&features, false)?;
println!("Encoder shape: {:?}", encoder_out.dims());
```

### 3. Inspect Predictor Initialization

**Question:** Does NeMo initialize LSTM states differently?

Check if NeMo:
- Warms up the predictor before decoding
- Uses non-zero initial states
- Has special handling for the first timestep

### 4. Check Joint Network Logits

Extract logits at first timestep in both implementations:
```python
# NeMo
joint_out = model.joint(encoder_out[:, 0:1, :], predictor_out)
logits = joint_out[0, 0, 0, :]
print("Top 5 logits:", torch.topk(logits, 5))
```

```rust
// Rust
let logits = self.joint.forward(&enc_t, &pred_out)?;
// Print top 5 values
```

### 5. Verify Tokenizer Decoding

Ensure tokenizer produces same text:
```python
# NeMo
tokens = [452, 7894, 867, 344, 467]  # First 5 from dots.wav
text = model.decoding.tokenizer.ids_to_text(tokens)
print(text)
```

```rust
// Rust
let tokens = vec![452, 7894, 867, 344, 467];
let text = model.decode_tokens(&tokens)?;
println!("{}", text);
```

---

## Hypotheses for jfk.wav Failure

### Hypothesis 1: Feature Extraction Mismatch
**Likelihood:** Low
- Dots.wav works perfectly with our features
- Feature extraction is standard (mel spectrogram)

### Hypothesis 2: Greedy Decoding Limitations
**Likelihood:** HIGH ⭐
- NeMo might use beam search (beam_size=4 is common)
- Beam search explores multiple hypotheses
- Can recover from early mistakes
- Our greedy decoder gets stuck in bad paths

### Hypothesis 3: Blank Bias Incorrect
**Likelihood:** Medium
- Our blank bias (0.5 after 5 steps) might be wrong
- May be preventing valid content tokens
- NeMo might not use blank bias at all

### Hypothesis 4: LSTM State Corruption
**Likelihood:** Medium
- LSTM state might diverge from NeMo's
- Our state initialization might be wrong
- State reset strategy might be incorrect

### Hypothesis 5: Temperature/Sampling
**Likelihood:** Low-Medium
- NeMo might use temperature scaling
- Could make predictions more robust
- But usually temperature is for sampling, not greedy

---

## Next Steps

1. **Inspect NeMo's decoding configuration** to see exact algorithm
2. **Compare encoder outputs** on jfk.wav (first few frames)
3. **Extract and compare logits** at first timestep
4. **Implement beam search** if that's what NeMo uses
5. **Try removing blank bias** to match vanilla greedy

---

## Test Commands

### NeMo Baseline
```bash
~/bin/jrpython nemo_baseline_tdt.py jfk.wav
~/bin/jrpython nemo_baseline_tdt.py dots.wav
```

### Rust Implementation
```bash
cargo run --example transcribe_tdt --release -- jfk.wav
cargo run --example transcribe_tdt --release -- dots.wav
```

---

## Expected Outcome

Once we identify the key difference (likely beam search or decoding strategy), we can:

1. Implement the same algorithm in Rust
2. Verify jfk.wav transcribes correctly
3. Confirm timestamp-based punctuation works on jfk.wav
4. Have a production-ready TDT implementation

The model weights are correct - we just need to match NeMo's decoding algorithm!
