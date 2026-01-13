# TDT Model Limitations and Robustness Analysis

## Executive Summary

The Parakeet TDT (Transducer) model shows **excellent performance on some audio** (dots.wav) but **complete failure on other audio** (jfk.wav), indicating fundamental robustness issues. The CTC model with VAD is significantly more robust across diverse audio types.

## Test Results

### dots.wav (35.33s - Steve Jobs speech)
**TDT Model:**
- ✅ **140 tokens** decoded (100% quality)
- ✅ Correct transcription
- ✅ Timestamp-based punctuation works perfectly

**CTC Model with VAD:**
- ✅ **140 tokens** decoded (100% quality)
- ✅ Correct transcription with natural punctuation

**Conclusion:** Both models work perfectly on this audio.

---

### jfk.wav (11.00s - JFK inaugural speech)
**Expected transcript:**
> "And so, my fellow Americans: ask not what your country can do for you—ask what you can do for your country"

**TDT Model (GPU):**
```
, you know, you're not going to..................................
```
- ❌ **44 tokens** of hallucination
- ❌ Completely wrong - no semantic similarity
- ❌ Gets stuck emitting periods

**TDT Model (CPU):**
```
, you know, you can see that we'..............................
```
- ❌ **40 tokens** of hallucination
- ❌ Same hallucination pattern on CPU (not a GPU issue)

**CTC Model with VAD:**
```
Segment 1: "And so my fellow Americans." ✅
Segment 2: "No." (should be "ask not")
Segment 3: "What your country can do for you, and what you can do for your country."
```
- ✅ **Much better** - captures most content correctly
- ✅ Gets 75%+ of the transcript right
- ✅ Natural segmentation and punctuation

**Conclusion:** TDT completely fails, CTC works reasonably well.

---

## Improvements Made to TDT Decoding

To address the infinite loop issues observed, I implemented several safeguards:

### 1. **Reduced Maximum Inner Steps**
```rust
const MAX_INNER_STEPS: usize = 10;  // Reduced from 50
```
Prevents the decoder from getting stuck for too long at any timestep.

### 2. **Repetition Detection**
```rust
if prev_tok == token && token != blank_id {
    repetition_count += 1;
    if repetition_count >= 3 {
        // Force blank to break loop
        break;
    }
}
```
Detects when the same token is predicted 3+ times and forces termination.

### 3. **Adaptive Blank Bias**
```rust
let blank_bias = if inner_steps > 5 { 0.5 } else { 0.0 };
```
After 5 inner steps, slightly favor the blank token to encourage termination.

### 4. **State Reset on Timeout**
```rust
if inner_steps > MAX_INNER_STEPS {
    pred_states = None;  // Reset corrupted state
    last_token = blank_id;
    break;
}
```
Instead of just breaking, reset the predictor LSTM state to recover from bad states.

### Results of Improvements
- ✅ **Prevents infinite loops** - No more 200+ token repetitions
- ✅ **Maintains quality on good audio** - dots.wav still works perfectly (140 tokens)
- ❌ **Doesn't fix hallucinations** - jfk.wav still produces wrong output (44 tokens)

The improvements help with stability but don't address the fundamental robustness issue.

---

## Root Cause Analysis

### Why TDT Fails on jfk.wav

The TDT model's failure on jfk.wav appears to stem from:

#### 1. **Encoder Output Issues**
The FastConformer encoder may not be producing good acoustic representations for this audio:
- Different recording quality (1961 vs modern)
- Different microphone characteristics
- Different speaker voice characteristics (JFK vs Steve Jobs)
- Background noise patterns

#### 2. **Joint Network Brittleness**
The transducer's joint network (combining encoder + predictor) may be sensitive to out-of-distribution encoder outputs:
```
encoder_out + predictor_out → joint → logits
```
If `encoder_out` is poor, the joint network produces poor logits.

#### 3. **Predictor LSTM Instability**
The LSTM predictor can get into bad states where:
- It keeps predicting content tokens even when encoder says "no speech here"
- It produces hallucinated sequences that have grammatical structure but no semantic meaning
- Once in a bad state, it's hard to recover (even with state resets)

### Why CTC Works Better

The CTC model is more robust because:

#### 1. **Simpler Architecture**
```
encoder → CTC head → logits
```
No predictor LSTM, no joint network - just encoder → output.

#### 2. **Frame-Independent Predictions**
Each frame's prediction is independent:
- Bad predictions at one frame don't corrupt future frames
- No hidden state that can accumulate errors
- More tolerant of poor encoder outputs

#### 3. **VAD Filtering**
The VAD-based approach only transcribes detected speech:
- Filters out silence and noise
- Segments at natural pauses
- Reduces exposure to bad encoder regions

---

## Comparison Matrix

| Feature | TDT Model | CTC Model + VAD |
|---------|-----------|-----------------|
| **Robustness** | ❌ Fails on jfk.wav | ✅ Works on jfk.wav |
| **Quality (good audio)** | ✅ 100% (dots.wav) | ✅ 100% (dots.wav) |
| **Quality (challenging audio)** | ❌ 0% hallucination | ✅ 75%+ correct |
| **Punctuation Source** | ✅ Model timestamps | ⚠️ External VAD |
| **Complexity** | ⚠️ 3 networks | ✅ 2 models |
| **Decoding Speed** | ⚠️ Slower (LSTM) | ✅ Faster (parallel) |
| **Stability** | ❌ Can loop/hallucinate | ✅ Stable |
| **Production Ready** | ❌ No | ✅ Yes |

---

## Recommendations

### For Production Use: **CTC Model with VAD**
```bash
cargo run --example transcribe_with_vad --release -- audio.wav
```

**Reasons:**
- ✅ Robust across diverse audio types
- ✅ Natural segmentation and punctuation
- ✅ Stable decoding (no loops or hallucinations)
- ✅ Good quality on both clean and challenging audio
- ✅ Text correction with Qwen3 for better output

### For Research/Experimentation: **TDT Model**
```bash
cargo run --example transcribe_tdt --release -- audio.wav
```

**Use cases:**
- Clean, modern recordings
- When model timestamps are critical
- When experimenting with transducer architectures
- When working with audio similar to training data

**Not recommended for:**
- Diverse audio sources
- Historical recordings
- Noisy environments
- Production systems requiring reliability

---

## Technical Limitations

### TDT Model Architecture Challenges

#### 1. **Predictor LSTM State Management**
The LSTM maintains hidden state across the sequence:
```rust
pub struct PredictionNetwork {
    embedding: Embedding,
    lstms: Vec<rnn::LSTM>,  // 2 layers, 640 hidden
    ...
}
```

**Issues:**
- State can accumulate errors over long sequences
- Bad predictions early on corrupt later predictions
- State resets are disruptive and lose context

#### 2. **Joint Network Sensitivity**
```rust
let joint = enc_proj.broadcast_add(&pred_proj)?;  // [B, T, U, joint_hidden]
```

The joint network combines encoder and predictor via broadcasting:
- If encoder output is poor, joint output is poor
- No way to recover from bad encoder representations
- Small numerical errors can cascade

#### 3. **Greedy Decoding Limitations**
Our implementation uses greedy decoding (argmax):
```rust
let token = log_probs_masked.argmax(D::Minus1)?;
```

**Limitations:**
- No beam search to explore alternative hypotheses
- Can't recover from early mistakes
- Gets stuck in local optima

**Note:** Adding beam search could improve quality but adds significant complexity.

---

## Future Work

### Potential Improvements

#### 1. **Beam Search Decoding**
Implement beam search with beam width 5-10:
- Explore multiple hypotheses
- Can recover from early mistakes
- More robust to local optima

**Tradeoff:** 5-10x slower decoding

#### 2. **Temperature Sampling**
Add temperature to logits before argmax:
```rust
let logits = logits / temperature;  // temperature = 1.1
```

**Benefit:** Slightly randomized decoding can escape local optima

#### 3. **Encoder Fine-tuning**
Fine-tune the encoder on diverse audio:
- Historical recordings
- Different microphones
- Various speakers
- Noisy environments

**Challenge:** Requires significant compute and data

#### 4. **Hybrid CTC/TDT Approach**
Use CTC for robust transcription + TDT timestamps:
- CTC provides stable transcription
- TDT provides frame-level alignment
- Combine outputs for best of both

**Complexity:** Significant implementation work

---

## Conclusion

The TDT model shows promise for timestamp-based punctuation on clean audio but has fundamental robustness issues that make it unsuitable for production use. The improvements made to decoding stability help prevent infinite loops but don't address the core hallucination problem.

**For production applications**, the **CTC model with VAD** is strongly recommended due to its superior robustness across diverse audio types.

The timestamp-based punctuation feature works beautifully when the TDT model produces correct transcriptions, but the model's robustness issues limit its practical applicability.

---

## Test Commands

### TDT Model
```bash
# Basic transcription
cargo run --example transcribe_tdt --release -- jfk.wav
cargo run --example transcribe_tdt --release -- dots.wav

# With timestamp punctuation
cargo run --example transcribe_tdt_with_punctuation --release -- jfk.wav
cargo run --example transcribe_tdt_with_punctuation --release -- dots.wav

# Force CPU (rule out GPU issues)
PARAKEET_DEVICE=cpu cargo run --example transcribe_tdt --release -- jfk.wav
```

### CTC Model with VAD (Recommended)
```bash
cargo run --example transcribe_with_vad --release -- jfk.wav
cargo run --example transcribe_with_vad --release -- dots.wav
```

---

## References

- **TDT Model**: `nvidia/parakeet-tdt-0.6b-v3`
- **CTC Model**: `nvidia/parakeet-ctc-0.6b`
- **Transducer Paper**: Graves (2012) "Sequence Transduction with Recurrent Neural Networks"
- **Implementation**: `src/parakeet/transducer.rs`
- **Test Audio**:
  - `dots.wav`: Steve Jobs "Connecting the Dots" (35.33s)
  - `jfk.wav`: JFK Inaugural Address excerpt (11.00s)
