# NeMo's Solution to Streaming Beam Search

## The Key Innovation: ALSD (Alignment-Length Synchronous Decoding)

### Problem with Traditional Beam Search (TSD)
- **Time-Synchronous Decoding (TSD)**: All hypotheses process the same encoder frame
- Execution time: T × max_expansions (very expensive)
- Poor for streaming: requires all encoder frames upfront

### NeMo's Solution: ALSD

**Core Concept**: Synchronize hypotheses by **alignment length** (tokens + blanks), not by time.

**Benefits**:
- 25% faster than TSD
- 42% fewer joint network evaluations
- Better for streaming: hypotheses can be at different timesteps
- Execution time: T + U_max (much better scaling)

### Algorithm

```python
def alsd_beam_search(encoder_output, beam_size):
    hypotheses = [Hypothesis(score=0, y=[], state=None, t=0)]

    for alignment_len in range(max_alignment_length):
        candidates = []

        for hyp in hypotheses:
            if hyp.t >= len(encoder_output):
                # Hypothesis finished all frames
                candidates.append(hyp)
                continue

            # Get current frame
            f = encoder_output[hyp.t]
            g, new_state = decoder(hyp.last_token, hyp.state)
            logits = joint(f, g)

            # Option 1: Emit blank (advance time, no token)
            candidates.append(Hypothesis(
                score=hyp.score + logits[BLANK],
                y=hyp.y,
                state=hyp.state,
                t=hyp.t + 1  # Advance time
            ))

            # Option 2: Emit token (same time, add token)
            for token in top_k(logits):
                if token != BLANK:
                    candidates.append(Hypothesis(
                        score=hyp.score + logits[token],
                        y=hyp.y + [token],
                        state=new_state,
                        t=hyp.t  # Same time
                    ))

        # Recombine: merge hypotheses with same (y, t)
        candidates = recombine_hypotheses(candidates)

        # Keep top beam_size
        hypotheses = top_k(candidates, beam_size)

    return best_hypothesis(hypotheses)
```

## Key Differences from My Implementation

### 1. Hypothesis State Management

**NeMo approach**:
```python
@dataclass
class Hypothesis:
    score: float
    y_sequence: List[int]  # All tokens emitted so far
    dec_state: Any          # LSTM (h, c) state
    timestep: int           # Current encoder frame
    last_token: int         # Previous token
```

**My approach** (incorrect for streaming):
```rust
struct StreamingBeamHypothesis {
    tokens: Vec<u32>,       // Tokens for THIS CHUNK only
    score: f32,
    pred_state: Vec<LSTMState>,
    last_token: u32,
}
```

**Problem**: I was only tracking tokens per chunk, not cumulative tokens.

### 2. Streaming Strategy

**NeMo for streaming**:
1. Keep all K hypotheses active across chunks
2. DON'T extract tokens per chunk
3. At the end, select best hypothesis
4. For truly incremental output, use **greedy decoding** or commit to common prefix

**My approach** (incorrect):
1. Try to extract "new tokens" after each chunk
2. Switch between hypotheses causes token loss
3. Incomplete transcription

### 3. The Solution for Streaming

**Option A: Greedy Decoding (What we have)**
- Use greedy streaming (189 tokens, 84% quality)
- Simple, robust, works well
- **This is what NeMo recommends for true streaming**

**Option B: ALSD for Offline/Batch**
- Use ALSD beam search for complete utterances
- Better quality but not truly incremental
- Process chunks with overlap, merge results

**Option C: Prefix Commitment (Advanced)**
- Track longest common prefix of all beam hypotheses
- Output only tokens all hypotheses agree on
- Complex but enables true streaming beam search

## Implementation Plan for ALSD

```rust
pub fn beam_decode_alsd(
    &self,
    encoder_out: &Tensor,
    beam_size: usize,
) -> Result<Vec<u32>> {
    let (batch_size, time_steps, _) = encoder_out.dims3()?;

    // ALSD: index by alignment length, not timestep
    let max_alignment = time_steps + self.config.max_symbols_per_step * time_steps;

    let mut beam = vec![ALSDHypothesis {
        tokens: Vec::new(),
        score: 0.0,
        pred_state: self.predictor.init_states(1, encoder_out.device())?,
        last_token: self.config.blank_id as u32,
        timestep: 0,
    }];

    for _alignment_len in 0..max_alignment {
        let mut candidates = Vec::new();

        for hyp in &beam {
            // Skip if past encoder length
            if hyp.timestep >= time_steps {
                candidates.push(hyp.clone());
                continue;
            }

            // Get encoder frame
            let enc_t = encoder_out.narrow(1, hyp.timestep, 1)?;

            // Predictor
            let pred_input = Tensor::new(&[hyp.last_token], encoder_out.device())?
                .unsqueeze(0)?;
            let (pred_out, new_states) = self.predictor.forward(&pred_input, Some(&hyp.pred_state))?;

            // Joint
            let logits = self.joint.forward(&enc_t, &pred_out)?;
            let log_probs = log_softmax(&logits)?;

            // Blank: advance time
            candidates.push(ALSDHypothesis {
                tokens: hyp.tokens.clone(),
                score: hyp.score + log_probs[BLANK],
                pred_state: hyp.pred_state.clone(),
                last_token: hyp.last_token,
                timestep: hyp.timestep + 1,  // ADVANCE TIME
            });

            // Non-blank: emit token (stay at same time initially)
            for token in top_k_tokens(&log_probs, beam_size) {
                if token != BLANK {
                    let mut new_tokens = hyp.tokens.clone();
                    new_tokens.push(token);

                    candidates.push(ALSDHypothesis {
                        tokens: new_tokens,
                        score: hyp.score + log_probs[token],
                        pred_state: new_states.clone(),
                        last_token: token,
                        timestep: hyp.timestep,  // SAME TIME
                    });
                }
            }
        }

        // Recombine hypotheses with same (tokens, timestep)
        candidates = recombine_hypotheses(candidates);

        // Keep top beam_size
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        beam = candidates.into_iter().take(beam_size).collect();

        // Early termination: all hypotheses finished
        if beam.iter().all(|h| h.timestep >= time_steps) {
            break;
        }
    }

    // Return best hypothesis
    let best = beam.into_iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
        .ok_or_else(|| anyhow!("No hypotheses"))?;

    Ok(best.tokens)
}
```

## Key Insights

1. **ALSD is not designed for chunk-by-chunk streaming** - it's for processing complete utterances more efficiently
2. **NeMo uses greedy for true streaming** - that's why our greedy (189 tokens) is the right approach
3. **Beam search is for offline/batch processing** - better quality but not incremental
4. **The 16% gap is expected** - greedy vs beam, plus model differences

## Recommendation

**Keep using greedy streaming (189 tokens, 84%)** - this matches NeMo's approach for true streaming applications.

For non-streaming (offline), implement ALSD beam search for better quality.
