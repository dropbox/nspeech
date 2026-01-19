# Why Beam Search Uses So Much Memory

## Quick Answer

Beam search with `beam_size=2` doesn't just use 2x the memory of greedy decode. During expansion, it temporarily creates **up to 20 hypothesis candidates** (2 beams × 10 max inner steps), each with a **full copy of the LSTM state**. Combined with temporary tensors for predictor and joint network computations, this causes memory usage to spike dramatically.

## Detailed Explanation

### Greedy Decode Memory Usage

With greedy decode, you maintain exactly **ONE** hypothesis with:
- Token sequence: `Vec<u32>` (small, ~100 tokens max)
- LSTM state: 2 layers × 2 states (h, c) × [1, 512] floats

**Memory per hypothesis:**
- 2 layers × 2 (h+c) × 512 × 2 bytes (BF16) = **4 KB**

Plus temporary tensors during forward pass:
- Predictor input: [1, 1] = negligible
- Predictor output: [1, 1, 512] = 1 KB
- Joint logits: [1, vocab_size] = ~2 KB
- **Total temporary: ~3 KB**

**Greedy total: ~7 KB per timestep**

### Beam Search Memory Usage

With beam search (beam_size=2), the memory usage explodes:

#### 1. Starting Beam
- 2 hypotheses × 4 KB LSTM state = **8 KB**

#### 2. Expansion Phase (The Problem!)

For each hypothesis in the beam:
```rust
for hyp in &beam {  // 2 hypotheses
    let mut current_hyp = hyp.clone();  // CLONE #1: 4 KB

    for _inner_step in 0..MAX_INNER_STEPS {  // Up to 10 iterations
        // Run predictor forward
        let (pred_out, new_states) = self.predictor.forward(...);  // ~3 KB temp

        // Run joint network
        let logits = self.joint.forward(...);  // ~2 KB temp

        if token == blank {
            let mut blank_hyp = current_hyp.clone();  // CLONE #2: 4 KB
            candidates.push(blank_hyp);
            break;
        } else {
            current_hyp.pred_state = new_states;  // CLONE #3: 4 KB
            // Continue inner loop...
        }
    }
}
```

**Per hypothesis expansion:**
- 1 initial clone: 4 KB
- Up to 10 inner steps × (3 KB predictor + 2 KB joint) = **50 KB temporaries**
- Up to 10 candidate clones: 10 × 4 KB = **40 KB**
- **Total per hypothesis: ~94 KB**

**For beam_size=2:**
- 2 hypotheses × 94 KB = **188 KB per timestep**

#### 3. Accumulation Over Time

For a 26-second segment:
- Encoder downsamples 8x: 26s × 16000 / 8 = **52,000 timesteps**
- Peak memory during expansion: **188 KB**
- But this happens at EVERY timestep!

The issue is that Python/Node.js garbage collection doesn't run fast enough, so memory from previous timesteps accumulates before being freed.

**Estimated accumulated memory:**
- If GC lags by even 100 timesteps: 100 × 188 KB = **18.8 MB** just for beam search
- Add to existing 900 MB model + buffers = **~920 MB**
- Node.js has limited heap (typically ~1-2 GB)
- **Result: OS kills process with SIGKILL**

### Why It's Worse in Node.js

1. **Model already embedded**: 849 MB quantized model is memory-mapped but still claims address space

2. **VAD model**: Additional ~50 MB for Silero VAD

3. **No manual memory management**: Rust would drop temporaries immediately, but Node.js FFI keeps references longer

4. **Heap fragmentation**: Repeated allocation/deallocation of LSTM states fragments the heap

5. **OS memory pressure**: macOS is aggressive about killing processes that use too much memory

## Visualization

### Greedy Decode Timeline
```
Time →
t=0:  [Hyp1] → Forward → [Hyp1'] ✓ (7 KB)
t=1:  [Hyp1'] → Forward → [Hyp1''] ✓ (7 KB)
t=2:  [Hyp1''] → Forward → [Hyp1'''] ✓ (7 KB)
```

### Beam Search Timeline (beam_size=2)
```
Time →
t=0:  [Hyp1, Hyp2] → Expand → [Cand1..Cand20] → Prune → [Hyp1', Hyp2'] ✓ (188 KB)
                      ↑
                      Creates up to 20 candidates!

t=1:  [Hyp1', Hyp2'] → Expand → [Cand1..Cand20] → Prune → [Hyp1'', Hyp2''] ✓ (188 KB)
      [Old memory from t=0 not yet freed...] ⚠️

t=2:  [Hyp1'', Hyp2''] → Expand → [Cand1..Cand20] → Prune → [Hyp1''', Hyp2'''] ✗ (OOM!)
      [Old memory from t=0, t=1 not yet freed...] 💥
```

## The Math

### Per-Segment Memory Breakdown

For a 26-second segment (52,000 encoder timesteps):

| Component | Greedy | Beam (size=2) |
|-----------|--------|---------------|
| Model (mmap'd) | 849 MB | 849 MB |
| VAD | 50 MB | 50 MB |
| Feature buffers | ~20 MB | ~20 MB |
| LSTM states | 4 KB | 8 KB (steady) |
| Temp tensors/frame | 3 KB | 50 KB |
| Candidate copies/frame | 0 KB | 40 KB |
| **Peak per frame** | **7 KB** | **188 KB** |
| **Peak if GC lags 100 frames** | **700 KB** | **18.8 MB** |
| **Total peak** | **~920 MB** | **~938 MB** |

With GC lag, beam search pushes past the ~1 GB threshold where macOS starts killing processes.

## Solutions (Not Implemented)

### 1. Reduce Beam Size
- Use beam_size=1.5 or adaptive pruning
- Reduces candidates but still better than greedy

### 2. Checkpoint-Based Decoding
- Don't expand all hypotheses at once
- Process one hypothesis at a time
- Slower but uses constant memory

### 3. State Sharing
- Share LSTM states between hypotheses when possible
- Only fork when predictions diverge
- Complex to implement

### 4. External Process
- Run beam search in separate Rust process
- Communicate via IPC
- Node.js just handles I/O

### 5. Smaller Model
- Use 4-bit quantization instead of 8-bit
- Reduces base model memory
- Leaves more room for beam search overhead

## Conclusion

Beam search doesn't just use 2x memory - it uses **~27x more memory per timestep** (188 KB vs 7 KB) due to:
1. Maintaining multiple hypotheses
2. Cloning LSTM states multiple times during expansion
3. Creating up to 20 temporary candidates before pruning
4. Accumulating unreleased memory across timesteps

This is why the Node.js process gets killed by the OS, while the Rust CLI handles it fine (better memory management, no FFI overhead, immediate dropping of temporaries).
