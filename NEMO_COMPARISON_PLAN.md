# NeMo Streaming Comparison Plan

## Objective

Identify why our streaming transcription achieves 54-61% quality vs 100% non-streaming baseline, by comparing layer-by-layer with NVIDIA NeMo's implementation.

## Hypothesis

Our implementation differs from NeMo in one or more critical aspects:
1. LSTM state management (reset strategy, state passing)
2. Frame-level processing (which frames to decode)
3. Token deduplication (LCS implementation)
4. Chunk boundaries and overlap handling
5. Predictor initialization/warm-up

## Comparison Layers

### Layer 1: Feature Extraction
- [x] Already validated - Rust mel extraction matches Python
- No issues here

### Layer 2: Chunk Segmentation
- [ ] Compare chunk boundaries
- [ ] Verify overlap calculation
- [ ] Check frame alignment

### Layer 3: Encoder Processing
- [ ] Compare encoder outputs for same chunks
- [ ] Verify encoder state (should be stateless per chunk)

### Layer 4: Predictor (LSTM) State Management
- [ ] **CRITICAL**: How does NeMo handle LSTM state across chunks?
- [ ] When does NeMo reset vs maintain state?
- [ ] What is the initial state for each chunk?

### Layer 5: Transducer Decoding
- [ ] Compare decoding loop (frame iteration)
- [ ] Verify blank token handling
- [ ] Check inner loop termination

### Layer 6: Token Deduplication
- [ ] Compare LCS implementation
- [ ] Verify overlap token removal
- [ ] Check merge logic

## NeMo Code to Examine

Key files in ~/src/NeMo:
1. `nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py` - Greedy decoder
2. `nemo/collections/asr/parts/submodules/rnnt_loop_labels_computer.py` - Decoding loop
3. `nemo/collections/asr/models/rnnt_models.py` - Model classes
4. `nemo/collections/asr/parts/utils/streaming_utils.py` - Streaming utilities

## Investigation Steps

### Step 1: Review NeMo Streaming Documentation
- Find NeMo examples of streaming transcription
- Identify recommended configuration
- Note any caveats or requirements

### Step 2: Create Diagnostic Tools
- Tool 1: Dump intermediate outputs at each layer
- Tool 2: Compare chunk-by-chunk token production
- Tool 3: Visualize LSTM state evolution
- Tool 4: Compare LCS deduplication results

### Step 3: Run Parallel Transcriptions
- Run NeMo streaming on dots.wav
- Run our streaming on dots.wav
- Compare outputs at each layer

### Step 4: Identify Divergence Point
- Start from features (known good)
- Compare encoder outputs per chunk
- Compare tokens per chunk before deduplication
- Compare tokens per chunk after deduplication
- Identify first layer where outputs differ

### Step 5: Root Cause Analysis
- Once divergence is found, analyze why
- Check NeMo implementation for that layer
- Identify missing logic or incorrect assumptions
- Propose fix

## Expected Outcomes

### Possible Findings:

1. **LSTM state reset is wrong**
   - NeMo might maintain state longer
   - Or reset more intelligently (VAD-based?)

2. **Frame processing is different**
   - NeMo might process overlapping frames differently
   - Or have better handling of chunk boundaries

3. **LCS deduplication is incorrect**
   - Our implementation might be too aggressive
   - Or missing edge cases

4. **Chunk boundaries are suboptimal**
   - NeMo might use different chunk sizes
   - Or align chunks to utterance boundaries

5. **Predictor initialization**
   - NeMo might warm up predictor differently
   - Or use better initial states

## Success Criteria

- Identify specific difference(s) between implementations
- Understand why NeMo's approach works better
- Have actionable fix to improve our quality
- Target: 80%+ quality (110+ tokens vs 140 baseline)

## Timeline

- Step 1: 30 minutes (review NeMo code)
- Step 2: 1 hour (create diagnostic tools)
- Step 3: 30 minutes (run parallel transcriptions)
- Step 4: 1 hour (compare and identify divergence)
- Step 5: 1 hour (analyze and propose fix)

Total: ~4 hours of investigation
