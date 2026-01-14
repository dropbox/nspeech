#!/usr/bin/env python3
"""Test NeMo streaming TDT inference"""

import nemo.collections.asr as nemo_asr
import time
import os

print("\nLoading model from HuggingFace...")
start = time.time()

asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-speech-streaming-en-0.6b"
)

load_time = time.time() - start
print(f"✓ Model loaded ({load_time:.2f}s)")

# Check model info
if hasattr(asr_model, 'decoder'):
    print(f"  Vocab size: {asr_model.decoder.vocab_size}")
    print(f"  Blank ID: {asr_model.decoder.blank_idx}")

# Find audio file
audio_paths = ["dots.wav", "/workspace/dots.wav", "assets/assets/dots.wav"]
audio_file = None
for path in audio_paths:
    if os.path.exists(path):
        audio_file = path
        break

if not audio_file:
    print(f"ERROR: dots.wav not found")
    print(f"Tried: {audio_paths}")
    import sys
    sys.exit(1)

print(f"\nTranscribing: {audio_file}")
start = time.time()

transcription = asr_model.transcribe([audio_file])[0]

trans_time = time.time() - start
print(f"✓ Transcription complete ({trans_time:.2f}s)")

# Display results
print("\n" + "="*60)
print("TRANSCRIPTION")
print("="*60)
print(transcription)
print("="*60)

# Show token info
if hasattr(asr_model, 'tokenizer'):
    try:
        tokens = asr_model.tokenizer.text_to_ids(transcription)
        print(f"\nTokens: {len(tokens)}")
        print(f"First 30 tokens: {tokens[:30]}")
    except Exception as e:
        print(f"\nCould not tokenize: {e}")

print("\n✓ Done")
