#!/usr/bin/env python3
"""Minimal NeMo streaming inference test"""

import sys
sys.path.insert(0, '/Users/jhansen/src/NeMo')

import torch
import nemo.collections.asr as nemo_asr

print("Loading streaming model...")
model = nemo_asr.models.ASRModel.restore_from(
    "/Users/jhansen/src/speech/.cache/parakeet-streaming-tdt/nemotron-speech-streaming-en-0.6b.nemo"
)
model.eval()

print(f"Model loaded: {type(model)}")
print(f"Encoder type: {type(model.encoder)}")
print(f"Vocab size: {model.decoder.vocab_size if hasattr(model, 'decoder') else 'N/A'}")
print(f"Blank ID: {model.decoder.blank_id if hasattr(model, 'decoder') else 'N/A'}")

# Check if encoder has streaming support
if hasattr(model.encoder, 'streaming_cfg'):
    print(f"Streaming config: {model.encoder.streaming_cfg}")
elif hasattr(model.encoder, 'setup_streaming_params'):
    model.encoder.setup_streaming_params()
    print(f"Streaming config: {model.encoder.streaming_cfg}")

# Transcribe
print("\nTranscribing dots.wav...")
result = model.transcribe(["dots.wav"])
print(f"\nResult: {result}")
