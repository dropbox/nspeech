#!/usr/bin/env python3
"""Compare our mel features with NeMo's"""

import numpy as np
import soundfile as sf
import torch

# Load audio
audio, sr = sf.read('dots.wav')
print(f"Audio: {len(audio)} samples, {sr} Hz\n")

# Load NeMo model
import nemo.collections.asr as nemo_asr
print("Loading NeMo model...")
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-speech-streaming-en-0.6b"
)
print("✓ Model loaded\n")

# Extract features using NeMo's preprocessor
print("Extracting features with NeMo...")
audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
audio_len = torch.tensor([len(audio)])

with torch.no_grad():
    processed, processed_len = asr_model.preprocessor(
        input_signal=audio_tensor,
        length=audio_len
    )

print(f"NeMo features shape: {processed.shape}")
print(f"  [batch, features, time] = {list(processed.shape)}")
print(f"  Feature mean: {processed.mean():.6f}")
print(f"  Feature std: {processed.std():.6f}")
print(f"  Feature min: {processed.min():.6f}")
print(f"  Feature max: {processed.max():.6f}")
print()

# Save first 10 frames for comparison
nemo_features = processed[0, :, :10].cpu().numpy()
print(f"First frame (10 values): {nemo_features[:10, 0]}")
print()

# Save to file for Rust to compare
np.save('/tmp/nemo_features_full.npy', processed[0].cpu().numpy())
print(f"✓ Saved NeMo features to /tmp/nemo_features_full.npy")
print(f"  Shape: {processed[0].shape} (features x time)")
