#!/usr/bin/env python3
"""Extract mel features using librosa (NeMo's basis) for comparison"""

import numpy as np
import soundfile as sf
import librosa

# Load audio
audio, sr = sf.read('dots.wav')
print(f"Audio: {len(audio)} samples, {sr} Hz\n")

# Extract mel spectrogram using librosa with NeMo-like settings
# NeMo uses: n_fft=512, hop_length=160, win_length=400, n_mels=136
# Using Slaney norm (default in librosa)
mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=512,
    hop_length=160,
    win_length=400,
    window='hann',
    center=True,
    pad_mode='reflect',
    n_mels=136,
    fmin=0.0,
    fmax=sr/2.0,
    norm='slaney',  # NeMo default
    htk=False       # Use Slaney mel scale, not HTK
)

# Convert to log scale (librosa uses natural log by default)
# But NeMo/Parakeet might use log10...let's compute both
log_mel = np.log10(mel_spec + 1e-10)
ln_mel = np.log(mel_spec + 1e-10)

# Apply per-feature normalization (like NeMo's normalize='per_feature')
# Normalize each mel bin (feature dimension) independently to mean=0, std=1
log_mel_per_feature = np.zeros_like(log_mel)
for i in range(log_mel.shape[0]):  # For each mel bin
    mean = log_mel[i, :].mean()
    std = log_mel[i, :].std()
    if std < 1e-10:
        std = 1.0
    log_mel_per_feature[i, :] = (log_mel[i, :] - mean) / std

print("=== Librosa Mel Features (log10, per-feature norm) ===")
print(f"  Shape: {log_mel_per_feature.shape} (features x time)")
print(f"  Mean: {log_mel_per_feature.mean():.6f}")
print(f"  Std:  {log_mel_per_feature.std():.6f}")
print(f"  Min:  {log_mel_per_feature.min():.6f}")
print(f"  Max:  {log_mel_per_feature.max():.6f}")
print()

print("First frame (first 10 values):")
for i in range(10):
    print(f"  [{i}]: {log_mel_per_feature[i, 0]:.6f}")
print()

print("=== Raw log_mel (no norm) ===")
print(f"  Mean: {log_mel.mean():.6f}")
print(f"  Std:  {log_mel.std():.6f}")
print(f"  Min:  {log_mel.min():.6f}")
print(f"  Max:  {log_mel.max():.6f}")
print()

# Save for Rust comparison
np.save('/tmp/librosa_mel_features.npy', log_mel_per_feature)
print("✓ Saved to /tmp/librosa_mel_features.npy")
