"""Inspect NeMo's preprocessor configuration for TDT model."""

import nemo.collections.asr as nemo_asr
import torch

# Load model
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")

# Print preprocessor config
print("=" * 60)
print("NeMo Preprocessor Configuration")
print("=" * 60)

preprocessor = model.preprocessor
print(f"\nPreprocessor type: {type(preprocessor).__name__}")
print(f"\nPreprocessor config:")

if hasattr(preprocessor, 'cfg'):
    cfg = preprocessor.cfg
    for key in dir(cfg):
        if not key.startswith('_'):
            try:
                val = getattr(cfg, key)
                if not callable(val):
                    print(f"  {key}: {val}")
            except:
                pass

# Check specific attributes
attrs = ['sample_rate', 'n_fft', 'n_window_size', 'n_window_stride',
         'window', 'normalize', 'preemph', 'nfilt', 'lowfreq', 'highfreq',
         'log', 'log_zero_guard_type', 'log_zero_guard_value', 'dither',
         'pad_to', 'frame_splicing', 'stft_conv', 'pad_value', 'mag_power',
         'exact_pad', 'use_grads']

print("\n" + "=" * 60)
print("Direct Attribute Access:")
print("=" * 60)
for attr in attrs:
    if hasattr(preprocessor, attr):
        val = getattr(preprocessor, attr)
        print(f"  {attr}: {val}")

# Try to get the actual preprocessing module
print("\n" + "=" * 60)
print("Preprocessor Module Details:")
print("=" * 60)
if hasattr(preprocessor, 'featurizer'):
    print("Has featurizer:")
    featurizer = preprocessor.featurizer
    for attr in dir(featurizer):
        if not attr.startswith('_') and not callable(getattr(featurizer, attr, None)):
            try:
                val = getattr(featurizer, attr)
                print(f"  {attr}: {val}")
            except:
                pass
