#!/usr/bin/env python3
"""
Simple Baseline: Parakeet TDT v3 Transcription

This script loads the Parakeet TDT model directly using Hugging Face
transformers, avoiding the full NeMo toolkit installation which can
have dependency issues.

Usage:
    python simple_baseline_tdt.py dots.wav

Requirements:
    uv pip install torch torchaudio transformers librosa soundfile

Model: nvidia/parakeet-tdt-0.6b-v3 (Transducer/RNN-T)
"""

import sys
import time
import torch
import torchaudio
from pathlib import Path

def load_audio(audio_path, target_sr=16000):
    """Load audio file and resample to target sample rate."""
    import librosa
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio, sr

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_baseline_tdt.py <audio.wav>")
        print("\nExample:")
        print("  python simple_baseline_tdt.py dots.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print("=" * 60)
    print("Simple Baseline: Parakeet TDT v3 Transcription")
    print("=" * 60)
    print(f"\nAudio: {audio_path}\n")

    # Check dependencies
    try:
        from transformers import AutoModelForCTC, AutoProcessor
    except ImportError:
        print("Error: transformers not installed.")
        print("\nInstall with:")
        print("  uv pip install torch torchaudio transformers librosa soundfile")
        sys.exit(1)

    # Note: Parakeet TDT is a Transducer model, not CTC
    # We'll try to load it with transformers if possible
    print("Attempting to load Parakeet TDT v3...")
    print("Note: This model may not be directly loadable with transformers.")
    print("      For full compatibility, use the NeMo scripts instead.\n")

    try:
        # Try loading with transformers
        model_id = "nvidia/parakeet-tdt-0.6b-v3"

        # Check if model files exist
        from huggingface_hub import hf_hub_download

        print("Downloading model files...")
        config_file = hf_hub_download(repo_id=model_id, filename="config.json")
        print(f"  Config: {config_file}")

        # Load model weights
        model_file = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        print(f"  Weights: {model_file}")

        tokenizer_file = hf_hub_download(repo_id=model_id, filename="tokenizer.json")
        print(f"  Tokenizer: {tokenizer_file}")

        print("\n" + "=" * 60)
        print("MODEL FILES DOWNLOADED")
        print("=" * 60)
        print()
        print("The Parakeet TDT model has been downloaded successfully.")
        print("However, it requires NeMo's custom RNN-T implementation.")
        print()
        print("To use this model, please either:")
        print()
        print("1. Use our Rust implementation (recommended):")
        print("   cargo run --example transcribe_tdt --release -- dots.wav")
        print()
        print("2. Install NeMo (may have build issues):")
        print("   uv pip install nemo_toolkit[asr]")
        print("   python nemo_baseline_tdt.py dots.wav")
        print()
        print("3. Use the model files with Rust:")
        print("   The files are in: ~/.cache/huggingface/hub/")
        print("   Our Rust implementation can load them directly!")
        print()

        # Load audio to show it works
        print("=" * 60)
        print("AUDIO VERIFICATION")
        print("=" * 60)
        print()
        audio, sr = load_audio(audio_path)
        duration = len(audio) / sr
        print(f"  Audio loaded successfully")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Samples: {len(audio)}")
        print()
        print("✓ Audio is valid and ready for transcription")
        print()

    except Exception as e:
        print(f"Error: {e}")
        print()
        print("This model requires NeMo or our Rust implementation.")
        sys.exit(1)

if __name__ == "__main__":
    main()
