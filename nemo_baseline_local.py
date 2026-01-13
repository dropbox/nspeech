#!/usr/bin/env python3
"""
NeMo Baseline: Parakeet TDT v3 Transcription (Local Model)

This script uses the local .nemo model file to avoid NeMo installation issues.
It loads the model directly from the .cache/parakeet-tdt directory.

Usage:
    python nemo_baseline_local.py dots.wav

Requirements:
    uv pip install torch torchaudio librosa nemo-toolkit

Model: .cache/parakeet-tdt/parakeet-tdt-0.6b-v3.nemo
"""

import sys
import time
import torch
from pathlib import Path

def load_audio(audio_path, target_sr=16000):
    """Load audio file and resample to target sample rate."""
    import librosa
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio, sr

def main():
    if len(sys.argv) < 2:
        print("Usage: python nemo_baseline_local.py <audio.wav>")
        print("\nExample:")
        print("  python nemo_baseline_local.py dots.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print("=" * 60)
    print("NeMo Baseline: Parakeet TDT v3 (Local Model)")
    print("=" * 60)
    print(f"\nAudio: {audio_path}\n")

    # Check for local model
    local_model_path = ".cache/parakeet-tdt/parakeet-tdt-0.6b-v3.nemo"
    if not Path(local_model_path).exists():
        print(f"Error: Local model not found: {local_model_path}")
        print("\nThe model file should be in .cache/parakeet-tdt/")
        print("Please download it first.")
        sys.exit(1)

    # Import NeMo ASR
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print("Error: NeMo toolkit not installed.")
        print("\nInstall with:")
        print("  uv pip install nemo-toolkit")
        print("\nNote: This may have build issues. If it fails, use our Rust implementation:")
        print("  cargo run --example transcribe_tdt --release -- dots.wav")
        sys.exit(1)

    # Load model from local .nemo file
    print(f"Loading model from: {local_model_path}")
    print("  Architecture: Transducer (RNN-T)")
    print("  - Encoder: FastConformer (24 layers)")
    print("  - Predictor: LSTM (2 layers, 512 hidden)")
    print("  - Joint Network: 512 hidden → 8193 vocab\n")

    start_load = time.time()

    try:
        # Load from local .nemo file
        model = nemo_asr.models.ASRModel.restore_from(local_model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nIf you get dependency errors, please use our Rust implementation:")
        print("  cargo run --example transcribe_tdt --release -- dots.wav")
        sys.exit(1)

    load_time = time.time() - start_load
    print(f"  Load time: {load_time:.2f}s\n")

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    if device == "cuda":
        model = model.cuda()
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print()

    # Transcribe
    print("Transcribing...")
    start_time = time.time()

    transcription = model.transcribe([audio_path])

    transcribe_time = time.time() - start_time

    # Get audio duration for RTF calculation
    audio, sr = load_audio(audio_path)
    duration = len(audio) / sr
    rtf = transcribe_time / duration

    # Print results
    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULT")
    print("=" * 60)
    print()

    if isinstance(transcription, list) and len(transcription) > 0:
        text = transcription[0]
        print(f'"{text}"')
        print()

        # Token count (approximate - split on spaces and punctuation)
        import re
        tokens = re.findall(r'\w+', text.lower())
        token_count = len(tokens)

        print("Statistics:")
        print(f"  Audio duration: {duration:.2f}s")
        print(f"  Transcription time: {transcribe_time:.2f}s")
        print(f"  Real-time factor: {rtf:.3f}x")
        print(f"  Word count: {token_count}")
        print(f"  Characters: {len(text)}")
    else:
        print("(No transcription returned)")

    print()

    # Compare with our Rust baseline if this is dots.wav
    if "dots.wav" in audio_path.lower():
        print("=" * 60)
        print("COMPARISON WITH RUST IMPLEMENTATION")
        print("=" * 60)
        print()
        print("Rust baseline (transcribe_tdt):")
        print("  Tokens: 140")
        print("  Quality: 100% (reference)")
        print()
        print("Rust VAD-based (transcribe_tdt_with_vad):")
        print("  Tokens: 140")
        print("  Quality: 100% ✓")
        print()
        print("Rust chunked streaming (transcribe_tdt_streaming):")
        print("  Tokens: 99")
        print("  Quality: 71%")
        print()
        print(f"NeMo baseline (this script):")
        print(f"  Words: {token_count}")
        print()
        print("Note: Word count may differ from token count due to")
        print("      different tokenization approaches.")
        print()

if __name__ == "__main__":
    main()
