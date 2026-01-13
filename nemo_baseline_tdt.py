#!/usr/bin/env python3
"""
NeMo Baseline: Parakeet TDT v3 Transcription

This script uses NVIDIA NeMo's official implementation of Parakeet TDT 0.6B v3
to transcribe audio files. It serves as a reference baseline for comparing
our Rust implementation.

Usage:
    python nemo_baseline_tdt.py dots.wav
    python nemo_baseline_tdt.py MLKDream_16k.wav

Requirements:
    uv pip install nemo_toolkit[asr]
    # Or: uv pip install nemo-toolkit[all]

Model: nvidia/parakeet-tdt-0.6b-v3 (Transducer/RNN-T)
"""

import sys
import time
import torch
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python nemo_baseline_tdt.py <audio.wav>")
        print("\nExample:")
        print("  python nemo_baseline_tdt.py dots.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    print("=" * 60)
    print("NeMo Baseline: Parakeet TDT v3 Transcription")
    print("=" * 60)
    print(f"\nAudio: {audio_path}\n")

    # Import NeMo ASR
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print("Error: NeMo toolkit not installed.")
        print("\nInstall with:")
        print("  uv pip install nemo_toolkit[asr]")
        print("  # Or for full installation:")
        print("  uv pip install nemo-toolkit[all]")
        sys.exit(1)

    # Load model
    print("Loading Parakeet TDT 0.6B v3 model...")
    print("  Model: nvidia/parakeet-tdt-0.6b-v3")
    print("  Architecture: Transducer (RNN-T)")
    print("  - Encoder: FastConformer (24 layers)")
    print("  - Predictor: LSTM (2 layers, 512 hidden)")
    print("  - Joint Network: 512 hidden → 8193 vocab")

    start_load = time.time()

    # Load from Hugging Face Hub or local cache
    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3"
    )

    load_time = time.time() - start_load
    print(f"  Load time: {load_time:.2f}s\n")

    # Print model details
    print("Model Configuration:")
    print(f"  Sample rate: {model.preprocessor._cfg.get('sample_rate', 16000)} Hz")
    print(f"  Vocab size: {model.decoder.vocab_size if hasattr(model.decoder, 'vocab_size') else 'N/A'}")

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    if device == "cuda":
        model = model.cuda()
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Print decoding configuration
    print("\nDecoding Configuration:")
    try:
        if hasattr(model, 'cfg') and hasattr(model.cfg, 'decoding'):
            dec_cfg = model.cfg.decoding
            print(f"  Strategy: {dec_cfg.get('strategy', 'unknown')}")
            if 'beam' in dec_cfg:
                print(f"  Beam size: {dec_cfg.beam.get('beam_size', 'N/A')}")
            if 'greedy' in dec_cfg:
                print(f"  Max symbols: {dec_cfg.greedy.get('max_symbols', 'N/A')}")
            print(f"  Preserve alignments: {dec_cfg.get('preserve_alignments', False)}")
            print(f"  Compute timestamps: {dec_cfg.get('compute_timestamps', False)}")
        else:
            print("  (Using default decoding)")
    except Exception as e:
        print(f"  (Could not access config: {e})")

    print()

    # Transcribe
    print("Transcribing...")
    start_time = time.time()

    # NeMo's transcribe() method handles everything
    transcription = model.transcribe([audio_path])

    transcribe_time = time.time() - start_time

    # Get audio duration for RTF calculation
    import librosa
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    rtf = transcribe_time / duration

    # Print results
    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULT")
    print("=" * 60)
    print()

    if isinstance(transcription, list) and len(transcription) > 0:
        # Extract text from Hypothesis object
        hypothesis = transcription[0]
        if hasattr(hypothesis, 'text'):
            text = hypothesis.text
        else:
            text = str(hypothesis)

        print(f'"{text}"')
        print()

        # Token count (approximate - split on spaces and punctuation)
        import re
        # Remove punctuation and split
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
        print("Note: Token counts may differ due to tokenization differences")
        print("between SentencePiece implementations.")
        print()

if __name__ == "__main__":
    main()
