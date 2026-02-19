#!/usr/bin/env python3
"""Phase 1: Test Moonshine V2 Medium Streaming model on dots.wav.

Establishes baseline transcription output for Rust implementation verification.
"""

import sys
import os
import wave
import struct

from moonshine_voice import ModelArch, get_model_for_language
from moonshine_voice.transcriber import Transcriber, load_wav_file


def load_wav_pcm16(path: str) -> list[float]:
    """Load a 16-bit PCM WAV file and return float samples in [-1, 1]."""
    with wave.open(path, 'rb') as wf:
        assert wf.getsampwidth() == 2, f"Expected 16-bit PCM, got {wf.getsampwidth()*8}-bit"
        assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} channels"
        n_frames = wf.getnframes()
        sample_rate = wf.getframerate()
        raw = wf.readframes(n_frames)
        samples = struct.unpack(f'<{n_frames}h', raw)
        # Normalize to [-1, 1]
        return [s / 32768.0 for s in samples], sample_rate


def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "dots.wav"

    if not os.path.exists(wav_path):
        print(f"Error: {wav_path} not found")
        sys.exit(1)

    print(f"Loading audio: {wav_path}")
    samples, sample_rate = load_wav_pcm16(wav_path)
    duration = len(samples) / sample_rate
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Samples: {len(samples)}")
    print(f"  Duration: {duration:.2f}s")

    print("\nDownloading/loading Moonshine V2 Medium Streaming EN model...")
    model_path, arch = get_model_for_language("en", ModelArch.MEDIUM_STREAMING)
    print(f"  Model path: {model_path}")
    print(f"  Architecture: {arch}")

    print("\nCreating transcriber...")
    transcriber = Transcriber(model_path, ModelArch.MEDIUM_STREAMING)

    print("Transcribing (non-streaming)...")
    result = transcriber.transcribe_without_streaming(samples, sample_rate)
    print(f"\n{'='*60}")
    print(f"TRANSCRIPTION RESULT:")
    print(f"{'='*60}")
    full_text = " ".join(line.text for line in result.lines)
    print(f"Full text: {full_text}")
    print(f"Lines: {len(result.lines)}")
    for i, line in enumerate(result.lines):
        end_time = line.start_time + line.duration
        print(f"  [{i}] ({line.start_time:.2f}s - {end_time:.2f}s): {line.text}")
    print(f"{'='*60}")

    transcriber.close()


if __name__ == "__main__":
    main()
