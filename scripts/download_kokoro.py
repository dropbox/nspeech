#!/usr/bin/env python3
"""Download Kokoro TTS model weights from HuggingFace."""

import os
import sys
from pathlib import Path

def main():
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("pip install huggingface_hub")
        sys.exit(1)

    repo_id = "hexgrad/Kokoro-82M"
    output_dir = Path("hf_kokoro")
    output_dir.mkdir(exist_ok=True)
    voices_dir = output_dir / "voices"
    voices_dir.mkdir(exist_ok=True)

    # Download config and model
    for filename in ["config.json", "kokoro-v1_0.pth"]:
        dest = output_dir / filename
        if dest.exists():
            print(f"  Already exists: {dest}")
            continue
        print(f"  Downloading {filename}...")
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(output_dir))

    # Download a default voice pack
    voice_file = "voices/af_heart.pt"
    dest = output_dir / voice_file
    if not dest.exists():
        print(f"  Downloading {voice_file}...")
        hf_hub_download(repo_id=repo_id, filename=voice_file, local_dir=str(output_dir))
    else:
        print(f"  Already exists: {dest}")

    print(f"\nModel files in {output_dir}/")
    print("\nNote: The model is in PyTorch .pth format.")
    print("Convert to safetensors with:")
    print("  python scripts/convert_kokoro_safetensors.py")

if __name__ == "__main__":
    main()
