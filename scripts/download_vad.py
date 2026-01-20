"""
Download Silero VAD models, quantize to GGUF Q8_0, and compress for deployment.

This script downloads the Silero VAD v4.0 model from GitHub in safetensors format,
quantizes it to GGUF Q8_0 format for efficient inference, and compresses both versions
for use in the speech recognition system.

Usage:
  python scripts/download_vad.py

  # Or specify custom directories:
  python scripts/download_vad.py --cache .cache/vad --assets assets

The VAD model is small (~1.2 MB) and will be quantized to ~194 KB (6.2x reduction).
No PyTorch required for download - safetensors format used directly.
Quantization requires Rust toolchain (cargo) to be available.
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import urllib.request
from compress import compress_file

# Updated URL - files moved from files/ to src/silero_vad/data/
# Using direct safetensors format (no PyTorch conversion needed)
SILERO_VAD_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad_16k.safetensors"

SILERO_CONFIG = {
    "model_type": "silero_vad",
    "version": "4.0",
    "sample_rate": 16000,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 100,
    "speech_pad_ms": 30,
    # STFT parameters (from GitHub model)
    "hop_length": 128,           # 8ms hop at 16kHz (GitHub model uses 128)
    "win_length": 256,           # 16ms window at 16kHz (basis kernel size)
    "n_fft": 512,                # FFT size (258 bins = 512/2 + 2)
    "stft_right_padding": 64,    # Reflection padding for streaming
    # Encoder parameters
    "encoder_padding": 1,        # Conv1d padding
    # Streaming parameters
    "context_size": 64,          # Context samples (4ms at 16kHz)
    "chunk_size": 512            # Processing chunk (32ms at 16kHz)
}


def download_file(url: str, dest_path: pathlib.Path) -> None:
    """Download a file from URL to dest_path."""
    print(f"  Downloading {url}")
    print(f"  → {dest_path.name}")

    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192

            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Progress: {percent:.1f}%", end='', flush=True)

            print()  # New line after progress

    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        raise


def convert_github_safetensors(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """Convert GitHub safetensors format to expected tensor names."""
    print(f"  Converting tensor names to expected format...")

    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
        import torch
    except ImportError:
        print("  ✗ Missing dependencies. Install with:")
        print("    pip install torch safetensors")
        sys.exit(1)

    # Load GitHub safetensors
    tensors = {}
    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # Map GitHub tensor names to expected names
    name_mapping = {
        # STFT basis (only weight, no bias)
        "stft_conv.weight": "stft.forward_basis_buffer",

        # Encoder convolutions (4 layers)
        "conv1.weight": "enc.0.weight",
        "conv1.bias": "enc.0.bias",
        "conv2.weight": "enc.1.weight",
        "conv2.bias": "enc.1.bias",
        "conv3.weight": "enc.2.weight",
        "conv3.bias": "enc.2.bias",
        "conv4.weight": "enc.3.weight",
        "conv4.bias": "enc.3.bias",

        # RNN/LSTM weights
        "lstm_cell.weight_ih": "rnn.weight_ih",
        "lstm_cell.weight_hh": "rnn.weight_hh",
        "lstm_cell.bias_ih": "rnn.bias_ih",
        "lstm_cell.bias_hh": "rnn.bias_hh",

        # Head (output layer)
        "final_conv.weight": "head.weight",
        "final_conv.bias": "head.bias",
    }

    # Create mapped state dict
    mapped_tensors = {}
    for old_name, tensor in tensors.items():
        if old_name in name_mapping:
            new_name = name_mapping[old_name]
            mapped_tensors[new_name] = tensor
            print(f"    {old_name} → {new_name}")
        else:
            print(f"    Warning: Unmapped tensor: {old_name}")

    # Verify all expected tensors are present
    expected_tensors = set(name_mapping.values())
    found_tensors = set(mapped_tensors.keys())
    missing = expected_tensors - found_tensors
    if missing:
        print(f"  ✗ Missing expected tensors: {missing}")
        sys.exit(1)

    # Save with mapped names
    save_file(mapped_tensors, str(output_path))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Converted safetensors ({size_mb:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare Silero VAD models"
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Cache directory for downloads (default: .cache/vad in project root)",
    )
    parser.add_argument(
        "--assets",
        default="assets",
        help="Assets directory for final compressed files (default: assets)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if files already exist in cache",
    )
    args = parser.parse_args()

    # Determine cache directory
    if args.cache:
        cache_dir = pathlib.Path(args.cache)
    else:
        # Default to .cache/vad in project root
        project_root = pathlib.Path(__file__).parent.parent
        cache_dir = project_root / ".cache" / "vad"

    assets_dir = pathlib.Path(args.assets)

    print("Silero VAD Model Download & Setup")
    print("=" * 60)
    print(f"Cache:  {cache_dir}")
    print(f"Assets: {assets_dir}\n")

    # Create directories
    cache_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    github_model_path = cache_dir / "silero_vad_16k_github.safetensors"  # Downloaded from GitHub
    model_path = cache_dir / "vad16.safetensors"  # Converted format
    config_path = cache_dir / "vad16.config.json"

    # Check if files exist in project root (already present in repo)
    project_root = pathlib.Path(__file__).parent.parent
    root_model = project_root / "vad16.safetensors"
    root_config = project_root / "vad16.config.json"

    # Copy from project root to cache if they exist there
    if root_model.exists() and not model_path.exists():
        print("Step 1: Found VAD model in project root, copying to cache...")
        shutil.copy2(root_model, model_path)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Copied vad16.safetensors ({size_mb:.2f} MB)\n")
    elif model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"Step 1: Using cached safetensors: {model_path} ({size_mb:.2f} MB)\n")
    else:
        # Download and convert GitHub safetensors
        print("Step 1: Downloading Silero VAD v4.0 safetensors model...")
        print("  Note: If download fails, you can manually place vad16.safetensors in the project root")
        download_file(SILERO_VAD_URL, github_model_path)
        print("\nStep 2: Converting tensor names to expected format...")
        convert_github_safetensors(github_model_path, model_path)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Model ready ({size_mb:.2f} MB)\n")

    if root_config.exists() and not config_path.exists():
        print("Step 3: Found VAD config in project root, copying to cache...")
        shutil.copy2(root_config, config_path)
        print(f"  ✓ Copied vad16.config.json\n")
    elif not config_path.exists():
        print("Step 3: Creating config file...")
        with open(config_path, 'w') as f:
            json.dump(SILERO_CONFIG, f, indent=2)
        print(f"  ✓ Created {config_path.name}\n")
    else:
        print(f"Step 3: Using cached config: {config_path}\n")

    # Compress files
    print("Step 4: Compressing safetensors with zstd (level 19)...")

    model_zst = assets_dir / "vad16.safetensors.zst"
    compress_file(model_path, model_zst)

    config_zst = assets_dir / "vad16.config.json.zst"
    compress_file(config_path, config_zst)

    # Quantize to GGUF Q8_0
    print("\nStep 5: Quantizing to GGUF Q8_0 format...")
    gguf_path = cache_dir / "vad16_q8_0.gguf.zst"

    try:
        # Run the Rust quantization tool
        result = subprocess.run(
            ["cargo", "run", "--example", "quantize_vad_gguf", "--release", "--",
             str(model_path), str(gguf_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True
        )

        print("  Quantization output:")
        # Print key lines from output
        for line in result.stdout.split('\n'):
            if any(x in line for x in ['Quantizing', 'Q8_0', 'FP32', 'compression', '✓']):
                print(f"    {line}")

        # Copy to assets directory
        assets_gguf = assets_dir / "vad16_q8_0.gguf.zst"
        shutil.copy2(gguf_path, assets_gguf)

        size_kb = assets_gguf.stat().st_size / 1024
        print(f"  ✓ Quantized GGUF created ({size_kb:.1f} KB)")

    except subprocess.CalledProcessError as e:
        print(f"  ✗ Quantization failed: {e}")
        print(f"  stderr: {e.stderr}")
        print("  Continuing without quantized model...")
    except FileNotFoundError:
        print("  ✗ Cargo not found. Skipping quantization.")
        print("  You can manually quantize later with:")
        print(f"    cargo run --example quantize_vad_gguf --release -- {model_path} {gguf_path}")

    # Summary
    print("\n" + "=" * 60)
    print("✓ Setup complete!")
    print(f"\nCache directory: {cache_dir}")
    print("  - Original files kept for future use")

    print(f"\nAssets directory: {assets_dir}")
    print("  - Compressed .zst files ready for inference:")
    for p in sorted(assets_dir.glob("vad*.zst")):
        size_kb = p.stat().st_size / 1024
        model_type = "quantized GGUF Q8_0" if "q8_0" in p.name else "safetensors FP32"
        print(f"    • {p.name:30s} ({size_kb:6.1f} KB) - {model_type}")

    print("\nYou can now run:")
    print("  cargo run --release --example transcribe_with_vad -- audio.wav")
    print("\nThe quantized GGUF model (vad16_q8_0.gguf.zst) is used by default for 6.2x smaller size.")


if __name__ == "__main__":
    main()
