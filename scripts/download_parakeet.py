"""
Download Parakeet CTC weights/config/tokenizer from Hugging Face.

Automatically:
1. Downloads model files from HuggingFace
2. Compresses JSON configs with zstd (for optional binary embedding)
3. Quantizes model to GGUF format with zstd compression

Usage:
  python scripts/download_parakeet.py --repo nvidia/parakeet-ctc-0.6b --output assets

If the repo is private, set HF_TOKEN or run `huggingface-cli login` first.
"""

import argparse
import pathlib
import subprocess
import sys
from huggingface_hub import snapshot_download


def compress_json_configs(out_path: pathlib.Path) -> None:
    """Compress JSON config files with zstd for optional binary embedding."""
    print("\nCompressing JSON configs with zstd...")

    json_files = [
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]

    for json_file in json_files:
        json_path = out_path / json_file
        if json_path.exists():
            zst_path = out_path / f"{json_file}.zst"
            try:
                result = subprocess.run(
                    ["zstd", "-19", "-f", str(json_path), "-o", str(zst_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                # Parse compression ratio from stderr
                if result.stderr:
                    print(f"  ✓ {json_file} -> {json_file}.zst")
                    # Extract compression info if available
                    for line in result.stderr.split('\n'):
                        if json_file in line and ':' in line:
                            print(f"    {line.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to compress {json_file}: {e}")
            except FileNotFoundError:
                print(f"  ✗ zstd command not found. Install with: brew install zstd")
                break
        else:
            print(f"  - Skipping {json_file} (not found)")


def quantize_model(out_path: pathlib.Path) -> None:
    """Quantize safetensors model to GGUF format with zstd compression."""
    print("\nQuantizing model to GGUF with zstd compression...")

    safetensors_path = out_path / "model.safetensors"
    gguf_path = out_path / "model_q8_0.gguf"

    if not safetensors_path.exists():
        print(f"  ✗ model.safetensors not found in {out_path}")
        return

    print(f"  Input:  {safetensors_path}")
    print(f"  Output: {gguf_path}")
    print(f"  Format: Q8_0 (recommended quality/size balance)")
    print(f"  Compression: zstd level 19\n")

    try:
        # Run cargo build first to ensure quantize_gguf is compiled
        print("  Building quantize_gguf tool...")
        subprocess.run(
            ["cargo", "build", "--example", "quantize_gguf", "--release"],
            check=True,
            cwd=pathlib.Path(__file__).parent.parent,
        )

        # Run quantization tool with zstd compression
        print("  Running quantization (this may take a few minutes)...")
        subprocess.run(
            [
                "cargo", "run", "--example", "quantize_gguf", "--release", "--",
                str(safetensors_path),
                str(gguf_path),
                "--format", "q8_0",
                "--compress",
            ],
            check=True,
            cwd=pathlib.Path(__file__).parent.parent,
        )
        print(f"  ✓ Quantized model saved to {gguf_path}")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Quantization failed: {e}")
        print(f"  You can manually quantize later with:")
        print(f"    cargo run --example quantize_gguf --release -- \\")
        print(f"      {safetensors_path} {gguf_path} --format q8_0 --compress")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Parakeet model and prepare for use"
    )
    parser.add_argument(
        "--repo",
        default="nvidia/parakeet-ctc-0.6b",
        help="HF repo id to download",
    )
    parser.add_argument(
        "--output",
        default="assets",
        help="Local directory to place the files (default: assets)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision/tag (defaults to latest).",
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Skip automatic quantization step",
    )
    parser.add_argument(
        "--skip-compress",
        action="store_true",
        help="Skip JSON config compression step",
    )
    args = parser.parse_args()

    print("Parakeet Model Download & Setup")
    print("=" * 40)
    print(f"Repository: {args.repo}")
    print(f"Output: {args.output}\n")

    # Download from HuggingFace
    print("Downloading from HuggingFace...")
    out_path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=args.output,
        local_dir_use_symlinks=False,
    )
    out_path = pathlib.Path(out_path)

    print(f"\n✓ Downloaded to: {out_path}")
    print("\nDownloaded files:")
    for p in sorted(out_path.glob("**/*")):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  - {p.name} ({size_mb:.1f} MB)")

    # Compress JSON configs
    if not args.skip_compress:
        compress_json_configs(out_path)

    # Quantize model
    if not args.skip_quantize:
        quantize_model(out_path)

    print("\n" + "=" * 40)
    print("✓ Setup complete!")
    print(f"\nModel ready to use from: {out_path}")
    print("\nTo use in your code:")
    print(f"  let model = parakeet::load_parakeet_ctc_from_gguf_local(\"{args.output}\", &device)?;")


if __name__ == "__main__":
    main()
