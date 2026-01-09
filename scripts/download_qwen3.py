"""
Download Qwen3-0.6B-Instruct model for text correction from Hugging Face.

Automatically:
1. Downloads pre-quantized GGUF model from bartowski's repo
2. Downloads tokenizer and config from official Qwen repo
3. Compresses all files with zstd level 19
4. Copies compressed .zst files to assets directory

Usage:
  python scripts/download_qwen3.py
  python scripts/download_qwen3.py --assets /path/to/assets

The model is used for correcting ASR transcriptions by adding proper
punctuation and capitalization.
"""

import argparse
import pathlib
import shutil
import subprocess
from huggingface_hub import hf_hub_download


def compress_file(file_path: pathlib.Path, output_name: str) -> pathlib.Path | None:
    """Compress a file with zstd level 19.

    Args:
        file_path: Path to file to compress
        output_name: Name for the compressed output file

    Returns:
        Path to compressed .zst file, or None if compression failed
    """
    zst_path = file_path.parent / output_name
    try:
        result = subprocess.run(
            ["zstd", "-19", "-f", str(file_path), "-o", str(zst_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse compression info from stderr
        if result.stderr:
            for line in result.stderr.split('\n'):
                if file_path.name in line and ':' in line:
                    print(f"    {line.strip()}")
        return zst_path
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to compress {file_path.name}: {e}")
        return None
    except FileNotFoundError:
        print(f"  ✗ zstd command not found. Install with: brew install zstd")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Qwen3 model and prepare for use"
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Cache directory for downloads (default: .cache/qwen3 in project root)",
    )
    parser.add_argument(
        "--assets",
        default="assets",
        help="Assets directory for final compressed files (default: assets)",
    )
    parser.add_argument(
        "--quantization",
        default="Q4_K_M",
        choices=["Q4_K_M", "Q8_0", "Q5_K_M"],
        help="GGUF quantization format (default: Q4_K_M for best size/quality)",
    )
    args = parser.parse_args()

    # Determine cache directory
    if args.cache:
        cache_dir = pathlib.Path(args.cache)
    else:
        # Default to .cache/qwen3 in project root
        project_root = pathlib.Path(__file__).parent.parent
        cache_dir = project_root / ".cache" / "qwen3"

    assets_dir = pathlib.Path(args.assets)
    cache_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    print("Qwen3 Model Download & Setup")
    print("=" * 40)
    print(f"Model:      Qwen3-0.6B-Instruct")
    print(f"Quant:      {args.quantization}")
    print(f"Cache:      {cache_dir}")
    print(f"Assets:     {assets_dir}\n")

    # Download files from HuggingFace
    print("Downloading from HuggingFace...\n")

    # 1. Download pre-quantized GGUF model from bartowski's repo
    gguf_repo = "bartowski/Qwen_Qwen3-0.6B-GGUF"
    gguf_filename = f"Qwen_Qwen3-0.6B-{args.quantization}.gguf"

    print(f"  Downloading quantized model from {gguf_repo}...")
    print(f"    File: {gguf_filename}")
    gguf_path = hf_hub_download(
        repo_id=gguf_repo,
        filename=gguf_filename,
        cache_dir=str(cache_dir / "huggingface"),
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )
    gguf_path = pathlib.Path(gguf_path)
    size_mb = gguf_path.stat().st_size / (1024 * 1024)
    print(f"    ✓ Downloaded ({size_mb:.1f} MB)")

    # 2. Download tokenizer from official Qwen repo
    qwen_repo = "Qwen/Qwen3-0.6B-Instruct"

    print(f"\n  Downloading tokenizer from {qwen_repo}...")
    tokenizer_path = hf_hub_download(
        repo_id=qwen_repo,
        filename="tokenizer.json",
        cache_dir=str(cache_dir / "huggingface"),
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )
    tokenizer_path = pathlib.Path(tokenizer_path)
    size_kb = tokenizer_path.stat().st_size / 1024
    print(f"    ✓ Downloaded ({size_kb:.1f} KB)")

    # 3. Download config from official Qwen repo
    print(f"\n  Downloading config from {qwen_repo}...")
    config_path = hf_hub_download(
        repo_id=qwen_repo,
        filename="config.json",
        cache_dir=str(cache_dir / "huggingface"),
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )
    config_path = pathlib.Path(config_path)
    size_kb = config_path.stat().st_size / 1024
    print(f"    ✓ Downloaded ({size_kb:.1f} KB)")

    # Compress all files with zstd
    print("\nCompressing files with zstd level 19...")

    compressed_files = []

    # Compress GGUF model
    print(f"  ✓ {gguf_path.name} -> qwen3-0.6b-instruct-q4_k_m.gguf.zst")
    gguf_zst = compress_file(gguf_path, "qwen3-0.6b-instruct-q4_k_m.gguf.zst")
    if gguf_zst:
        compressed_files.append(gguf_zst)

    # Compress tokenizer
    print(f"  ✓ tokenizer.json -> qwen3-0.6b-instruct-tokenizer.json.zst")
    tokenizer_zst = compress_file(tokenizer_path, "qwen3-0.6b-instruct-tokenizer.json.zst")
    if tokenizer_zst:
        compressed_files.append(tokenizer_zst)

    # Compress config
    print(f"  ✓ config.json -> qwen3-0.6b-instruct-config.json.zst")
    config_zst = compress_file(config_path, "qwen3-0.6b-instruct-config.json.zst")
    if config_zst:
        compressed_files.append(config_zst)

    # Copy compressed files to assets directory
    print(f"\nCopying compressed files to assets directory...")

    if not compressed_files:
        print("  ✗ No compressed files to copy")
    else:
        for src_file in compressed_files:
            dst_file = assets_dir / src_file.name
            print(f"  ✓ {src_file.name}")
            shutil.copy2(src_file, dst_file)

    # Summary
    print("\n" + "=" * 40)
    print("✓ Setup complete!")
    print(f"\nCache directory: {cache_dir}")
    print(f"  - Original files kept for future use")
    print(f"\nAssets directory: {assets_dir}")
    print(f"  - Compressed .zst files ready for use")
    print("\nAssets contents:")
    for p in sorted(assets_dir.glob("qwen3-*.zst")):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  - {p.name} ({size_mb:.1f} MB)")

    print("\nTo use in your code:")
    print(f"  cargo run --example transcribe_with_vad --release -- audio.wav --use-qwen")
    print("\nFor more information:")
    print(f"  - See src/qwen.rs for implementation details")
    print(f"  - See QWEN_INTEGRATION.md for usage guide")


if __name__ == "__main__":
    main()
