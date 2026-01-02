"""
Download Silero VAD models and compress for deployment.

The Silero VAD models need to be obtained from the official repository and
converted to safetensors format. This script will compress existing VAD files
or help you set them up.

Usage:
  # If you already have vad16.safetensors and vad16.config.json:
  python scripts/download_vad.py

  # This will compress them and copy to assets directory

The VAD model is small (~1.2 MB) and will be compressed to ~948 KB.

To obtain the VAD model:
1. Download from: https://github.com/snakers4/silero-vad
2. Convert PyTorch JIT model to safetensors (or use existing converted version)
3. Place vad16.safetensors and vad16.config.json in cache directory
"""

import argparse
import pathlib
import shutil
import subprocess
import sys


def compress_file(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """Compress a file with zstd level 19."""
    try:
        result = subprocess.run(
            ["zstd", "-19", "-f", str(input_path), "-o", str(output_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stderr:
            for line in result.stderr.split('\n'):
                if input_path.name in line and ':' in line:
                    print(f"    {line.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Compression failed: {e}")
        raise
    except FileNotFoundError:
        print(f"  ✗ zstd command not found. Install with: brew install zstd")
        raise


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
    print("=" * 40)
    print(f"Cache:  {cache_dir}")
    print(f"Assets: {assets_dir}\n")

    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if files already exist in cache or assets
    model_path = cache_dir / "vad16.safetensors"
    config_path = cache_dir / "vad16.config.json"

    # If not in cache, check assets directory (user might have them there already)
    if not model_path.exists():
        assets_model = assets_dir / "vad16.safetensors"
        if assets_model.exists():
            print("Found existing VAD model in assets, copying to cache...")
            shutil.copy2(assets_model, model_path)

    if not config_path.exists():
        assets_config = assets_dir / "vad16.config.json"
        if assets_config.exists():
            print("Found existing VAD config in assets, copying to cache...")
            shutil.copy2(assets_config, config_path)

    # Check if we have the files now
    if not model_path.exists() or not config_path.exists():
        print("✗ VAD model files not found in cache or assets")
        print("\nTo use this script, you need to obtain the Silero VAD model:")
        print("  1. Download from: https://github.com/snakers4/silero-vad")
        print("  2. Convert to safetensors format (or obtain pre-converted version)")
        print("  3. Place the files in the cache directory:")
        print(f"     {cache_dir}/vad16.safetensors")
        print(f"     {cache_dir}/vad16.config.json")
        print("\nOr place them directly in assets and re-run this script.")
        sys.exit(1)

    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    config_size_b = config_path.stat().st_size

    print(f"✓ Found VAD files:")
    print(f"  vad16.safetensors ({model_size_mb:.2f} MB)")
    print(f"  vad16.config.json ({config_size_b} B)\n")

    # Compress files
    print("Compressing with zstd (level 19)...")

    compressed_files = []

    # Compress model
    model_zst = cache_dir / "vad16.safetensors.zst"
    print(f"  ✓ {model_path.name} -> {model_zst.name}")
    compress_file(model_path, model_zst)
    compressed_files.append(model_zst)

    # Compress config
    config_zst = cache_dir / "vad16.config.json.zst"
    print(f"  ✓ {config_path.name} -> {config_zst.name}")
    compress_file(config_path, config_zst)
    compressed_files.append(config_zst)

    # Copy compressed files to assets directory
    print(f"\nCopying compressed files to assets: {assets_dir}")
    assets_dir.mkdir(parents=True, exist_ok=True)

    for src_file in compressed_files:
        dst_file = assets_dir / src_file.name
        print(f"  ✓ Copying {src_file.name}")
        shutil.copy2(src_file, dst_file)

    # Summary
    print("\n" + "=" * 40)
    print("✓ Setup complete!")
    print(f"\nCache directory: {cache_dir}")
    print(f"  - Original vad16.safetensors and vad16.config.json kept for future use")
    print(f"\nAssets directory: {assets_dir}")
    print(f"  - Only compressed .zst files needed for inference")
    print("\nAssets contents:")
    for p in sorted(assets_dir.glob("vad*.zst")):
        size_kb = p.stat().st_size / 1024
        print(f"  - {p.name} ({size_kb:.1f} KB)")

    print("\nTo use in your code:")
    print(f"  let vad = SileroVad::load(\"{args.assets}\", &device)?;")


if __name__ == "__main__":
    main()
