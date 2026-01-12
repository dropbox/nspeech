"""
Download Parakeet CTC weights/config/tokenizer from Hugging Face.

Automatically:
1. Downloads model files from HuggingFace to cache
2. Compresses JSON configs with zstd (for optional binary embedding)
3. Converts model to FP16 safetensors with zstd compression (for unquantized inference)
4. Quantizes model to GGUF Q8_0 format with zstd compression (for quantized inference)
5. Copies only compressed .zst files needed for inference to assets directory

Usage:
  python scripts/download_parakeet.py --repo nvidia/parakeet-ctc-0.6b

Requirements:
  - safetensors and torch for FP16 conversion: uv pip install safetensors torch
  - zstd for compression: brew install zstd

If the repo is private, set HF_TOKEN or run `huggingface-cli login` first.
"""

import argparse
import pathlib
import shutil
import subprocess
import sys
from huggingface_hub import snapshot_download

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def convert_to_fp16(cache_path: pathlib.Path) -> pathlib.Path | None:
    """Convert FP32 safetensors model to FP16 and compress with zstd.

    Returns path to compressed FP16 model, or None if conversion failed.
    """
    if not SAFETENSORS_AVAILABLE:
        print("\n⚠ Skipping FP16 conversion: safetensors and torch not installed")
        print("  Install with: uv pip install safetensors torch")
        return None

    print("\nConverting model to FP16 format...")

    fp32_path = cache_path / "model.safetensors"
    fp16_path = cache_path / "model_fp16.safetensors"
    fp16_zst_path = cache_path / "model_fp16.safetensors.zst"

    if not fp32_path.exists():
        print(f"  ✗ model.safetensors not found in {cache_path}")
        return None

    try:
        print(f"  Input:  {fp32_path.name}")
        print(f"  Output: {fp16_zst_path.name}")

        # Load FP32 tensors
        print("  Loading FP32 model...")
        tensors = {}
        with safe_open(str(fp32_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        # Convert to FP16
        print("  Converting to FP16...")
        fp16_tensors = {}
        for key, tensor in tensors.items():
            if tensor.dtype == torch.float32:
                fp16_tensors[key] = tensor.half()
            else:
                # Keep non-float tensors as-is (e.g., int32)
                fp16_tensors[key] = tensor

        # Save as safetensors
        print("  Saving FP16 safetensors...")
        save_file(fp16_tensors, str(fp16_path))

        fp32_size_mb = fp32_path.stat().st_size / (1024 * 1024)
        fp16_size_mb = fp16_path.stat().st_size / (1024 * 1024)
        print(f"  Size: {fp32_size_mb:.1f} MB → {fp16_size_mb:.1f} MB ({fp16_size_mb/fp32_size_mb*100:.1f}%)")

        # Compress with zstd
        print("  Compressing with zstd level 19...")
        result = subprocess.run(
            ["zstd", "-19", "-f", str(fp16_path), "-o", str(fp16_zst_path)],
            capture_output=True,
            text=True,
            check=True,
        )

        zst_size_mb = fp16_zst_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Compressed: {zst_size_mb:.1f} MB ({zst_size_mb/fp32_size_mb*100:.1f}% of original)")

        # Clean up uncompressed FP16 file
        fp16_path.unlink()

        return fp16_zst_path

    except Exception as e:
        print(f"  ✗ FP16 conversion failed: {e}")
        return None


def compress_json_configs(cache_path: pathlib.Path) -> list[pathlib.Path]:
    """Compress JSON config files with zstd for optional binary embedding.

    Returns list of compressed .zst file paths.
    """
    print("\nCompressing JSON configs with zstd...")

    json_files = [
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]

    compressed_files = []
    for json_file in json_files:
        json_path = cache_path / json_file
        if json_path.exists():
            zst_path = cache_path / f"{json_file}.zst"
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
                compressed_files.append(zst_path)
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to compress {json_file}: {e}")
            except FileNotFoundError:
                print(f"  ✗ zstd command not found. Install with: brew install zstd")
                break
        else:
            print(f"  - Skipping {json_file} (not found)")

    return compressed_files


def quantize_model(cache_path: pathlib.Path) -> pathlib.Path | None:
    """Quantize safetensors model to GGUF format with zstd compression.

    Returns path to quantized .gguf.zst file, or None if quantization failed.
    """
    print("\nQuantizing model to GGUF with zstd compression...")

    safetensors_path = cache_path / "model.safetensors"
    gguf_path = cache_path / "model_q8_0.gguf.zst"

    if not safetensors_path.exists():
        print(f"  ✗ model.safetensors not found in {cache_path}")
        return None

    print(f"  Input:  {safetensors_path}")
    print(f"  Output: {gguf_path}")
    print(f"  Format: Q8_0 (recommended quality/size balance)")
    print(f"  Compression: zstd level 19 (inline)\n")

    try:
        # Run cargo build first to ensure quantize_gguf is compiled
        print("  Building quantize_gguf tool...")
        subprocess.run(
            ["cargo", "build", "--example", "quantize_gguf", "--release"],
            check=True,
            cwd=pathlib.Path(__file__).parent.parent,
        )

        # Run quantization tool (compression is always enabled)
        print("  Running quantization (this may take a few minutes)...")
        subprocess.run(
            [
                "cargo", "run", "--example", "quantize_gguf", "--release", "--",
                str(safetensors_path),
                str(gguf_path),
                "--format", "q8_0",
            ],
            check=True,
            cwd=pathlib.Path(__file__).parent.parent,
        )
        print(f"  ✓ Quantized model saved to {gguf_path}")
        return gguf_path
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Quantization failed: {e}")
        print(f"  You can manually quantize later with:")
        print(f"    cargo run --example quantize_gguf --release -- \\")
        print(f"      {safetensors_path} {gguf_path} --format q8_0")
        return None


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
        "--cache",
        default=None,
        help="Cache directory for downloads (default: .cache/parakeet in project root)",
    )
    parser.add_argument(
        "--assets",
        default="assets",
        help="Assets directory for final compressed files (default: assets)",
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

    # Determine cache directory
    if args.cache:
        cache_dir = pathlib.Path(args.cache)
    else:
        # Default to .cache/parakeet in project root
        project_root = pathlib.Path(__file__).parent.parent
        cache_dir = project_root / ".cache" / "parakeet"

    assets_dir = pathlib.Path(args.assets)

    print("Parakeet Model Download & Setup")
    print("=" * 40)
    print(f"Repository: {args.repo}")
    print(f"Cache:      {cache_dir}")
    print(f"Assets:     {assets_dir}\n")

    # Download from HuggingFace to cache
    print("Downloading from HuggingFace to cache...")
    cache_path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )
    cache_path = pathlib.Path(cache_path)

    print(f"\n✓ Downloaded to cache: {cache_path}")
    print("\nDownloaded files:")
    for p in sorted(cache_path.glob("**/*")):
        if p.is_file() and not p.name.endswith(".zst"):
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  - {p.name} ({size_mb:.1f} MB)")

    # Compress JSON configs in cache
    compressed_files = []
    if not args.skip_compress:
        compressed_files = compress_json_configs(cache_path)

    # Convert model to FP16 and compress
    fp16_file = convert_to_fp16(cache_path)

    # Quantize model in cache
    gguf_file = None
    if not args.skip_quantize:
        gguf_file = quantize_model(cache_path)

    # Copy compressed files to assets directory
    print(f"\nCopying compressed files to assets directory: {assets_dir}")
    assets_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = []
    if compressed_files:
        files_to_copy.extend(compressed_files)
    if fp16_file:
        files_to_copy.append(fp16_file)
    if gguf_file:
        files_to_copy.append(gguf_file)

    if not files_to_copy:
        print("  ✗ No compressed files to copy")
    else:
        for src_file in files_to_copy:
            dst_file = assets_dir / src_file.name
            print(f"  ✓ Copying {src_file.name}")
            shutil.copy2(src_file, dst_file)

    # Summary
    print("\n" + "=" * 40)
    print("✓ Setup complete!")
    print(f"\nCache directory: {cache_path}")
    print(f"  - Original model.safetensors and JSON files kept for future use")
    print(f"\nAssets directory: {assets_dir}")
    print(f"  - Only compressed .zst files needed for inference")
    print("\nAssets contents:")
    for p in sorted(assets_dir.glob("*.zst")):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  - {p.name} ({size_mb:.1f} MB)")

    print("\nTo use in your code:")
    print(f"  let model = parakeet::load_parakeet_ctc_from_gguf_local(\"{args.assets}\", &device)?;")


if __name__ == "__main__":
    main()
