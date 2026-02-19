#!/usr/bin/env python3
"""Download Moonshine V2 Medium Streaming model and prepare for Rust.

Downloads from HuggingFace:
- model.safetensors (FP32 weights, ~1GB)
- config.json
- tokenizer.json

Optionally quantizes to GGUF Q8_0 format.
Optionally compresses with zstd for asset embedding.
"""

import os
import sys
import json
import struct
import argparse
import numpy as np
from pathlib import Path


def download_from_hf(output_dir: str):
    """Download model files from HuggingFace."""
    from huggingface_hub import hf_hub_download

    repo_id = "UsefulSensors/moonshine-streaming-medium"
    files = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "generation_config.json",
    ]

    os.makedirs(output_dir, exist_ok=True)

    for fname in files:
        print(f"Downloading {fname}...")
        src = hf_hub_download(repo_id, fname)
        dst = os.path.join(output_dir, fname)
        if not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)
            size = os.path.getsize(dst)
            print(f"  Saved: {dst} ({size/1024/1024:.1f} MB)")
        else:
            print(f"  Already exists: {dst}")


def create_streaming_config(output_dir: str):
    """Create streaming_config.json for Rust from HF config.json."""
    with open(os.path.join(output_dir, "config.json")) as f:
        hf_config = json.load(f)

    enc = hf_config.get("encoder_config", {})

    streaming_config = {
        "encoder_dim": enc.get("hidden_size", 768),
        "encoder_intermediate_size": enc.get("intermediate_size", 3072),
        "encoder_num_heads": enc.get("num_attention_heads", 10),
        "encoder_num_kv_heads": enc.get("num_key_value_heads", 10),
        "encoder_head_dim": enc.get("head_dim", 64),
        "encoder_hidden_act": enc.get("hidden_act", "gelu"),
        "encoder_num_layers": enc.get("num_hidden_layers", 14),

        "decoder_dim": hf_config.get("hidden_size", 640),
        "decoder_intermediate_size": hf_config.get("intermediate_size", 2560),
        "decoder_num_heads": hf_config.get("num_attention_heads", 10),
        "decoder_num_kv_heads": hf_config.get("num_key_value_heads", 10),
        "decoder_head_dim": hf_config.get("head_dim", 64),
        "decoder_hidden_act": hf_config.get("hidden_act", "silu"),
        "decoder_num_layers": hf_config.get("num_hidden_layers", 14),

        "vocab_size": hf_config.get("vocab_size", 32768),
        "bos_id": hf_config.get("bos_token_id", 1),
        "eos_id": hf_config.get("eos_token_id", 2),
        "pad_id": hf_config.get("pad_token_id", 0),
        "max_position_embeddings": hf_config.get("max_position_embeddings", 4096),

        "frame_len": int(round(
            enc.get("sample_rate", 16000) * enc.get("frame_ms", 5.0) / 1000.0
        )),
        "sample_rate": enc.get("sample_rate", 16000),

        "partial_rotary_factor": hf_config.get("rope_parameters", {}).get("partial_rotary_factor", 0.5),
        "rope_theta": hf_config.get("rope_parameters", {}).get("rope_theta", 10000.0),

        "sliding_windows": enc.get("sliding_windows", []),

        "frontend": {
            "d_model": enc.get("hidden_size", 768),
            "c1": enc.get("hidden_size", 768) * 2,  # conv1 out channels
            "c2": enc.get("hidden_size", 768),       # conv2 out channels
            "kernel_size": 5,
            "stride": 2,
        },

        "tie_word_embeddings": hf_config.get("tie_word_embeddings", False),
    }

    dst = os.path.join(output_dir, "streaming_config.json")
    with open(dst, "w") as f:
        json.dump(streaming_config, f, indent=2)
    print(f"Created: {dst}")


def quantize_to_gguf(output_dir: str, fmt: str = "q8_0"):
    """Quantize safetensors to GGUF format."""
    from safetensors import safe_open
    import struct as st

    src_path = os.path.join(output_dir, "model.safetensors")
    dst_path = os.path.join(output_dir, f"model_{fmt}.gguf")

    if os.path.exists(dst_path):
        print(f"GGUF already exists: {dst_path}")
        return

    print(f"\nQuantizing to GGUF {fmt}...")
    print(f"  Input: {src_path}")
    print(f"  Output: {dst_path}")
    print("  (Use the Rust quantize_gguf example for proper quantization)")
    print("  This script just prepares the safetensors file.")


def compress_with_zstd(output_dir: str, assets_dir: str):
    """Compress files with zstd for asset embedding."""
    import zstandard as zstd

    os.makedirs(assets_dir, exist_ok=True)

    files_to_compress = [
        ("streaming_config.json", "moonshine-config.json.zst"),
        ("tokenizer.json", "moonshine-tokenizer.json.zst"),
    ]

    for src_name, dst_name in files_to_compress:
        src = os.path.join(output_dir, src_name)
        dst = os.path.join(assets_dir, dst_name)
        if not os.path.exists(src):
            print(f"  Skipping {src_name} (not found)")
            continue

        with open(src, 'rb') as f_in:
            data = f_in.read()

        cctx = zstd.ZstdCompressor(level=19)
        compressed = cctx.compress(data)

        with open(dst, 'wb') as f_out:
            f_out.write(compressed)

        ratio = len(data) / len(compressed)
        print(f"  {src_name} -> {dst_name} ({len(data):,} -> {len(compressed):,} bytes, {ratio:.1f}x)")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Moonshine V2 model")
    parser.add_argument("--output", default="hf_moonshine", help="Output directory for model files")
    parser.add_argument("--assets", default="assets", help="Assets directory for compressed files")
    parser.add_argument("--quantize", action="store_true", help="Quantize to GGUF Q8_0")
    parser.add_argument("--compress", action="store_true", help="Compress for asset embedding")
    parser.add_argument("--all", action="store_true", help="Download, create config, quantize, and compress")
    args = parser.parse_args()

    if args.all:
        args.quantize = True
        args.compress = True

    print("=" * 60)
    print("Moonshine V2 Medium Streaming EN - Download & Prepare")
    print("=" * 60)

    # Step 1: Download from HuggingFace
    download_from_hf(args.output)

    # Step 2: Create streaming config for Rust
    create_streaming_config(args.output)

    # Step 3: Optional quantization
    if args.quantize:
        quantize_to_gguf(args.output)

    # Step 4: Optional zstd compression
    if args.compress:
        compress_with_zstd(args.output, args.assets)

    print("\nDone! Files in", args.output)
    for f in sorted(os.listdir(args.output)):
        size = os.path.getsize(os.path.join(args.output, f))
        if size > 1024 * 1024:
            print(f"  {f}: {size/1024/1024:.1f} MB")
        else:
            print(f"  {f}: {size/1024:.1f} KB")


if __name__ == "__main__":
    main()
