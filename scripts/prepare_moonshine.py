#!/usr/bin/env python3
"""Download Moonshine V2 model from HuggingFace and prepare assets for Rust.

This is the single script needed to go from a clean checkout to ready-to-run:

    python scripts/prepare_moonshine.py

It will:
  1. Download model.safetensors, config.json, tokenizer.json from HuggingFace
  2. Create the streaming config JSON for Rust
  3. Copy config and tokenizer into assets/ (plain files, memory-mapped at runtime)
  4. Build and run the Rust quantizer to create assets/moonshine_q8_0.gguf

After this, you can run:
    cargo run --example transcribe_moonshine --release -- audio.wav
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ID = "UsefulSensors/moonshine-streaming-medium"
HF_DIR = "hf_moonshine"
ASSETS_DIR = "assets"

HF_FILES = [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "generation_config.json",
]


def download_from_hf(output_dir: str):
    """Download model files from HuggingFace."""
    from huggingface_hub import hf_hub_download

    os.makedirs(output_dir, exist_ok=True)

    for fname in HF_FILES:
        dst = os.path.join(output_dir, fname)
        if os.path.exists(dst):
            print(f"  {fname}: already exists")
            continue
        print(f"  Downloading {fname}...")
        src = hf_hub_download(REPO_ID, fname)
        shutil.copy2(src, dst)
        size = os.path.getsize(dst)
        if size > 1024 * 1024:
            print(f"    -> {size / 1024 / 1024:.1f} MB")
        else:
            print(f"    -> {size / 1024:.1f} KB")


def create_streaming_config(output_dir: str):
    """Create streaming_config.json for Rust from HF config.json."""
    dst = os.path.join(output_dir, "streaming_config.json")
    if os.path.exists(dst):
        print(f"  streaming_config.json: already exists")
        return

    with open(os.path.join(output_dir, "config.json")) as f:
        hf = json.load(f)

    enc = hf.get("encoder_config", {})

    cfg = {
        "encoder_dim": enc.get("hidden_size", 768),
        "encoder_intermediate_size": enc.get("intermediate_size", 3072),
        "encoder_num_heads": enc.get("num_attention_heads", 10),
        "encoder_num_kv_heads": enc.get("num_key_value_heads", 10),
        "encoder_head_dim": enc.get("head_dim", 64),
        "encoder_hidden_act": enc.get("hidden_act", "gelu"),
        "encoder_num_layers": enc.get("num_hidden_layers", 14),

        "decoder_dim": hf.get("hidden_size", 640),
        "decoder_intermediate_size": hf.get("intermediate_size", 2560),
        "decoder_num_heads": hf.get("num_attention_heads", 10),
        "decoder_num_kv_heads": hf.get("num_key_value_heads", 10),
        "decoder_head_dim": hf.get("head_dim", 64),
        "decoder_hidden_act": hf.get("hidden_act", "silu"),
        "decoder_num_layers": hf.get("num_hidden_layers", 14),

        "vocab_size": hf.get("vocab_size", 32768),
        "bos_id": hf.get("bos_token_id", 1),
        "eos_id": hf.get("eos_token_id", 2),
        "pad_id": hf.get("pad_token_id", 0),
        "max_position_embeddings": hf.get("max_position_embeddings", 4096),

        "frame_len": int(round(
            enc.get("sample_rate", 16000) * enc.get("frame_ms", 5.0) / 1000.0
        )),
        "sample_rate": enc.get("sample_rate", 16000),

        "partial_rotary_factor": hf.get("rope_parameters", {}).get("partial_rotary_factor", 0.5),
        "rope_theta": hf.get("rope_parameters", {}).get("rope_theta", 10000.0),

        "sliding_windows": enc.get("sliding_windows", []),

        "frontend": {
            "d_model": enc.get("hidden_size", 768),
            "c1": enc.get("hidden_size", 768) * 2,
            "c2": enc.get("hidden_size", 768),
            "kernel_size": 5,
            "stride": 2,
        },

        "tie_word_embeddings": hf.get("tie_word_embeddings", False),
    }

    with open(dst, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Created streaming_config.json")


def copy_assets(hf_dir: str, assets_dir: str):
    """Copy config and tokenizer to assets/ (plain files, no compression)."""
    os.makedirs(assets_dir, exist_ok=True)

    copies = [
        ("streaming_config.json", "moonshine-config.json"),
        ("tokenizer.json", "moonshine-tokenizer.json"),
    ]

    for src_name, dst_name in copies:
        src = os.path.join(hf_dir, src_name)
        dst = os.path.join(assets_dir, dst_name)
        if os.path.exists(dst):
            print(f"  {dst_name}: already exists")
            continue
        if not os.path.exists(src):
            print(f"  ERROR: {src} not found", file=sys.stderr)
            sys.exit(1)
        shutil.copy2(src, dst)
        size = os.path.getsize(dst)
        print(f"  {src_name} -> {dst_name} ({size:,} bytes)")


def quantize_gguf(hf_dir: str, assets_dir: str):
    """Build and run the Rust quantizer to create the GGUF file."""
    gguf_path = os.path.join(assets_dir, "moonshine_q8_0.gguf")
    safetensors_path = os.path.join(hf_dir, "model.safetensors")

    if os.path.exists(gguf_path):
        size = os.path.getsize(gguf_path)
        print(f"  moonshine_q8_0.gguf: already exists ({size / 1024 / 1024:.1f} MB)")
        return

    if not os.path.exists(safetensors_path):
        print(f"  ERROR: {safetensors_path} not found", file=sys.stderr)
        sys.exit(1)

    # Build the quantizer
    print("  Building quantizer...")
    result = subprocess.run(
        ["cargo", "build", "-p", "quantize-moonshine", "--release"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: cargo build failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Run the quantizer
    print(f"  Quantizing {safetensors_path} -> {gguf_path}...")
    result = subprocess.run(
        [
            "cargo", "run", "-p", "quantize-moonshine", "--release",
            "--", safetensors_path, gguf_path,
        ],
    )
    if result.returncode != 0:
        print("  ERROR: quantization failed", file=sys.stderr)
        sys.exit(1)

    size = os.path.getsize(gguf_path)
    print(f"  -> {size / 1024 / 1024:.1f} MB")


def main():
    print("=" * 60)
    print("Moonshine V2 Medium Streaming — Prepare Assets")
    print("=" * 60)

    print(f"\n[1/4] Downloading from HuggingFace ({REPO_ID})...")
    download_from_hf(HF_DIR)

    print("\n[2/4] Creating streaming config...")
    create_streaming_config(HF_DIR)

    print(f"\n[3/4] Copying config and tokenizer to {ASSETS_DIR}/...")
    copy_assets(HF_DIR, ASSETS_DIR)

    print(f"\n[4/4] Quantizing to GGUF Q8_0...")
    quantize_gguf(HF_DIR, ASSETS_DIR)

    print("\n" + "=" * 60)
    print("Done! Assets ready in assets/:")
    for f in sorted(Path(ASSETS_DIR).glob("moonshine*")):
        size = f.stat().st_size
        if size > 1024 * 1024:
            print(f"  {f.name}: {size / 1024 / 1024:.1f} MB")
        else:
            print(f"  {f.name}: {size / 1024:.1f} KB")
    print()
    print("Run transcription with:")
    print("  cargo run --example transcribe_moonshine --release -- audio.wav")


if __name__ == "__main__":
    main()
