#!/usr/bin/env python3
"""Compress Moonshine config and tokenizer to zstd for assets/."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from compress import compress_file


def main():
    src_dir = "hf_moonshine"
    dst_dir = "assets"

    files = [
        ("streaming_config.json", "moonshine-config.json.zst"),
        ("tokenizer.json", "moonshine-tokenizer.json.zst"),
    ]

    for src_name, dst_name in files:
        src_path = os.path.join(src_dir, src_name)
        dst_path = os.path.join(dst_dir, dst_name)

        if not os.path.exists(src_path):
            print(f"ERROR: {src_path} not found")
            sys.exit(1)

        compress_file(src_path, dst_path)

        src_size = os.path.getsize(src_path)
        dst_size = os.path.getsize(dst_path)
        print(f"  {src_size:,} -> {dst_size:,} bytes ({src_size / dst_size:.1f}x)\n")

    print("Done! Moonshine assets ready in assets/")


if __name__ == "__main__":
    main()
