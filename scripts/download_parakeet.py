"""
Download Parakeet CTC weights/config/tokenizer from Hugging Face.

Usage:
  python scripts/download_parakeet.py --repo nvidia/parakeet-ctc-0.6b --output ./hf_parakeet_0_6b

If the repo is private, set HF_TOKEN or run `huggingface-cli login` first.
"""

import argparse
import pathlib
from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default="nvidia/parakeet-ctc-0.6b",
        help="HF repo id to download",
    )
    parser.add_argument(
        "--output",
        default="hf_parakeet",
        help="Local directory to place the snapshot",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision/tag (defaults to latest).",
    )
    args = parser.parse_args()

    out_path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=args.output,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {out_path}")
    print("Files:")
    for p in sorted(pathlib.Path(out_path).glob("**/*")):
        if p.is_file():
            print(" -", p.relative_to(out_path))


if __name__ == "__main__":
    main()
