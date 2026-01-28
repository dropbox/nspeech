"""
Download Parakeet TDT (Transducer) model from Hugging Face and convert to safetensors.

NeMo (.nemo) files are tar archives containing:
- model_config.yaml: Model configuration
- model_weights.ckpt: PyTorch checkpoint
- tokenizer files

This script:
1. Downloads parakeet-tdt-0.6b-v3.nemo from Hugging Face
2. Extracts the archive to get config and weights
3. Converts PyTorch weights to safetensors format
4. Compresses with zstd for binary embedding

Usage:
  python scripts/download_parakeet_tdt.py --output assets

Requirements:
  pip install huggingface_hub torch safetensors pyyaml
  brew install zstd
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import tarfile
import tempfile
from typing import Dict, Any
from compress import compress_file

try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file
    import torch
    import yaml
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


def extract_nemo_file(nemo_path: pathlib.Path, extract_dir: pathlib.Path) -> Dict[str, pathlib.Path]:
    """Extract .nemo file (tar archive) to get model config and weights.

    Returns:
        Dict with paths to extracted files
    """
    print(f"\nExtracting {nemo_path.name}...")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(nemo_path, 'r') as tar:
        tar.extractall(extract_dir)

    # Find extracted files
    files = {
        'config': None,
        'weights': None,
        'tokenizer': None,
    }

    for file in extract_dir.rglob('*'):
        if file.is_file():
            if 'model_config' in file.name and file.suffix in ['.yaml', '.yml']:
                files['config'] = file
                print(f"  ✓ Found config: {file.name}")
            elif file.suffix == '.ckpt' or 'weights' in file.name:
                files['weights'] = file
                print(f"  ✓ Found weights: {file.name}")
            elif 'tokenizer' in file.name and file.suffix == '.model':
                files['tokenizer'] = file
                print(f"  ✓ Found tokenizer: {file.name}")

    return files


def convert_nemo_config_to_json(yaml_config_path: pathlib.Path) -> Dict[str, Any]:
    """Convert NeMo YAML config to our JSON format."""
    with open(yaml_config_path, 'r') as f:
        nemo_config = yaml.safe_load(f)

    # Extract relevant fields
    # NeMo config structure varies, so we need to handle different formats
    encoder_cfg = nemo_config.get('encoder', {})
    decoder_cfg = nemo_config.get('decoder', {})
    joint_cfg = nemo_config.get('joint', {})

    # Extract FFN expansion factor (typically 4x for Conformer)
    d_model = encoder_cfg.get('d_model', 1024)
    ff_expansion = encoder_cfg.get('ff_expansion_factor', 4)  # Conformer uses 4x expansion

    # NeMo nests predictor/joint configs inside prednet/jointnet
    prednet_cfg = decoder_cfg.get('prednet', {})
    jointnet_cfg = joint_cfg.get('jointnet', {})

    config = {
        'architectures': ['ParakeetForTransducer'],
        'model_type': 'parakeet_transducer',
        'vocab_size': decoder_cfg.get('vocab_size', 8192),
        'blank_id': decoder_cfg.get('blank_id', 0),
        'joint_vocab_size': 8198,  # Joint output includes special tokens (verified from model weights)
        'encoder_config': {
            'hidden_size': d_model,
            'num_hidden_layers': encoder_cfg.get('n_layers', 24),
            'num_attention_heads': encoder_cfg.get('n_heads', 8),
            'intermediate_size': d_model * ff_expansion,  # FFN intermediate dimension
            'num_mel_bins': encoder_cfg.get('feat_in', 128),
            'subsampling_factor': encoder_cfg.get('subsampling_factor', 8),
            'conv_kernel_size': encoder_cfg.get('conv_kernel_size', 9),
            'dropout': encoder_cfg.get('dropout', 0.1),
            'dropout_positions': encoder_cfg.get('dropout_emb', 0.0),
            'subsampling_conv_channels': encoder_cfg.get('subsampling_conv_channels', 256),
            'subsampling_conv_stride': 2,
            'scale_input': True,
        },
        'predictor_config': {
            'pred_hidden': prednet_cfg.get('pred_hidden', 640),
            'pred_rnn_layers': prednet_cfg.get('pred_rnn_layers', 2),
        },
        'joint_config': {
            'joint_hidden': jointnet_cfg.get('joint_hidden', 640),
            'activation': jointnet_cfg.get('activation', 'relu'),
        }
    }

    return config


def convert_weights_to_safetensors(ckpt_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """Convert PyTorch checkpoint to safetensors format (BF16 for space efficiency)."""
    print(f"\nConverting weights to safetensors (BF16)...")
    print(f"  Input:  {ckpt_path.name}")
    print(f"  Output: {output_path.name}")

    # Load PyTorch checkpoint
    print("  Loading checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Extract state dict (may be nested)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Remove any module. prefix if present
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('model.', '').replace('module.', '')
        cleaned_state_dict[new_key] = value

    print(f"  Found {len(cleaned_state_dict)} tensors")

    # Convert to BF16 to save disk space (1.5GB vs 2.4GB)
    print("  Converting to BF16...")
    bf16_state_dict = {}
    for key, tensor in cleaned_state_dict.items():
        if tensor.dtype != torch.bfloat16:
            bf16_state_dict[key] = tensor.bfloat16()
        else:
            bf16_state_dict[key] = tensor

    # Save as safetensors
    print("  Saving safetensors...")
    save_file(bf16_state_dict, str(output_path))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {size_mb:.1f} MB")


def quantize_tdt_model(cache_dir: pathlib.Path, safetensors_path: pathlib.Path) -> pathlib.Path | None:
    """Quantize TDT safetensors model to GGUF format with zstd compression.

    Returns path to quantized .gguf.zst file, or None if quantization failed.
    """
    print("\nQuantizing TDT model to GGUF with zstd compression...")

    gguf_path = cache_dir / "parakeet-tdt-model_q8_0.gguf.zst"

    if not safetensors_path.exists():
        print(f"  ✗ model.safetensors not found at {safetensors_path}")
        return None

    print(f"  Input:  {safetensors_path}")
    print(f"  Output: {gguf_path}")
    print(f"  Format: Q8_0 (recommended quality/size balance)")
    print(f"  Compression: zstd level 19 (inline)\n")

    try:
        # Run cargo build first to ensure quantize_gguf is compiled
        print("  Building quantize_gguf tool...")
        project_root = pathlib.Path(__file__).parent.parent
        subprocess.run(
            ["cargo", "build", "--example", "quantize_gguf", "--release"],
            check=True,
            cwd=project_root,
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
            cwd=project_root,
        )
        print(f"  ✓ Quantized TDT model saved to {gguf_path}")
        return gguf_path
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Quantization failed: {e}")
        print(f"  You can manually quantize later with:")
        print(f"    cargo run --example quantize_gguf --release -- \\")
        print(f"      {safetensors_path} {gguf_path} --format q8_0")
        return None


def main():
    if not DEPS_AVAILABLE:
        print("Error: Required dependencies not installed")
        print("Install with: pip install huggingface_hub torch safetensors pyyaml")
        return 1

    parser = argparse.ArgumentParser(
        description="Download and convert Parakeet TDT model"
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Cache directory (default: .cache/parakeet-tdt)",
    )
    parser.add_argument(
        "--assets",
        default="assets",
        help="Assets directory for compressed files (default: assets)",
    )
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help="Keep extracted NeMo files after conversion",
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Skip automatic quantization step",
    )
    args = parser.parse_args()

    # Setup directories
    if args.cache:
        cache_dir = pathlib.Path(args.cache)
    else:
        project_root = pathlib.Path(__file__).parent.parent
        cache_dir = project_root / ".cache" / "parakeet-tdt"

    assets_dir = pathlib.Path(args.assets)
    cache_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    print("Parakeet TDT Model Download & Conversion")
    print("=" * 50)
    print(f"Model:  nvidia/parakeet-tdt-0.6b-v3")
    print(f"Cache:  {cache_dir}")
    print(f"Assets: {assets_dir}\n")

    # Download .nemo file
    print("Downloading from Hugging Face...")
    nemo_path = hf_hub_download(
        repo_id="nvidia/parakeet-tdt-0.6b-v3",
        filename="parakeet-tdt-0.6b-v3.nemo",
        cache_dir=str(cache_dir / "huggingface"),
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )
    nemo_path = pathlib.Path(nemo_path)
    size_gb = nemo_path.stat().st_size / (1024 * 1024 * 1024)
    print(f"✓ Downloaded: {nemo_path.name} ({size_gb:.2f} GB)\n")

    # Extract NeMo file
    extract_dir = cache_dir / "extracted"
    files = extract_nemo_file(nemo_path, extract_dir)

    if not files['config'] or not files['weights']:
        print("✗ Error: Could not find config or weights in .nemo file")
        return 1

    # Convert config
    print("\nConverting config...")
    config = convert_nemo_config_to_json(files['config'])
    config_path = cache_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved: {config_path}")

    # Convert weights
    safetensors_path = cache_dir / "model.safetensors"
    convert_weights_to_safetensors(files['weights'], safetensors_path)

    # Copy tokenizer if found and convert to JSON format
    if files['tokenizer']:
        tokenizer_model_path = cache_dir / "tokenizer.model"
        shutil.copy2(files['tokenizer'], tokenizer_model_path)
        print(f"✓ Tokenizer copied: {tokenizer_model_path}")

        # Convert to tokenizer.json for easier loading in Rust
        try:
            from tokenizers import Tokenizer, models as tokenizer_models
            from sentencepiece import SentencePieceProcessor

            sp = SentencePieceProcessor()
            sp.load(str(tokenizer_model_path))

            # Get vocabulary
            vocab = {}
            for i in range(sp.vocab_size()):
                piece = sp.id_to_piece(i)
                score = sp.get_score(i)
                vocab[piece] = score

            # Create Unigram tokenizer
            tokenizer = Tokenizer(tokenizer_models.Unigram(list(vocab.items())))

            # Add decoder to handle SentencePiece special characters
            from tokenizers import decoders
            tokenizer.decoder = decoders.Metaspace(replacement="▁")

            # Save as JSON
            tokenizer_json_path = cache_dir / "tokenizer.json"
            tokenizer.save(str(tokenizer_json_path))
            print(f"✓ Created tokenizer.json with {len(vocab)} tokens")
        except ImportError:
            print("⚠ Warning: sentencepiece or tokenizers not installed, skipping tokenizer.json conversion")
            print("  The model will still work but may have issues loading the tokenizer")
            print("  Install with: pip install sentencepiece tokenizers")

    # Compress files
    print("\nCompressing files with zstd...")
    compressed_files = []

    # Compress config
    config_zst = compress_file(config_path, "parakeet-tdt-config.json.zst")
    compressed_files.append(config_zst)

    # Compress weights
    weights_zst = compress_file(safetensors_path, "parakeet-tdt-model.safetensors.zst")
    compressed_files.append(weights_zst)

    # Compress tokenizers if they exist
    if files['tokenizer']:
        # Compress tokenizer.model
        tokenizer_model_zst = compress_file(tokenizer_model_path, "parakeet-tdt-tokenizer.model.zst")
        compressed_files.append(tokenizer_model_zst)

        # Compress tokenizer.json if it was created
        if (cache_dir / "tokenizer.json").exists():
            tokenizer_json_zst = compress_file(cache_dir / "tokenizer.json", "parakeet-tdt-tokenizer.json.zst")
            compressed_files.append(tokenizer_json_zst)

    # Quantize model to GGUF format
    if not args.skip_quantize:
        gguf_file = quantize_tdt_model(cache_dir, safetensors_path)
        if gguf_file:
            compressed_files.append(gguf_file)

    # Copy to assets
    print(f"\nCopying to assets directory...")
    for src_file in compressed_files:
        dst_file = assets_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        print(f"  ✓ {src_file.name}")

    # Cleanup
    if not args.keep_extracted:
        print("\nCleaning up extracted files...")
        shutil.rmtree(extract_dir)

    # Summary
    print("\n" + "=" * 50)
    print("✓ Conversion complete!")
    print(f"\nCache: {cache_dir}")
    print(f"Assets: {assets_dir}")
    print("\nAssets contents:")
    for p in sorted(assets_dir.glob("parakeet-tdt-*.zst")):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  - {p.name} ({size_mb:.1f} MB)")

    print("\nYou can now run:")
    print("  cargo run --example transcribe_tdt_with_vad --release -- audio.wav")

    return 0


if __name__ == "__main__":
    exit(main())
