#!/bin/bash
# Download and prepare Qwen3-0.6B-Instruct model for text correction
#
# This script downloads the quantized Qwen3-0.6B-Instruct Q4_K_M model
# and prepares it for use with the Parakeet speech recognition pipeline.

set -e

ASSETS_DIR="${1:-assets}"
TEMP_DIR=$(mktemp -d)

echo "Downloading Qwen3-0.6B-Instruct Q4_K_M model..."
echo "Assets directory: $ASSETS_DIR"
echo

# Create assets directory if it doesn't exist
mkdir -p "$ASSETS_DIR"

# Download from Hugging Face (bartowski's GGUF collection)
MODEL_REPO="bartowski/Qwen_Qwen3-0.6B-GGUF"
MODEL_FILE="Qwen_Qwen3-0.6B-Q4_K_M.gguf"

echo "Downloading model file (~400MB)..."
curl -L "https://huggingface.co/$MODEL_REPO/resolve/main/$MODEL_FILE" \
  -o "$TEMP_DIR/$MODEL_FILE"

echo "Downloading tokenizer..."
curl -L "https://huggingface.co/Qwen/Qwen3-0.6B-Instruct/resolve/main/tokenizer.json" \
  -o "$TEMP_DIR/tokenizer.json"

echo "Downloading config..."
curl -L "https://huggingface.co/Qwen/Qwen3-0.6B-Instruct/resolve/main/config.json" \
  -o "$TEMP_DIR/config.json"

echo
echo "Compressing files with zstd..."

# Compress with zstd level 19 (matches Parakeet assets)
zstd -19 "$TEMP_DIR/$MODEL_FILE" -o "$ASSETS_DIR/qwen3-0.6b-instruct-q4_k_m.gguf.zst" --force
zstd -19 "$TEMP_DIR/tokenizer.json" -o "$ASSETS_DIR/qwen3-0.6b-instruct-tokenizer.json.zst" --force
zstd -19 "$TEMP_DIR/config.json" -o "$ASSETS_DIR/qwen3-0.6b-instruct-config.json.zst" --force

echo
echo "Cleaning up..."
rm -rf "$TEMP_DIR"

echo
echo "✓ Qwen3 model files prepared successfully!"
echo
echo "Files created in $ASSETS_DIR:"
ls -lh "$ASSETS_DIR"/qwen3-0.6b-instruct-*.zst

echo
echo "You can now use Qwen3 text correction in your transcriptions."
echo "See src/qwen.rs and QWEN_INTEGRATION.md for usage examples."
