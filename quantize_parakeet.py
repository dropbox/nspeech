#!/usr/bin/env python3
"""
Quantize Parakeet CTC model weights to GGUF format for Candle.

Supports Q8_0 (8-bit) and Q4_0 (4-bit) quantization formats.
"""
import numpy as np
import safetensors.torch
import json
import struct
from pathlib import Path
from typing import Dict, Tuple
import argparse


# GGML Quantization formats (compatible with llama.cpp/GGUF)
class QuantType:
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9


def quantize_q8_0(tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize to Q8_0 format: 8-bit integers with per-block scaling.
    Block size: 32 elements per block.

    Format: For each block of 32 values:
        - 1 float16 scale factor
        - 32 int8 quantized values

    Returns: (scales, qweights) where scales is float16 and qweights is int8
    """
    # Flatten if multi-dimensional
    orig_shape = tensor.shape
    tensor = tensor.flatten()

    block_size = 32
    n_blocks = (len(tensor) + block_size - 1) // block_size

    # Pad to multiple of block_size
    pad_size = n_blocks * block_size - len(tensor)
    if pad_size > 0:
        tensor = np.concatenate([tensor, np.zeros(pad_size, dtype=tensor.dtype)])

    # Reshape into blocks
    blocks = tensor.reshape(n_blocks, block_size)

    # Compute scale per block (max absolute value)
    scales = np.abs(blocks).max(axis=1, keepdims=True)
    # Avoid division by zero
    scales = np.where(scales == 0, 1.0, scales)

    # Quantize: scale to [-127, 127] range
    qweights = np.clip(np.round(blocks / scales * 127.0), -127, 127).astype(np.int8)

    # Convert scales to float16 and squeeze
    scales = scales.squeeze().astype(np.float16)

    return scales, qweights, orig_shape


def quantize_q4_0(tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize to Q4_0 format: 4-bit integers with per-block scaling.
    Block size: 32 elements per block.

    Format: For each block of 32 values:
        - 1 float16 scale factor
        - 32 values packed into 16 bytes (2 values per byte)

    Returns: (scales, qweights) where scales is float16 and qweights is uint8 (packed)
    """
    orig_shape = tensor.shape
    tensor = tensor.flatten()

    block_size = 32
    n_blocks = (len(tensor) + block_size - 1) // block_size

    # Pad to multiple of block_size
    pad_size = n_blocks * block_size - len(tensor)
    if pad_size > 0:
        tensor = np.concatenate([tensor, np.zeros(pad_size, dtype=tensor.dtype)])

    blocks = tensor.reshape(n_blocks, block_size)

    # Compute scale per block
    scales = np.abs(blocks).max(axis=1, keepdims=True)
    scales = np.where(scales == 0, 1.0, scales)

    # Quantize to 4-bit range [-8, 7]
    qweights = np.clip(np.round(blocks / scales * 7.0), -8, 7).astype(np.int8)

    # Pack two 4-bit values into one byte
    # Reshape to (n_blocks, 16, 2) to pair values
    qweights_reshaped = qweights.reshape(n_blocks, 16, 2)

    # Pack: low nibble = first value + 8, high nibble = second value + 8
    # (Add 8 to convert from signed to unsigned 4-bit range)
    packed = ((qweights_reshaped[:, :, 0] + 8) & 0xF) | (((qweights_reshaped[:, :, 1] + 8) & 0xF) << 4)
    qweights_packed = packed.astype(np.uint8)

    scales = scales.squeeze().astype(np.float16)

    return scales, qweights_packed, orig_shape


def quantize_weight(name: str, tensor: np.ndarray, quant_type: int) -> Dict:
    """Quantize a single weight tensor."""
    if quant_type == QuantType.F32:
        return {
            'name': name,
            'type': quant_type,
            'shape': tensor.shape,
            'data': tensor.astype(np.float32)
        }
    elif quant_type == QuantType.Q8_0:
        scales, qweights, shape = quantize_q8_0(tensor)
        return {
            'name': name,
            'type': quant_type,
            'shape': shape,
            'scales': scales,
            'qweights': qweights
        }
    elif quant_type == QuantType.Q4_0:
        scales, qweights_packed, shape = quantize_q4_0(tensor)
        return {
            'name': name,
            'type': quant_type,
            'shape': shape,
            'scales': scales,
            'qweights': qweights_packed
        }
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}")


def should_quantize_layer(name: str) -> bool:
    """Determine if a layer should be quantized.

    Generally quantize:
    - Linear layer weights (large matrices)
    - Convolution weights

    Don't quantize:
    - Biases (small, sensitive)
    - LayerNorm weights (small, sensitive)
    - BatchNorm parameters
    - Embedding lookups
    """
    # Don't quantize biases, norms, or small tensors
    if any(x in name for x in ['bias', 'norm', 'bn', 'ln', 'layernorm', 'batchnorm']):
        return False

    # Quantize linear and conv weights
    if any(x in name for x in ['weight', 'linear', 'conv', 'proj']):
        return True

    return False


def quantize_model(input_path: str, output_prefix: str, quant_type: int):
    """Quantize a safetensors model."""
    print(f"Loading model from {input_path}...")
    weights = safetensors.torch.load_file(input_path)

    print(f"Model has {len(weights)} tensors")

    # Convert to numpy
    weights_np = {k: v.numpy() for k, v in weights.items()}

    # Quantize
    quantized = {}
    total_size_fp32 = 0
    total_size_quant = 0
    n_quantized = 0
    n_kept_fp32 = 0

    print(f"\nQuantizing to {'Q8_0' if quant_type == QuantType.Q8_0 else 'Q4_0'}...")

    for name, tensor in weights_np.items():
        total_size_fp32 += tensor.nbytes

        if should_quantize_layer(name) and tensor.size >= 1024:  # Only quantize larger tensors
            q_tensor = quantize_weight(name, tensor, quant_type)
            quantized[name] = q_tensor

            # Calculate quantized size
            if quant_type == QuantType.Q8_0:
                # scales: float16 per 32 elements + int8 per element
                n_blocks = (tensor.size + 31) // 32
                size = n_blocks * 2 + n_blocks * 32  # scales + qweights
            else:  # Q4_0
                n_blocks = (tensor.size + 31) // 32
                size = n_blocks * 2 + n_blocks * 16  # scales + packed qweights

            total_size_quant += size
            n_quantized += 1
            print(f"  Quantized: {name:60s} {tensor.shape} -> {size/1024/1024:.2f} MB")
        else:
            # Keep as FP32
            q_tensor = quantize_weight(name, tensor, QuantType.F32)
            quantized[name] = q_tensor
            total_size_quant += tensor.nbytes
            n_kept_fp32 += 1
            print(f"  Kept FP32: {name:60s} {tensor.shape}")

    print(f"\nQuantization summary:")
    print(f"  Quantized layers: {n_quantized}")
    print(f"  FP32 layers: {n_kept_fp32}")
    print(f"  Original size: {total_size_fp32 / 1024 / 1024:.2f} MB")
    print(f"  Quantized size: {total_size_quant / 1024 / 1024:.2f} MB")
    print(f"  Compression ratio: {total_size_fp32 / total_size_quant:.2f}x")

    # Save as npz for easy loading in Rust (via numpy or direct binary reading)
    output_file = f"{output_prefix}_{'q8_0' if quant_type == QuantType.Q8_0 else 'q4_0'}.npz"
    print(f"\nSaving to {output_file}...")

    # Prepare data for npz
    save_dict = {}
    for name, q_tensor in quantized.items():
        if q_tensor['type'] == QuantType.F32:
            save_dict[f"{name}||data"] = q_tensor['data']
            save_dict[f"{name}||type"] = np.array([QuantType.F32], dtype=np.int32)
            save_dict[f"{name}||shape"] = np.array(q_tensor['shape'], dtype=np.int64)
        else:
            save_dict[f"{name}||type"] = np.array([q_tensor['type']], dtype=np.int32)
            save_dict[f"{name}||shape"] = np.array(q_tensor['shape'], dtype=np.int64)
            save_dict[f"{name}||scales"] = q_tensor['scales']
            save_dict[f"{name}||qweights"] = q_tensor['qweights']

    np.savez_compressed(output_file, **save_dict)
    print(f"Saved quantized model to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Quantize Parakeet CTC model')
    parser.add_argument('--input', default='hf_parakeet/model.safetensors',
                        help='Input safetensors file')
    parser.add_argument('--output', default='hf_parakeet/model',
                        help='Output prefix (will add _q8_0.npz or _q4_0.npz)')
    parser.add_argument('--quant', choices=['q8_0', 'q4_0', 'both'], default='both',
                        help='Quantization type')

    args = parser.parse_args()

    if args.quant in ['q8_0', 'both']:
        print("=" * 80)
        print("Quantizing to Q8_0 (8-bit)")
        print("=" * 80)
        quantize_model(args.input, args.output, QuantType.Q8_0)

    if args.quant in ['q4_0', 'both']:
        print("\n" + "=" * 80)
        print("Quantizing to Q4_0 (4-bit)")
        print("=" * 80)
        quantize_model(args.input, args.output, QuantType.Q4_0)

    print("\nDone!")


if __name__ == '__main__':
    main()
