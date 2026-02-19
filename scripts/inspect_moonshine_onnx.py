#!/usr/bin/env python3
"""Phase 2a: Inspect Moonshine V2 ONNX models to understand architecture.

Prints tensor names, shapes, dtypes, and graph structure for all 5 components.
"""

import os
import sys
import json
import onnx
import numpy as np
from collections import defaultdict

MODEL_DIR = "/Users/jhansen/Library/Caches/moonshine_voice/download.moonshine.ai/model/medium-streaming-en/quantized"

COMPONENTS = ["frontend", "encoder", "adapter", "cross_kv", "decoder_kv"]


def inspect_model(name: str, path: str):
    print(f"\n{'='*80}")
    print(f"Component: {name}")
    print(f"File: {path} ({os.path.getsize(path)/1024/1024:.1f} MB)")
    print(f"{'='*80}")

    model = onnx.load(path)
    graph = model.graph

    # Print inputs
    print(f"\n  INPUTS ({len(graph.input)}):")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        print(f"    {inp.name}: {shape} ({dtype})")

    # Print outputs
    print(f"\n  OUTPUTS ({len(graph.output)}):")
    for out in graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
        print(f"    {out.name}: {shape} ({dtype})")

    # Print initializers (weights)
    print(f"\n  INITIALIZERS (weights) ({len(graph.initializer)}):")
    total_params = 0
    weight_info = []
    for init in graph.initializer:
        shape = list(init.dims)
        dtype = onnx.TensorProto.DataType.Name(init.data_type)
        num_params = 1
        for d in shape:
            num_params *= d
        total_params += num_params
        weight_info.append((init.name, shape, dtype, num_params))

    # Sort by name for readability
    for wname, shape, dtype, num_params in sorted(weight_info):
        print(f"    {wname}: {shape} ({dtype}, {num_params:,} params)")

    print(f"\n  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Print operations summary
    op_counts = defaultdict(int)
    for node in graph.node:
        op_counts[node.op_type] += 1

    print(f"\n  OPERATIONS ({len(graph.node)} nodes):")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"    {op}: {count}")

    # Print full graph (nodes) for small models, summary for large ones
    if len(graph.node) <= 200:
        print(f"\n  GRAPH NODES:")
        for i, node in enumerate(graph.node):
            inputs = ", ".join(node.input[:4])
            if len(node.input) > 4:
                inputs += f", ... ({len(node.input)} total)"
            outputs = ", ".join(node.output)
            attrs = ""
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.INT:
                    attrs += f" {attr.name}={attr.i}"
                elif attr.type == onnx.AttributeProto.FLOAT:
                    attrs += f" {attr.name}={attr.f:.4f}"
                elif attr.type == onnx.AttributeProto.INTS:
                    attrs += f" {attr.name}={list(attr.ints)}"
                elif attr.type == onnx.AttributeProto.STRING:
                    attrs += f" {attr.name}={attr.s.decode()}"
            print(f"    [{i:3d}] {node.op_type}({inputs}) -> {outputs}{attrs}")
    else:
        # For large models, print first 30 and last 10
        print(f"\n  GRAPH NODES (first 30 of {len(graph.node)}):")
        for i, node in enumerate(graph.node[:30]):
            inputs = ", ".join(node.input[:4])
            if len(node.input) > 4:
                inputs += f", ... ({len(node.input)} total)"
            outputs = ", ".join(node.output)
            attrs = ""
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.INT:
                    attrs += f" {attr.name}={attr.i}"
                elif attr.type == onnx.AttributeProto.FLOAT:
                    attrs += f" {attr.name}={attr.f:.4f}"
                elif attr.type == onnx.AttributeProto.INTS:
                    attrs += f" {attr.name}={list(attr.ints)}"
                elif attr.type == onnx.AttributeProto.STRING:
                    attrs += f" {attr.name}={attr.s.decode()}"
            print(f"    [{i:3d}] {node.op_type}({inputs}) -> {outputs}{attrs}")

        print(f"    ... ({len(graph.node) - 40} more nodes) ...")

        print(f"\n  GRAPH NODES (last 10):")
        for i in range(max(30, len(graph.node)-10), len(graph.node)):
            node = graph.node[i]
            inputs = ", ".join(node.input[:4])
            if len(node.input) > 4:
                inputs += f", ... ({len(node.input)} total)"
            outputs = ", ".join(node.output)
            attrs = ""
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.INT:
                    attrs += f" {attr.name}={attr.i}"
                elif attr.type == onnx.AttributeProto.FLOAT:
                    attrs += f" {attr.name}={attr.f:.4f}"
                elif attr.type == onnx.AttributeProto.INTS:
                    attrs += f" {attr.name}={list(attr.ints)}"
                elif attr.type == onnx.AttributeProto.STRING:
                    attrs += f" {attr.name}={attr.s.decode()}"
            print(f"    [{i:3d}] {node.op_type}({inputs}) -> {outputs}{attrs}")

    return total_params


def main():
    print("Moonshine V2 Medium Streaming EN - ONNX Model Inspection")
    print(f"Model directory: {MODEL_DIR}")

    # Print streaming config
    config_path = os.path.join(MODEL_DIR, "streaming_config.json")
    with open(config_path) as f:
        config = json.load(f)
    print(f"\nStreaming Config:")
    print(json.dumps(config, indent=2))

    grand_total = 0
    for component in COMPONENTS:
        path = os.path.join(MODEL_DIR, f"{component}.ort")
        if os.path.exists(path):
            total = inspect_model(component, path)
            grand_total += total
        else:
            print(f"\n  WARNING: {path} not found!")

    print(f"\n{'='*80}")
    print(f"GRAND TOTAL: {grand_total:,} parameters ({grand_total/1e6:.1f}M)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
