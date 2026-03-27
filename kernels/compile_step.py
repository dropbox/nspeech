#!/usr/bin/env python3
"""Per-file compilation step for ninja.

Usage:
    python compile_step.py msl_apple  INPUT.ttir  OUTPUT.metal
    python compile_step.py msl_intel  INPUT.ttir  OUTPUT.metal
    python compile_step.py metallib   INPUT.metal OUTPUT.metallib
    python compile_step.py hlsl       INPUT.ttir  OUTPUT.hlsl
"""
import json
import os
import sys
from pathlib import Path

TRITON_METAL_DIR = Path(__file__).resolve().parent.parent.parent / "triton" / "third_party" / "metal"
sys.path.insert(0, str(TRITON_METAL_DIR))


def main():
    cmd, inp, out = sys.argv[1], sys.argv[2], sys.argv[3]

    if cmd == "msl_apple":
        os.environ["TRITON_METAL_SIMDGROUP"] = "1"
        from backend.codegen import ttir_to_msl_with_metadata
        msl, _, _, _ = ttir_to_msl_with_metadata(
            Path(inp).read_text(), block_size=256, use_simdgroup=True)
        Path(out).write_text(msl)

    elif cmd == "msl_intel":
        os.environ["TRITON_METAL_SIMDGROUP"] = "0"
        from backend.codegen import ttir_to_msl_with_metadata
        msl, _, _, _ = ttir_to_msl_with_metadata(
            Path(inp).read_text(), block_size=256, use_simdgroup=False)
        Path(out).write_text(msl)

    elif cmd == "metallib":
        from backend.compiler import _compile_msl_runtime
        data = _compile_msl_runtime(Path(inp).read_text())
        if data:
            Path(out).write_bytes(data)
        else:
            sys.exit(1)

    elif cmd == "hlsl":
        os.environ["TRITON_METAL_SIMDGROUP"] = "0"
        from backend.codegen import ttir_to_hlsl_with_metadata
        # Check metadata for force_acc_fp16
        meta_path = Path(inp).with_suffix(".json")
        force_fp16 = False
        if meta_path.exists():
            force_fp16 = json.loads(meta_path.read_text()).get("force_acc_fp16", False)
        hlsl, name, _, threads, half4_args = ttir_to_hlsl_with_metadata(
            Path(inp).read_text(), block_size=256, force_acc_fp16=force_fp16)
        Path(out).write_text(hlsl)

    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
