#!/usr/bin/env python3
"""Per-file compilation step for ninja.

Usage:
    python compile_step.py msl_metal         INPUT.ttir  OUTPUT.metal
    python compile_step.py msl_metal_nosimd  INPUT.ttir  OUTPUT.metal
    python compile_step.py air_metal         INPUT.ttir  OUTPUT.ll
    python compile_step.py hlsl       INPUT.ttir  OUTPUT.hlsl

Metallib compilation is handled directly by xcrun in build.ninja:
    .metal -> xcrun metal -> .metallib
    .ll    -> xcrun metal-as -> .air -> xcrun metallib -> .metallib
"""
import json
import sys
from pathlib import Path

TRITON_METAL_DIR = Path(__file__).resolve().parent.parent.parent / "triton" / "third_party" / "metal"
sys.path.insert(0, str(TRITON_METAL_DIR))


def write_if_changed(path: Path, content: str) -> bool:
    """Write only if content differs, preserving mtime for ninja."""
    if path.exists():
        try:
            if path.read_text() == content:
                return False
        except Exception:
            pass
    path.write_text(content)
    return True


def main():
    cmd, inp, out = sys.argv[1], sys.argv[2], sys.argv[3]

    if cmd == "msl_metal":
        from backend.codegen import ttir_to_msl_with_metadata
        msl, _, _, _ = ttir_to_msl_with_metadata(
            Path(inp).read_text(), block_size=256, use_simdgroup=True)
        write_if_changed(Path(out), msl)

    elif cmd == "msl_metal_nosimd":
        from backend.codegen import ttir_to_msl_with_metadata
        msl, _, _, _ = ttir_to_msl_with_metadata(
            Path(inp).read_text(), block_size=256, use_simdgroup=False)
        write_if_changed(Path(out), msl)

    elif cmd == "air_metal":
        from backend.codegen.air_emitter import ttgir_to_air
        ir_text, kname, tg_mem, bs = ttgir_to_air(Path(inp).read_text(), block_size=256)
        write_if_changed(Path(out), ir_text)
        # Write sidecar with AIR-specific metadata
        meta_out = Path(out).with_suffix(".json")
        write_if_changed(meta_out, json.dumps({
            "kernel_name": kname,
            "block_size": bs,
            "tg_mem_bytes": tg_mem,
        }))

    elif cmd == "hlsl":
        from backend.codegen import ttir_to_hlsl_with_metadata
        meta_path = Path(inp).with_suffix(".json")
        force_fp16 = False
        if meta_path.exists():
            force_fp16 = json.loads(meta_path.read_text()).get("force_acc_fp16", False)
        hlsl, name, _, threads, half4_args = ttir_to_hlsl_with_metadata(
            Path(inp).read_text(), block_size=256, force_acc_fp16=force_fp16)
        write_if_changed(Path(out), hlsl)

    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
