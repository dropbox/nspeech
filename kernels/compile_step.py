#!/usr/bin/env python3
"""Per-file compilation step for ninja.

Usage:
    python compile_step.py msl_apple  INPUT.ttir  OUTPUT.metal
    python compile_step.py msl_intel  INPUT.ttir  OUTPUT.metal
    python compile_step.py metallib_apple  INPUT.metal OUTPUT.metallib
    python compile_step.py metallib_intel  INPUT.metal OUTPUT.metallib
    python compile_step.py hlsl       INPUT.ttir  OUTPUT.hlsl
"""
import json
import os
import sys
from pathlib import Path

TRITON_METAL_DIR = Path(__file__).resolve().parent.parent.parent / "triton" / "third_party" / "metal"
sys.path.insert(0, str(TRITON_METAL_DIR))


def compile_metallib(msl_source, language_version, patch_deploy_target=None):
    """Compile MSL to metallib bytes using Metal runtime with specified language version.

    If patch_deploy_target is set, binary-patches the metallib VERS deployment target
    to the given value. This is needed because the Metal runtime always embeds the
    host SDK's deployment target, which may be too new for older GPUs (e.g. Intel).
    """
    import Metal
    from Foundation import NSURL
    import struct, tempfile

    device = Metal.MTLCreateSystemDefaultDevice()
    options = Metal.MTLCompileOptions.alloc().init()
    options.setFastMathEnabled_(True)
    options.setLanguageVersion_(language_version)

    library, error = device.newLibraryWithSource_options_error_(msl_source, options, None)
    if error is not None:
        print(f"Metal compile error: {error}", file=sys.stderr)
        return None

    with tempfile.NamedTemporaryFile(suffix='.metallib', delete=True) as tmp:
        url = NSURL.fileURLWithPath_(tmp.name)
        _, error = library.serializeToURL_error_(url, None)
        if error is not None:
            print(f"Metal serialize error: {error}", file=sys.stderr)
            return None
        data = bytearray(Path(tmp.name).read_bytes())

    if patch_deploy_target is not None:
        # Find VERS block and patch the deployment target field
        vers_tag = b'VERS'
        idx = data.find(vers_tag)
        if idx >= 0:
            # Layout: VERS <1-byte len> <4-byte deploy_target BE> <4-byte lang_version BE>
            dt_offset = idx + len(vers_tag) + 1
            struct.pack_into('>I', data, dt_offset, patch_deploy_target)

    return bytes(data)


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

    elif cmd == "metallib_apple":
        import Metal
        data = compile_metallib(Path(inp).read_text(), Metal.MTLLanguageVersion3_1)
        if data:
            Path(out).write_bytes(data)
        else:
            sys.exit(1)

    elif cmd == "metallib_intel":
        import Metal
        # Patch deployment target to Metal 2.4 so Intel GPUs can load the metallib
        data = compile_metallib(Path(inp).read_text(), Metal.MTLLanguageVersion2_4,
                                patch_deploy_target=Metal.MTLLanguageVersion2_4)
        if data:
            Path(out).write_bytes(data)
        else:
            sys.exit(1)

    elif cmd == "hlsl":
        os.environ["TRITON_METAL_SIMDGROUP"] = "0"
        from backend.codegen import ttir_to_hlsl_with_metadata
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
