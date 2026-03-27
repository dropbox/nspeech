#!/usr/bin/env python3
"""Build all Moonshine Triton kernels.

Generates TTIR from @triton.jit, writes build.ninja, runs ninja.
Output: out/{apple,intel}/*.metallib, out/hlsl/*.hlsl

    python build.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "out"
TRITON = SCRIPT_DIR.parent.parent / "triton"
TRITON_METAL = TRITON / "third_party" / "metal"
PYTHON = str(TRITON / "env" / "bin" / "python")
NINJA = str(TRITON / "env" / "bin" / "ninja")
COMPILE_STEP = str(SCRIPT_DIR / "compile_step.py")


def gen_ttir():
    """Generate TTIR for every kernel config (one Python process)."""
    sys.path.insert(0, str(TRITON_METAL))
    sys.path.insert(0, str(SCRIPT_DIR))
    os.environ.setdefault("TRITON_METAL_SIMDGROUP", "0")

    from aot_compile import compile_kernel
    import moonshine_kernels as K
    from kernel_configs import METAL_KERNELS, HLSL_EXTRA_KERNELS

    ttir_dir = OUT / "ttir"
    ttir_dir.mkdir(parents=True, exist_ok=True)

    # Collect all unique configs by name
    configs = {}
    for cfg in METAL_KERNELS:
        configs[cfg[0]] = cfg
    for cfg in HLSL_EXTRA_KERNELS:
        configs.setdefault(cfg[0], cfg)

    print(f"Generating TTIR for {len(configs)} kernels...")
    t0 = time.time()
    ok = 0

    for name, cfg in sorted(configs.items()):
        func_name, sig, nw, grid = cfg[1], cfg[2], cfg[3], cfg[4]
        opts = cfg[5] if len(cfg) > 5 else {}
        fn = getattr(K, func_name, None)
        if fn is None:
            print(f"  {name}: SKIP (no {func_name})")
            continue
        try:
            r = compile_kernel(fn=fn, signature=sig, num_warps=nw, grid=grid)
            ir = r.ttgir_text or r.ttir_text
            (ttir_dir / f"{name}.ttir").write_text(ir)
            (ttir_dir / f"{name}.json").write_text(json.dumps({
                "kernel_name": r.kernel_name,
                "params": r.params,
                "constants": r.constants,
                "threadgroup_size": r.threadgroup_size,
                "grid": grid,
                "force_acc_fp16": opts.get("force_acc_fp16", False),
            }, indent=2))
            ok += 1
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    print(f"TTIR: {ok}/{len(configs)} in {time.time()-t0:.1f}s\n")


def gen_ninja():
    """Write build.ninja for TTIR → MSL/metallib/HLSL."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from kernel_configs import METAL_KERNELS, HLSL_EXTRA_KERNELS, get_hlsl_kernels

    ttir = OUT / "ttir"
    apple = OUT / "apple"
    intel = OUT / "intel"
    hlsl = OUT / "hlsl"
    for d in [apple, intel, hlsl]:
        d.mkdir(parents=True, exist_ok=True)

    w = []
    w.append("# Auto-generated — do not edit")
    w.append(f"python = {PYTHON}")
    w.append(f"step = {COMPILE_STEP}")
    w.append("")
    w.append("rule msl_apple\n  command = $python $step msl_apple $in $out\n  description = MSL(apple) $out")
    w.append("rule msl_intel\n  command = $python $step msl_intel $in $out\n  description = MSL(intel) $out")
    w.append("rule metallib_apple\n  command = xcrun metal -std=metal3.1 -O3 -ffast-math -w -o $out $in\n  description = METALLIB(apple) $out")
    w.append("rule metallib_intel\n  command = xcrun metal -std=macos-metal2.4 -mmacosx-version-min=14.0 -ffast-math -w -o $out $in\n  description = METALLIB(intel) $out")
    w.append("rule hlsl\n  command = $python $step hlsl $in $out\n  description = HLSL $out")
    w.append("")

    apple_libs, intel_libs, hlsl_files = [], [], []

    for cfg in METAL_KERNELS:
        name = cfg[0]
        t = ttir / f"{name}.ttir"
        if not t.exists():
            continue
        am = apple / f"{name}.metal"
        al = apple / f"{name}.metallib"
        im = intel / f"{name}.metal"
        il = intel / f"{name}.metallib"
        w.append(f"build {am}: msl_apple {t}")
        w.append(f"build {al}: metallib_apple {am}")
        w.append(f"build {im}: msl_intel {t}")
        w.append(f"build {il}: metallib_intel {im}")
        apple_libs.append(str(al))
        intel_libs.append(str(il))

    hlsl_seen = set()
    for cfg in get_hlsl_kernels():
        name = cfg[0]
        if name in hlsl_seen:
            continue
        hlsl_seen.add(name)
        t = ttir / f"{name}.ttir"
        if not t.exists():
            continue
        h = hlsl / f"{name}.hlsl"
        w.append(f"build {h}: hlsl {t}")
        hlsl_files.append(str(h))

    w.append("")
    w.append(f"build apple: phony {' '.join(apple_libs)}")
    w.append(f"build intel: phony {' '.join(intel_libs)}")
    w.append(f"build hlsl_all: phony {' '.join(hlsl_files)}")
    w.append("default apple intel hlsl_all")
    w.append("")

    (OUT / "build.ninja").write_text("\n".join(w))
    print(f"build.ninja: {len(apple_libs)} apple, {len(intel_libs)} intel, {len(hlsl_files)} hlsl")


def run_ninja():
    ninja = NINJA if Path(NINJA).exists() else "ninja"
    t0 = time.time()
    r = subprocess.run([ninja, "-f", str(OUT / "build.ninja")])
    print(f"ninja: {time.time()-t0:.1f}s (exit={r.returncode})")
    return r.returncode == 0


if __name__ == "__main__":
    gen_ttir()
    gen_ninja()
    if not run_ninja():
        sys.exit(1)
