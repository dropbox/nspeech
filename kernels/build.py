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


def write_if_changed(path: Path, content: str) -> bool:
    """Write content to path only if it differs from existing content.

    Returns True if the file was written (content changed or new file).
    Avoids touching mtime when content is unchanged, so ninja/cargo
    won't trigger unnecessary downstream rebuilds.
    """
    if path.exists():
        try:
            if path.read_text() == content:
                return False
        except Exception:
            pass
    path.write_text(content)
    return True

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
            write_if_changed(ttir_dir / f"{name}.ttir", ir)
            write_if_changed(ttir_dir / f"{name}.json", json.dumps({
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
    from kernel_configs import METAL_KERNELS, HLSL_EXTRA_KERNELS, get_hlsl_kernels, KERNEL_METADATA

    ttir = OUT / "ttir"
    apple = OUT / "apple"
    intel = OUT / "intel"
    hlsl = OUT / "hlsl"
    for d in [apple, intel, hlsl]:
        d.mkdir(parents=True, exist_ok=True)

    # Compiler source files — ninja rebuilds when these change
    codegen_dir = TRITON_METAL / "backend" / "codegen"
    compiler_deps = " ".join(str(p) for p in sorted(codegen_dir.glob("*.py")))
    implicit = f"| {compiler_deps} {COMPILE_STEP}"

    w = []
    w.append("# Auto-generated — do not edit")
    w.append(f"python = {PYTHON}")
    w.append(f"step = {COMPILE_STEP}")
    w.append("")
    w.append("rule msl_apple\n  command = $python $step msl_apple $in $out\n  restat = 1\n  description = MSL(apple) $out")
    w.append("rule msl_intel\n  command = $python $step msl_intel $in $out\n  restat = 1\n  description = MSL(intel) $out")
    w.append("rule metallib_apple\n  command = xcrun metal -std=metal3.1 -O3 -ffast-math -w -o $out $in\n  description = METALLIB(apple) $out")
    w.append("rule metallib_intel\n  command = xcrun metal -std=macos-metal2.4 -mmacosx-version-min=14.0 -O3 -ffast-math -w -o $out $in\n  description = METALLIB(intel) $out")
    # AIR disabled for now — re-enable when emitter covers all ops
    # w.append("rule air_emit\n  command = $python $step air_apple $in $out\n  restat = 1\n  description = AIR(emit) $out")
    # w.append("rule air_asm\n  command = xcrun metal-as -o $out $in\n  description = AIR(asm) $out")
    # w.append("rule air_link\n  command = xcrun metallib -o $out $in\n  description = AIR(link) $out")
    w.append("rule hlsl\n  command = $python $step hlsl $in $out\n  restat = 1\n  description = HLSL $out")

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
        w.append(f"build {am}: msl_apple {t} {implicit}")
        w.append(f"build {al}: metallib_apple {am}")
        w.append(f"build {im}: msl_intel {t} {implicit}")
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
        w.append(f"build {h}: hlsl {t} {implicit}")
        hlsl_files.append(str(h))

    w.append("")
    w.append(f"build apple: phony {' '.join(apple_libs)}")
    w.append(f"build intel: phony {' '.join(intel_libs)}")
    w.append(f"build hlsl_all: phony {' '.join(hlsl_files)}")
    # DXIL compilation happens after ninja, via compile_dxil()
    w.append(f"default apple intel hlsl_all")
    w.append("")

    write_if_changed(OUT / "build.ninja", "\n".join(w))
    print(f"build.ninja: {len(apple_libs)} apple, {len(intel_libs)} intel, {len(hlsl_files)} hlsl")


def run_ninja():
    ninja = NINJA if Path(NINJA).exists() else "ninja"
    t0 = time.time()
    r = subprocess.run([ninja, "-f", str(OUT / "build.ninja")])
    dt = time.time() - t0
    # Critical: all apple + intel metallibs must exist
    from kernel_configs import METAL_KERNELS
    ttir = OUT / "ttir"
    missing = [c[0] for c in METAL_KERNELS
               if (ttir / f"{c[0]}.ttir").exists()
               and (not (OUT / "apple" / f"{c[0]}.metallib").exists()
                    or not (OUT / "intel" / f"{c[0]}.metallib").exists())]
    if missing:
        print(f"ninja: {dt:.1f}s - FATAL: missing metallibs for {missing}")
        return False
    print(f"ninja: {dt:.1f}s (exit={r.returncode})")
    return True


def compile_dxil():
    """Compile HLSL → DXIL, preferring Windows DXC via SSH for optimal Intel GPU code.

    Falls back to local Mac DXC if Windows is unreachable.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    from kernel_configs import KERNEL_METADATA

    hlsl_dir = OUT / "hlsl"
    ttir_dir = OUT / "ttir"
    dxil_dir = OUT / "dxil"
    dxil_dir.mkdir(parents=True, exist_ok=True)

    # Collect HLSL → entry point mapping for d3d12-capable kernels
    kernels = {}
    for name, meta in KERNEL_METADATA.items():
        if not meta.get("d3d12"):
            continue
        hlsl_path = hlsl_dir / f"{name}.hlsl"
        if not hlsl_path.exists():
            continue
        json_path = ttir_dir / f"{name}.json"
        if json_path.exists():
            entry = json.loads(json_path.read_text()).get("kernel_name", name)
        else:
            entry = name
        kernels[name] = entry

    if not kernels:
        return

    # Try Windows DXC via SSH (produces optimal code for Intel GPUs)
    if _compile_dxil_remote(kernels, hlsl_dir, dxil_dir):
        return

    # Fall back to local Mac DXC
    dxc_bin = SCRIPT_DIR / "dxc" / "dxc"
    if dxc_bin.exists():
        print("DXIL: using local Mac DXC (fallback)")
        _compile_dxil_local(kernels, hlsl_dir, dxil_dir, dxc_bin)
    else:
        print("DXIL: no DXC available, skipping")


def _compile_dxil_remote(kernels, hlsl_dir, dxil_dir):
    """Compile DXIL on Windows via SSH. Returns True on success."""
    # Check if Windows host is reachable
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=3", "windows", "echo ok"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

    t0 = time.time()
    remote_dir = "candle/dxil_build"

    # Copy HLSL files to Windows
    hlsl_files = [hlsl_dir / f"{name}.hlsl" for name in kernels]
    subprocess.run(["ssh", "windows", f"mkdir -p {remote_dir}"], check=True)
    subprocess.run(["scp", "-q"] + [str(f) for f in hlsl_files] +
                   [f"windows:{remote_dir}/"], check=True)

    # Build PowerShell compile script
    dxc = r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\dxc.exe"
    ps_lines = [f'$DXC = "{dxc}"']
    ps_lines.append(f'cd {remote_dir}')
    ps_lines.append(f'if (-not (Test-Path dxil)) {{ New-Item -ItemType Directory dxil | Out-Null }}')
    for name, entry in sorted(kernels.items()):
        ps_lines.append(
            f'& $DXC -T cs_6_2 -E {entry} -enable-16bit-types -O3 '
            f'-Fo "dxil\\{name}.dxil" "{name}.hlsl" 2>&1 | Out-Null'
        )
    ps_lines.append('Write-Host "done"')
    ps_script = "\n".join(ps_lines)

    # Write and run script
    script_path = Path("/tmp/dxil_build.ps1")
    script_path.write_text(ps_script)
    subprocess.run(["scp", "-q", str(script_path), f"windows:{remote_dir}/build.ps1"], check=True)

    r = subprocess.run(
        ["ssh", "windows", f"powershell -ExecutionPolicy Bypass -File {remote_dir}/build.ps1"],
        capture_output=True, text=True, timeout=120
    )
    if r.returncode != 0:
        print(f"DXIL: Windows DXC failed: {r.stderr[:200]}")
        return False

    # Copy DXIL files back
    subprocess.run(
        ["scp", "-q", f"windows:{remote_dir}/dxil/*.dxil", str(dxil_dir) + "/"],
        check=True
    )

    # Verify
    compiled = sum(1 for name in kernels if (dxil_dir / f"{name}.dxil").exists()
                   and (dxil_dir / f"{name}.dxil").stat().st_size > 0)
    dt = time.time() - t0
    print(f"DXIL: {compiled}/{len(kernels)} compiled on Windows in {dt:.1f}s")
    return compiled > 0


def _compile_dxil_local(kernels, hlsl_dir, dxil_dir, dxc_bin):
    """Compile DXIL with local Mac DXC (fallback)."""
    dxc_lib = SCRIPT_DIR / "dxc"
    env = {**os.environ, "DYLD_LIBRARY_PATH": str(dxc_lib)}
    ok = 0
    for name, entry in sorted(kernels.items()):
        hlsl_path = hlsl_dir / f"{name}.hlsl"
        dxil_path = dxil_dir / f"{name}.dxil"
        r = subprocess.run(
            [str(dxc_bin), "-T", "cs_6_2", "-E", entry,
             "-enable-16bit-types", "-O3", "-Fo", str(dxil_path), str(hlsl_path)],
            env=env, capture_output=True, text=True
        )
        if r.returncode == 0 and dxil_path.exists() and dxil_path.stat().st_size > 0:
            ok += 1
    print(f"DXIL: {ok}/{len(kernels)} compiled locally (Mac DXC)")


def gen_rust():
    """Generate Rust kernel embedding code from metadata."""
    from gen_rust import main as gen_rust_main
    gen_rust_main()


if __name__ == "__main__":
    gen_ttir()
    gen_ninja()
    if not run_ninja():
        sys.exit(1)
    compile_dxil()
    gen_rust()
