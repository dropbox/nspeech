#!/usr/bin/env python3
"""Build Moonshine Triton kernels.

Generates TTIR from @triton.jit, writes per-platform ninja files, runs ninja.
Output: out/{metal,metal_nosimd}/*.metallib, out/hlsl/*.hlsl

    python build.py                    # all platforms
    python build.py metal              # Apple Silicon metallibs only
    python build.py metal_nosimd       # Intel Mac metallibs only
    python build.py hlsl               # HLSL + DXIL only
    python build.py metal hlsl         # multiple platforms
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
_VENV = TRITON / "env"
_BIN = _VENV / "Scripts" if sys.platform == "win32" else _VENV / "bin"
PYTHON = str(_BIN / "python")
NINJA = str(_BIN / "ninja")
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
            # Serialize constants — replace non-JSON-serializable objects (e.g.
            # @triton.jit functions used as constexpr meta-parameters) with their name.
            serializable_constants = {}
            for k, v in r.constants.items():
                if hasattr(v, '__name__'):
                    serializable_constants[k] = v.__name__
                elif v is None:
                    serializable_constants[k] = None
                else:
                    serializable_constants[k] = v
            write_if_changed(ttir_dir / f"{name}.json", json.dumps({
                "kernel_name": r.kernel_name,
                "params": r.params,
                "constants": serializable_constants,
                "threadgroup_size": r.threadgroup_size,
                "grid": grid,
                "force_acc_fp16": opts.get("force_acc_fp16", False),
            }, indent=2))
            ok += 1
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    print(f"TTIR: {ok}/{len(configs)} in {time.time()-t0:.1f}s\n")


def _ninja_preamble():
    """Common ninja preamble: python path and compile_step path."""
    codegen_dir = TRITON_METAL / "backend" / "codegen"
    compiler_deps = " ".join(str(p) for p in sorted(codegen_dir.glob("*.py")))
    implicit = f"| {compiler_deps} {COMPILE_STEP}"
    return [
        "# Auto-generated — do not edit",
        f"python = {PYTHON}",
        f"step = {COMPILE_STEP}",
        "",
    ], implicit


def gen_ninja_metal():
    """Write build_metal.ninja for TTIR → Apple Silicon metallib (simdgroup_matrix)."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from kernel_configs import METAL_KERNELS

    ttir = OUT / "ttir"
    metal = OUT / "metal"
    metal.mkdir(parents=True, exist_ok=True)

    w, implicit = _ninja_preamble()
    w.append("rule msl_metal\n  command = $python $step msl_metal $in $out\n  restat = 1\n  description = MSL(metal) $out")
    w.append("rule metallib_metal\n  command = xcrun metal -std=metal3.1 -O3 -ffast-math -w -o $out $in\n  description = METALLIB(metal) $out")
    w.append("")

    libs = []
    for cfg in METAL_KERNELS:
        name = cfg[0]
        t = ttir / f"{name}.ttir"
        if not t.exists():
            continue
        am = metal / f"{name}.metal"
        al = metal / f"{name}.metallib"
        w.append(f"build {am}: msl_metal {t} {implicit}")
        w.append(f"build {al}: metallib_metal {am}")
        libs.append(str(al))

    w.append("")
    w.append(f"build metal: phony {' '.join(libs)}")
    w.append(f"default metal")
    w.append("")

    write_if_changed(OUT / "build_metal.ninja", "\n".join(w))
    print(f"build_metal.ninja: {len(libs)} metallibs")


def gen_ninja_metal_nosimd():
    """Write build_metal_nosimd.ninja for TTIR → Metal metallib (no simdgroup_matrix)."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from kernel_configs import METAL_KERNELS

    ttir = OUT / "ttir"
    metal_nosimd = OUT / "metal_nosimd"
    metal_nosimd.mkdir(parents=True, exist_ok=True)

    w, implicit = _ninja_preamble()
    w.append("rule msl_metal_nosimd\n  command = $python $step msl_metal_nosimd $in $out\n  restat = 1\n  description = MSL(metal_nosimd) $out")
    w.append("rule metallib_metal_nosimd\n  command = xcrun metal -std=macos-metal2.4 -mmacosx-version-min=14.0 -O3 -ffast-math -w -o $out $in\n  description = METALLIB(metal_nosimd) $out")
    w.append("")

    libs = []
    for cfg in METAL_KERNELS:
        name = cfg[0]
        t = ttir / f"{name}.ttir"
        if not t.exists():
            continue
        im = metal_nosimd / f"{name}.metal"
        il = metal_nosimd / f"{name}.metallib"
        w.append(f"build {im}: msl_metal_nosimd {t} {implicit}")
        w.append(f"build {il}: metallib_metal_nosimd {im}")
        libs.append(str(il))

    w.append("")
    w.append(f"build metal_nosimd: phony {' '.join(libs)}")
    w.append(f"default metal_nosimd")
    w.append("")

    write_if_changed(OUT / "build_metal_nosimd.ninja", "\n".join(w))
    print(f"build_metal_nosimd.ninja: {len(libs)} metallibs")


def gen_ninja_hlsl():
    """Write build_hlsl.ninja for TTIR → HLSL."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from kernel_configs import get_hlsl_kernels

    ttir = OUT / "ttir"
    hlsl = OUT / "hlsl"
    hlsl.mkdir(parents=True, exist_ok=True)

    w, implicit = _ninja_preamble()
    w.append("rule hlsl\n  command = $python $step hlsl $in $out\n  restat = 1\n  description = HLSL $out")
    w.append("")

    hlsl_files = []
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
    w.append(f"build hlsl_all: phony {' '.join(hlsl_files)}")
    w.append(f"default hlsl_all")
    w.append("")

    write_if_changed(OUT / "build_hlsl.ninja", "\n".join(w))
    print(f"build_hlsl.ninja: {len(hlsl_files)} hlsl")


def run_ninja(platform):
    """Run ninja for a specific platform."""
    ninja_file = OUT / f"build_{platform}.ninja"
    if not ninja_file.exists():
        print(f"ninja: no {ninja_file.name}, skipping")
        return True
    ninja = NINJA if Path(NINJA).exists() else "ninja"
    t0 = time.time()
    r = subprocess.run([ninja, "-f", str(ninja_file)])
    dt = time.time() - t0
    if platform in ("metal", "metal_nosimd"):
        from kernel_configs import METAL_KERNELS
        ttir = OUT / "ttir"
        subdir = OUT / platform
        missing = [c[0] for c in METAL_KERNELS
                   if (ttir / f"{c[0]}.ttir").exists()
                   and not (subdir / f"{c[0]}.metallib").exists()]
        if missing:
            print(f"ninja({platform}): {dt:.1f}s - FATAL: missing metallibs for {missing}")
            return False
    print(f"ninja({platform}): {dt:.1f}s (exit={r.returncode})")
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

    # On Windows, use the local Windows SDK DXC directly
    if sys.platform == "win32":
        dxc_win = Path(r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\dxc.exe")
        if dxc_win.exists():
            _compile_dxil_local(kernels, hlsl_dir, dxil_dir, dxc_win)
            return
        print("DXIL: Windows SDK DXC not found, skipping")
        return

    # From Mac: try Windows DXC via SSH (produces optimal code for Intel GPUs)
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
    """Compile DXIL with local DXC."""
    env = os.environ.copy()
    if sys.platform == "darwin":
        env["DYLD_LIBRARY_PATH"] = str(SCRIPT_DIR / "dxc")
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
    print(f"DXIL: {ok}/{len(kernels)} compiled locally")


def gen_rust():
    """Generate Rust kernel embedding code from metadata."""
    from gen_rust import main as gen_rust_main
    gen_rust_main()


VALID_PLATFORMS = ("metal", "metal_nosimd", "hlsl")

if __name__ == "__main__":
    platforms = sys.argv[1:] or list(VALID_PLATFORMS)
    for p in platforms:
        if p not in VALID_PLATFORMS:
            print(f"Unknown platform: {p} (valid: {', '.join(VALID_PLATFORMS)})")
            sys.exit(1)

    gen_ttir()
    for p in platforms:
        {"metal": gen_ninja_metal, "metal_nosimd": gen_ninja_metal_nosimd, "hlsl": gen_ninja_hlsl}[p]()
    for p in platforms:
        if not run_ninja(p):
            sys.exit(1)
    if "hlsl" in platforms:
        compile_dxil()
    gen_rust()
