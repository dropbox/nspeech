#!/usr/bin/env python3
"""Build Kokoro TTS Triton kernels.

Generates TTIR from @triton.jit, compiles to Metal metallibs.

    python build_kokoro.py
"""
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "out"
TRITON = SCRIPT_DIR.parent.parent / "triton"
TRITON_METAL = TRITON / "third_party" / "metal"
_VENV = TRITON / "env"
_BIN = _VENV / "Scripts" if sys.platform == "win32" else _VENV / "bin"
PYTHON = str(_BIN / "python")
NINJA = str(_BIN / "ninja")
COMPILE_STEP = str(SCRIPT_DIR / "compile_step.py")


def write_if_changed(path: Path, content: str) -> bool:
    if path.exists():
        try:
            if path.read_text() == content:
                return False
        except Exception:
            pass
    path.write_text(content)
    return True


def gen_ttir():
    """Generate TTIR for Kokoro kernels."""
    sys.path.insert(0, str(TRITON_METAL))
    sys.path.insert(0, str(SCRIPT_DIR))

    from aot_compile import compile_kernel
    import kokoro_kernels as K
    from kokoro_configs import KOKORO_KERNELS

    ttir_dir = OUT / "ttir"
    ttir_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating TTIR for {len(KOKORO_KERNELS)} Kokoro kernels...")
    t0 = time.time()
    ok = 0

    for cfg in KOKORO_KERNELS:
        name, func_name, sig, nw, grid = cfg[0], cfg[1], cfg[2], cfg[3], cfg[4]
        opts = cfg[5] if len(cfg) > 5 else {}
        fn = getattr(K, func_name, None)
        if fn is None:
            print(f"  {name}: SKIP (no {func_name})")
            continue
        try:
            r = compile_kernel(fn=fn, signature=sig, num_warps=nw, grid=grid)
            ir = r.ttgir_text or r.ttir_text
            write_if_changed(ttir_dir / f"{name}.ttir", ir)
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
            import traceback
            traceback.print_exc()

    print(f"TTIR: {ok}/{len(KOKORO_KERNELS)} in {time.time()-t0:.1f}s\n")
    return ok


def gen_ninja():
    """Write build_kokoro.ninja for TTIR → metallib + HLSL."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from kokoro_configs import KOKORO_KERNELS

    ttir = OUT / "ttir"
    metal = OUT / "metal"
    metal.mkdir(parents=True, exist_ok=True)

    codegen_dir = TRITON_METAL / "backend" / "codegen"
    compiler_deps = " ".join(str(p) for p in sorted(codegen_dir.glob("*.py")))
    implicit = f"| {compiler_deps} {COMPILE_STEP}"

    w = [
        "# Auto-generated — do not edit",
        f"python = {PYTHON}",
        f"step = {COMPILE_STEP}",
        "",
        "rule msl_metal\n  command = $python $step msl_metal $in $out\n  restat = 1\n  description = MSL(metal) $out",
        "rule metallib_metal\n  command = xcrun metal -std=metal3.1 -O3 -ffast-math -w -o $out $in\n  description = METALLIB(metal) $out",
        "rule msl_metal_nosimd\n  command = $python $step msl_metal_nosimd $in $out\n  restat = 1\n  description = MSL(metal_nosimd) $out",
        "rule metallib_metal_nosimd\n  command = xcrun metal -std=macos-metal2.4 -mmacosx-version-min=14.0 -O3 -ffast-math -w -o $out $in\n  description = METALLIB(metal_nosimd) $out",
        "rule hlsl\n  command = $python $step hlsl $in $out\n  restat = 1\n  description = HLSL $out",
        "",
    ]

    metal_nosimd = OUT / "metal_nosimd"
    metal_nosimd.mkdir(parents=True, exist_ok=True)
    hlsl_dir = OUT / "hlsl"
    hlsl_dir.mkdir(parents=True, exist_ok=True)

    libs = []
    for cfg in KOKORO_KERNELS:
        name = cfg[0]
        t = ttir / f"{name}.ttir"
        if not t.exists():
            continue
        am = metal / f"{name}.metal"
        al = metal / f"{name}.metallib"
        w.append(f"build {am}: msl_metal {t} {implicit}")
        w.append(f"build {al}: metallib_metal {am}")
        libs.append(str(al))
        # nosimd variant for Intel Mac
        im = metal_nosimd / f"{name}.metal"
        il = metal_nosimd / f"{name}.metallib"
        w.append(f"build {im}: msl_metal_nosimd {t} {implicit}")
        w.append(f"build {il}: metallib_metal_nosimd {im}")
        libs.append(str(il))
        # HLSL for D3D12
        ah = hlsl_dir / f"{name}.hlsl"
        w.append(f"build {ah}: hlsl {t} {implicit}")
        libs.append(str(ah))

    w.append("")
    w.append(f"build kokoro: phony {' '.join(libs)}")
    w.append(f"default kokoro")
    w.append("")

    write_if_changed(OUT / "build_kokoro.ninja", "\n".join(w))
    print(f"build_kokoro.ninja: {len(libs)} targets (metallibs + HLSL)")


def run_ninja():
    """Run ninja to compile metallibs."""
    import subprocess
    ninja_file = OUT / "build_kokoro.ninja"
    if not ninja_file.exists():
        print("No ninja file, skipping")
        return
    ninja = NINJA if Path(NINJA).exists() else "ninja"
    t0 = time.time()
    r = subprocess.run([ninja, "-f", str(ninja_file)])
    dt = time.time() - t0
    print(f"ninja: {dt:.1f}s (exit={r.returncode})")
    return r.returncode == 0


def gen_rust():
    """Generate Rust embedding code for Kokoro kernels via gen_rust.py."""
    from gen_rust import main as gen_rust_main
    gen_rust_main()


def compile_dxil():
    """Compile Kokoro HLSL → DXIL via Windows DXC (SSH) or local fallback."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from kokoro_configs import KOKORO_KERNELS

    hlsl_dir = OUT / "hlsl"
    dxil_dir = OUT / "dxil"
    dxil_dir.mkdir(parents=True, exist_ok=True)

    # Collect kernels with HLSL files
    kernels = {}
    for cfg in KOKORO_KERNELS:
        name = cfg[0]
        hlsl_path = hlsl_dir / f"{name}.hlsl"
        if not hlsl_path.exists():
            continue
        json_path = OUT / "ttir" / f"{name}.json"
        if json_path.exists():
            entry = json.loads(json_path.read_text()).get("kernel_name", name)
        else:
            entry = name
        kernels[name] = entry

    if not kernels:
        print("DXIL: no HLSL files found")
        return

    # Try Windows DXC via SSH
    import subprocess
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=3", "windows", "echo ok"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            if _compile_dxil_remote(kernels, hlsl_dir, dxil_dir):
                return
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: local Mac DXC
    dxc_bin = SCRIPT_DIR / "dxc" / "dxc"
    if dxc_bin.exists():
        print("DXIL: using local Mac DXC (fallback)")
        _compile_dxil_local(kernels, hlsl_dir, dxil_dir, dxc_bin)
    else:
        print("DXIL: no DXC available, skipping")


def _compile_dxil_remote(kernels, hlsl_dir, dxil_dir):
    """Compile DXIL on Windows via SSH."""
    import subprocess
    t0 = time.time()
    remote_dir = "candle/dxil_build"

    hlsl_files = [hlsl_dir / f"{name}.hlsl" for name in kernels]
    subprocess.run(["ssh", "windows", f"mkdir -p {remote_dir}"], check=True)
    subprocess.run(["scp", "-q"] + [str(f) for f in hlsl_files] +
                   [f"windows:{remote_dir}/"], check=True)

    dxc = r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\dxc.exe"
    ps_lines = [f'$DXC = "{dxc}"']
    ps_lines.append(f'cd {remote_dir}')
    ps_lines.append('if (-not (Test-Path dxil)) { New-Item -ItemType Directory dxil | Out-Null }')
    for name, entry in sorted(kernels.items()):
        ps_lines.append(
            f'& $DXC -T cs_6_2 -E {entry} -enable-16bit-types -O3 '
            f'-Fo "dxil\\{name}.dxil" "{name}.hlsl" 2>&1 | Out-Null'
        )
    ps_lines.append('Write-Host "done"')

    script_path = Path("/tmp/kokoro_dxil_build.ps1")
    script_path.write_text("\n".join(ps_lines))
    subprocess.run(["scp", "-q", str(script_path), f"windows:{remote_dir}/build.ps1"], check=True)

    r = subprocess.run(
        ["ssh", "windows", f"powershell -ExecutionPolicy Bypass -File {remote_dir}/build.ps1"],
        capture_output=True, text=True, timeout=120
    )
    if r.returncode != 0:
        print(f"DXIL: Windows DXC failed: {r.stderr[:200]}")
        return False

    subprocess.run(
        ["scp", "-q", f"windows:{remote_dir}/dxil/*.dxil", str(dxil_dir) + "/"],
        check=True
    )

    compiled = sum(1 for name in kernels if (dxil_dir / f"{name}.dxil").exists()
                   and (dxil_dir / f"{name}.dxil").stat().st_size > 0)
    dt = time.time() - t0
    print(f"DXIL: {compiled}/{len(kernels)} Kokoro kernels compiled on Windows in {dt:.1f}s")
    return compiled > 0


def _compile_dxil_local(kernels, hlsl_dir, dxil_dir, dxc_bin):
    """Compile DXIL with local DXC."""
    import subprocess, os
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
        else:
            print(f"  {name}: FAILED - {r.stderr[:100]}")
    print(f"DXIL: {ok}/{len(kernels)} compiled locally")


if __name__ == "__main__":
    ok = gen_ttir()
    if ok > 0:
        gen_ninja()
        if run_ninja():
            compile_dxil()
            gen_rust()
