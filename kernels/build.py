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


def _load_env():
    """Load .env from the project root (parent of kernels/) if present."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        import os
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()


def _select_full_xcode_for_metal():
    """Use the standard full Xcode install when xcode-select points at CLT."""
    if sys.platform != "darwin" or "DEVELOPER_DIR" in os.environ:
        return
    selected_has_metal = subprocess.run(
        ["xcrun", "--find", "metal"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    full_xcode = Path("/Applications/Xcode.app/Contents/Developer")
    if not selected_has_metal and full_xcode.is_dir():
        os.environ["DEVELOPER_DIR"] = str(full_xcode)


_select_full_xcode_for_metal()


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
TRITON = Path(os.environ.get("TRITON_DIR", SCRIPT_DIR.parent.parent / "triton")).resolve()
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
    import triton
    import moonshine_kernels as K
    import kokoro_kernels as KK
    from kernel_configs import METAL_KERNELS, HLSL_EXTRA_KERNELS
    from kokoro_configs import KOKORO_KERNELS

    # AOT compilation does not need a live GPU, but JITFunction.create_binder()
    # asks Triton's runtime for one before aot_compile.py selects its offline
    # Metal target. Let the sibling checkout compile on hosts whose venv does
    # not include the optional PyObjC Metal driver.
    try:
        triton.runtime.driver.active
    except RuntimeError:
        from triton.compiler import ASTSource

        def prepare_for_offline_aot(fn):
            fn.ASTSource = ASTSource
            fn.create_binder = lambda: None
    else:
        def prepare_for_offline_aot(fn):
            pass

    ttir_dir = OUT / "ttir"
    ttir_dir.mkdir(parents=True, exist_ok=True)

    # Collect all unique configs by name, with their source module
    configs = {}
    for cfg in METAL_KERNELS:
        configs[cfg[0]] = (cfg, K)
    for cfg in HLSL_EXTRA_KERNELS:
        configs.setdefault(cfg[0], (cfg, K))
    for cfg in KOKORO_KERNELS:
        configs.setdefault(cfg[0], (cfg, KK))

    print(f"Generating TTIR for {len(configs)} kernels...")
    t0 = time.time()
    ok = 0

    for name, (cfg, module) in sorted(configs.items()):
        func_name, sig, nw, grid = cfg[1], cfg[2], cfg[3], cfg[4]
        opts = cfg[5] if len(cfg) > 5 else {}
        fn = getattr(module, func_name, None)
        if fn is None:
            print(f"  {name}: SKIP (no {func_name})")
            continue
        try:
            prepare_for_offline_aot(fn)
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


def _pack_tar_zst_rule():
    """Ninja rule: tar + zstd compress, no-op if output unchanged."""
    return ("rule pack_tar_zst\n"
            "  command = COPYFILE_DISABLE=1 tar --format ustar -cf - -C $dir $names "
            "| zstd -19 -f -o $out.tmp - && if cmp -s $out.tmp $out; then rm $out.tmp; else mv $out.tmp $out; fi\n"
            "  restat = 1\n"
            "  description = TAR+ZST $out")


def gen_ninja_metal():
    """Write build_metal.ninja for TTIR → Apple Silicon metallib (simdgroup_matrix)."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from kernel_configs import METAL_KERNELS
    from kokoro_configs import KOKORO_KERNELS

    ttir = OUT / "ttir"
    metal = OUT / "metal"
    metal.mkdir(parents=True, exist_ok=True)

    w, implicit = _ninja_preamble()
    w.append("rule msl_metal\n  command = $python $step msl_metal $in $out\n  restat = 1\n  description = MSL(metal) $out")
    w.append("rule metallib_metal\n  command = xcrun metal -std=metal3.1 -O3 -ffast-math -w -o $out $in\n  description = METALLIB(metal) $out")
    w.append(_pack_tar_zst_rule())
    w.append("")

    libs = []
    for cfg in list(METAL_KERNELS) + list(KOKORO_KERNELS):
        name = cfg[0]
        t = ttir / f"{name}.ttir"
        if not t.exists():
            continue
        am = metal / f"{name}.metal"
        al = metal / f"{name}.metallib"
        w.append(f"build {am}: msl_metal {t} {implicit}")
        w.append(f"build {al}: metallib_metal {am}")
        libs.append(str(al))

    tar_path = OUT / "kernels_metal.tar.zst"
    names = " ".join(Path(l).name for l in libs)
    w.append("")
    w.append(f"build {tar_path}: pack_tar_zst {' '.join(libs)}")
    w.append(f"  dir = {metal}")
    w.append(f"  names = {names}")
    w.append("")
    w.append(f"build metal: phony {tar_path}")
    w.append(f"default metal")
    w.append("")

    write_if_changed(OUT / "build_metal.ninja", "\n".join(w))
    print(f"build_metal.ninja: {len(libs)} metallibs")


def gen_ninja_metal_nosimd():
    """Write build_metal_nosimd.ninja for TTIR → Metal metallib (no simdgroup_matrix)."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from kernel_configs import METAL_KERNELS
    from kokoro_configs import KOKORO_KERNELS

    ttir = OUT / "ttir"
    metal_nosimd = OUT / "metal_nosimd"
    metal_nosimd.mkdir(parents=True, exist_ok=True)

    w, implicit = _ninja_preamble()
    w.append("rule msl_metal_nosimd\n  command = $python $step msl_metal_nosimd $in $out\n  restat = 1\n  description = MSL(metal_nosimd) $out")
    w.append("rule metallib_metal_nosimd\n  command = xcrun metal -std=macos-metal2.4 -mmacosx-version-min=14.0 -O3 -ffast-math -w -o $out $in\n  description = METALLIB(metal_nosimd) $out")
    w.append(_pack_tar_zst_rule())
    w.append("")

    libs = []
    for cfg in list(METAL_KERNELS) + list(KOKORO_KERNELS):
        name = cfg[0]
        t = ttir / f"{name}.ttir"
        if not t.exists():
            continue
        im = metal_nosimd / f"{name}.metal"
        il = metal_nosimd / f"{name}.metallib"
        w.append(f"build {im}: msl_metal_nosimd {t} {implicit}")
        w.append(f"build {il}: metallib_metal_nosimd {im}")
        libs.append(str(il))

    tar_path = OUT / "kernels_metal_nosimd.tar.zst"
    names = " ".join(Path(l).name for l in libs)
    w.append("")
    w.append(f"build {tar_path}: pack_tar_zst {' '.join(libs)}")
    w.append(f"  dir = {metal_nosimd}")
    w.append(f"  names = {names}")
    w.append("")
    w.append(f"build metal_nosimd: phony {tar_path}")
    w.append(f"default metal_nosimd")
    w.append("")

    write_if_changed(OUT / "build_metal_nosimd.ninja", "\n".join(w))
    print(f"build_metal_nosimd.ninja: {len(libs)} metallibs")


def gen_ninja_hlsl():
    """Write build_hlsl.ninja for TTIR → HLSL → DXIL → tar."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from kernel_configs import get_hlsl_kernels, KERNEL_METADATA
    from kokoro_configs import KOKORO_KERNELS

    ttir = OUT / "ttir"
    hlsl = OUT / "hlsl"
    dxil = OUT / "dxil"
    hlsl.mkdir(parents=True, exist_ok=True)
    dxil.mkdir(parents=True, exist_ok=True)

    import os
    dxc_host = os.environ.get("DXC_HOST", "")
    dxc_path = os.environ.get("DXC_PATH",
        "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/dxc.exe")
    dxc_flags = "-T cs_6_2 -enable-16bit-types -O3 -Wno-for-redefinition"

    w, implicit = _ninja_preamble()
    w.append("rule hlsl\n  command = $python $step hlsl $in $out\n  restat = 1\n  description = HLSL $out")
    if dxc_host:
        remote_dir = "dxil_build"
        # Single-quote the ssh command so parens in the Windows path don't get interpreted locally.
        # Inside single quotes, the remote shell sees: "C:/Program Files (x86)/..." with double quotes.
        remote_cmd = f'"{dxc_path}" {dxc_flags} -E $entry -Fo {remote_dir}/$out_name {remote_dir}/$in_name'
        dxil_cmd = (f"scp -q $in {dxc_host}:{remote_dir}/$in_name "
                    f"&& ssh {dxc_host} '{remote_cmd}' "
                    f"&& scp -q {dxc_host}:{remote_dir}/$out_name $out")
        w.append("pool dxc_pool\n  depth = 4")
        w.append(f"rule dxil\n  command = {dxil_cmd}\n  pool = dxc_pool\n  description = DXIL $out")
    else:
        local_dxc = SCRIPT_DIR / "dxc" / "dxc"
        dxil_cmd = f'DYLD_LIBRARY_PATH={SCRIPT_DIR / "dxc"} {local_dxc} {dxc_flags} -E $entry -Fo $out $in'
        w.append(f"rule dxil\n  command = {dxil_cmd}\n  description = DXIL $out")
    w.append(_pack_tar_zst_rule())
    w.append("")

    hlsl_files = []
    dxil_files = []
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
        # DXIL compilation
        d = dxil / f"{name}.dxil"
        entry = _get_entry_point(name)
        w.append(f"build {d}: dxil {h}")
        w.append(f"  entry = {entry}")
        w.append(f"  in_name = {name}.hlsl")
        w.append(f"  out_name = {name}.dxil")
        dxil_files.append(str(d))
    # Kokoro D3D12 kernels
    for cfg in KOKORO_KERNELS:
        name = cfg[0]
        meta = KERNEL_METADATA.get(name)
        if not meta or not meta.get("d3d12"):
            continue
        if name in hlsl_seen:
            continue
        hlsl_seen.add(name)
        t = ttir / f"{name}.ttir"
        if not t.exists():
            continue
        h = hlsl / f"{name}.hlsl"
        w.append(f"build {h}: hlsl {t} {implicit}")
        hlsl_files.append(str(h))
        # DXIL compilation
        d = dxil / f"{name}.dxil"
        entry = _get_entry_point(name)
        w.append(f"build {d}: dxil {h}")
        w.append(f"  entry = {entry}")
        w.append(f"  in_name = {name}.hlsl")
        w.append(f"  out_name = {name}.dxil")
        dxil_files.append(str(d))

    tar_path = OUT / "kernels_dxil.tar.zst"
    names = " ".join(Path(d).name for d in dxil_files)
    w.append("")
    w.append(f"build {tar_path}: pack_tar_zst {' '.join(dxil_files)}")
    w.append(f"  dir = {dxil}")
    w.append(f"  names = {names}")
    w.append("")
    w.append(f"build hlsl: phony {tar_path}")
    w.append(f"default hlsl")
    w.append("")

    write_if_changed(OUT / "build_hlsl.ninja", "\n".join(w))
    print(f"build_hlsl.ninja: {len(hlsl_files)} hlsl, {len(dxil_files)} dxil")


def _get_entry_point(name):
    """Get the DXIL entry point for a kernel from its TTIR JSON."""
    json_path = OUT / "ttir" / f"{name}.json"
    if json_path.exists():
        return json.loads(json_path.read_text()).get("kernel_name", name)
    return name


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
    gen_rust()
