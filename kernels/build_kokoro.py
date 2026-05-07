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
    """Write build_kokoro.ninja for TTIR → metallib."""
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
        "",
    ]

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

    w.append("")
    w.append(f"build kokoro: phony {' '.join(libs)}")
    w.append(f"default kokoro")
    w.append("")

    write_if_changed(OUT / "build_kokoro.ninja", "\n".join(w))
    print(f"build_kokoro.ninja: {len(libs)} metallibs")


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
    """Generate Rust embedding code for Kokoro kernels."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from kokoro_configs import KOKORO_KERNELS, KOKORO_METADATA

    gen_dir = OUT / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("// Auto-generated by build_kokoro.py -- do not edit")
    lines.append("")

    # kernel_data module
    lines.append("mod kernel_data {")
    lines.append("    pub fn load_kernel(name: &str) -> Option<&'static [u8]> {")
    lines.append("        match name {")
    for cfg in KOKORO_KERNELS:
        name = cfg[0]
        path = f"../metal/{name}.metallib"
        lines.append(f'            "{name}" => Some(include_bytes!("{path}")),')
    lines.append("            _ => None,")
    lines.append("        }")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # KokoroKernels struct
    lines.append("/// Compiled Triton kernel pipelines for the Kokoro TTS decoder.")
    lines.append("pub struct KokoroKernels {")
    for name in [cfg[0] for cfg in KOKORO_KERNELS]:
        meta = KOKORO_METADATA.get(name)
        if meta:
            alias = meta["alias"]
            lines.append(f"    pub {alias}: ComputePipeline,")
    lines.append("}")
    lines.append("")

    # KokoroKernels::load()
    lines.append("impl KokoroKernels {")
    lines.append("    pub fn load(metal_device: &MetalDevice) -> Result<Self> {")
    lines.append("        let device = metal_device.device();")
    lines.append("")
    lines.append("        let load = |name: &str, func_name: &str| -> Result<ComputePipeline> {")
    lines.append("            let data = kernel_data::load_kernel(name)")
    lines.append("                .ok_or_else(|| anyhow::anyhow!(\"No embedded kernel for {name}\"))?;")
    lines.append("            let lib = device.new_library_with_data(data)")
    lines.append("                .map_err(|e| anyhow::anyhow!(\"Failed to load metallib {name}: {e}\"))?;")
    lines.append("            let func = lib")
    lines.append("                .get_function(func_name, None)")
    lines.append("                .map_err(|e| anyhow::anyhow!(\"Function {func_name} not found in {name}: {e}\"))?;")
    lines.append("            let pipeline = device")
    lines.append("                .new_compute_pipeline_state_with_function(&func)")
    lines.append("                .map_err(|e| anyhow::anyhow!(\"Pipeline failed for {name}: {e}\"))?;")
    lines.append("            Ok(pipeline)")
    lines.append("        };")
    lines.append("")
    lines.append("        Ok(Self {")
    for cfg in KOKORO_KERNELS:
        name = cfg[0]
        func_name = cfg[1]
        meta = KOKORO_METADATA.get(name)
        if meta:
            alias = meta["alias"]
            lines.append(f'            {alias}: load("{name}", "{func_name}")?,')
    lines.append("        })")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    code = "\n".join(lines) + "\n"
    write_if_changed(gen_dir / "kokoro_metal_gen.rs", code)
    print(f"Generated kokoro_metal_gen.rs ({len([c for c in KOKORO_KERNELS if c[0] in KOKORO_METADATA])} kernels)")


if __name__ == "__main__":
    ok = gen_ttir()
    if ok > 0:
        gen_ninja()
        if run_ninja():
            gen_rust()
