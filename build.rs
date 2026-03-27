use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    napi_build::setup();

    // ── Triton Metal kernel compilation ──────────────────────────────────────
    // Automatically recompile Triton kernels to .metal when sources change.
    // This ensures hand-edited .metal files can NEVER slip into the build.
    //
    // The triton compiler (compile_moonshine.py) is the single source of truth.
    // If you need to change kernel behavior, edit:
    //   - moonshine_kernels.py          (kernel definitions)
    //   - backend/codegen/*.py           (compiler codegen)
    //   - compile_moonshine.py           (kernel configurations/signatures)
    //
    // Do NOT edit .metal files directly — they will be overwritten on next build.

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let triton_metal_dir = manifest_dir
        .parent()
        .unwrap()
        .join("triton/third_party/metal");

    // Only run if the triton repo exists (runs on macOS even for cross-compilation)
    if triton_metal_dir.exists() {
        compile_triton_kernels(&triton_metal_dir);
    }
}

fn compile_triton_kernels(triton_metal_dir: &Path) {
    let kernel_sources = &[
        "moonshine_kernels.py",
        "compile_moonshine.py",
        "compile_moonshine_hlsl.py",
        "aot_compile.py",
        "backend/codegen/__init__.py",
        "backend/codegen/msl_emitter.py",
        "backend/codegen/hlsl_emitter.py",
        "backend/codegen/lowering.py",
        "backend/codegen/emitter_base.py",
        "backend/codegen/mlir_walker.py",
        "backend/codegen/ir.py",
        "backend/compiler.py",
    ];

    // Tell cargo to rerun build.rs when any kernel source changes
    for src in kernel_sources {
        let path = triton_metal_dir.join(src);
        println!("cargo:rerun-if-changed={}", path.display());
    }

    // Check if the stamp file is newer than all sources — skip if up to date
    let stamp = triton_metal_dir.join("moonshine_metal/.compiled");
    if stamp.exists() {
        let stamp_mtime = std::fs::metadata(&stamp)
            .and_then(|m| m.modified())
            .ok();
        if let Some(stamp_time) = stamp_mtime {
            let all_up_to_date = kernel_sources.iter().all(|src| {
                let path = triton_metal_dir.join(src);
                match std::fs::metadata(&path).and_then(|m| m.modified()) {
                    Ok(src_time) => src_time <= stamp_time,
                    Err(_) => true, // missing source = don't force rebuild
                }
            });
            if all_up_to_date {
                // Kernels are up to date, skip compilation
                return;
            }
        }
    }

    // Find Python — prefer the triton venv
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let venv_python = manifest_dir
        .parent()
        .unwrap()
        .join("triton/env/bin/python");
    let python = if venv_python.exists() {
        venv_python
    } else {
        PathBuf::from("python3")
    };

    println!(
        "cargo:warning=Compiling Triton kernels → Metal (sources changed)..."
    );

    // Step 1: Compile Apple Silicon kernels (simdgroup_matrix)
    let status = Command::new(&python)
        .arg("compile_moonshine.py")
        .arg("--output-dir")
        .arg("moonshine_metal")
        .current_dir(triton_metal_dir)
        .status();

    match status {
        Ok(s) if s.success() => {
            let stamp = triton_metal_dir.join("moonshine_metal/.compiled");
            let _ = std::fs::write(
                &stamp,
                format!(
                    "compiled={}\ngenerator=build.rs\n",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs().to_string())
                        .unwrap_or_default()
                ),
            );
            println!("cargo:warning=Triton Metal kernels compiled successfully.");
        }
        Ok(s) => {
            panic!(
                "Triton kernel compilation failed (exit code: {:?}). \
                 Fix the compiler or kernel sources — do NOT hand-edit .metal files.",
                s.code()
            );
        }
        Err(e) => {
            println!(
                "cargo:warning=Could not run Triton compiler ({e}). \
                 Using existing .metal files if available."
            );
        }
    }

    // Step 2: Compile Intel kernels (scalar, no simdgroup_matrix)
    let intel_script = triton_metal_dir.join("compile_intel_kernels.py");
    if intel_script.exists() {
        println!("cargo:warning=Compiling Triton kernels → Intel Metal (scalar)...");
        let status = Command::new(&python)
            .arg("compile_intel_kernels.py")
            .current_dir(triton_metal_dir)
            .status();
        match status {
            Ok(s) if s.success() => {
                println!("cargo:warning=Intel Metal kernels compiled successfully.");
            }
            Ok(s) => {
                println!(
                    "cargo:warning=Intel kernel compilation failed (exit {:?}), \
                     using existing files.",
                    s.code()
                );
            }
            Err(e) => {
                println!("cargo:warning=Could not compile Intel kernels ({e}).");
            }
        }
    }

    // Step 3: Compile HLSL kernels for D3D12
    let hlsl_script = triton_metal_dir.join("compile_moonshine_hlsl.py");
    if hlsl_script.exists() {
        println!("cargo:warning=Compiling Triton kernels → HLSL (sources changed)...");
        let status = Command::new(&python)
            .arg("compile_moonshine_hlsl.py")
            .arg("--output-dir")
            .arg("moonshine_hlsl")
            .current_dir(triton_metal_dir)
            .status();
        match status {
            Ok(s) if s.success() => {
                println!("cargo:warning=HLSL kernels compiled successfully.");
            }
            Ok(s) => {
                println!(
                    "cargo:warning=HLSL kernel compilation failed (exit {:?}), \
                     using existing files.",
                    s.code()
                );
            }
            Err(e) => {
                println!("cargo:warning=Could not compile HLSL kernels ({e}).");
            }
        }
    }

    // Tell cargo to rerun when HLSL files change — include_str!() doesn't auto-track
    let hlsl_dir = triton_metal_dir.join("moonshine_hlsl");
    if hlsl_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&hlsl_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "hlsl") {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
        }
    }
}
