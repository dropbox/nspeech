use std::path::PathBuf;
use std::process::Command;

fn main() {
    napi_build::setup();

    // ── Triton kernel compilation via kernels/build.py ─────────────────────
    // Compiles @triton.jit → TTIR → MSL/HLSL → metallib/dxil via ninja.
    // Only builds for the current target platform:
    //   aarch64 + macOS → metal metallibs (simdgroup_matrix)
    //   x86_64 + macOS  → metal_nosimd metallibs (scalar fallback)
    //   Windows          → HLSL + DXIL

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let kernels_dir = manifest_dir.join("kernels");
    let triton_dir = manifest_dir.parent().unwrap().join("triton");
    let triton_metal_dir = triton_dir.join("third_party/metal");

    // Determine which platform(s) to build
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let has_metal = std::env::var("CARGO_FEATURE_TRITON_METAL").is_ok();
    let has_d3d12 = std::env::var("CARGO_FEATURE_TRITON_D3D12").is_ok();

    let mut platforms: Vec<&str> = Vec::new();
    if has_metal {
        match target_arch.as_str() {
            "aarch64" => platforms.push("metal"),
            "x86_64" if target_os == "macos" => platforms.push("metal_nosimd"),
            _ => platforms.push("metal"), // fallback
        }
    }
    if has_d3d12 {
        platforms.push("hlsl");
    }
    if platforms.is_empty() {
        return;
    }

    // Sources that affect kernel output — rerun build.rs when these change
    let kernel_sources = &[
        "kernels/moonshine_kernels.py",
        "kernels/kernel_configs.py",
        "kernels/build.py",
        "kernels/compile_step.py",
        "kernels/gen_rust.py",
    ];
    let compiler_sources = &[
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

    for src in kernel_sources {
        println!("cargo:rerun-if-changed={}", manifest_dir.join(src).display());
    }
    for src in compiler_sources {
        println!("cargo:rerun-if-changed={}", triton_metal_dir.join(src).display());
    }

    // Track output directories for current platform only
    println!("cargo:rerun-if-changed={}", manifest_dir.join("kernels/out/generated").display());
    for p in &platforms {
        let subdir = match *p {
            "metal" => "kernels/out/metal",
            "metal_nosimd" => "kernels/out/metal_nosimd",
            "hlsl" => "kernels/out/hlsl",
            _ => continue,
        };
        println!("cargo:rerun-if-changed={}", manifest_dir.join(subdir).display());
    }
    if platforms.contains(&"hlsl") {
        println!("cargo:rerun-if-changed={}", manifest_dir.join("kernels/out/dxil").display());
    }

    // Per-platform stamp — skip if up to date
    let stamp_name = format!(".stamp_{}", platforms.join("_"));
    let stamp = kernels_dir.join("out").join(&stamp_name);
    if stamp.exists() {
        let stamp_time = std::fs::metadata(&stamp)
            .and_then(|m| m.modified())
            .ok();
        if let Some(st) = stamp_time {
            let all_fresh = kernel_sources.iter().all(|s| {
                std::fs::metadata(manifest_dir.join(s))
                    .and_then(|m| m.modified())
                    .map(|t| t <= st)
                    .unwrap_or(true)
            }) && compiler_sources.iter().all(|s| {
                std::fs::metadata(triton_metal_dir.join(s))
                    .and_then(|m| m.modified())
                    .map(|t| t <= st)
                    .unwrap_or(true)
            });
            if all_fresh {
                return;
            }
        }
    }

    // Find python
    let python = triton_dir.join("env/bin/python");
    let python = if python.exists() { python } else { PathBuf::from("python3") };

    let platform_str = platforms.join(", ");
    println!("cargo:warning=Compiling Triton kernels for [{platform_str}] (sources changed)...");

    let status = Command::new(&python)
        .arg("build.py")
        .args(&platforms)
        .current_dir(&kernels_dir)
        .status();

    match status {
        Ok(s) if s.success() => {
            let _ = std::fs::write(&stamp, format!("{:?}", std::time::SystemTime::now()));
            println!("cargo:warning=Triton kernels compiled successfully ({platform_str}).");
        }
        Ok(s) => {
            panic!(
                "Kernel compilation failed (exit {:?}). Fix kernels/build.py output.",
                s.code()
            );
        }
        Err(e) => {
            println!("cargo:warning=Could not run kernel build ({e}). Using existing outputs.");
        }
    }
}
