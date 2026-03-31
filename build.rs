use std::path::PathBuf;
use std::process::Command;

fn main() {
    napi_build::setup();

    // ── Triton kernel compilation via kernels/build.py ─────────────────────
    // Compiles @triton.jit → TTIR → MSL/HLSL → metallib/dxil via ninja.
    // Output: kernels/out/{apple,intel}/*.metallib, kernels/out/hlsl/*.hlsl,
    //         kernels/out/dxil/*.dxil
    // Embedded via include_bytes!.

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let kernels_dir = manifest_dir.join("kernels");
    let triton_dir = manifest_dir.parent().unwrap().join("triton");
    let triton_metal_dir = triton_dir.join("third_party/metal");

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

    // Track output directories so cargo recompiles when metallib/hlsl/dxil files change
    for subdir in &["kernels/out/apple", "kernels/out/intel", "kernels/out/hlsl",
                     "kernels/out/dxil", "kernels/out/generated"] {
        println!("cargo:rerun-if-changed={}", manifest_dir.join(subdir).display());
    }

    // Check stamp — skip if up to date
    let stamp = kernels_dir.join("out/.stamp");
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

    println!("cargo:warning=Compiling Triton kernels (sources changed)...");

    let status = Command::new(&python)
        .arg("build.py")
        .current_dir(&kernels_dir)
        .status();

    match status {
        Ok(s) if s.success() => {
            let _ = std::fs::write(&stamp, format!("{:?}", std::time::SystemTime::now()));
            println!("cargo:warning=Triton kernels compiled successfully.");
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
