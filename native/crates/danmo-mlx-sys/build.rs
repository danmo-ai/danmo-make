use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn feature(name: &str) -> bool {
    env::var(format!("CARGO_FEATURE_{}", name.to_uppercase())).is_ok()
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mlx_c_src = manifest.join("../../third_party/mlx-c");
    let mlx_src = manifest.join("../../third_party/mlx");
    let fmt_src = manifest.join("../../third_party/fmt");
    let json_src = manifest.join("../../third_party/json");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_dir = out.join("mlx-c-build");
    let lib_dir = out.join("lib");
    let metal_dir = out.join("metal");

    for path in [
        mlx_c_src.join("CMakeLists.txt"),
        mlx_c_src.join("mlx/c/mlx.h"),
        manifest.join("wrapper.h"),
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    for source in [&mlx_src, &fmt_src, &json_src] {
        println!("cargo:rerun-if-changed={}", source.display());
        if !source.join("CMakeLists.txt").is_file() {
            panic!(
                "vendored MLX dependency is incomplete: {}",
                source.display()
            );
        }
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let metal = feature("metal");
    let cuda = feature("cuda");
    println!("cargo:rerun-if-env-changed=DANQING_MLX_PREBUILT");
    let prebuilt = env::var_os("DANQING_MLX_PREBUILT").map(PathBuf::from);
    if metal && cuda {
        panic!("danmo-mlx-sys builds one GPU backend per artifact; disable metal or cuda");
    }
    if metal && target_os != "macos" {
        panic!("MLX Metal is only supported on macOS");
    }
    if cuda && target_os != "linux" {
        panic!("MLX CUDA is only supported on Linux");
    }
    if !mlx_c_src.join("CMakeLists.txt").is_file() {
        panic!("mlx-c sources are missing; initialize native/third_party/mlx-c at v0.8.0");
    }
    if let Some(prefix) = &prebuilt {
        if !metal {
            panic!("DANQING_MLX_PREBUILT currently requires the metal feature");
        }
        for relative in [
            "lib/libmlx.dylib",
            "share/cmake/MLX/MLXConfig.cmake",
            "include/mlx/version.h",
        ] {
            if !prefix.join(relative).is_file() {
                panic!(
                    "prebuilt MLX prefix is missing {relative}: {}",
                    prefix.display()
                );
            }
        }
        let version = std::fs::read_to_string(prefix.join("include/mlx/version.h"))
            .expect("read prebuilt MLX version");
        for declaration in [
            "#define MLX_VERSION_MAJOR 0",
            "#define MLX_VERSION_MINOR 32",
            "#define MLX_VERSION_PATCH 4",
        ] {
            assert!(
                version.contains(declaration),
                "prebuilt MLX must be 0.32.4: {}",
                prefix.display()
            );
        }
    }

    std::fs::create_dir_all(&lib_dir).expect("create MLX library directory");
    std::fs::create_dir_all(&metal_dir).expect("create MLX Metal directory");

    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(&mlx_c_src)
        .arg("-B")
        .arg(&build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DBUILD_SHARED_LIBS=ON")
        .arg("-DMLX_C_BUILD_EXAMPLES=OFF")
        .arg("-DMLX_BUILD_TESTS=OFF")
        .arg("-DMLX_BUILD_EXAMPLES=OFF")
        .arg("-DMLX_BUILD_BENCHMARKS=OFF")
        .arg("-DMLX_BUILD_PYTHON_BINDINGS=OFF")
        .arg("-DMLX_BUILD_GGUF=OFF")
        .arg("-DMLX_BUILD_SAFETENSORS=ON")
        .arg("-DMLX_USE_CCACHE=OFF")
        .arg(format!("-DMLX_BUILD_METAL={}", on(metal)))
        .arg(format!("-DMLX_BUILD_CUDA={}", on(cuda)))
        .arg(format!(
            "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}",
            lib_dir.display()
        ))
        .arg(format!(
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}",
            lib_dir.display()
        ))
        .arg(format!(
            "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={}",
            lib_dir.display()
        ));
    if let Some(prefix) = &prebuilt {
        configure
            .arg("-DMLX_C_USE_SYSTEM_MLX=ON")
            .arg(format!("-DCMAKE_PREFIX_PATH={}", prefix.display()));
    } else {
        if metal {
            configure.arg(format!("-DMLX_METAL_PATH={}", metal_dir.display()));
        }
        for (name, source) in [("MLX", &mlx_src), ("FMT", &fmt_src), ("JSON", &json_src)] {
            configure.arg(format!(
                "-DFETCHCONTENT_SOURCE_DIR_{name}={}",
                source.display()
            ));
        }
        if cuda {
            let archs = env::var("MLX_CUDA_ARCHITECTURES")
                .or_else(|_| env::var("CMAKE_CUDA_ARCHITECTURES"))
                .unwrap_or_else(|_| "75;80;86;89;90".into());
            configure.arg(format!("-DMLX_CUDA_ARCHITECTURES={archs}"));
        }
    }

    run(&mut configure, "configure mlx-c");
    let jobs = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "1".into());
    let mut build = Command::new("cmake");
    build
        .arg("--build")
        .arg(&build_dir)
        .arg("--target")
        .arg("mlxc")
        .arg("-j")
        .arg(jobs);
    run(&mut build, "build mlx-c");
    if let Some(prefix) = &prebuilt {
        stage_prebuilt(&prefix.join("lib"), &lib_dir);
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    if let Some(prefix) = &prebuilt {
        let prefix_lib = prefix.join("lib");
        println!("cargo:rustc-link-search=native={}", prefix_lib.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", prefix_lib.display());
    }
    println!("cargo:rustc-link-lib=dylib=mlxc");
    println!("cargo:rustc-link-lib=dylib=mlx");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=m");
    }

    let bindings = bindgen::Builder::default()
        .header(manifest.join("wrapper.h").to_string_lossy())
        .clang_arg(format!("-I{}", mlx_c_src.display()))
        .allowlist_function("mlx_.*")
        .allowlist_function("_mlx_.*")
        .allowlist_type("mlx_.*")
        .allowlist_var("MLX_.*")
        .layout_tests(false)
        .derive_debug(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("generate mlx-c bindings");
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("write mlx-c bindings");

    if prebuilt.is_none() {
        copy_metal_library(&build_dir, &metal_dir);
    }
}

fn stage_prebuilt(source: &Path, destination: &Path) {
    for entry in std::fs::read_dir(source).expect("read prebuilt MLX libraries") {
        let entry = entry.expect("read prebuilt MLX library entry");
        let source = entry.path();
        if !source.is_file() {
            continue;
        }
        let target = destination.join(entry.file_name());
        std::fs::copy(&source, &target)
            .unwrap_or_else(|error| panic!("failed to stage {}: {error}", source.display()));
    }
}

fn copy_metal_library(build_dir: &Path, metal_dir: &Path) {
    if !feature("metal") {
        return;
    }
    let candidate = build_dir
        .join("_deps")
        .join("mlx-build")
        .join("mlx")
        .join("backend")
        .join("metal")
        .join("kernels")
        .join("mlx.metallib");
    if candidate.is_file() {
        std::fs::copy(&candidate, metal_dir.join("mlx.metallib")).expect("copy MLX Metal library");
    }
}

fn run(command: &mut Command, description: &str) {
    let status = command.status().unwrap_or_else(|error| {
        panic!("failed to {description}: {error}");
    });
    if !status.success() {
        panic!("failed to {description}: process exited with {status}");
    }
}

fn on(enabled: bool) -> &'static str {
    if enabled {
        "ON"
    } else {
        "OFF"
    }
}
