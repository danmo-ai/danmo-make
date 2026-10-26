use std::env;
use std::path::PathBuf;
use std::process::Command;

fn feature(name: &str) -> bool {
    env::var(format!("CARGO_FEATURE_{}", name.to_uppercase())).is_ok()
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let ggml_src = manifest.join("../../third_party/ggml");
    println!(
        "cargo:rerun-if-changed={}",
        ggml_src.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest.join("wrapper.h").display()
    );

    let cuda = feature("cuda");
    let vulkan = feature("vulkan");
    let metal = feature("metal");
    if metal && env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() != "macos" {
        panic!("ggml Metal backend is only built on macOS");
    }

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_dir = out.join("ggml-build");
    std::fs::create_dir_all(&build_dir).expect("create ggml build dir");

    let jobs = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "1".to_string());

    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(&ggml_src)
        .arg("-B")
        .arg(&build_dir)
        .arg("-DGGML_NATIVE=OFF")
        .arg("-DGGML_CCACHE=OFF")
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DGGML_BUILD_TESTS=OFF")
        .arg("-DGGML_BUILD_EXAMPLES=OFF")
        .arg("-DGGML_BLAS=OFF")
        .arg("-DGGML_OPENMP=OFF")
        .arg("-DGGML_BACKEND_DL=OFF")
        .arg(format!("-DGGML_CUDA={}", on(cuda)))
        .arg(format!("-DGGML_VULKAN={}", on(vulkan)))
        .arg(format!("-DGGML_METAL={}", on(metal)));
    let status = configure.status().expect("cmake configure");
    if !status.success() {
        panic!("cmake configure of ggml failed");
    }

    let status = Command::new("cmake")
        .arg("--build")
        .arg(&build_dir)
        .arg("--target")
        .arg("ggml")
        .arg("-j")
        .arg(&jobs)
        .status()
        .expect("cmake build");
    if !status.success() {
        panic!("cmake build of ggml failed");
    }

    let lib_dir = build_dir.join("src");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");
    if cuda {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
    }
    if vulkan {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
        println!("cargo:rustc-link-lib=vulkan");
    }
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=m");

    let mut clang_args = vec![format!("-I{}", ggml_src.join("include").display())];
    if cuda {
        clang_args.push("-DDANMO_GGML_CUDA".into());
    }
    if vulkan {
        clang_args.push("-DDANMO_GGML_VULKAN".into());
    }
    if metal {
        clang_args.push("-DDANMO_GGML_METAL".into());
    }

    let mut builder = bindgen::Builder::default()
        .header(manifest.join("wrapper.h").to_string_lossy())
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("GGML_.*")
        .layout_tests(false)
        .derive_debug(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    for arg in &clang_args {
        builder = builder.clang_arg(arg);
    }
    let bindings = builder.generate().expect("bindgen ggml headers");
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("write bindings");
}

fn on(enabled: bool) -> &'static str {
    if enabled {
        "ON"
    } else {
        "OFF"
    }
}
