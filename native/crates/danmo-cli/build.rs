//! Re-emit the MLX-C library directory as an LC_RPATH for the danmo-*
//! binaries. `cargo:rustc-link-arg` from `danmo-mlx-sys` only covers that
//! package's own targets, so each bin crate must add the rpath itself. The
//! `DEP_MLXC_*` variables come from the `links = "mlxc"` metadata.
//!
//! Also add `@executable_path` / `$ORIGIN` so a binary that ships next to
//! `libmlx` / `libmlxc` keeps resolving them outside the build tree.

fn main() {
    for key in ["DEP_MLXC_LIB_DIR", "DEP_MLXC_PREBUILT_LIB_DIR"] {
        if let Ok(dir) = std::env::var(key) {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }
    if std::env::var_os("DEP_MLXC_LIB_DIR").is_some()
        || std::env::var_os("DEP_MLXC_PREBUILT_LIB_DIR").is_some()
    {
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if target_os == "macos" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
        } else if target_os == "linux" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        }
    }
}
