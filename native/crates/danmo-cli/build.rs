//! Re-emit the MLX-C library directory as an LC_RPATH for the danmo-*
//! binaries. `cargo:rustc-link-arg` from `danmo-mlx-sys` only covers that
//! package's own targets, so each bin crate must add the rpath itself. The
//! `DEP_MLXC_*` variables come from the `links = "mlxc"` metadata.

fn main() {
    for key in ["DEP_MLXC_LIB_DIR", "DEP_MLXC_PREBUILT_LIB_DIR"] {
        if let Ok(dir) = std::env::var(key) {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }
}
