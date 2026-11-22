//! Forward build-time metadata from `danmo-mlx-sys` into `rustc-env`.
//!
//! `links` metadata (`DEP_MLXC_*`) is visible only inside build scripts, never
//! inside the dependent crate's own `env!`. The capability module reads the
//! re-emitted `DANMO_MLX_*` variables at compile time instead.

fn main() {
    for (metadata, rustc_env) in [
        ("DEP_MLXC_DEPLOYMENT_TARGET", "DANMO_MLX_DEPLOYMENT_TARGET"),
        ("DEP_MLXC_NAX_BUILT", "DANMO_MLX_NAX_BUILT"),
    ] {
        println!("cargo:rerun-if-env-changed={metadata}");
        if let Ok(value) = std::env::var(metadata) {
            println!("cargo:rustc-env={rustc_env}={value}");
        }
    }
}
