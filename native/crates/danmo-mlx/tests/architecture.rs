use std::path::{Path, PathBuf};

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files.extend(rust_files(&path));
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            files.push(path);
        }
    }
    files
}

#[test]
fn engine_never_depends_on_raw_mlx_bindings() {
    let crates = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    for path in rust_files(&crates.join("danmo-engine/src")) {
        let source = std::fs::read_to_string(&path).unwrap();
        let production = source.split("#[cfg(test)]").next().unwrap();
        for forbidden in ["danmo_mlx_sys", "danmo-mlx-sys", "mlx_sys::"] {
            assert!(
                !production.contains(forbidden),
                "{} contains raw MLX binding reference `{forbidden}`",
                path.display()
            );
        }
    }
}

#[test]
fn safe_wrapper_is_the_only_mlx_sys_consumer() {
    let crates = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    for entry in std::fs::read_dir(crates).unwrap() {
        let path = entry.unwrap().path().join("Cargo.toml");
        if !path.is_file() {
            continue;
        }
        let source = std::fs::read_to_string(&path).unwrap();
        let is_wrapper = source.contains("name = \"danmo-mlx\"");
        let is_sys = source.contains("name = \"danmo-mlx-sys\"");
        let has_dependency = source
            .lines()
            .any(|line| line.trim_start().starts_with("danmo-mlx-sys") && line.contains("path"));
        assert!(
            !has_dependency || is_wrapper,
            "{} exposes raw MLX bindings outside danmo-mlx",
            path.display()
        );
        assert!(!(is_sys && source.contains("danmo-mlx =")));
    }
}
