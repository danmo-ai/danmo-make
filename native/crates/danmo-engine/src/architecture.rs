use std::path::{Path, PathBuf};

fn source_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

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

fn production_source(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap()
        .split("#[cfg(test)]")
        .next()
        .unwrap()
        .to_string()
}

#[test]
fn registry_and_family_factory_stay_in_sync() {
    let registry_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../default_config/models_registry.json");
    let registry: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(registry_path).unwrap()).unwrap();
    let mut registry_families: Vec<String> = registry["models"]
        .as_object()
        .unwrap()
        .values()
        .filter_map(|model| model["runtime"]["family"].as_str().map(str::to_string))
        .collect();
    registry_families.sort();
    registry_families.dedup();
    let mut factory_families: Vec<String> = crate::families::all_specs()
        .iter()
        .map(|spec| spec.id.to_string())
        .collect();
    factory_families.sort();
    assert_eq!(factory_families, registry_families);
}

#[test]
fn shared_layers_do_not_dispatch_families() {
    for directory in ["pipeline", "session", "common", "runtime", "weights"] {
        for path in rust_files(&source_root().join(directory)) {
            if path.file_name().and_then(|name| name.to_str()) == Some("legacy.rs") {
                continue;
            }
            let source = production_source(&path);
            for forbidden in ["family ==", "match family", "families::ace_step"] {
                assert!(
                    !source.contains(forbidden),
                    "{} contains forbidden dispatch `{forbidden}`",
                    path.display()
                );
            }
        }
    }
}

/// A family never imports another family's module. Shared traits arrive
/// through `families::driver` re-exports; everything else under
/// `crate::families::<id>` inside a different family directory is forbidden.
#[test]
fn families_do_not_import_each_other() {
    let families_root = source_root().join("families");
    let family_dirs: Vec<String> = std::fs::read_dir(&families_root)
        .unwrap()
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();
    for family in &family_dirs {
        for path in rust_files(&families_root.join(family)) {
            let source = production_source(&path);
            for other in &family_dirs {
                if other == family {
                    continue;
                }
                for forbidden in [
                    format!("families::{other}::"),
                    format!("families::{other},"),
                    format!("families::{{*{other}"),
                ] {
                    assert!(
                        !source.contains(forbidden.as_str()),
                        "{} imports family `{other}` via `{forbidden}`",
                        path.display()
                    );
                }
            }
        }
    }
}

/// Only `danmo-mlx` may depend on `danmo-mlx-sys`. Engine code reaches MLX
/// through the `danmo-mlx` wrapper, never through raw C handles.
#[test]
fn engine_code_does_not_touch_mlx_sys() {
    for directory in ["families", "common", "pipeline", "session", "runtime", "weights"] {
        for path in rust_files(&source_root().join(directory)) {
            let source = production_source(&path);
            for forbidden in ["danmo_mlx_sys", "danmo-mlx-sys"] {
                assert!(
                    !source.contains(forbidden),
                    "{} references `{forbidden}`; go through danmo-mlx",
                    path.display()
                );
            }
        }
    }
}

#[test]
fn production_families_do_not_call_ggml_directly() {
    for path in rust_files(&source_root().join("families")) {
        let source = production_source(&path);
        assert!(
            !source.contains("danmo_ggml_sys"),
            "{} directly depends on danmo_ggml_sys",
            path.display()
        );
    }
}

/// Families and shared layers get MLX weights only from `weights::load_mlx_directory`
/// and run linears only through `LinearWeight::apply`, so a packed bundle cannot
/// be re-read or dequantized into a dense resident copy.
#[test]
fn mlx_consumers_cannot_bypass_packed_linears() {
    for directory in ["families", "common", "pipeline", "session", "runtime"] {
        for path in rust_files(&source_root().join(directory)) {
            let source = production_source(&path);
            for forbidden in [
                "SafeTensors::load",
                ".dequantize(",
                "quantize_affine(",
                "quantized_matmul(",
                "QuantizedArray",
            ] {
                assert!(
                    !source.contains(forbidden),
                    "{} uses `{forbidden}`; load through weights::load_mlx_directory and call LinearWeight::apply",
                    path.display()
                );
            }
        }
    }
}

#[test]
fn server_adapter_has_no_family_implementation_types() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../danmo-server/src/engine_host.rs");
    let source = production_source(&path);
    for forbidden in ["ace_step::", "CoverMode", "LmMode", "CreateOutput"] {
        assert!(
            !source.contains(forbidden),
            "server adapter contains {forbidden}"
        );
    }
}
