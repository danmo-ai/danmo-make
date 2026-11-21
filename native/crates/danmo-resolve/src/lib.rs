//! Installed-bundle resolution shared by the HTTP server and the CLI.
//!
//! A model id (optionally `id:version`) resolves through the registry to an
//! on-disk bundle under the workspace. The rules live here exactly once:
//! `danmo-server` and `danmo-cli` must not grow private copies.

use danmo_catalog::Catalog;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

/// Marker file written while an install hook (e.g. checkpoint conversion) is
/// still running. A bundle carrying it is not ready.
pub const INSTALL_HOOK_INCOMPLETE: &str = ".install-hook-incomplete";

pub fn resolve_model_path(workspace: &Path, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        workspace.join(path)
    }
}

pub fn version_ready(path: &Path) -> bool {
    if path.is_file() {
        return is_bundle_weight_name(path);
    }
    if path.join(INSTALL_HOOK_INCOMPLETE).is_file() {
        return false;
    }
    let manifest = path.join("bundle.manifest.json");
    if manifest.is_file() {
        return manifest_ready(path, &manifest);
    }
    let mut weight = false;
    let mut partial = false;
    walk_bundle_weights(path, &mut weight, &mut partial);
    weight && !partial
}

/// A published weight file, matching the files main treats as an installed bundle.
pub fn is_bundle_weight_name(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".gguf")
        || lower.ends_with(".safetensors")
        || lower.ends_with(".bin")
        || lower.ends_with(".pt")
        || lower.ends_with(".ckpt")
        || name == "model_index.json"
        || name == "config.json"
        || name == "pytorch_model.bin.index.json"
}

pub fn manifest_ready(root: &Path, manifest: &Path) -> bool {
    let Ok(text) = fs::read_to_string(manifest) else {
        return false;
    };
    let Ok(body) = serde_json::from_str::<Value>(&text) else {
        return false;
    };
    let listing = body.get("files").or_else(|| {
        body.get("components")
            .filter(|components| components.is_object())
    });
    let mut weights = Vec::new();
    collect_manifest_weights(listing, &mut weights);
    if weights.is_empty() {
        return false;
    }
    if has_partial(root) {
        return false;
    }
    listing
        .and_then(|files| files.get("transformer"))
        .is_some_and(|transformer| {
            let mut required = Vec::new();
            collect_manifest_weights(Some(transformer), &mut required);
            !required.is_empty()
        })
        && weights.iter().all(|relative| {
            let path = Path::new(relative);
            !relative.is_empty()
                && !path.is_absolute()
                && !path.components().any(|part| part.as_os_str() == "..")
                && root.join(path).is_file()
        })
}

fn collect_manifest_weights(value: Option<&Value>, out: &mut Vec<String>) {
    match value {
        Some(Value::String(path)) => out.push(path.clone()),
        Some(Value::Array(rows)) => {
            for row in rows {
                collect_manifest_weights(Some(row), out);
            }
        }
        Some(Value::Object(rows)) => {
            for row in rows.values() {
                collect_manifest_weights(Some(row), out);
            }
        }
        _ => {}
    }
}

fn has_partial(path: &Path) -> bool {
    let Ok(entries) = fs::read_dir(path) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if has_partial(&path) {
                return true;
            }
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".partial"))
        {
            return true;
        }
    }
    false
}

fn walk_bundle_weights(path: &Path, weight: &mut bool, partial: &mut bool) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_bundle_weights(&path, weight, partial);
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".partial"))
        {
            *partial = true;
        } else if is_bundle_weight_name(&path) {
            *weight = true;
        }
    }
}

/// One safetensors adapter inside a registry LoRA install.
/// A directory with several weight files and no single preferred name fails.
pub fn lora_weight_file(root: &Path) -> Result<PathBuf, String> {
    if root.is_file() {
        let safetensors = root
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"));
        if safetensors {
            return Ok(root.to_path_buf());
        }
        return Err(format!(
            "lora weight {} is not a safetensors file",
            root.display()
        ));
    }
    if !root.is_dir() {
        return Err(format!("lora weight {} is missing", root.display()));
    }
    let mut files = Vec::new();
    collect_safetensors(root, &mut files);
    if files.is_empty() {
        return Err(format!(
            "lora directory {} has no safetensors file",
            root.display()
        ));
    }
    if files.len() == 1 {
        return Ok(files.remove(0));
    }
    const PREFERRED: &[&str] = &[
        "final_adapters.safetensors",
        "adapters.safetensors",
        "adapter_model.safetensors",
        "pytorch_lora_weights.safetensors",
        "model.safetensors",
    ];
    for name in PREFERRED {
        let mut hits = files
            .iter()
            .filter(|path| path.file_name().and_then(|file| file.to_str()) == Some(*name));
        let Some(first) = hits.next() else {
            continue;
        };
        if hits.next().is_some() {
            return Err(format!(
                "lora directory {} has more than one {name}",
                root.display()
            ));
        }
        return Ok(first.clone());
    }
    Err(format!(
        "lora directory {} has {} safetensors files",
        root.display(),
        files.len()
    ))
}

fn collect_safetensors(path: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_safetensors(&path, files);
        } else if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
        {
            files.push(path);
        }
    }
}

/// Installed weight for one registry LoRA card. `Ok(None)` means that version is not on disk.
pub fn registry_lora_weight(
    workspace: &Path,
    entry: &Value,
    version: Option<&str>,
) -> Result<Option<PathBuf>, String> {
    let versions = entry
        .pointer("/distribution/versions")
        .and_then(|value| value.as_object())
        .ok_or_else(|| "lora card has no versions".to_string())?;
    let spec = if let Some(version) = version.filter(|version| !version.is_empty()) {
        versions
            .get(version)
            .cloned()
            .ok_or_else(|| format!("lora version {version} is not published"))?
    } else {
        let ready: Vec<Value> = versions
            .values()
            .filter(|spec| version_spec_ready(workspace, spec))
            .cloned()
            .collect();
        if ready.is_empty() {
            return Ok(None);
        }
        if let Some(spec) = ready
            .iter()
            .find(|spec| spec.get("default").and_then(|value| value.as_bool()) == Some(true))
        {
            spec.clone()
        } else if ready.len() == 1 {
            ready.into_iter().next().expect("one ready version")
        } else {
            return Err("lora has more than one installed version; select id:version".into());
        }
    };
    let local = spec
        .get("local_path")
        .and_then(|value| value.as_str())
        .filter(|path| !path.is_empty())
        .ok_or_else(|| "lora version has no local_path".to_string())?;
    let root = resolve_model_path(workspace, local);
    if !version_ready(&root) {
        return Ok(None);
    }
    lora_weight_file(&root).map(Some)
}

fn version_spec_ready(workspace: &Path, spec: &Value) -> bool {
    spec.get("local_path")
        .and_then(|value| value.as_str())
        .is_some_and(|path| version_ready(&resolve_model_path(workspace, path)))
}

/// Resolve a model id to its installed bundle. `id:version` pins the tier.
/// A bare id picks the first **installed** tier, so a registry `default` the
/// user never downloaded cannot win over one they did. Product preference is
/// safetensors/MLX; GGUF tiers are only used when that is what is installed
/// (or explicitly requested).
pub fn resolve_bundle(workspace: &Path, registry: &Path, model: &str) -> Result<PathBuf, String> {
    let catalog = Catalog::open(registry).map_err(|e| e.to_string())?;
    let (bare, requested) = match model.split_once(':') {
        Some((bare, version)) if !bare.is_empty() && !version.is_empty() => (bare, Some(version)),
        _ => (model, None),
    };
    if let Some(version) = requested {
        if danmo_catalog::GGUF_VERSION_KEYS.contains(&version) {
            let path = catalog.gguf_local_path(model).map_err(|e| e.to_string())?;
            return Ok(resolve_model_path(workspace, &path));
        }
        return safetensors_bundle(workspace, catalog.root(), bare, version);
    }
    if let Some(version) = installed_safetensors_version(workspace, catalog.root(), bare) {
        return safetensors_bundle(workspace, catalog.root(), bare, &version);
    }
    if let Some(version) = installed_gguf_version(workspace, catalog.root(), bare) {
        let path = catalog
            .gguf_local_path(&format!("{bare}:{version}"))
            .map_err(|e| e.to_string())?;
        return Ok(resolve_model_path(workspace, &path));
    }
    Err(format!(
        "{bare} has no installed weight bundle under the workspace (checked safetensors/MLX and GGUF tiers)"
    ))
}

fn safetensors_bundle(
    workspace: &Path,
    root: &Value,
    model: &str,
    version: &str,
) -> Result<PathBuf, String> {
    let path = version_local_path(root, model, version)?;
    let resolved = resolve_model_path(workspace, &path);
    if danmo_engine::is_safetensors_bundle(&resolved) {
        return Ok(resolved);
    }
    Err(format!(
        "{model}:{version} is not an installed safetensors bundle at {}",
        resolved.display()
    ))
}

fn version_local_path(root: &Value, model: &str, version: &str) -> Result<String, String> {
    root.pointer(&format!(
        "/models/{model}/distribution/versions/{version}/local_path"
    ))
    .and_then(|value| value.as_str())
    .filter(|path| !path.is_empty())
    .map(|path| path.to_string())
    .ok_or_else(|| format!("{model}:{version} has no local_path"))
}

fn installed_safetensors_version(workspace: &Path, root: &Value, model_id: &str) -> Option<String> {
    let versions = root
        .pointer(&format!("/models/{model_id}/distribution/versions"))?
        .as_object()?;
    let ready = |key: &str| {
        let path = versions
            .get(key)
            .and_then(|version| version.get("local_path"))
            .and_then(|value| value.as_str())
            .filter(|path| !path.is_empty())?;
        {
            let resolved = resolve_model_path(workspace, path);
            danmo_engine::is_safetensors_bundle(&resolved).then_some(key.to_string())
        }
    };
    for key in danmo_catalog::SAFETENSORS_VERSION_KEYS {
        if versions
            .get(*key)
            .and_then(|version| version.get("default"))
            .and_then(|value| value.as_bool())
            == Some(true)
        {
            if let Some(key) = ready(key) {
                return Some(key);
            }
        }
    }
    danmo_catalog::SAFETENSORS_VERSION_KEYS
        .iter()
        .find_map(|key| ready(key))
}

/// First tier whose directory is a ready GGUF bundle, in bf16 / q8_0 / q4_k order.
pub fn installed_gguf_version(workspace: &Path, root: &Value, model_id: &str) -> Option<String> {
    let versions = root
        .pointer(&format!("/models/{model_id}/distribution/versions"))?
        .as_object()?;
    for key in danmo_catalog::GGUF_VERSION_KEYS {
        let Some(path) = versions
            .get(*key)
            .and_then(|v| v.get("local_path"))
            .and_then(|v| v.as_str())
            .filter(|path| !path.is_empty())
        else {
            continue;
        };
        if version_ready(&resolve_model_path(workspace, path)) {
            return Some((*key).to_string());
        }
    }
    None
}
