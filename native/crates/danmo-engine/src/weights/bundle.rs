//! Installed bundle: `transformer.gguf`, `text_encoder.gguf`, `vae.gguf`,
//! tokenizer, and `bundle.manifest.json`. Old MLX directories are refused.

use super::gguf::write_f32_gguf;
use super::WeightError;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

pub struct GgufBundle {
    pub root: PathBuf,
    pub quant: String,
    pub transformer: Vec<PathBuf>,
    pub text_encoder: Vec<PathBuf>,
    pub vae: Vec<PathBuf>,
}

#[derive(Deserialize)]
struct Manifest {
    format: String,
    quant: String,
    files: ManifestFiles,
}

#[derive(Deserialize)]
struct ManifestFiles {
    transformer: FileList,
    text_encoder: FileList,
    vae: FileList,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum FileList {
    One(String),
    Many(Vec<String>),
}

impl FileList {
    fn names(&self) -> &[String] {
        match self {
            FileList::One(name) => std::slice::from_ref(name),
            FileList::Many(names) => names,
        }
    }
}

pub fn assert_gguf_bundle(dir: &Path) -> Result<GgufBundle, WeightError> {
    let root = dir.to_path_buf();
    if !root.is_dir() {
        return Err(WeightError::Bundle {
            path: root.display().to_string(),
            detail: "directory is missing".into(),
        });
    }
    let dir_name = root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if is_legacy_dir(dir_name) {
        return Err(WeightError::LegacyBundle {
            path: root.display().to_string(),
        });
    }
    if contains_safetensors(&root) {
        return Err(WeightError::LegacyBundle {
            path: root.display().to_string(),
        });
    }
    let manifest_path = root.join("bundle.manifest.json");
    let text = fs::read_to_string(&manifest_path).map_err(|err| WeightError::Bundle {
        path: root.display().to_string(),
        detail: format!("bundle.manifest.json: {err}"),
    })?;
    let manifest: Manifest = serde_json::from_str(&text).map_err(|err| WeightError::Bundle {
        path: root.display().to_string(),
        detail: format!("bundle.manifest.json: {err}"),
    })?;
    if manifest.format != "gguf" {
        return Err(WeightError::Bundle {
            path: root.display().to_string(),
            detail: format!("format is {}, expected gguf", manifest.format),
        });
    }
    if !matches!(manifest.quant.as_str(), "bf16" | "q8_0" | "q4_k") {
        return Err(WeightError::Bundle {
            path: root.display().to_string(),
            detail: format!("quant {} is not bf16, q8_0, or q4_k", manifest.quant),
        });
    }
    if !tokenizer_ready(&root) {
        return Err(WeightError::Bundle {
            path: root.display().to_string(),
            detail: "tokenizer.json is missing".into(),
        });
    }
    Ok(GgufBundle {
        transformer: resolve_role(&root, "transformer", manifest.files.transformer.names())?,
        text_encoder: resolve_role(&root, "text_encoder", manifest.files.text_encoder.names())?,
        vae: resolve_role(&root, "vae", manifest.files.vae.names())?,
        quant: manifest.quant,
        root,
    })
}

fn resolve_role(root: &Path, role: &str, names: &[String]) -> Result<Vec<PathBuf>, WeightError> {
    if names.is_empty() {
        return Err(WeightError::Bundle {
            path: root.display().to_string(),
            detail: format!("{role} file list is empty"),
        });
    }
    let mut paths = Vec::with_capacity(names.len());
    for name in names {
        if name.contains("..") || Path::new(name).is_absolute() {
            return Err(WeightError::Bundle {
                path: root.display().to_string(),
                detail: format!("{role} path {name} escapes the bundle"),
            });
        }
        let path = root.join(name);
        if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
            return Err(WeightError::NotGguf {
                path: path.display().to_string(),
            });
        }
        if !path.is_file() {
            return Err(WeightError::Bundle {
                path: root.display().to_string(),
                detail: format!("missing {name}"),
            });
        }
        paths.push(path);
    }
    Ok(paths)
}

fn tokenizer_ready(root: &Path) -> bool {
    root.join("tokenizer.json").is_file() || root.join("tokenizer").join("tokenizer.json").is_file()
}

fn is_legacy_dir(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("mlx-")
        || name.ends_with("-fp16")
        || name.ends_with("-int8")
        || name.ends_with("-int4")
        || name.ends_with("-fp8")
}

fn contains_safetensors(dir: &Path) -> bool {
    fn walk(dir: &Path, depth: usize) -> bool {
        if depth > 3 {
            return false;
        }
        let Ok(entries) = fs::read_dir(dir) else {
            return false;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
                return true;
            }
            if path.is_dir() && walk(&path, depth + 1) {
                return true;
            }
        }
        false
    }
    walk(dir, 0)
}

/// Tiny bf16-tier bundle whose DiT file contains `velocity.weight` under that exact name.
pub fn write_smoke_bundle(dir: &Path, dim: i64) -> Result<(), WeightError> {
    fs::create_dir_all(dir).map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;
    let n = (dim as usize).saturating_mul(dim as usize);
    let mut weight = vec![0f32; n];
    for i in 0..dim as usize {
        weight[i * dim as usize + i] = 0.1;
    }
    write_f32_gguf(
        &dir.join("transformer.gguf"),
        "velocity.weight",
        dim,
        dim,
        &weight,
    )?;
    write_f32_gguf(
        &dir.join("text_encoder.gguf"),
        "token_embed.weight",
        1,
        1,
        &[0.0],
    )?;
    write_f32_gguf(&dir.join("vae.gguf"), "decoder.conv.weight", 1, 1, &[1.0])?;
    fs::write(dir.join("tokenizer.json"), "{}\n").map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;
    let manifest = r#"{
  "format": "gguf",
  "quant": "bf16",
  "files": {
    "transformer": "transformer.gguf",
    "text_encoder": "text_encoder.gguf",
    "vae": "vae.gguf"
  }
}
"#;
    fs::write(dir.join("bundle.manifest.json"), manifest).map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;
    Ok(())
}
