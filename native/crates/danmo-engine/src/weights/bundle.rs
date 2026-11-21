//! Installed bundle layouts.
//!
//! Product MLX paths are safetensors (image create tree, SeedVR2 flat files,
//! ACE-Step, ControlNet). [`assert_gguf_bundle`] / [`assert_hunyuan_bundle`]
//! remain only for families that still load a GGUF DiT; they never treat a
//! safetensors install as "legacy" or ask the user to re-download GGUF.

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

/// Hunyuan i2v bundle: the base GGUF bundle (GGUF DiT + Qwen2.5-VL text
/// encoder + video VAE + tokenizer) plus the SigLIP image encoder used for
/// the vision conditioning tokens.
pub struct HunyuanBundle {
    pub base: GgufBundle,
    pub image_encoder: Vec<PathBuf>,
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
    #[serde(default)]
    text_encoder: Option<FileList>,
    #[serde(default)]
    vae: Option<FileList>,
    #[serde(default)]
    image_encoder: Option<FileList>,
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

/// Safetensors image layout: `transformer/`, `text_encoder/`, and `vae/` each
/// hold a `.safetensors` file, and the tree has no GGUF. A GGUF bundle with
/// safetensors companions stays on the GGUF path.
pub fn is_safetensors_image_bundle(dir: &Path) -> bool {
    dir.is_dir()
        && !contains_gguf(dir)
        && ["transformer", "text_encoder", "vae"]
            .iter()
            .all(|role| contains_extension(&dir.join(role), "safetensors"))
}

/// SeedVR2 flat layout: DiT + VAE safetensors at the bundle root (no GGUF).
pub fn is_seedvr2_safetensors_bundle(dir: &Path) -> bool {
    if !dir.is_dir() || contains_gguf(dir) {
        return false;
    }
    let dit = dir.join("seedvr2_ema_7b_fp16.safetensors").is_file()
        || dir.join("seedvr2_ema_3b_fp16.safetensors").is_file();
    dit && dir.join("ema_vae_fp16.safetensors").is_file()
}

/// ControlNet (or similar) adapter directory: one or more top-level
/// `*.safetensors` files without the full image create layout.
pub fn is_controlnet_safetensors_dir(dir: &Path) -> bool {
    if !dir.is_dir()
        || is_safetensors_image_bundle(dir)
        || is_seedvr2_safetensors_bundle(dir)
        || contains_gguf(dir)
    {
        return false;
    }
    contains_extension(dir, "safetensors")
}

/// ACE-Step 1.5 safetensors tree: `vae/`, `Qwen3-Embedding-0.6B/`, and an XL
/// DiT directory (this directory or one child) with `model.safetensors.index.json`.
pub fn is_ace_step_safetensors_bundle(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let text = dir.join("Qwen3-Embedding-0.6B");
    let vae = dir.join("vae");
    text.join("config.json").is_file()
        && text.join("tokenizer.json").is_file()
        && contains_safetensors(&vae)
        && (dir.join("model.safetensors.index.json").is_file()
            || std::fs::read_dir(dir).ok().is_some_and(|entries| {
                entries
                    .flatten()
                    .any(|entry| entry.path().join("model.safetensors.index.json").is_file())
            }))
}

/// True when the directory is a product safetensors / MLX install (any known layout).
pub fn is_safetensors_bundle(dir: &Path) -> bool {
    is_safetensors_image_bundle(dir)
        || is_seedvr2_safetensors_bundle(dir)
        || is_ace_step_safetensors_bundle(dir)
        || is_controlnet_safetensors_dir(dir)
}

pub fn assert_gguf_bundle(dir: &Path) -> Result<GgufBundle, WeightError> {
    let (root, manifest) = read_gguf_manifest(dir)?;
    if !tokenizer_ready(&root) {
        return Err(WeightError::Bundle {
            path: root.display().to_string(),
            detail: "tokenizer is missing".into(),
        });
    }
    let text_encoder = manifest
        .files
        .text_encoder
        .as_ref()
        .ok_or_else(|| WeightError::Bundle {
            path: root.display().to_string(),
            detail: "text_encoder file list is missing".into(),
        })?;
    let vae = manifest
        .files
        .vae
        .as_ref()
        .ok_or_else(|| WeightError::Bundle {
            path: root.display().to_string(),
            detail: "vae file list is missing".into(),
        })?;
    Ok(GgufBundle {
        transformer: resolve_role(&root, "transformer", manifest.files.transformer.names())?,
        text_encoder: resolve_role(&root, "text_encoder", text_encoder.names())?,
        vae: resolve_role(&root, "vae", vae.names())?,
        quant: manifest.quant,
        root,
    })
}

/// The Hunyuan i2v bundle: everything [`assert_gguf_bundle`] checks plus the
/// mandatory SigLIP image encoder list.
pub fn assert_hunyuan_bundle(dir: &Path) -> Result<HunyuanBundle, WeightError> {
    let base = assert_gguf_bundle(dir)?;
    let (_, manifest) = read_gguf_manifest(dir)?;
    let image_encoder =
        manifest
            .files
            .image_encoder
            .as_ref()
            .ok_or_else(|| WeightError::Bundle {
                path: base.root.display().to_string(),
                detail: "image_encoder file list is missing".into(),
            })?;
    Ok(HunyuanBundle {
        image_encoder: resolve_role(&base.root, "image_encoder", image_encoder.names())?,
        base,
    })
}

fn read_gguf_manifest(dir: &Path) -> Result<(PathBuf, Manifest), WeightError> {
    let root = dir.to_path_buf();
    if !root.is_dir() {
        return Err(WeightError::Bundle {
            path: root.display().to_string(),
            detail: "directory is missing".into(),
        });
    }
    if is_safetensors_bundle(&root) {
        return Err(WeightError::Bundle {
            path: root.display().to_string(),
            detail: "directory is a safetensors/MLX bundle; this family still expects a GGUF DiT layout"
                .into(),
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
    if !contains_gguf(&root) {
        return Err(WeightError::Bundle {
            path: root.display().to_string(),
            detail: "no .gguf files under the bundle".into(),
        });
    }
    Ok((root, manifest))
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
        let ext = path.extension().and_then(|ext| ext.to_str());
        if !role_file(role, ext) {
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

fn role_file(role: &str, ext: Option<&str>) -> bool {
    match (role, ext) {
        ("transformer", Some("gguf")) => true,
        ("text_encoder" | "vae", Some("gguf" | "safetensors")) => true,
        ("image_encoder", Some("gguf" | "safetensors")) => true,
        _ => false,
    }
}

fn tokenizer_ready(root: &Path) -> bool {
    // Hugging Face `tokenizer.json` (Qwen3 / Flux / Z-Image) or Qwen2.5-VL
    // `vocab.json` + `merges.txt` under `tokenizer/`.
    if root.join("tokenizer.json").is_file()
        || root.join("tokenizer").join("tokenizer.json").is_file()
    {
        return true;
    }
    let tok = root.join("tokenizer");
    tok.join("vocab.json").is_file() && tok.join("merges.txt").is_file()
}

fn contains_safetensors(dir: &Path) -> bool {
    contains_extension(dir, "safetensors")
}

fn contains_gguf(dir: &Path) -> bool {
    contains_extension(dir, "gguf")
}

fn contains_extension(dir: &Path, extension: &str) -> bool {
    fn walk(dir: &Path, extension: &str, depth: usize) -> bool {
        if depth > 3 {
            return false;
        }
        let Ok(entries) = fs::read_dir(dir) else {
            return false;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some(extension) {
                return true;
            }
            if path.is_dir() && walk(&path, extension, depth + 1) {
                return true;
            }
        }
        false
    }
    walk(dir, extension, 0)
}
