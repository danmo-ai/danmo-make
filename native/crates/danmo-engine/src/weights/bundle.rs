//! Installed bundle: GGUF DiT, GGUF or safetensors text encoder and VAE,
//! tokenizer, and `bundle.manifest.json`. Old MLX directories are refused.

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

/// LTX bundle: the base GGUF bundle plus the audio VAE, the text-embedding
/// connectors, and (distilled stage 2) the latent upscaler.
pub struct LtxBundle {
    pub base: GgufBundle,
    pub audio_vae: Vec<PathBuf>,
    pub connectors: Vec<PathBuf>,
    pub upscaler: Vec<PathBuf>,
    pub stage2_transformer: Vec<PathBuf>,
}

/// Upscale bundle (SeedVR2): GGUF DiT plus a GGUF or safetensors VAE. There
/// is no text encoder and no tokenizer.
pub struct UpscaleBundle {
    pub root: PathBuf,
    pub quant: String,
    pub transformer: Vec<PathBuf>,
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
    audio_vae: Option<FileList>,
    #[serde(default)]
    connectors: Option<FileList>,
    #[serde(default)]
    upscaler: Option<FileList>,
    #[serde(default)]
    transformer_stage2: Option<FileList>,
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
                entries.flatten().any(|entry| {
                    entry.path().join("model.safetensors.index.json").is_file()
                })
            }))
}

pub fn assert_gguf_bundle(dir: &Path) -> Result<GgufBundle, WeightError> {
    let (root, manifest) = read_manifest(dir)?;
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

/// The LTX bundle: everything [`assert_gguf_bundle`] checks plus the audio
/// VAE, connectors, and (distilled) latent upscaler lists. Audio VAE and
/// connectors are mandatory; the upscaler is required only by the distilled
/// two-stage pipeline, so it stays optional here and the distilled session
/// asserts it.
pub fn assert_ltx_bundle(dir: &Path) -> Result<LtxBundle, WeightError> {
    let base = assert_gguf_bundle(dir)?;
    let (_, manifest) = read_manifest(dir)?;
    let audio_vae = manifest
        .files
        .audio_vae
        .as_ref()
        .ok_or_else(|| WeightError::Bundle {
            path: base.root.display().to_string(),
            detail: "audio_vae file list is missing".into(),
        })?;
    let connectors = manifest
        .files
        .connectors
        .as_ref()
        .ok_or_else(|| WeightError::Bundle {
            path: base.root.display().to_string(),
            detail: "connectors file list is missing".into(),
        })?;
    let upscaler = match &manifest.files.upscaler {
        Some(list) => resolve_role(&base.root, "upscaler", list.names())?,
        None => Vec::new(),
    };
    let stage2_transformer = match &manifest.files.transformer_stage2 {
        Some(list) => resolve_role(&base.root, "transformer_stage2", list.names())?,
        None => Vec::new(),
    };
    Ok(LtxBundle {
        audio_vae: resolve_role(&base.root, "audio_vae", audio_vae.names())?,
        connectors: resolve_role(&base.root, "connectors", connectors.names())?,
        upscaler,
        stage2_transformer,
        base,
    })
}

/// The Hunyuan i2v bundle: everything [`assert_gguf_bundle`] checks plus the
/// mandatory SigLIP image encoder list.
pub fn assert_hunyuan_bundle(dir: &Path) -> Result<HunyuanBundle, WeightError> {
    let base = assert_gguf_bundle(dir)?;
    let (_, manifest) = read_manifest(dir)?;
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

pub fn assert_upscale_bundle(dir: &Path) -> Result<UpscaleBundle, WeightError> {
    let (root, manifest) = read_manifest(dir)?;
    let vae = manifest
        .files
        .vae
        .as_ref()
        .ok_or_else(|| WeightError::Bundle {
            path: root.display().to_string(),
            detail: "vae file list is missing".into(),
        })?;
    Ok(UpscaleBundle {
        transformer: resolve_role(&root, "transformer", manifest.files.transformer.names())?,
        vae: resolve_role(&root, "vae", vae.names())?,
        quant: manifest.quant,
        root,
    })
}

fn read_manifest(dir: &Path) -> Result<(PathBuf, Manifest), WeightError> {
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
    if contains_safetensors(&root) && !contains_gguf(&root) {
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
        ("transformer_stage2", Some("gguf")) => true,
        ("text_encoder" | "vae", Some("gguf" | "safetensors")) => true,
        ("audio_vae" | "connectors" | "upscaler", Some("gguf" | "safetensors")) => true,
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

fn is_legacy_dir(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("mlx-")
        || name.ends_with("-fp16")
        || name.ends_with("-int8")
        || name.ends_with("-int4")
        || name.ends_with("-fp8")
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
