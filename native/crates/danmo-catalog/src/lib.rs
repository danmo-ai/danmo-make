//! Registry reader. Product catalog stays `default_config/models_registry.json`.

use serde_json::{json, Value};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CatalogError {
    #[error("registry not readable at {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("registry JSON is invalid: {0}")]
    Json(#[from] serde_json::Error),
    #[error("model {0} is not in the registry")]
    UnknownModel(String),
    #[error("model {model} has no family")]
    MissingFamily { model: String },
    #[error(
        "{model} has no GGUF download (bf16, q8_0, q4_k). Re-download; safetensors and MLX weights are not loaded. Registry versions: {found}"
    )]
    NoGguf { model: String, found: String },
    #[error("model {model} GGUF version {version} has no local_path")]
    MissingPath { model: String, version: String },
    #[error("{0}")]
    Invalid(String),
}

pub const GGUF_VERSION_KEYS: &[&str] = &["bf16", "q8_0", "q4_k"];

/// Published safetensors / MLX version keys. Resolution still requires the
/// directory to be a safetensors image bundle; these keys are not GGUF tiers.
pub const SAFETENSORS_VERSION_KEYS: &[&str] = &[
    "fp16", "mlx-q8", "mlx-q4", "int8", "int4", "mlx-bf16", "bf16",
];

#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub id: String,
    pub family: String,
    pub media: String,
    pub backends: Vec<String>,
}

pub struct Catalog {
    root: Value,
}

impl Catalog {
    pub fn open(path: &Path) -> Result<Self, CatalogError> {
        let text = std::fs::read_to_string(path).map_err(|source| CatalogError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Ok(Self {
            root: serde_json::from_str(&text)?,
        })
    }

    pub fn from_value(root: Value) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Value {
        &self.root
    }

    pub fn model(&self, id: &str) -> Result<ModelEntry, CatalogError> {
        let bare = id.split(':').next().unwrap_or(id);
        let models = self
            .root
            .get("models")
            .and_then(|m| m.as_object())
            .ok_or_else(|| CatalogError::UnknownModel(id.to_string()))?;
        let entry = models
            .get(bare)
            .ok_or_else(|| CatalogError::UnknownModel(id.to_string()))?;
        let runtime = entry.get("runtime");
        let catalog = entry.get("catalog");
        let family = runtime
            .and_then(|r| r.get("family"))
            .and_then(|v| v.as_str())
            .or_else(|| entry.get("family").and_then(|v| v.as_str()))
            .ok_or_else(|| CatalogError::MissingFamily {
                model: bare.to_string(),
            })?
            .to_string();
        let media = catalog
            .and_then(|c| c.get("media"))
            .and_then(|v| v.as_str())
            .or_else(|| entry.get("media").and_then(|v| v.as_str()))
            .unwrap_or("image")
            .to_string();
        let backends = runtime
            .and_then(|r| r.get("backends"))
            .or_else(|| entry.get("backends"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        Ok(ModelEntry {
            id: bare.to_string(),
            family,
            media,
            backends,
        })
    }

    /// Install path for a published GGUF tier. `id:version` selects that download.
    /// Legacy MLX version keys are not returned.
    pub fn gguf_local_path(&self, id: &str) -> Result<String, CatalogError> {
        let (bare, requested) = split_model_version(id);
        let entry = self
            .root
            .get("models")
            .and_then(|m| m.as_object())
            .and_then(|models| models.get(bare))
            .ok_or_else(|| CatalogError::UnknownModel(id.to_string()))?;
        let versions = entry
            .pointer("/distribution/versions")
            .and_then(|v| v.as_object());
        let found = versions
            .map(|map| map.keys().cloned().collect::<Vec<_>>().join(", "))
            .unwrap_or_else(|| "(none)".into());
        let versions = versions.ok_or_else(|| CatalogError::NoGguf {
            model: bare.to_string(),
            found: found.clone(),
        })?;
        let present: Vec<&str> = GGUF_VERSION_KEYS
            .iter()
            .copied()
            .filter(|key| versions.contains_key(*key))
            .collect();
        if present.is_empty() {
            return Err(CatalogError::NoGguf {
                model: bare.to_string(),
                found,
            });
        }
        let key = if let Some(version) = requested {
            if !GGUF_VERSION_KEYS.contains(&version) || !versions.contains_key(version) {
                return Err(CatalogError::NoGguf {
                    model: id.to_string(),
                    found,
                });
            }
            version
        } else {
            present
                .iter()
                .copied()
                .find(|key| {
                    versions
                        .get(*key)
                        .and_then(|ver| ver.get("default"))
                        .and_then(|v| v.as_bool())
                        == Some(true)
                })
                .unwrap_or(present[0])
        };
        let local = versions
            .get(key)
            .and_then(|ver| ver.get("local_path"))
            .and_then(|v| v.as_str())
            .filter(|path| !path.is_empty())
            .ok_or_else(|| CatalogError::MissingPath {
                model: bare.to_string(),
                version: key.to_string(),
            })?;
        Ok(local.to_string())
    }

    /// Registry served to the UI.
    ///
    /// Schema v3 is projected the way main's `build_catalog_response` does:
    /// each model is flattened (ui profile + catalog + runtime + actions +
    /// every published version), and `_index` carries the API action names.
    /// Older documents keep the GGUF-only lift used by the ggml engine tests.
    pub fn public_registry(&self) -> Result<Value, CatalogError> {
        if schema_version(&self.root) >= 3 {
            return project_v3(&self.root);
        }
        Ok(self.legacy_public_registry())
    }

    fn legacy_public_registry(&self) -> Value {
        let mut root = self.root.clone();
        let Some(models) = root.get_mut("models").and_then(|m| m.as_object_mut()) else {
            return root;
        };
        for model in models.values_mut() {
            if let Some(versions) = model
                .pointer_mut("/distribution/versions")
                .and_then(|v| v.as_object_mut())
            {
                versions.retain(|key, _| GGUF_VERSION_KEYS.contains(&key.as_str()));
            }
            lift_model_view(model);
        }
        root
    }
}

fn schema_version(root: &Value) -> i64 {
    match root.get("schema_version") {
        Some(Value::Number(number)) => number.as_i64().unwrap_or(2),
        Some(Value::String(text)) => text.parse().unwrap_or(2),
        _ => 2,
    }
}

fn project_v3(root: &Value) -> Result<Value, CatalogError> {
    let models = root
        .get("models")
        .and_then(|value| value.as_object())
        .ok_or_else(|| CatalogError::Invalid("'models' must be an object".into()))?;
    let mut expanded = serde_json::Map::new();
    let mut index = serde_json::Map::new();
    for (model_id, raw) in models {
        let flat = flatten_v3_model(model_id, raw, root)?;
        index.insert(model_id.clone(), index_entry(&flat));
        expanded.insert(model_id.clone(), flat);
    }
    let mut response = serde_json::Map::new();
    response.insert("schema_version".into(), Value::from(schema_version(root)));
    response.insert(
        "engines".into(),
        root.get("engines").cloned().unwrap_or_else(|| json!({})),
    );
    response.insert(
        "categories".into(),
        root.get("categories").cloned().unwrap_or_else(|| json!({})),
    );
    response.insert("models".into(), Value::Object(expanded));
    response.insert("_index".into(), Value::Object(index));
    response.insert(
        "ui_profiles".into(),
        root.get("ui_profiles")
            .cloned()
            .unwrap_or_else(|| json!({})),
    );
    response.insert(
        "families".into(),
        root.get("families").cloned().unwrap_or_else(|| json!({})),
    );
    Ok(Value::Object(response))
}

/// Project one v3 model to the flat record the Vue create page and model library read.
fn flatten_v3_model(model_id: &str, record: &Value, doc: &Value) -> Result<Value, CatalogError> {
    let record = record.as_object().ok_or_else(|| {
        CatalogError::Invalid(format!("Model {model_id:?}: entry must be an object"))
    })?;
    let ui_profiles = doc.get("ui_profiles").and_then(|value| value.as_object());
    let catalog = record.get("catalog").and_then(|value| value.as_object());
    let runtime = record.get("runtime").and_then(|value| value.as_object());
    let ui = record.get("ui").and_then(|value| value.as_object());
    let distribution = record
        .get("distribution")
        .and_then(|value| value.as_object());

    let mut merged = serde_json::Map::new();
    if let Some(extends) = ui.and_then(|ui| ui.get("extends")).and_then(|v| v.as_str()) {
        let extends = extends.trim();
        if !extends.is_empty() {
            if let Some(profile) = ui_profiles
                .and_then(|profiles| profiles.get(extends))
                .and_then(|value| value.as_object())
            {
                merged = profile.clone();
            }
        }
    }
    if let Some(catalog) = catalog {
        for (key, value) in catalog {
            merged.insert(key.clone(), value.clone());
        }
    }

    let family = runtime
        .and_then(|runtime| runtime.get("family"))
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|family| !family.is_empty())
        .ok_or_else(|| {
            CatalogError::Invalid(format!(
                "Model {model_id:?}: runtime.family is required in schema v3"
            ))
        })?;
    merged.insert("family".into(), Value::String(family.to_string()));

    if let Some(backends) = runtime
        .and_then(|runtime| runtime.get("backends"))
        .and_then(|value| value.as_array())
        .filter(|rows| !rows.is_empty())
    {
        merged.insert(
            "backends".into(),
            Value::Array(
                backends
                    .iter()
                    .map(|value| Value::String(value.as_str().unwrap_or("").to_string()))
                    .collect(),
            ),
        );
    }

    merged.insert(
        "actions".into(),
        record.get("actions").cloned().unwrap_or_else(|| json!({})),
    );
    if let Some(versions) = distribution
        .and_then(|dist| dist.get("versions"))
        .filter(|value| value.is_object())
    {
        merged.insert("versions".into(), versions.clone());
    }
    if let Some(distribution) = distribution {
        if let Some(dependencies) = distribution.get("dependencies") {
            merged.insert("dependencies".into(), dependencies.clone());
        }
    }

    let mut params = merged
        .get("parameters")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    if let Some(ui_params) = ui
        .and_then(|ui| ui.get("parameters"))
        .filter(|value| value.is_object())
    {
        params = deep_merge(&params, ui_params);
    }
    if let Some(overrides) = runtime
        .and_then(|runtime| runtime.get("overrides"))
        .and_then(|value| value.as_object())
    {
        let Some(params_obj) = params.as_object_mut() else {
            return Err(CatalogError::Invalid(format!(
                "Model {model_id:?}: parameters must be an object"
            )));
        };
        for (key, value) in overrides {
            params_obj.insert(key.clone(), value.clone());
        }
    }
    merged.insert("parameters".into(), params);
    Ok(Value::Object(merged))
}

fn deep_merge(base: &Value, override_value: &Value) -> Value {
    let (Some(base_obj), Some(over_obj)) = (base.as_object(), override_value.as_object()) else {
        return override_value.clone();
    };
    let mut out = base_obj.clone();
    for (key, value) in over_obj {
        if key == "profile" {
            continue;
        }
        if out.get(key).is_some_and(|existing| existing.is_object()) && value.is_object() {
            let merged = deep_merge(&out[key], value);
            out.insert(key.clone(), merged);
            continue;
        }
        out.insert(key.clone(), value.clone());
    }
    Value::Object(out)
}

fn index_entry(flat: &Value) -> Value {
    let media = media_from_record(flat);
    let family = flat
        .get("family")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string();
    let engine = flat
        .get("engine")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string();
    let mut actions = api_actions(flat.get("actions"), &media);
    actions.sort();
    json!({
        "media": media,
        "family": family,
        "engine": engine,
        "actions": actions,
    })
}

fn media_from_record(raw: &Value) -> String {
    let media = raw
        .get("media")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    if matches!(media, "image" | "video" | "audio" | "llm") {
        return media.to_string();
    }
    if raw.get("category").and_then(|value| value.as_str()) == Some("video_models") {
        "video".into()
    } else {
        "image".into()
    }
}

/// Registry verbs → the API action names main puts in `_index`.
fn api_actions(actions: Option<&Value>, media: &str) -> Vec<String> {
    let Some(actions) = actions.and_then(|value| value.as_object()) else {
        return Vec::new();
    };
    let has = |key: &str| actions.get(key).is_some_and(|value| !value.is_null());
    let mut out = Vec::new();
    match media {
        "video" => {
            if has("create") {
                out.push("generate".into());
            }
            if has("animate") {
                out.push("edit".into());
            }
            if has("upscale") {
                out.push("upscale".into());
            }
            if has("avatar") {
                out.push("avatar".into());
            }
            if has("avatar_script") {
                out.push("avatar_script".into());
            }
        }
        "audio" => {
            if has("create") {
                out.push("create_music".into());
            }
            if has("cover") || has("repaint") {
                out.push("edit".into());
            }
        }
        "llm" => {
            if has("chat") {
                out.push("chat".into());
            }
            if has("enhance") {
                out.push("enhance".into());
            }
            if has("describe") {
                out.push("describe".into());
            }
        }
        _ => {
            if has("create") {
                out.push("generate".into());
            }
            if has("rewrite") || has("retouch") || has("extend") {
                out.push("edit".into());
            }
            if has("upscale") {
                out.push("upscale".into());
            }
        }
    }
    out
}

fn split_model_version(id: &str) -> (&str, Option<&str>) {
    match id.split_once(':') {
        Some((bare, version)) if !bare.is_empty() && !version.is_empty() => (bare, Some(version)),
        _ => (id, None),
    }
}

/// Copy the fields the Vue create page reads onto the model object.
fn lift_model_view(model: &mut Value) {
    let fields = [
        ("media", "/catalog/media"),
        ("category", "/catalog/category"),
        ("name", "/catalog/name"),
        ("description", "/catalog/description"),
        ("recommended", "/catalog/recommended"),
        ("commercial_use_allowed", "/catalog/commercial_use_allowed"),
        ("engine", "/catalog/engine"),
        ("successor", "/catalog/successor"),
        ("distilled_from", "/catalog/distilled_from"),
        ("distilled_variant", "/catalog/distilled_variant"),
        ("family", "/runtime/family"),
        ("parameters", "/ui/parameters"),
        ("versions", "/distribution/versions"),
    ];
    let copied: Vec<(String, Value)> = fields
        .iter()
        .filter_map(|(key, pointer)| {
            model
                .pointer(pointer)
                .filter(|value| !value.is_null())
                .cloned()
                .map(|value| ((*key).to_string(), value))
        })
        .collect();
    let Some(obj) = model.as_object_mut() else {
        return;
    };
    for (key, value) in copied {
        obj.insert(key, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_versions_are_not_installable() {
        let catalog = Catalog::from_value(json!({
            "models": {
                "flux2-klein-4b": {
                    "distribution": {
                        "versions": {
                            "fp16": { "local_path": "models/old-fp16" },
                            "mlx-q4": { "local_path": "models/old-q4" }
                        }
                    }
                }
            }
        }));
        let err = catalog.gguf_local_path("flux2-klein-4b").unwrap_err();
        assert!(err.to_string().contains("Re-download"), "{err}");
        let public = catalog.public_registry().unwrap();
        let versions = public
            .pointer("/models/flux2-klein-4b/distribution/versions")
            .unwrap();
        assert!(versions.as_object().unwrap().is_empty());
    }

    #[test]
    fn gguf_default_version_is_selected() {
        let catalog = Catalog::from_value(json!({
            "models": {
                "flux2-klein-4b": {
                    "distribution": {
                        "versions": {
                            "bf16": { "local_path": "models/bf16" },
                            "q4_k": { "local_path": "models/q4", "default": true }
                        }
                    }
                }
            }
        }));
        assert_eq!(
            catalog.gguf_local_path("flux2-klein-4b").unwrap(),
            "models/q4"
        );
        assert_eq!(
            catalog.gguf_local_path("flux2-klein-4b:bf16").unwrap(),
            "models/bf16"
        );
        assert_eq!(
            catalog.gguf_local_path("flux2-klein-4b:q4_k").unwrap(),
            "models/q4"
        );
    }

    #[test]
    fn public_registry_lifts_fields_the_ui_reads() {
        let catalog = Catalog::from_value(json!({
            "models": {
                "flux2-klein-4b": {
                    "catalog": {
                        "media": "image",
                        "name": { "zh": "K", "en": "K" },
                        "recommended": true
                    },
                    "ui": { "parameters": { "steps": { "default": 4 } } },
                    "runtime": { "family": "flux2" },
                    "distribution": {
                        "versions": {
                            "bf16": { "local_path": "models/bf16", "default": true },
                            "fp16": { "local_path": "models/old" }
                        }
                    }
                }
            }
        }));
        let public = catalog.public_registry().unwrap();
        let model = &public["models"]["flux2-klein-4b"];
        assert_eq!(model["media"], "image");
        assert_eq!(model["family"], "flux2");
        assert_eq!(model["parameters"]["steps"]["default"], 4);
        assert!(model["versions"]["bf16"].is_object());
        assert!(model["versions"].get("fp16").is_none());
        assert!(model["distribution"]["versions"].get("fp16").is_none());
    }

    #[test]
    fn v3_registry_keeps_every_version_and_merges_the_ui_profile() {
        let catalog = Catalog::from_value(json!({
            "schema_version": 3,
            "engines": { "danqing-image": { "media": "image" } },
            "categories": { "base_models": { "icon": "x" } },
            "ui_profiles": {
                "image_dit_standard": {
                    "engine": "danqing-image",
                    "category": "base_models",
                    "parameters": {
                        "steps": { "default": 20, "min": 1, "max": 50 },
                        "lora_support": true
                    }
                }
            },
            "families": { "z_image": { "paradigm": "flow_matching" } },
            "models": {
                "z-image-turbo": {
                    "catalog": {
                        "media": "image",
                        "name": { "zh": "Z-Image-Turbo", "en": "Z-Image-Turbo" },
                        "category": "base_models",
                        "engine": "danqing-image",
                        "recommended": true
                    },
                    "runtime": {
                        "family": "z_image",
                        "backends": ["mlx"],
                        "overrides": { "supports_guidance": false }
                    },
                    "actions": { "create": {}, "rewrite": {} },
                    "ui": {
                        "extends": "image_dit_standard",
                        "parameters": {
                            "steps": { "default": 9, "note": "turbo" }
                        }
                    },
                    "distribution": {
                        "versions": {
                            "fp16": { "local_path": "models/fp16", "default": true },
                            "mlx-q4": { "local_path": "models/q4" },
                            "mlx-q8": { "local_path": "models/q8" }
                        }
                    }
                }
            }
        }));
        let public = catalog.public_registry().unwrap();
        let model = &public["models"]["z-image-turbo"];
        assert!(model.get("catalog").is_none());
        assert_eq!(model["media"], "image");
        assert_eq!(model["family"], "z_image");
        assert_eq!(model["name"]["zh"], "Z-Image-Turbo");
        assert_eq!(model["parameters"]["steps"]["default"], 9);
        assert_eq!(model["parameters"]["steps"]["min"], 1);
        assert_eq!(model["parameters"]["steps"]["note"], "turbo");
        assert_eq!(model["parameters"]["lora_support"], true);
        assert_eq!(model["parameters"]["supports_guidance"], false);
        assert!(model["versions"]["fp16"].is_object());
        assert!(model["versions"]["mlx-q4"].is_object());
        assert!(model["versions"]["mlx-q8"].is_object());
        assert_eq!(model["actions"]["create"], json!({}));
        assert_eq!(public["_index"]["z-image-turbo"]["family"], "z_image");
        assert_eq!(
            public["_index"]["z-image-turbo"]["actions"],
            json!(["edit", "generate"])
        );
        assert!(public["ui_profiles"]["image_dit_standard"].is_object());
        assert!(public["families"]["z_image"].is_object());
    }

    #[test]
    fn factory_registry_projects_z_image_turbo_for_the_ui() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../default_config/models_registry.json");
        let catalog = Catalog::open(&path).expect("factory registry");
        let public = catalog.public_registry().expect("v3 projection");
        let model = &public["models"]["z-image-turbo"];
        assert_eq!(model["media"], "image");
        assert_eq!(model["family"], "z_image");
        assert_eq!(model["category"], "base_models");
        assert_eq!(model["name"]["en"], "Z-Image-Turbo");
        assert_eq!(model["parameters"]["steps"]["default"], 9);
        assert_eq!(model["parameters"]["supports_guidance"], false);
        assert_eq!(model["parameters"]["lora_support"], true);
        assert_eq!(
            model["versions"]["fp16"]["local_path"],
            "models/Image/z-image-turbo-fp16"
        );
        assert_eq!(
            model["versions"]["mlx-q4"]["repo_id"],
            "OsaurusAI/Z-Image-Turbo-mflux-4bit"
        );
        assert_eq!(
            model["versions"]["mlx-q8"]["repo_id"],
            "OsaurusAI/Z-Image-Turbo-mflux-8bit"
        );
        assert!(model.get("catalog").is_none());
        let actions = public["_index"]["z-image-turbo"]["actions"]
            .as_array()
            .unwrap();
        assert!(actions.iter().any(|action| action == "generate"));
        assert!(actions.iter().any(|action| action == "edit"));
        assert!(public["models"].as_object().unwrap().len() > 50);
        assert!(public["categories"]["base_models"].is_object());
    }
}
