//! Registry reader. Product catalog stays `default_config/models_registry.json`.

use serde_json::Value;
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
}

pub const GGUF_VERSION_KEYS: &[&str] = &["bf16", "q8_0", "q4_k"];

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

    /// Install path for a published GGUF tier. Legacy MLX version keys are not returned.
    pub fn gguf_local_path(&self, id: &str) -> Result<String, CatalogError> {
        let bare = id.split(':').next().unwrap_or(id);
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
        let key = present
            .iter()
            .copied()
            .find(|key| {
                versions
                    .get(*key)
                    .and_then(|ver| ver.get("default"))
                    .and_then(|v| v.as_bool())
                    == Some(true)
            })
            .unwrap_or(present[0]);
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

    /// Registry served to the UI. Version entries other than bf16 / q8_0 / q4_k are dropped.
    pub fn public_registry(&self) -> Value {
        let mut root = self.root.clone();
        let Some(models) = root.get_mut("models").and_then(|m| m.as_object_mut()) else {
            return root;
        };
        for model in models.values_mut() {
            let Some(versions) = model
                .pointer_mut("/distribution/versions")
                .and_then(|v| v.as_object_mut())
            else {
                continue;
            };
            versions.retain(|key, _| GGUF_VERSION_KEYS.contains(&key.as_str()));
        }
        root
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
        let public = catalog.public_registry();
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
    }
}
