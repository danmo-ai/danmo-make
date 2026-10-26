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
}

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
}
