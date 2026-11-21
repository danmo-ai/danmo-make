//! Strict loader for the checkpoint-derived `bundle_config.json`.
//!
//! This is family configuration, not a shared bundle format. Every component
//! section remains JSON because the DiT, VAE, vocoder, upsampler, and Gemma
//! builders each interpret a different checkpoint schema.

use crate::weights::WeightError;
use serde_json::Value;
use std::path::Path;

pub const BUNDLE_CONFIG_NAME: &str = "bundle_config.json";

/// Affine linear quantization metadata from the MLX install hook / bundle config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quantization {
    pub bits: i32,
    pub group_size: i32,
}

#[derive(Debug, Clone)]
pub struct BundleConfig {
    pub transformer: Value,
    pub vae: Value,
    pub audio_vae: Value,
    pub vocoder: Value,
    pub upsampler: Value,
    pub gemma: Value,
    pub quantization: Option<Quantization>,
}

impl BundleConfig {
    pub fn load(bundle_root: &Path) -> Result<Self, WeightError> {
        let path = bundle_root.join(BUNDLE_CONFIG_NAME);
        if !path.is_file() {
            return Err(WeightError::Open {
                path: path.display().to_string(),
                detail: "LTX 2.5 bundle config is missing; the MLX install must produce bundle_config.json".into(),
            });
        }
        let text = std::fs::read_to_string(&path).map_err(|error| WeightError::Open {
            path: path.display().to_string(),
            detail: error.to_string(),
        })?;
        Self::parse(&text).map_err(|detail| WeightError::Open {
            path: path.display().to_string(),
            detail,
        })
    }

    fn parse(text: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(text)
            .map_err(|error| format!("LTX 2.5 bundle config is not valid JSON: {error}"))?;
        let root = value
            .as_object()
            .ok_or_else(|| "LTX 2.5 bundle config must be a JSON object".to_string())?;
        let quantization = match root.get("quantization") {
            Some(value) => {
                let quantization = value
                    .as_object()
                    .ok_or_else(|| "LTX 2.5 quantization metadata must be an object".to_string())?;
                let bits = quantization
                    .get("bits")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| "LTX 2.5 quantization bits missing".to_string())?;
                let group_size = quantization
                    .get("group_size")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| "LTX 2.5 quantization group_size missing".to_string())?;
                Some(Quantization {
                    bits: bits as i32,
                    group_size: group_size as i32,
                })
            }
            None => None,
        };
        Ok(Self {
            transformer: section(root, "transformer")?,
            vae: section(root, "vae")?,
            audio_vae: section(root, "audio_vae")?,
            vocoder: section(root, "vocoder")?,
            upsampler: section(root, "upsampler")?,
            gemma: section(root, "gemma")?,
            quantization,
        })
    }
}

fn section(root: &serde_json::Map<String, Value>, name: &str) -> Result<Value, String> {
    let value = root
        .get(name)
        .ok_or_else(|| format!("LTX 2.5 bundle config lacks the {name} section"))?;
    if !value.is_object() {
        return Err(format!(
            "LTX 2.5 bundle config section {name} must be a JSON object"
        ));
    }
    Ok(value.clone())
}

#[cfg(test)]
mod tests {
    use super::BundleConfig;
    use serde_json::{json, Value};

    fn valid() -> String {
        json!({
            "transformer": {"num_layers": 48},
            "vae": {"decoder_blocks": [[2, 2, 2]]},
            "audio_vae": {"mel_bins": 16},
            "vocoder": {"sampling_rate": 48000},
            "upsampler": {"in_channels": 128},
            "gemma": {"hidden_size": 3840},
            "quantization": {"bits": 4, "group_size": 32, "mode": "affine"}
        })
        .to_string()
    }

    #[test]
    fn parses_every_checkpoint_section_without_defaults() {
        let config = BundleConfig::parse(&valid()).unwrap();
        assert_eq!(config.transformer["num_layers"], 48);
        assert_eq!(config.audio_vae["mel_bins"], 16);
        assert_eq!(config.upsampler["in_channels"], 128);
        assert_eq!(config.gemma["hidden_size"], 3840);
    }

    #[test]
    fn absent_or_non_object_sections_fail() {
        let error = BundleConfig::parse("{}").err().unwrap();
        assert!(error.contains("transformer"), "{error}");
        let mut non_object: Value = serde_json::from_str(&valid()).unwrap();
        non_object["gemma"] = json!([]);
        let error = BundleConfig::parse(&non_object.to_string()).err().unwrap();
        assert!(error.contains("section gemma"), "{error}");
        assert!(BundleConfig::parse("[]").is_err());
        assert!(BundleConfig::parse("not json").is_err());
    }

    #[test]
    fn missing_file_names_the_required_bundle_config() {
        let root =
            std::env::temp_dir().join(format!("danmo-ltx25-missing-config-{}", std::process::id()));
        let error = BundleConfig::load(&root).err().unwrap().to_string();
        assert!(error.contains("bundle_config.json"), "{error}");
        assert!(error.contains("MLX install"), "{error}");
    }
}
