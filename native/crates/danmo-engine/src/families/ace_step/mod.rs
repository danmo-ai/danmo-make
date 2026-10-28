//! ACE-Step 1.5 music DiT (GenerationFacade).
//!
//! This slice locks the XL weight table against acestep.cpp `dit.h` and main
//! `AceStepConfig`. The denoise graph, text tower, 5Hz LM, VAE, and cover path
//! are not built. Create still fails loud after a successful layout probe.

mod weights;

use crate::weights::{GgufWeights, WeightError};
use std::path::{Path, PathBuf};
pub use weights::{
    check_layout, config_from_weights, layout_matches, weight_layout, AceStepConfig, TIME_FREQ_DIM,
};

/// Find the DiT GGUF and confirm the XL / smoke layout.
pub fn probe_layout(bundle: &Path) -> Result<AceStepConfig, WeightError> {
    let path = dit_gguf(bundle)?;
    let loaded = GgufWeights::open(&path)?;
    let config = config_from_weights(&loaded)?;
    check_layout(&loaded, config)?;
    Ok(config)
}

fn dit_gguf(bundle: &Path) -> Result<PathBuf, WeightError> {
    if !bundle.is_dir() {
        return Err(WeightError::Bundle {
            path: bundle.display().to_string(),
            detail: "directory is missing".into(),
        });
    }
    let mut matches = Vec::new();
    let entries = std::fs::read_dir(bundle).map_err(|err| WeightError::Bundle {
        path: bundle.display().to_string(),
        detail: err.to_string(),
    })?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
            continue;
        }
        let Ok(weights) = GgufWeights::open(&path) else {
            continue;
        };
        if weights.contains("decoder.proj_in.1.weight") {
            matches.push(path);
        }
    }
    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => Err(WeightError::NotBuilt(
            "ACE-Step DiT GGUF with decoder.proj_in.1.weight is missing".into(),
        )),
        _ => Err(WeightError::NotBuilt(format!(
            "ACE-Step bundle has {} DiT GGUF files; keep one",
            matches.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        config_from_weights, dit_gguf, layout_matches, probe_layout, weight_layout, AceStepConfig,
        TIME_FREQ_DIM,
    };
    use crate::weights::{write_f32_nd, F32Nd, GgufWeights};
    use std::path::Path;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-ace-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_dit(dir: &Path, config: AceStepConfig) {
        let owned: Vec<(String, [i64; 4], Vec<f32>)> = weight_layout(config)
            .into_iter()
            .map(|row| {
                let n = row.ne.iter().product::<i64>() as usize;
                (row.name, row.ne, vec![0f32; n])
            })
            .collect();
        let tensors: Vec<F32Nd<'_>> = owned
            .iter()
            .map(|(name, ne, data)| F32Nd {
                name,
                ne: *ne,
                data,
            })
            .collect();
        write_f32_nd(&dir.join("acestep-v15-smoke.gguf"), &tensors).unwrap();
    }

    #[test]
    fn xl_layout_has_thirty_two_layers_and_the_acestep_names() {
        let config = AceStepConfig::xl(false);
        assert_eq!(config.hidden, 2560);
        assert_eq!(config.layers, 32);
        assert_eq!(config.intermediate, 9728);
        assert_eq!(config.heads, 32);
        assert_eq!(config.kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.in_channels, 192);
        assert_eq!(config.out_channels, 64);
        assert_eq!(config.patch, 2);
        assert_eq!(config.sliding_window, 128);
        assert_eq!(config.condition_dim, 2048);
        assert_eq!(TIME_FREQ_DIM, 256);
        let names: Vec<_> = weight_layout(config)
            .into_iter()
            .map(|row| row.name)
            .collect();
        assert!(names
            .iter()
            .any(|name| name == "decoder.layers.0.self_attn.q_proj.weight"));
        assert!(names
            .iter()
            .any(|name| name == "decoder.layers.31.mlp.down_proj.weight"));
        assert!(names.iter().any(|name| name == "decoder.proj_in.1.weight"));
        assert!(names
            .iter()
            .any(|name| name == "decoder.time_embed_r.time_proj.weight"));
        assert!(names.iter().all(|name| !name.starts_with("layers.")));
        assert!(!names.iter().any(|name| name.contains("null_condition")));
        assert_eq!(
            names
                .iter()
                .filter(|name| name.contains("self_attn.q_proj.weight"))
                .count(),
            32
        );
    }

    #[test]
    fn smoke_probe_accepts_the_layout_then_refuses_generate() {
        let dir = scratch("smoke");
        write_dit(&dir, AceStepConfig::smoke());
        let path = dit_gguf(&dir).unwrap();
        let loaded = GgufWeights::open(&path).unwrap();
        let config = config_from_weights(&loaded).unwrap();
        assert_eq!(config, AceStepConfig::smoke());
        assert!(layout_matches(&loaded, config));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn base_2b_layout_is_refused() {
        let dir = scratch("2b");
        write_dit(&dir, AceStepConfig::base_2b());
        let err = probe_layout(&dir).unwrap_err();
        assert!(
            err.to_string().contains("2B") && err.to_string().contains("XL"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_proj_in_fails_loud() {
        let dir = scratch("empty");
        let err = probe_layout(&dir).unwrap_err();
        assert!(
            err.to_string().contains("decoder.proj_in.1.weight"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn audio_create_refuses_after_a_valid_smoke_dit() {
        let dir = scratch("create");
        write_dit(&dir, AceStepConfig::smoke());
        let err = crate::pipeline::audio_create("ace_step", &dir).unwrap_err();
        assert!(err.to_string().contains("DiT graph is not built"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
