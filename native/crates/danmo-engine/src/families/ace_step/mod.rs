//! ACE-Step 1.5 music DiT (GenerationFacade).
//!
//! Full text2music create (acestep.cpp `pipeline-synth`): text encode → cond →
//! silence context → DiT Euler (APG CFG) → VAE → WAV. Cover-nofsq encodes
//! source audio into context; full cover runs FSQ roundtrip. 5Hz LM fails loud.

mod apg;
mod ca_capture;
mod cond;
mod cover;
mod create;
mod fsq;
mod fsq_quant;
mod graph;
mod lyrics;
mod lyrics_dtw;
mod repaint;
mod schedule;
mod text;
mod token_decode;
mod transformer;
mod vae;
mod vae_enc;
pub(crate) mod wav;
mod weights;

use crate::runtime::{BackendKind, GgmlContext};
use crate::weights::{GgufWeights, WeightError};
use std::path::{Path, PathBuf};
#[allow(unused_imports)]
pub use cover::{CoverMode, CoverRequest};
#[allow(unused_imports)]
pub use create::{duration_to_latent_frames, snap_latent_frames, CreateOutput, CreateRequest};
#[allow(unused_imports)]
pub use schedule::{
    sft_schedule, shift_schedule, turbo_schedule, SFT_DEFAULT_SHIFT, SFT_DEFAULT_STEPS,
    TURBO_SHIFT, TURBO_TIMESTEPS,
};
#[allow(unused_imports)]
pub use transformer::{prove_smoke_denoise, AceStepDit};
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

/// Prefer [`run_cover_edit`] with an explicit [`CoverRequest`].
pub fn run_cover(_bundle: &Path) -> Result<(), WeightError> {
    Err(WeightError::NotBuilt(
        "ACE-Step cover needs CoverRequest via audio.edit (task=cover, cover-nofsq, or repaint)".into(),
    ))
}

/// Cover / cover-nofsq / repaint: encode → optional FSQ → context → DiT → VAE.
pub fn run_cover_edit(
    bundle: &Path,
    request: &CoverRequest<'_>,
) -> Result<CreateOutput, WeightError> {
    if crate::weights::assert_gguf_bundle(bundle).is_ok() {
        return cover::run_full_cover(bundle, request);
    }
    Err(WeightError::NotBuilt(
        "ACE-Step cover needs a GGUF bundle with text encoder + VAE + FSQ weights".into(),
    ))
}

/// Alias kept for call sites that only want the no-FSQ path name.
#[allow(dead_code)]
pub fn run_cover_nofsq(
    bundle: &Path,
    request: &CoverRequest<'_>,
) -> Result<CreateOutput, WeightError> {
    run_cover_edit(bundle, request)
}

/// 5Hz LM lyric/caption expansion is not built (acestep.cpp `pipeline-lm`).
pub fn run_lm_expand(_bundle: &Path) -> Result<(), WeightError> {
    Err(WeightError::NotBuilt(
        "ACE-Step 5Hz LM expansion is not built (acestep.cpp pipeline-lm)".into(),
    ))
}

/// Product create. A full `bundle.manifest.json` (DiT + text + VAE + tokenizer)
/// runs the acestep.cpp text2music path. A DiT-only smoke card proves Euler
/// then refuses until companions are installed.
pub fn run_create(bundle: &Path, request: &CreateRequest<'_>) -> Result<CreateOutput, WeightError> {
    if crate::weights::assert_gguf_bundle(bundle).is_ok() {
        return create::run_full_create(bundle, request);
    }
    let config = probe_layout(bundle)?;
    if config == AceStepConfig::smoke() {
        let ctx = GgmlContext::new(BackendKind::Cpu)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        let _ = prove_smoke_denoise(&ctx, bundle, config)?;
        return Err(WeightError::NotBuilt(
            "ACE-Step needs text_encoder/, vae/, and tokenizer/ companions (install the registry components)".into(),
        ));
    }
    Err(WeightError::NotBuilt(
        "ACE-Step XL create needs a GGUF bundle with text encoder + VAE (see registry components)".into(),
    ))
}

pub(crate) fn dit_gguf(bundle: &Path) -> Result<PathBuf, WeightError> {
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
        config_from_weights, dit_gguf, layout_matches, probe_layout, run_cover, run_create,
        weight_layout, AceStepConfig, CreateRequest, TIME_FREQ_DIM,
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
    fn smoke_probe_accepts_the_layout() {
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
    fn audio_create_runs_smoke_denoise_then_refuses_companions() {
        let dir = scratch("create");
        write_dit(&dir, AceStepConfig::smoke());
        let err = crate::pipeline::audio_create(
            "ace_step",
            &dir,
            &crate::pipeline::AudioCreateRequest {
                prompt: "test",
                lyrics: "[inst]",
                duration_sec: 5.0,
                seed: Some(0),
                guidance: 1.0,
                steps: None,
                vocal_language: "en",
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("companions") || err.to_string().contains("text_encoder"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cover_and_lm_fail_loud_with_acestep_pointers() {
        let err = run_cover(Path::new("/tmp")).unwrap_err();
        assert!(
            err.to_string().contains("CoverRequest") || err.to_string().contains("audio.edit"),
            "{err}"
        );
        let err = super::run_lm_expand(Path::new("/tmp")).unwrap_err();
        assert!(err.to_string().contains("pipeline-lm"), "{err}");
    }

    #[test]
    fn run_create_on_smoke_proves_denoise_then_refuses() {
        let dir = scratch("prove");
        write_dit(&dir, AceStepConfig::smoke());
        let err = run_create(
            &dir,
            &CreateRequest {
                prompt: "test",
                lyrics: "[inst]",
                duration_sec: 5.0,
                seed: Some(0),
                guidance: 1.0,
                steps: None,
                vocal_language: "en",
                bpm: None,
                keyscale: "",
                timesignature: "",
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("companions") || err.to_string().contains("text_encoder"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
