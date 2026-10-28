//! Product create-path 5Hz LM expansion (`lm_expansion` = auto | format | off).
//!
//! Official Custom path: run `format_sample` on caption+lyrics before DiT.
//! Fail loud when expansion is requested but no LM GGUF is installed.

use super::lm;
use super::lm_prompt::LmMode;
use super::{run_lm, LmRequest};
use crate::weights::WeightError;
use std::path::{Path, PathBuf};

/// Normalized expansion mode for music create.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmExpansionMode {
    Off,
    /// Same as format today (registry default).
    Auto,
    Format,
}

impl LmExpansionMode {
    pub fn parse(raw: &str) -> Result<Self, WeightError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "format" => Ok(Self::Format),
            "off" | "false" | "0" | "none" => Ok(Self::Off),
            other => Err(WeightError::NotBuilt(format!(
                "ACE-Step lm_expansion '{other}' is invalid (use auto, format, or off)"
            ))),
        }
    }

    pub fn is_off(self) -> bool {
        matches!(self, Self::Off)
    }
}

pub struct LmExpandRequest<'a> {
    pub mode: &'a str,
    pub caption: &'a str,
    pub lyrics: &'a str,
    pub duration: f32,
    pub bpm: i32,
    pub keyscale: &'a str,
    pub timesignature: &'a str,
    pub vocal_language: &'a str,
    pub seed: u64,
    /// Explicit LM card path (file or dir). Used when DiT has no companion LM.
    pub lm_bundle: Option<&'a Path>,
    /// Caller-supplied codes (kept when LM format emits none).
    pub existing_codes: Option<&'a [i32]>,
}

#[derive(Debug, Clone)]
pub struct LmExpandOutput {
    pub caption: String,
    pub lyrics: String,
    pub bpm: Option<i32>,
    pub duration: Option<i32>,
    pub keyscale: String,
    pub timesignature: String,
    pub language: String,
    pub audio_code_indices: Vec<i32>,
    pub did_expand: bool,
}

/// Resolve LM weights for create: DiT companions → explicit override → sibling card.
pub fn resolve_lm_bundle_for_create(
    dit_bundle: &Path,
    override_lm: Option<&Path>,
) -> Result<PathBuf, WeightError> {
    if let Ok(path) = find_lm_companion(dit_bundle) {
        return Ok(path);
    }
    if let Some(path) = override_lm {
        if path.exists() {
            return Ok(path.to_path_buf());
        }
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step 5Hz LM override {} is missing (install ace-step-5hz-lm or a DiT companions/lm)",
            path.display()
        )));
    }
    if let Ok(path) = find_sibling_lm_card(dit_bundle) {
        return Ok(path);
    }
    Err(WeightError::NotBuilt(
        "ACE-Step lm_expansion needs 5Hz LM weights: install model ace-step-5hz-lm, \
         or place GGUF under the DiT bundle at lm/, companions/lm/, or 5hz-lm/ \
         (set lm_expansion=off to skip)"
            .into(),
    ))
}

fn find_lm_companion(bundle: &Path) -> Result<PathBuf, WeightError> {
    if !bundle.is_dir() {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step DiT bundle {} is not a directory",
            bundle.display()
        )));
    }
    for sub in ["lm", "companions/lm", "5hz-lm"] {
        let dir = bundle.join(sub);
        if dir.is_dir() {
            if let Ok(path) = lm::find_lm_gguf_in_dir(&dir) {
                return Ok(path);
            }
        }
    }
    Err(WeightError::NotBuilt(format!(
        "no LM companion under {}",
        bundle.display()
    )))
}

fn find_sibling_lm_card(dit_bundle: &Path) -> Result<PathBuf, WeightError> {
    // DiT layouts: models/<id>/<tier> or models/<id>
    let mut roots = Vec::new();
    if let Some(p) = dit_bundle.parent() {
        roots.push(p.to_path_buf());
        if let Some(gp) = p.parent() {
            roots.push(gp.to_path_buf());
        }
    }
    for root in roots {
        if !root.is_dir() {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if !(name.contains("5hz-lm") || name.contains("ace-step-5hz")) {
                continue;
            }
            // Prefer q8_0 / default tier dirs when present.
            for tier in ["q8_0", "q4_k", "q5_k", "q6_k", "f16"] {
                let tier_dir = path.join(tier);
                if tier_dir.is_dir() {
                    if let Ok(gguf) = lm::find_lm_gguf_in_dir(&tier_dir) {
                        return Ok(gguf);
                    }
                }
            }
            if path.is_dir() {
                if let Ok(gguf) = lm::find_lm_gguf_in_dir(&path) {
                    return Ok(gguf);
                }
            }
        }
    }
    Err(WeightError::NotBuilt(
        "no sibling ace-step-5hz-lm card under models/".into(),
    ))
}

/// Run format_sample when `lm_expansion` is auto/format; skip when off.
pub fn maybe_expand_for_create(
    dit_bundle: &Path,
    request: &LmExpandRequest<'_>,
) -> Result<LmExpandOutput, WeightError> {
    let mode = LmExpansionMode::parse(request.mode)?;
    let passthrough = || LmExpandOutput {
        caption: request.caption.to_string(),
        lyrics: request.lyrics.to_string(),
        bpm: (request.bpm > 0).then_some(request.bpm),
        duration: (request.duration > 0.0).then_some(request.duration.round() as i32),
        keyscale: request.keyscale.to_string(),
        timesignature: request.timesignature.to_string(),
        language: if request.vocal_language.is_empty() {
            "en".into()
        } else {
            request.vocal_language.into()
        },
        audio_code_indices: request
            .existing_codes
            .map(|c| c.to_vec())
            .unwrap_or_default(),
        did_expand: false,
    };
    if mode.is_off() {
        return Ok(passthrough());
    }

    let lm_path = resolve_lm_bundle_for_create(dit_bundle, request.lm_bundle)?;
    // Prefer the LM directory (tokenizer lives beside GGUF).
    let lm_root = if lm_path.is_file() {
        lm_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| lm_path.clone())
    } else {
        lm_path.clone()
    };

    let use_fsm = super::text::load_tokenizer_json(&lm_root).is_ok();
    let out = run_lm(
        &lm_root,
        &LmRequest {
            mode: LmMode::Format,
            caption: request.caption,
            lyrics: request.lyrics,
            duration: request.duration,
            bpm: request.bpm,
            keyscale: request.keyscale,
            timesignature: request.timesignature,
            vocal_language: request.vocal_language,
            max_new_tokens: 512,
            tokenizer_json: None,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: request.seed,
            use_fsm,
            cfg_scale: 1.0,
            negative_prompt: "",
        },
    )?;

    let mut codes = out.audio_code_indices;
    if codes.is_empty() {
        if let Some(existing) = request.existing_codes {
            codes = existing.to_vec();
        }
    }

    Ok(LmExpandOutput {
        caption: out.caption,
        lyrics: out.lyrics,
        bpm: out.bpm.or_else(|| (request.bpm > 0).then_some(request.bpm)),
        duration: out
            .duration
            .or_else(|| (request.duration > 0.0).then_some(request.duration.round() as i32)),
        keyscale: if out.keyscale.is_empty() {
            request.keyscale.to_string()
        } else {
            out.keyscale
        },
        timesignature: if out.timesignature.is_empty() {
            request.timesignature.to_string()
        } else {
            out.timesignature
        },
        language: if out.language.is_empty() {
            if request.vocal_language.is_empty() {
                "en".into()
            } else {
                request.vocal_language.into()
            }
        } else {
            out.language
        },
        audio_code_indices: codes,
        did_expand: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::ace_step::{write_smoke_lm, AceLmConfig};

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("danmo-lm-expand-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parse_modes() {
        assert_eq!(LmExpansionMode::parse("").unwrap(), LmExpansionMode::Auto);
        assert_eq!(LmExpansionMode::parse("AUTO").unwrap(), LmExpansionMode::Auto);
        assert_eq!(LmExpansionMode::parse("format").unwrap(), LmExpansionMode::Format);
        assert_eq!(LmExpansionMode::parse("off").unwrap(), LmExpansionMode::Off);
        assert!(LmExpansionMode::parse("inspire").is_err());
    }

    #[test]
    fn off_skips_missing_lm() {
        let dit = scratch("off-dit");
        let out = maybe_expand_for_create(
            &dit,
            &LmExpandRequest {
                mode: "off",
                caption: "lofi",
                lyrics: "[inst]",
                duration: 30.0,
                bpm: 90,
                keyscale: "C major",
                timesignature: "4",
                vocal_language: "en",
                seed: 0,
                lm_bundle: None,
                existing_codes: None,
            },
        )
        .unwrap();
        assert!(!out.did_expand);
        assert_eq!(out.caption, "lofi");
        assert_eq!(out.bpm, Some(90));
        let _ = std::fs::remove_dir_all(&dit);
    }

    #[test]
    fn auto_fails_loud_without_lm() {
        let dit = scratch("auto-missing");
        let err = maybe_expand_for_create(
            &dit,
            &LmExpandRequest {
                mode: "auto",
                caption: "lofi",
                lyrics: "[inst]",
                duration: 30.0,
                bpm: 0,
                keyscale: "",
                timesignature: "",
                vocal_language: "en",
                seed: 0,
                lm_bundle: None,
                existing_codes: None,
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("5Hz LM") || err.to_string().contains("lm_expansion"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dit);
    }

    #[test]
    fn auto_uses_companion_lm_under_dit() {
        let dit = scratch("auto-companion");
        let lm_dir = dit.join("companions/lm");
        std::fs::create_dir_all(&lm_dir).unwrap();
        write_smoke_lm(&lm_dir, AceLmConfig::smoke()).unwrap();
        let out = maybe_expand_for_create(
            &dit,
            &LmExpandRequest {
                mode: "auto",
                caption: "lofi beat",
                lyrics: "[inst]",
                duration: 10.0,
                bpm: 100,
                keyscale: "A minor",
                timesignature: "4",
                vocal_language: "en",
                seed: 1,
                lm_bundle: None,
                existing_codes: Some(&[1, 2, 3]),
            },
        )
        .unwrap();
        assert!(out.did_expand);
        // Smoke LM has no real tokenizer decode → keep user caption fallbacks via format assemble.
        assert!(!out.caption.is_empty());
        // Existing codes kept when format emits none.
        assert_eq!(out.audio_code_indices, vec![1, 2, 3]);
        let _ = std::fs::remove_dir_all(&dit);
    }

    #[test]
    fn resolve_prefers_companion_over_override() {
        let dit = scratch("prefer-companion");
        let lm_dir = dit.join("lm");
        std::fs::create_dir_all(&lm_dir).unwrap();
        write_smoke_lm(&lm_dir, AceLmConfig::smoke()).unwrap();
        let other = scratch("other-lm");
        write_smoke_lm(&other, AceLmConfig::smoke()).unwrap();
        let resolved = resolve_lm_bundle_for_create(&dit, Some(&other)).unwrap();
        assert!(resolved.starts_with(&lm_dir) || resolved.parent() == Some(lm_dir.as_path()));
        let _ = std::fs::remove_dir_all(&dit);
        let _ = std::fs::remove_dir_all(&other);
    }
}
