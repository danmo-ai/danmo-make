//! Qwen-Image create/edit prompt conditioning on the shared Qwen2.5-VL text
//! trunk (`common::qwen25vl`). The family owns the chat templates, the
//! 34/64-token system prefix drops, and the 1058-token cap.
//!
//! Create positions are one index on all three MRoPE axes; see the shared
//! module for why that matches half-split RoPE.

use crate::common::qwen25vl;
use crate::runtime::RuntimeError;
use crate::weights::WeightError;

pub use qwen25vl::{Qwen25VlConfig, TextForward, TowerOutput};

pub const DROP_IDX: usize = 34;
pub const MAX_LENGTH: usize = 1058;

/// Hardcoded Qwen-Image create template from main `QwenImageTextEncoder`.
pub const PROMPT_TEMPLATE: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";

/// Edit template from main `QwenVisionLanguageTokenizer` with
/// `use_picture_prefix=True`. The `{}` is
/// `Picture 1: <|vision_start|><|image_pad|><|vision_end|>` plus the prompt.
/// The hidden-state trim drops [`EDIT_DROP_IDX`] tokens.
pub const EDIT_TEMPLATE: &str = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";
pub const EDIT_DROP_IDX: usize = 64;

pub use qwen25vl::{layout_matches, text_layout};

/// Drop the 34-token create system prefix from a `[hidden, seq]` buffer.
pub fn trim_system_prefix(
    hidden: &[f32],
    width: usize,
    valid: usize,
) -> Result<Vec<f32>, RuntimeError> {
    qwen25vl::trim_prefix(hidden, width, valid, DROP_IDX)
}

pub fn config_from_weights(
    weights: &crate::weights::GgufWeights,
) -> Result<Qwen25VlConfig, WeightError> {
    for candidate in [
        Qwen25VlConfig::published(),
        Qwen25VlConfig::create_smoke(),
        Qwen25VlConfig::edit_smoke(),
        Qwen25VlConfig::smoke(),
    ] {
        if layout_matches(weights, candidate) {
            return Ok(candidate);
        }
    }
    Err(WeightError::NotBuilt(
        "Qwen2.5-VL text encoder layout is not the published or smoke tower".into(),
    ))
}

/// Token ids → final hidden state with the 34-token system prefix removed.
/// Layout is `[hidden, trimmed_seq]` with hidden on `ne0`.
pub fn encode(
    ctx: &crate::runtime::GgmlContext,
    weights: &crate::weights::GgufWeights,
    config: Qwen25VlConfig,
    ids: &[i32],
) -> Result<Vec<f32>, WeightError> {
    if ids.is_empty() {
        return Err(WeightError::NotBuilt("prompt has no tokens".into()));
    }
    if !layout_matches(weights, config) {
        return Err(WeightError::NotBuilt(
            "Qwen2.5-VL text encoder layout does not match the tower config".into(),
        ));
    }
    let seq = ids.len() as i64;
    let mut tower = TextForward::build(ctx, config, seq, TowerOutput::FinalNormed)?;
    qwen25vl::load_tower(&mut tower, weights, config, TowerOutput::FinalNormed)?;
    let hidden = tower.compute(ids, &qwen25vl::causal_mask(seq))?;
    Ok(trim_system_prefix(
        &hidden,
        config.hidden as usize,
        ids.len(),
    )?)
}

pub fn encode_with_vision(
    ctx: &crate::runtime::GgmlContext,
    weights: &crate::weights::GgufWeights,
    config: Qwen25VlConfig,
    ids: &[i32],
    vision: &[f32],
    positions: &[i64],
) -> Result<Vec<f32>, WeightError> {
    if ids.is_empty() || positions.is_empty() {
        return Err(WeightError::NotBuilt(
            "qwen edit prompt or vision positions are empty".into(),
        ));
    }
    if !layout_matches(weights, config) {
        return Err(WeightError::NotBuilt(
            "Qwen2.5-VL text encoder layout does not match the tower config".into(),
        ));
    }
    let seq = ids.len() as i64;
    let mut tower = TextForward::build_with_vision(
        ctx,
        config,
        seq,
        Some((positions.len() as i64, positions.len() as i64)),
        TowerOutput::FinalNormed,
    )?;
    qwen25vl::load_tower(&mut tower, weights, config, TowerOutput::FinalNormed)?;
    let hidden = tower.compute_with_vision(ids, &qwen25vl::causal_mask(seq), vision, positions)?;
    Ok(qwen25vl::trim_prefix(
        &hidden,
        config.hidden as usize,
        ids.len(),
        EDIT_DROP_IDX,
    )?)
}

#[cfg(test)]
mod tests {
    use super::{DROP_IDX, EDIT_TEMPLATE, MAX_LENGTH, PROMPT_TEMPLATE};

    #[test]
    fn published_template_drops_thirty_four_and_skips_vision() {
        assert_eq!(DROP_IDX, 34);
        assert_eq!(MAX_LENGTH, 1058);
        assert!(PROMPT_TEMPLATE.contains(
            "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"
        ));
        assert!(PROMPT_TEMPLATE.contains("<|im_start|>user\n{}<|im_end|>"));
        assert!(EDIT_TEMPLATE.contains("Describe the key features of the input image"));
    }
}
