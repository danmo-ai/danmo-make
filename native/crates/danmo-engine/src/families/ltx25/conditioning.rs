//! Reference-token assembly for LTX 2.5 IC-LoRA / Ref2VA / animate I2V.
//!
//! Ref2VA: reference tokens stay clean at timestep 0 and are prepended to the
//! noisy target sequence. Only the target suffix is updated during denoise.
//! Slot embeddings and negative temporal RoPE offsets live with the MSR path.
//!
//! Animate (official ImageReplace at frame 0): first-frame tokens stay inside
//! the target sequence. Per-token sigma is `denoise_mask * sigma` with
//! `denoise_mask = 1 - strength`, and after each step
//! `x = denoised * mask + clean * (1 - mask)`.

use danmo_mlx::{Array, Result, Stream};

/// Distilled two-stage I2V strengths from the published Comfy / TI2Vid path.
pub const STAGE_1_IMAGE_STRENGTH: f32 = 0.7;
pub const STAGE_2_IMAGE_STRENGTH: f32 = 1.0;

/// Build per-token diffusion sigmas: `0` for each reference token, `sigma` for
/// each target token.
pub fn reference_then_target_sigmas(
    reference_tokens: usize,
    target_tokens: usize,
    sigma: f32,
) -> Vec<f32> {
    let mut values = Vec::with_capacity(reference_tokens + target_tokens);
    values.resize(reference_tokens, 0.0);
    values.extend(std::iter::repeat(sigma).take(target_tokens));
    values
}

/// Concatenate reference and target token batches on the sequence axis.
///
/// Both inputs must be `[B, T, C]` with matching batch and channel sizes.
pub fn prepend_reference_tokens(
    reference: &Array,
    target: &Array,
    stream: &Stream,
) -> Result<Array> {
    let r = reference.shape();
    let t = target.shape();
    if r.len() != 3 || t.len() != 3 {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 conditioning",
            format!("expected [B,T,C] tokens, got reference {r:?} target {t:?}"),
        ));
    }
    if r[0] != t[0] || r[2] != t[2] {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 conditioning",
            format!("batch/channel mismatch: reference {r:?} target {t:?}"),
        ));
    }
    Array::concatenate_axis(&[reference.clone(), target.clone()], 1, stream)
}

/// Keep only the target suffix after a joint forward (drop prepended refs).
pub fn take_target_suffix(tokens: &Array, target_tokens: usize, stream: &Stream) -> Result<Array> {
    let shape = tokens.shape();
    if shape.len() != 3 {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 conditioning",
            format!("expected [B,T,C], got {shape:?}"),
        ));
    }
    let total = shape[1] as usize;
    if target_tokens == 0 || target_tokens > total {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 conditioning",
            format!("target_tokens {target_tokens} is outside sequence length {total}"),
        ));
    }
    let start = (total - target_tokens) as i32;
    tokens.slice(
        &[0, start, 0],
        &[shape[0], shape[1], shape[2]],
        &[1, 1, 1],
        stream,
    )
}

/// MSR / Ref2VA negative temporal offsets: `-(K-i+1)/fps` seconds for slot
/// order `i = 1..K` among the selected references.
pub fn msr_temporal_offsets(slot_count: usize, fps: f32) -> Result<Vec<f32>> {
    if !fps.is_finite() || fps <= 0.0 {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 conditioning",
            format!("MSR temporal offsets need a positive finite fps, got {fps}"),
        ));
    }
    if slot_count == 0 {
        return Ok(Vec::new());
    }
    Ok((1..=slot_count)
        .map(|index| -((slot_count - index + 1) as f32) / fps)
        .collect())
}

/// Token count for latent frame 0 (`H * W` after spatial compression).
pub fn first_frame_token_count(shape: crate::common::av_latent::VideoLatentShape) -> Result<usize> {
    let height = shape.height as usize;
    let width = shape.width as usize;
    height.checked_mul(width).ok_or_else(|| {
        danmo_mlx::Error::new("LTX 2.5 conditioning", "first-frame token count overflow")
    })
}

/// Per-token video sigmas for ImageReplace at latent index 0.
///
/// First-frame tokens use `(1 - strength) * sigma`; remaining tokens use `sigma`.
pub fn replace_first_frame_sigmas(
    total_tokens: usize,
    first_frame_tokens: usize,
    sigma: f32,
    strength: f32,
) -> Result<Vec<f32>> {
    if first_frame_tokens == 0 || first_frame_tokens > total_tokens {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 conditioning",
            format!(
                "first_frame_tokens {first_frame_tokens} is outside total token count {total_tokens}"
            ),
        ));
    }
    if !(0.0..=1.0).contains(&strength) || !strength.is_finite() {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 conditioning",
            format!("image conditioning strength must be finite in [0, 1], got {strength}"),
        ));
    }
    if !sigma.is_finite() || sigma < 0.0 {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 conditioning",
            format!("sigma must be finite and non-negative, got {sigma}"),
        ));
    }
    let mask = 1.0 - strength;
    let mut values = Vec::with_capacity(total_tokens);
    values.resize(first_frame_tokens, mask * sigma);
    values.extend(std::iter::repeat(sigma).take(total_tokens - first_frame_tokens));
    Ok(values)
}

/// `lerp(clean, noisy, 1 - strength)` on the leading first-frame token span.
pub fn init_replace_first_frame(
    noise: &Array,
    clean_first: &Array,
    strength: f32,
    stream: &Stream,
) -> Result<Array> {
    let (prefix, rest) = split_prefix(noise, clean_first, stream)?;
    let blended = lerp_toward_clean(&prefix, clean_first, strength, stream)?;
    Array::concatenate_axis(&[blended, rest], 1, stream)
}

/// `denoised * (1 - strength) + clean * strength` on the leading first-frame span.
pub fn blend_replace_first_frame(
    tokens: &Array,
    clean_first: &Array,
    strength: f32,
    stream: &Stream,
) -> Result<Array> {
    let (prefix, rest) = split_prefix(tokens, clean_first, stream)?;
    let blended = lerp_toward_clean(&prefix, clean_first, strength, stream)?;
    Array::concatenate_axis(&[blended, rest], 1, stream)
}

fn split_prefix(tokens: &Array, clean_first: &Array, stream: &Stream) -> Result<(Array, Array)> {
    let shape = tokens.shape();
    let clean = clean_first.shape();
    if shape.len() != 3 || clean.len() != 3 {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 conditioning",
            format!("expected [B,T,C] tokens, got video {shape:?} clean {clean:?}"),
        ));
    }
    if shape[0] != clean[0] || shape[2] != clean[2] {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 conditioning",
            format!("batch/channel mismatch: video {shape:?} clean {clean:?}"),
        ));
    }
    let first = clean[1] as usize;
    let total = shape[1] as usize;
    if first == 0 || first > total {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 conditioning",
            format!("clean first-frame tokens {first} outside sequence length {total}"),
        ));
    }
    let prefix = tokens.slice(
        &[0, 0, 0],
        &[shape[0], first as i32, shape[2]],
        &[1, 1, 1],
        stream,
    )?;
    let rest = tokens.slice(
        &[0, first as i32, 0],
        &[shape[0], shape[1], shape[2]],
        &[1, 1, 1],
        stream,
    )?;
    Ok((prefix, rest))
}

fn lerp_toward_clean(
    noisy: &Array,
    clean: &Array,
    strength: f32,
    stream: &Stream,
) -> Result<Array> {
    if !(0.0..=1.0).contains(&strength) || !strength.is_finite() {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 conditioning",
            format!("image conditioning strength must be finite in [0, 1], got {strength}"),
        ));
    }
    // post = noisy * (1 - strength) + clean * strength
    let keep = crate::common::mlx_ops::scalar(1.0 - strength)?;
    let pull = crate::common::mlx_ops::scalar(strength)?;
    let a = Array::multiply(noisy, &keep, stream)?;
    let b = Array::multiply(clean, &pull, stream)?;
    Array::add(&a, &b, stream)
}

/// Absolute MSR-AVref audio RoPE midpoints for one slot of `token_count` tokens.
///
/// Windows end at `-((image_slots-slot_id)*duration)-margin`, matching the
/// published sparse absolute layout.
pub fn msr_audio_slot_positions(
    token_count: usize,
    slot_id: usize,
    image_slot_count: usize,
    slot_duration: f32,
    end_margin: f32,
    spec: crate::common::av_latent::AudioLatentSpec,
) -> Result<Vec<f32>> {
    if slot_id == 0 || image_slot_count == 0 || slot_id > image_slot_count {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 conditioning",
            format!("audio slot {slot_id} is outside image slot count {image_slot_count}"),
        ));
    }
    if !slot_duration.is_finite()
        || slot_duration <= 0.0
        || !end_margin.is_finite()
        || end_margin <= 0.0
    {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 conditioning",
            "MSR audio slot duration/margin must be positive and finite",
        ));
    }
    let local = crate::common::av_latent::audio_positions(token_count, spec)
        .map_err(|detail| danmo_mlx::Error::new("LTX 2.5 conditioning", detail))?;
    if local.is_empty() {
        return Ok(local);
    }
    let downsample = spec.downsample_factor as f32;
    let seconds_per_hop = spec.hop_length as f32 / spec.sample_rate as f32;
    let local_end =
        ((token_count as f32) * downsample + 1.0 - downsample).max(0.0) * seconds_per_hop;
    let slot_end = -((image_slot_count - slot_id) as f32 * slot_duration) - end_margin;
    let shift = slot_end - local_end;
    Ok(local.into_iter().map(|mid| mid + shift).collect())
}

#[cfg(test)]
mod tests {
    use super::{msr_temporal_offsets, reference_then_target_sigmas, replace_first_frame_sigmas};

    #[test]
    fn reference_sigmas_are_zero_then_sigma() {
        assert_eq!(
            reference_then_target_sigmas(2, 3, 0.5),
            vec![0.0, 0.0, 0.5, 0.5, 0.5]
        );
    }

    #[test]
    fn replace_first_frame_sigmas_scale_by_one_minus_strength() {
        assert_eq!(
            replace_first_frame_sigmas(5, 2, 1.0, 0.7).unwrap(),
            vec![0.3, 0.3, 1.0, 1.0, 1.0]
        );
        assert_eq!(
            replace_first_frame_sigmas(4, 1, 0.5, 1.0).unwrap(),
            vec![0.0, 0.5, 0.5, 0.5]
        );
    }

    #[test]
    fn msr_offsets_match_published_three_slot_example() {
        let offsets = msr_temporal_offsets(3, 25.0).unwrap();
        assert_eq!(offsets.len(), 3);
        assert!((offsets[0] - (-0.12)).abs() < 1e-6);
        assert!((offsets[1] - (-0.08)).abs() < 1e-6);
        assert!((offsets[2] - (-0.04)).abs() < 1e-6);
    }

    #[test]
    fn msr_audio_slot_ends_at_published_window() {
        use super::msr_audio_slot_positions;
        use crate::common::av_latent::AudioLatentSpec;
        let spec = AudioLatentSpec {
            downsample_factor: 4,
            hop_length: 160,
            sample_rate: 16_000,
        };
        let positions = msr_audio_slot_positions(2, 1, 2, 5.0, 0.04, spec).unwrap();
        assert_eq!(positions.len(), 2);
        // slot_end = -5.04; local_end = 0.05; shift = -5.09
        // local mids ≈ 0.005, 0.030 → -5.085, -5.060
        assert!((positions[0] - (-5.085)).abs() < 1e-4);
        assert!((positions[1] - (-5.060)).abs() < 1e-4);
    }
}
