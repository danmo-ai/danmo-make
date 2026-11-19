//! Z-Image Fun ControlNet Union — MLX control branch.
//!
//! Hint injection into the main DiT: embed 33-ch context → control noise
//! refiner → noise_refiner hints → control layers → main-layer hints.
//! Create uses the eager DiT path when a guide is active (compiled DiT has
//! no hint hooks).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use danmo_mlx::{Array, Dtype, Error, LinearWeight, Result, Stream};

use super::super::geom::{self, TokenGrid};
use super::attention::AttentionWeights;
use super::blocks::{self, BlockWeights};
use super::MlxZImageConfig;
use crate::weights::{MlxWeights, WeightDtype, WeightError};

/// Control latent packing: 16 control + 1 preserve mask + 16 inpaint.
pub const CONTROL_IN_CHANNELS: i64 = 33;
/// Main DiT layers that receive control hints (every other of 30).
pub const CONTROL_LAYER_PLACES: [usize; 15] =
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28];
/// Noise-refiner layers that receive control hints.
pub const CONTROL_REFINER_PLACES: [usize; 2] = [0, 1];

/// Remapped ControlNet weights plus the VAE-packed control context.
#[derive(Clone, Copy)]
pub struct ControlConditioning<'a> {
    pub weights: &'a MlxWeights,
    /// `[1, 33, H, W]` control context (non-inpaint zeros the mask/inpaint).
    pub context: &'a Array,
    pub scale: f32,
}

/// `x + hint * context_scale` (main `apply_control_hint`).
pub fn apply_control_hint(
    x: &Array,
    hint: &Array,
    context_scale: f32,
    stream: &Stream,
) -> Result<Array> {
    if !context_scale.is_finite() {
        return Err(Error::new(
            "Z-Image control",
            format!("context_scale {context_scale} is not finite"),
        ));
    }
    if context_scale == 0.0 {
        return Ok(x.clone());
    }
    if x.shape() != hint.shape() {
        return Err(Error::new(
            "Z-Image control",
            format!(
                "control hint shape {:?} does not match stream {:?}",
                hint.shape(),
                x.shape()
            ),
        ));
    }
    let scale = Array::from_f32_slice(&[context_scale], &[])?.astype(x.dtype(), stream)?;
    let delta = Array::multiply(hint, &scale, stream)?;
    let delta = if delta.dtype() == x.dtype() {
        delta
    } else {
        delta.astype(x.dtype(), stream)?
    };
    Array::add(x, &delta, stream)
}

/// Pack a VAE-encoded control latent `[1,16,H,W]` into the 33-channel
/// ControlNet context. Non-inpaint path zeros the preserve mask and inpaint
/// latents (main `build_z_image_control_context_nchw`).
pub fn build_control_context_nchw(control_latent: &Array, stream: &Stream) -> Result<Array> {
    let shape = control_latent.shape();
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 16 {
        return Err(Error::new(
            "Z-Image control",
            format!("control latent must be [1,16,H,W], got {:?}", shape),
        ));
    }
    let height = shape[2];
    let width = shape[3];
    let zeros_mask = zeros(&[1, 1, height, width], control_latent.dtype(), stream)?;
    let zeros_inpaint = zeros(&[1, 16, height, width], control_latent.dtype(), stream)?;
    Array::concatenate_axis(
        &[control_latent.clone(), zeros_mask, zeros_inpaint],
        1,
        stream,
    )
}

/// Pack control + preserve mask + inpaint latents into 33-channel context.
/// `preserve_mask` is `[1,1,H,W]` at latent resolution (1 = keep source).
/// `inpaint_latent` is `[1,16,H,W]` VAE-encoded inpaint/source image.
pub fn build_control_context_inpaint_nchw(
    control_latent: &Array,
    preserve_mask: &Array,
    inpaint_latent: &Array,
    stream: &Stream,
) -> Result<Array> {
    let shape = control_latent.shape();
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 16 {
        return Err(Error::new(
            "Z-Image control",
            format!("control latent must be [1,16,H,W], got {:?}", shape),
        ));
    }
    let height = shape[2];
    let width = shape[3];
    let mask_shape = preserve_mask.shape();
    if mask_shape != [1, 1, height, width] {
        return Err(Error::new(
            "Z-Image control",
            format!(
                "preserve mask must be [1,1,{height},{width}], got {:?}",
                mask_shape
            ),
        ));
    }
    let inpaint_shape = inpaint_latent.shape();
    if inpaint_shape != [1, 16, height, width] {
        return Err(Error::new(
            "Z-Image control",
            format!(
                "inpaint latent must be [1,16,{height},{width}], got {:?}",
                inpaint_shape
            ),
        ));
    }
    let mask = if preserve_mask.dtype() == control_latent.dtype() {
        preserve_mask.clone()
    } else {
        preserve_mask.astype(control_latent.dtype(), stream)?
    };
    let inpaint = if inpaint_latent.dtype() == control_latent.dtype() {
        inpaint_latent.clone()
    } else {
        inpaint_latent.astype(control_latent.dtype(), stream)?
    };
    Array::concatenate_axis(&[control_latent.clone(), mask, inpaint], 1, stream)
}

fn zeros(shape: &[i32], dtype: Dtype, stream: &Stream) -> Result<Array> {
    Array::from_f32_slice(&[0.0], &[])?
        .astype(dtype, stream)?
        .broadcast_to(shape, stream)
}

pub fn control_layer_hint_index(layer_idx: usize) -> Option<usize> {
    CONTROL_LAYER_PLACES
        .iter()
        .position(|&place| place == layer_idx)
}

pub fn control_refiner_hint_index(layer_idx: usize) -> Option<usize> {
    CONTROL_REFINER_PLACES
        .iter()
        .position(|&place| place == layer_idx)
}

pub fn control_embed_in_dim(patch: i64) -> Result<i64> {
    if patch <= 0 {
        return Err(Error::new(
            "Z-Image control",
            format!("patch size must be positive, got {patch}"),
        ));
    }
    Ok(patch * patch * CONTROL_IN_CHANNELS)
}

/// Load a ControlNet safetensors directory (or a single file). Multiple
/// variant shards with the same keys are not merged — the canonical non-lite
/// `*-8steps.safetensors` (no date suffix) is preferred; otherwise fail loud.
pub fn load_directory(path: &Path, patch: i64, dtype: WeightDtype) -> Result<MlxWeights> {
    let shard = select_control_shard(path)?;
    let weights = load_single_safetensors(&shard, dtype)?;
    let mut names = BTreeMap::new();
    for name in weights.names() {
        match super::weights::remap_control_name(name, patch)
            .map_err(|error| Error::new("Z-Image control remap", error.to_string()))?
        {
            Some(target) => {
                names.insert(name.to_string(), target);
            }
            None => {
                return Err(Error::new(
                    "Z-Image control",
                    format!(
                        "unexpected non-control tensor {name} in {}",
                        shard.display()
                    ),
                ));
            }
        }
    }
    if names.is_empty() {
        return Err(Error::new(
            "Z-Image control",
            format!("no control_* tensors in {}", shard.display()),
        ));
    }
    weights.rename(&names)
}

fn load_single_safetensors(path: &Path, dtype: WeightDtype) -> Result<MlxWeights> {
    // load_mlx_directory only accepts directories. Use a private sibling
    // directory that contains a symlink/copy is heavy; instead read via the
    // same SafeTensors path by staging a one-file temp dir under the parent.
    let staging = path.parent().map(|p| p.join(".danmo-control-load")).ok_or_else(|| {
        Error::new(
            "Z-Image control",
            format!("{} has no parent", path.display()),
        )
    })?;
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|error| {
        Error::new(
            "Z-Image control",
            format!("{}: {error}", staging.display()),
        )
    })?;
    let link = staging.join(
        path.file_name()
            .ok_or_else(|| Error::new("Z-Image control", "control shard has no file name"))?,
    );
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(path, &link).map_err(|error| {
            Error::new(
                "Z-Image control",
                format!("symlink {}: {error}", link.display()),
            )
        })?;
    }
    #[cfg(not(unix))]
    {
        std::fs::copy(path, &link).map_err(|error| {
            Error::new(
                "Z-Image control",
                format!("copy {}: {error}", link.display()),
            )
        })?;
    }
    let result = crate::weights::load_mlx_directory(&staging, dtype, None);
    let _ = std::fs::remove_dir_all(&staging);
    result
}

pub fn select_control_shard(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        if path.extension().and_then(|v| v.to_str()) != Some("safetensors") {
            return Err(Error::new(
                "Z-Image control",
                format!("{} is not a safetensors file", path.display()),
            ));
        }
        return Ok(path.to_path_buf());
    }
    if !path.is_dir() {
        return Err(Error::new(
            "Z-Image control",
            format!("{} is not an installed controlnet directory", path.display()),
        ));
    }
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(path).map_err(|error| {
        Error::new(
            "Z-Image control",
            format!("{}: {error}", path.display()),
        )
    })? {
        let entry = entry.map_err(|error| {
            Error::new(
                "Z-Image control",
                format!("{}: {error}", path.display()),
            )
        })?;
        let candidate = entry.path();
        if candidate.extension().and_then(|v| v.to_str()) == Some("safetensors") {
            shards.push(candidate);
        }
    }
    shards.sort();
    if shards.is_empty() {
        return Err(Error::new(
            "Z-Image control",
            format!("no safetensors shards in {}", path.display()),
        ));
    }
    if shards.len() == 1 {
        return Ok(shards.remove(0));
    }
    let non_lite: Vec<_> = shards
        .iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| !n.to_ascii_lowercase().contains("lite"))
        })
        .cloned()
        .collect();
    if non_lite.is_empty() {
        return Err(Error::new(
            "Z-Image control",
            format!(
                "only lite controlnet shards in {} ({}); install a non-lite *-8steps.safetensors",
                path.display(),
                shards
                    .iter()
                    .filter_map(|p| p.file_name()?.to_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    }
    if non_lite.len() == 1 {
        return Ok(non_lite[0].clone());
    }
    // Prefer newest dated 8-step (2602 > 2601 > undated). HF: 2601+ fix mask
    // leakage; 2602 adds Gray — needed for companion retouch without pose.
    let mut ranked = non_lite;
    ranked.sort_by_key(|p| {
        let name = p
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        std::cmp::Reverse(dated_8steps_rank(name))
    });
    Ok(ranked[0].clone())
}

fn has_dated_8steps_suffix(name: &str) -> bool {
    dated_8steps_rank(name) > 0
}

/// `…-2602-8steps` → 2602; undated `…-8steps` → 0.
fn dated_8steps_rank(name: &str) -> u32 {
    let Some(stem) = name.strip_suffix(".safetensors") else {
        return 0;
    };
    let Some(rest) = stem.strip_suffix("-8steps") else {
        return 0;
    };
    rest.rsplit_once('-')
        .and_then(|(_, tail)| {
            if tail.len() == 4 && tail.bytes().all(|b| b.is_ascii_digit()) {
                tail.parse().ok()
            } else {
                None
            }
        })
        .unwrap_or(0)
}

/// Patch-embed the 33-ch control context. Uses the main DiT `x_pad_token`.
pub fn embed_control_context(
    control_weights: &MlxWeights,
    dit_weights: &MlxWeights,
    config: MlxZImageConfig,
    grid: TokenGrid,
    context: &Array,
    stream: &Stream,
) -> Result<Array> {
    let shape = context.shape();
    if shape.len() != 4 || shape[0] != 1 || shape[1] != CONTROL_IN_CHANNELS as i32 {
        return Err(Error::new(
            "Z-Image control",
            format!(
                "control context must be [1,33,H,W], got {:?}",
                shape
            ),
        ));
    }
    if shape[2] as i64 != grid.latent_h || shape[3] as i64 != grid.latent_w {
        return Err(Error::new(
            "Z-Image control",
            format!(
                "control context spatial {:?} does not match latent grid {}x{}",
                &shape[2..],
                grid.latent_h,
                grid.latent_w
            ),
        ));
    }
    let token_dim = config.patch * config.patch * CONTROL_IN_CHANNELS;
    let indices = Array::from_i32_slice(
        &geom::patch_index(grid, CONTROL_IN_CHANNELS),
        &[grid.img_real as i32 * token_dim as i32],
    )?;
    let latent_size = CONTROL_IN_CHANNELS * grid.latent_w * grid.latent_h;
    let gathered = context
        .reshape(&[latent_size as i32], stream)?
        .take_axis(&indices, 0, stream)?;
    let mut patches = gathered.reshape(&[grid.img_real as i32, token_dim as i32], stream)?;
    if grid.img_pad > 0 {
        let last = patches.slice(
            &[grid.img_real as i32 - 1, 0],
            &[grid.img_real as i32, token_dim as i32],
            &[1, 1],
            stream,
        )?;
        let repeats = last.broadcast_to(&[grid.img_pad as i32, token_dim as i32], stream)?;
        patches = Array::concatenate_axis(&[patches, repeats], 0, stream)?;
    }
    let patches = patches.reshape(&[1, grid.img_len as i32, token_dim as i32], stream)?;
    let embedding_weight = linear(control_weights, "control_x_embedder.weight")?;
    let patches = cast_to(&patches, embedding_weight.compute_dtype(), stream)?;
    let mut embedded = embedding_weight.apply(
        &patches,
        Some(weight(control_weights, "control_x_embedder.bias")?),
        stream,
    )?;
    embedded = cast_to(&embedded, context.dtype(), stream)?;
    if grid.img_pad > 0 {
        let pad = weight(dit_weights, "x_pad_token")?
            .broadcast_to(&[1, grid.img_pad as i32, config.dim as i32], stream)?;
        let pad = cast_to(&pad, embedded.dtype(), stream)?;
        let real = embedded.slice(
            &[0, 0, 0],
            &[1, grid.img_real as i32, config.dim as i32],
            &[1, 1, 1],
            stream,
        )?;
        embedded = Array::concatenate_axis(&[real, pad], 1, stream)?;
    }
    Ok(embedded)
}

struct ControlBranch {
    hints: Vec<Array>,
    stream: Array,
}

fn control_block_forward(
    weights: &MlxWeights,
    config: MlxZImageConfig,
    prefix: &str,
    block_id: usize,
    branch: ControlBranch,
    x_main: &Array,
    ids: &[[i32; 3]],
    rope: &Array,
    timestep: &Array,
    stream: &Stream,
) -> Result<ControlBranch> {
    let control = if block_id == 0 {
        let projected = linear(weights, &format!("{prefix}.before_proj.weight"))?.apply(
            &branch.stream,
            Some(weight(weights, &format!("{prefix}.before_proj.bias"))?),
            stream,
        )?;
        Array::add(&projected, x_main, stream)?
    } else {
        branch.stream
    };
    let control = blocks::forward_with_table(
        &control,
        BlockWeights {
            attention: AttentionWeights {
                to_q: linear(weights, &format!("{prefix}.attention.to_q.weight"))?,
                to_k: linear(weights, &format!("{prefix}.attention.to_k.weight"))?,
                to_v: linear(weights, &format!("{prefix}.attention.to_v.weight"))?,
                to_out: linear(weights, &format!("{prefix}.attention.to_out.weight"))?,
                norm_q: weight(weights, &format!("{prefix}.attention.norm_q.weight"))?,
                norm_k: weight(weights, &format!("{prefix}.attention.norm_k.weight"))?,
            },
            attention_norm1: weight(weights, &format!("{prefix}.attn_norm1.weight"))?,
            attention_norm2: weight(weights, &format!("{prefix}.attn_norm2.weight"))?,
            ffn_norm1: weight(weights, &format!("{prefix}.ffn_norm1.weight"))?,
            ffn_norm2: weight(weights, &format!("{prefix}.ffn_norm2.weight"))?,
            w1: linear(weights, &format!("{prefix}.feed_forward.w1.weight"))?,
            w2: linear(weights, &format!("{prefix}.feed_forward.w2.weight"))?,
            w3: linear(weights, &format!("{prefix}.feed_forward.w3.weight"))?,
            adaln: Some((
                linear(weights, &format!("{prefix}.adaLN_modulation.0.weight"))?,
                weight(weights, &format!("{prefix}.adaLN_modulation.0.bias"))?,
            )),
        },
        config.heads as i32,
        ids,
        config.axes.map(|value| value as i32),
        config.rope_theta,
        config.norm_eps,
        Some(timestep),
        None,
        Some(rope),
        stream,
    )?;
    let hint = linear(weights, &format!("{prefix}.after_proj.weight"))?.apply(
        &control,
        Some(weight(weights, &format!("{prefix}.after_proj.bias"))?),
        stream,
    )?;
    let mut hints = branch.hints;
    hints.push(hint);
    Ok(ControlBranch {
        hints,
        stream: control,
    })
}

/// Control noise-refiner branch. Returns (refiner hints, refined_control).
/// `refined_control` is the last after_proj hint (main's `c[num_hints-1]`).
pub fn forward_control_refiner(
    control_weights: &MlxWeights,
    config: MlxZImageConfig,
    grid: TokenGrid,
    noise_stream: &Array,
    control_embed: &Array,
    image_rope: &Array,
    timestep: &Array,
    stream: &Stream,
) -> Result<(Vec<Array>, Array)> {
    let ids = geom::positions_image_first(grid);
    let image_ids = &ids[..grid.img_len as usize];
    let mut branch = ControlBranch {
        hints: Vec::new(),
        stream: control_embed.clone(),
    };
    for index in 0..CONTROL_REFINER_PLACES.len() {
        branch = control_block_forward(
            control_weights,
            config,
            &format!("control_noise_refiner.{index}"),
            index,
            branch,
            noise_stream,
            image_ids,
            image_rope,
            timestep,
            stream,
        )?;
    }
    if branch.hints.is_empty() {
        return Err(Error::new(
            "Z-Image control",
            "control noise refiner produced no hints",
        ));
    }
    let refined = branch.hints[branch.hints.len() - 1].clone();
    Ok((branch.hints, refined))
}

/// Control main-layer branch. Returns one hint per CONTROL_LAYER_PLACES entry.
pub fn forward_control_layers(
    control_weights: &MlxWeights,
    config: MlxZImageConfig,
    grid: TokenGrid,
    unified: &Array,
    refined_control: &Array,
    caption: &Array,
    unified_rope: &Array,
    timestep: &Array,
    stream: &Stream,
) -> Result<Vec<Array>> {
    let control_unified = Array::concatenate_axis(&[refined_control.clone(), caption.clone()], 1, stream)?;
    let ids = geom::positions_image_first(grid);
    let mut branch = ControlBranch {
        hints: Vec::new(),
        stream: control_unified,
    };
    for index in 0..CONTROL_LAYER_PLACES.len() {
        branch = control_block_forward(
            control_weights,
            config,
            &format!("control_layers.{index}"),
            index,
            branch,
            unified,
            &ids,
            unified_rope,
            timestep,
            stream,
        )?;
    }
    if branch.hints.len() != CONTROL_LAYER_PLACES.len() {
        return Err(Error::new(
            "Z-Image control",
            format!(
                "control layers produced {} hints, expected {}",
                branch.hints.len(),
                CONTROL_LAYER_PLACES.len()
            ),
        ));
    }
    Ok(branch.hints)
}

pub fn hint_for_refiner(layer_idx: usize, hints: &[Array]) -> Option<&Array> {
    control_refiner_hint_index(layer_idx).and_then(|idx| hints.get(idx))
}

pub fn hint_for_layer(layer_idx: usize, hints: &[Array]) -> Option<&Array> {
    control_layer_hint_index(layer_idx).and_then(|idx| hints.get(idx))
}

fn cast_to(input: &Array, dtype: Dtype, stream: &Stream) -> Result<Array> {
    if input.dtype() == dtype {
        Ok(input.clone())
    } else {
        input.astype(dtype, stream)
    }
}

fn weight<'a>(weights: &'a MlxWeights, name: &str) -> Result<&'a Array> {
    weights.tensor(name)
}

fn linear<'a>(weights: &'a MlxWeights, name: &str) -> Result<&'a LinearWeight> {
    weights.linear(name)
}

/// Map a loader/IO failure into [`WeightError`] for session boundaries.
pub fn weight_error(error: Error) -> WeightError {
    WeightError::NotBuilt(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use danmo_mlx::{Dtype, Stream};

    #[test]
    fn places_match_main() {
        assert_eq!(CONTROL_LAYER_PLACES.len(), 15);
        assert_eq!(CONTROL_REFINER_PLACES, [0, 1]);
        assert_eq!(control_layer_hint_index(0), Some(0));
        assert_eq!(control_layer_hint_index(2), Some(1));
        assert_eq!(control_layer_hint_index(1), None);
        assert_eq!(control_refiner_hint_index(1), Some(1));
        assert_eq!(control_embed_in_dim(2).unwrap(), 2 * 2 * 33);
    }

    #[test]
    fn hint_scales_and_rejects_shape_mismatch() {
        let stream = Stream::cpu().unwrap();
        let x = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let hint = Array::from_f32_slice(&[0.5, 0.5, 0.5, 0.5], &[1, 4]).unwrap();
        let out = apply_control_hint(&x, &hint, 2.0, &stream).unwrap();
        assert_eq!(out.to_vec_f32(&stream).unwrap(), vec![2.0, 3.0, 4.0, 5.0]);
        let zero = apply_control_hint(&x, &hint, 0.0, &stream).unwrap();
        assert_eq!(zero.to_vec_f32(&stream).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        let bad = Array::from_f32_slice(&[1.0], &[1, 1]).unwrap();
        assert!(apply_control_hint(&x, &bad, 1.0, &stream).is_err());
    }

    #[test]
    fn packs_33_channel_context() {
        let stream = Stream::cpu().unwrap();
        let latent = Array::from_f32_slice(&[1.0f32; 16 * 2 * 2], &[1, 16, 2, 2])
            .unwrap()
            .astype(Dtype::Float32, &stream)
            .unwrap();
        let ctx = build_control_context_nchw(&latent, &stream).unwrap();
        assert_eq!(ctx.shape(), vec![1, 33, 2, 2]);
        let values = ctx.to_vec_f32(&stream).unwrap();
        assert!(values[..64].iter().all(|v| (*v - 1.0).abs() < 1e-6));
        assert!(values[64..].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn packs_33_channel_inpaint_context() {
        let stream = Stream::cpu().unwrap();
        let control = Array::from_f32_slice(&[1.0f32; 16 * 2 * 2], &[1, 16, 2, 2]).unwrap();
        let mask = Array::from_f32_slice(&[0.25f32, 0.5, 0.75, 1.0], &[1, 1, 2, 2]).unwrap();
        let inpaint = Array::from_f32_slice(&[2.0f32; 16 * 2 * 2], &[1, 16, 2, 2]).unwrap();
        let ctx = build_control_context_inpaint_nchw(&control, &mask, &inpaint, &stream).unwrap();
        assert_eq!(ctx.shape(), vec![1, 33, 2, 2]);
        let values = ctx.to_vec_f32(&stream).unwrap();
        assert!(values[..64].iter().all(|v| (*v - 1.0).abs() < 1e-6));
        assert!((values[64] - 0.25).abs() < 1e-6);
        assert!((values[65] - 0.5).abs() < 1e-6);
        assert!((values[66] - 0.75).abs() < 1e-6);
        assert!((values[67] - 1.0).abs() < 1e-6);
        assert!(values[68..].iter().all(|v| (*v - 2.0).abs() < 1e-6));
        let bad_mask = Array::from_f32_slice(&[1.0f32], &[1, 1, 1, 1]).unwrap();
        assert!(build_control_context_inpaint_nchw(&control, &bad_mask, &inpaint, &stream).is_err());
    }

    #[test]
    fn dated_8steps_suffix_detection() {
        assert!(has_dated_8steps_suffix(
            "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2601-8steps.safetensors"
        ));
        assert!(!has_dated_8steps_suffix(
            "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors"
        ));
    }

    #[test]
    fn selects_newest_dated_shard_among_variants() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-cn-select-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for name in [
            "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2601-8steps.safetensors",
            "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors",
            "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors",
            "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors",
        ] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }
        let picked = select_control_shard(&dir).unwrap();
        assert!(
            picked.ends_with("Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors"),
            "{}",
            picked.display()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
