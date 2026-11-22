//! Qwen-Image edit denoise on the process executor.
//!
//! VL edit (Qwen-Image-Edit / FireRed) encodes each reference through the
//! vision tower and the VAE, concatenates the packed reference tokens after
//! the target noise, and steps only the target prefix. Base `qwen-image`
//! rewrite is fidelity img2img: the source latent is blended with noise at
//! `sigmas[init]` and the create text tower runs with no reference tokens.

use danmo_mlx::{Array, Dtype, Error, Stream};

use std::path::Path;

use super::forward::{self, RopeTables};
use super::resident::{self, cache_key};
use super::session::{self, resident_weights};
use super::text::{self, TextSpec};
use super::vision::{self, VisionGrid};
use crate::contract::ExecutionContext;
use crate::error::EngineError;
use crate::weights::{LoraFile, MlxWeights, WeightError};

pub(crate) struct PreparedEdit {
    pub width: u32,
    pub height: u32,
    pub source: (u32, u32),
    pub bundle: std::path::PathBuf,
    pub job: EditDenoise,
}

#[derive(Clone)]
pub(crate) struct EditDenoise {
    pub ids: Vec<i32>,
    pub neg_ids: Option<Vec<i32>>,
    pub image_pad: i32,
    pub pixels: Vec<f32>,
    pub grids: Vec<VisionGrid>,
    pub grid_h: i32,
    pub grid_w: i32,
    pub cond_grids: Vec<(i32, i32)>,
    pub cond: Vec<f32>,
    /// Fidelity img2img source, packed to the target grid. Empty for VL edit.
    pub reference: Vec<f32>,
    pub init: usize,
    pub seed: u64,
    pub sigmas: Vec<f32>,
    pub guidance: f32,
    pub loras: Vec<LoraFile>,
}

pub(crate) fn prepare(
    bundle: &Path,
    images: &[(Vec<u8>, u32, u32)],
    prompt: &str,
    negative: &str,
    vl: bool,
    picture_prefix: bool,
    output: (u32, u32),
    steps: u32,
    init: usize,
    guidance: f32,
    seed: i64,
    scheduler: Option<&str>,
    loras: Vec<LoraFile>,
) -> Result<PreparedEdit, EngineError> {
    if images.is_empty() {
        return Err(EngineError::bad_request(
            "qwen image edit has no source image",
        ));
    }
    let (width, height) = output;
    if width % 16 != 0 || height % 16 != 0 {
        return Err(
            WeightError::NotBuilt(format!("size {width}x{height} is not divisible by 16")).into(),
        );
    }
    resident::load_transformer(&bundle.join("transformer")).map_err(from_mlx)?;
    resident::load_text_encoder(&bundle.join("text_encoder")).map_err(from_mlx)?;
    resident::load_vae(&bundle.join("vae")).map_err(from_mlx)?;
    if vl {
        let text_key = cache_key(&bundle.join("text_encoder")).map_err(from_mlx)?;
        crate::runtime::mlx_executor()
            .map_err(from_mlx)?
            .execute("qwen_image edit vision", move |_, _, cache, _| {
                let weights = resident_weights(cache, &text_key, "text encoder")?;
                if !weights.has("encoder.visual.patch_embed.proj.weight") {
                    return Err(Error::new(
                        "Qwen-Image edit",
                        "text encoder has no vision tower; this bundle cannot run VL edit",
                    ));
                }
                Ok(())
            })
            .map_err(from_mlx)?;
    }
    let vocab = text_vocab(bundle)?;
    let (source_w, source_h) = (images[0].1, images[0].2);
    let (cond, cond_grids, pixels, grids, image_pad, ids, neg_ids, reference) = if vl {
        let vae_inputs = images
            .iter()
            .map(|(rgb, w, h)| {
                let (dw, dh) = vision::vae_condition_size(*w, *h)?;
                let rgb = resize(rgb, *w, *h, dw, dh)?;
                planar(&rgb, dw, dh)
            })
            .collect::<Result<Vec<_>, EngineError>>()?;
        let encoded = encode_vae(bundle, &vae_inputs)?;
        let mut cond = Vec::new();
        let mut cond_grids = Vec::new();
        for (packed, grid_h, grid_w) in encoded {
            cond.extend(packed);
            cond_grids.push((grid_h, grid_w));
        }
        let vl_size = if picture_prefix {
            None
        } else {
            Some(vision::condition_size(source_w, source_h)?)
        };
        let views: Vec<(&[u8], u32, u32)> = images
            .iter()
            .map(|(rgb, w, h)| (rgb.as_slice(), *w, *h))
            .collect();
        let (pixels, grids) = vision::preprocess(&views, picture_prefix, vl_size)?;
        let pad = text::image_pad_id(bundle, vocab)?;
        let counts: Vec<usize> = grids.iter().map(|grid| grid.tokens()).collect();
        let ids = edit_ids(
            bundle,
            prompt,
            images.len(),
            picture_prefix,
            pad,
            &counts,
            vocab,
        )?;
        let neg_ids = if guidance > 1.0 {
            let negative = if negative.trim().is_empty() {
                ""
            } else {
                negative
            };
            Some(edit_ids(
                bundle,
                negative,
                images.len(),
                picture_prefix,
                pad,
                &counts,
                vocab,
            )?)
        } else {
            None
        };
        (
            cond,
            cond_grids,
            pixels,
            grids,
            pad,
            ids,
            neg_ids,
            Vec::new(),
        )
    } else {
        let rgb = resize(&images[0].0, source_w, source_h, width, height)?;
        let planar = planar(&rgb, width, height)?;
        let encoded = encode_vae(bundle, &[planar])?;
        let (packed, grid_h, grid_w) = encoded
            .into_iter()
            .next()
            .ok_or_else(|| EngineError::message("qwen image edit encoded no latent"))?;
        if grid_h != (height / 16) as i32 || grid_w != (width / 16) as i32 {
            return Err(WeightError::NotBuilt(format!(
                "vae grid {grid_w}x{grid_h} does not match {width}x{height}"
            ))
            .into());
        }
        let ids = text::prompt_ids(bundle, prompt, vocab)?;
        let neg_ids = if guidance > 1.0 {
            let negative = if negative.trim().is_empty() {
                " "
            } else {
                negative
            };
            Some(text::prompt_ids(bundle, negative, vocab)?)
        } else {
            None
        };
        (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            0,
            ids,
            neg_ids,
            packed,
        )
    };
    let sigmas = super::super::schedule(steps, width, height, 16, scheduler, None, false)?;
    if init + 1 >= sigmas.len() {
        return Err(WeightError::NotBuilt(format!(
            "edit starts at step {init} of a {}-step schedule",
            sigmas.len().saturating_sub(1)
        ))
        .into());
    }
    let mut job = EditDenoise {
        ids,
        neg_ids,
        image_pad,
        pixels,
        grids,
        grid_h: (height / 16) as i32,
        grid_w: (width / 16) as i32,
        cond_grids,
        cond,
        reference,
        init,
        seed: seed as u64,
        sigmas,
        guidance,
        loras,
    };
    Ok(PreparedEdit {
        width,
        height,
        source: (source_w, source_h),
        bundle: bundle.to_path_buf(),
        job,
    })
}

fn edit_ids(
    bundle: &Path,
    prompt: &str,
    images: usize,
    picture_prefix: bool,
    pad: i32,
    counts: &[usize],
    vocab: i64,
) -> Result<Vec<i32>, EngineError> {
    let text = text::edit_prompt_text(prompt, images, picture_prefix);
    let ids = crate::common::qwen3::encode_plain(bundle, &text, vocab)?;
    text::expand_image_pads(&ids, pad, counts).map_err(EngineError::from)
}

fn encode_vae(
    bundle: &Path,
    images: &[(Vec<f32>, u32, u32)],
) -> Result<Vec<(Vec<f32>, i32, i32)>, EngineError> {
    let vae_key = cache_key(&bundle.join("vae")).map_err(from_mlx)?;
    let owned: Vec<(Vec<f32>, i32, i32)> = images
        .iter()
        .map(|(pixels, w, h)| (pixels.clone(), *w as i32, *h as i32))
        .collect();
    crate::runtime::mlx_executor()
        .map_err(from_mlx)?
        .execute("qwen_image edit vae", move |_, stream, cache, _| {
            let weights = resident_weights(cache, &vae_key, "vae")?;
            let mut encoded = Vec::with_capacity(owned.len());
            for (pixels, width, height) in &owned {
                let image = Array::from_f32_slice(pixels, &[1, 3, *height, *width])?;
                encoded.push(super::vae::encode_packed(weights, &image, stream)?);
            }
            Ok(encoded)
        })
        .map_err(from_mlx)
}

fn text_vocab(bundle: &Path) -> Result<i64, EngineError> {
    let key = cache_key(&bundle.join("text_encoder")).map_err(from_mlx)?;
    crate::runtime::mlx_executor()
        .map_err(from_mlx)?
        .execute("qwen_image edit text spec", move |_, _, cache, _| {
            let weights = resident_weights(cache, &key, "text encoder")?;
            Ok(text::text_spec(weights)?.vocab)
        })
        .map_err(from_mlx)
}

fn planar(rgb: &[u8], width: u32, height: u32) -> Result<(Vec<f32>, u32, u32), EngineError> {
    let pixels = crate::common::image::rgb_to_n11_planar(rgb, width, height)?;
    Ok((pixels, width, height))
}

fn resize(rgb: &[u8], width: u32, height: u32, dw: u32, dh: u32) -> Result<Vec<u8>, EngineError> {
    if width == dw && height == dh {
        return Ok(rgb.to_vec());
    }
    crate::common::image::resize_bicubic(rgb, width, height, dw, dh).map_err(EngineError::from)
}

fn from_mlx(error: Error) -> EngineError {
    WeightError::NotBuilt(error.to_string()).into()
}

pub(crate) fn denoise(
    cache: &mut danmo_mlx::ModelCache,
    text_key: &danmo_mlx::CacheKey,
    dit_key: &danmo_mlx::CacheKey,
    job: &EditDenoise,
    exec: &ExecutionContext,
    stream: &Stream,
) -> danmo_mlx::Result<Vec<f32>> {
    if job.sigmas.len() < 2 || job.init + 1 >= job.sigmas.len() {
        return Err(Error::new(
            "Qwen-Image edit",
            "sigma schedule has no step left",
        ));
    }
    let spec = {
        let weights = resident_weights(cache, dit_key, "transformer")?;
        forward::assert_contract(weights)?
    };
    let (text, txt_mask, seq_txt) = encode_text(cache, text_key, job, stream)?;
    text.eval()?;
    let weights = resident_weights(cache, dit_key, "transformer")?.clone();
    let weights = session::attach_dit_lora(weights, &job.loras, stream)?;

    let target = (job.grid_h * job.grid_w) as usize;
    let cond_tokens = job.cond.len() / spec.in_channels as usize;
    if !job.cond.is_empty() && job.cond.len() != cond_tokens * spec.in_channels as usize {
        return Err(Error::new(
            "Qwen-Image edit",
            "reference latent is truncated",
        ));
    }
    let seq_img = target + cond_tokens;
    let rope = rope_for(job, seq_txt, stream)?;
    if rope.img_cos.shape()[0] != seq_img as i32 {
        return Err(Error::new(
            "Qwen-Image edit",
            format!(
                "RoPE image length {} is not the target plus reference length {seq_img}",
                rope.img_cos.shape()[0]
            ),
        ));
    }

    let cfg = job.neg_ids.is_some();
    let noise = danmo_mlx::RandomKey::new(job.seed)?
        .normal(
            &[1, target as i32, spec.in_channels],
            Dtype::Float32,
            stream,
        )?
        .astype(Dtype::BFloat16, stream)?;
    let mut latent = if job.reference.is_empty() || job.init == 0 {
        noise
    } else {
        let sigma = job.sigmas[job.init];
        let clean = Array::from_f32_slice(&job.reference, &[1, target as i32, spec.in_channels])?
            .astype(Dtype::BFloat16, stream)?;
        let sigma = Array::from_f32_slice(&[sigma], &[])?.astype(Dtype::BFloat16, stream)?;
        let one = Array::from_f32_slice(&[1.0], &[])?.astype(Dtype::BFloat16, stream)?;
        let keep = Array::subtract(&one, &sigma, stream)?;
        Array::add(
            &Array::multiply(&clean, &keep, stream)?,
            &Array::multiply(&noise, &sigma, stream)?,
            stream,
        )?
    };
    let cond = if job.cond.is_empty() {
        None
    } else {
        Some(
            Array::from_f32_slice(&job.cond, &[1, cond_tokens as i32, spec.in_channels])?
                .astype(latent.dtype(), stream)?,
        )
    };

    let steps = job.sigmas.len() - 1 - job.init;
    for (index, pair) in job.sigmas[job.init..].windows(2).enumerate() {
        exec.check_cancelled()
            .map_err(|error| Error::new("Qwen-Image edit", error.to_string()))?;
        let packed = match &cond {
            Some(cond) => Array::concatenate_axis(&[latent.clone(), cond.clone()], 1, stream)?,
            None => latent.clone(),
        };
        let packed = if cfg {
            Array::concatenate_axis(&[packed.clone(), packed], 0, stream)?
        } else {
            packed
        };
        let velocity = forward::forward(
            &weights,
            spec,
            &packed,
            &text,
            pair[0],
            &rope,
            txt_mask.as_ref(),
            stream,
        )?;
        let velocity = velocity.slice(
            &[0, 0, 0],
            &[velocity.shape()[0], target as i32, spec.in_channels],
            &[1, 1, 1],
            stream,
        )?;
        let velocity = if cfg {
            let cond_v = velocity.slice(
                &[0, 0, 0],
                &[1, target as i32, spec.in_channels],
                &[1, 1, 1],
                stream,
            )?;
            let uncond = velocity.slice(
                &[1, 0, 0],
                &[2, target as i32, spec.in_channels],
                &[1, 1, 1],
                stream,
            )?;
            session::guided_velocity(&cond_v, &uncond, job.guidance, spec.in_channels, stream)?
        } else {
            velocity
        };
        velocity.eval()?;
        let finite = velocity
            .isfinite_all(stream)?
            .astype(Dtype::Float32, stream)?
            .to_vec_f32(stream)?;
        if finite.first().copied() != Some(1.0) {
            return Err(Error::new(
                "Qwen-Image edit",
                format!(
                    "dit velocity at step {} is non-finite",
                    job.init + index + 1
                ),
            ));
        }
        let dt = pair[1] - pair[0];
        let step = Array::from_f32_slice(&[dt], &[])?.astype(latent.dtype(), stream)?;
        let velocity = velocity.astype(latent.dtype(), stream)?;
        latent = Array::add(&latent, &Array::multiply(&velocity, &step, stream)?, stream)?;
        latent.eval()?;
        exec.progress("denoise", index as u32 + 1, steps as u32)
            .map_err(|error| Error::new("Qwen-Image edit", error.to_string()))?;
    }
    let values = latent.to_vec_f32(stream)?;
    if values.len() != target * spec.in_channels as usize {
        return Err(Error::new(
            "Qwen-Image edit",
            format!(
                "latent has {} values, expected {}",
                values.len(),
                target * spec.in_channels as usize
            ),
        ));
    }
    Ok(values)
}

fn encode_text(
    cache: &mut danmo_mlx::ModelCache,
    text_key: &danmo_mlx::CacheKey,
    job: &EditDenoise,
    stream: &Stream,
) -> danmo_mlx::Result<(Array, Option<Array>, usize)> {
    let weights = resident_weights(cache, text_key, "text encoder")?;
    let spec = text::text_spec(weights)?;
    let image = if job.pixels.is_empty() {
        None
    } else {
        let embeds = vision::forward(weights, &job.pixels, &job.grids, stream)?;
        if embeds.shape()[1] != spec.hidden {
            return Err(Error::new(
                "Qwen-Image edit",
                format!(
                    "vision width {} does not match text hidden {}",
                    embeds.shape()[1],
                    spec.hidden
                ),
            ));
        }
        Some(embeds)
    };
    let pos = encode_branch(
        weights,
        spec,
        &job.ids,
        job.image_pad,
        image.as_ref(),
        stream,
    )?;
    let pos_len = pos.shape()[1];
    match &job.neg_ids {
        None => Ok((pos, None, pos_len as usize)),
        Some(neg) => {
            let neg = encode_branch(weights, spec, neg, job.image_pad, image.as_ref(), stream)?;
            let neg_len = neg.shape()[1];
            let joint = pos_len.max(neg_len);
            let pos = session::pad_sequence(&pos, joint, stream)?;
            let neg = session::pad_sequence(&neg, joint, stream)?;
            let text = Array::concatenate_axis(&[pos, neg], 0, stream)?;
            let seq_img =
                job.grid_h * job.grid_w + job.cond_grids.iter().map(|(h, w)| h * w).sum::<i32>();
            let mask = forward::joint_text_bias(
                &[
                    session::valid_prefix(joint, pos_len),
                    session::valid_prefix(joint, neg_len),
                ],
                seq_img,
                stream,
            )?;
            Ok((text, Some(mask), joint as usize))
        }
    }
}

fn encode_branch(
    weights: &MlxWeights,
    spec: TextSpec,
    ids: &[i32],
    image_pad: i32,
    image: Option<&Array>,
    stream: &Stream,
) -> danmo_mlx::Result<Array> {
    match image {
        Some(image) => text::encode_edit_prompt(weights, spec, ids, image_pad, image, stream),
        None => text::encode_prompt(weights, spec, ids, stream),
    }
}

fn rope_for(job: &EditDenoise, seq_txt: usize, stream: &Stream) -> danmo_mlx::Result<RopeTables> {
    let mut grids = vec![(i64::from(job.grid_h), i64::from(job.grid_w))];
    grids.extend(
        job.cond_grids
            .iter()
            .map(|(h, w)| (i64::from(*h), i64::from(*w))),
    );
    let ids = super::super::ids::position_ids_frames(&grids, seq_txt as i64)
        .map_err(|error| Error::new("Qwen-Image edit", error.to_string()))?;
    let (img_cos, img_sin) = forward::rope_from_positions(&ids.image, stream)?;
    let (txt_cos, txt_sin) = forward::rope_from_positions(&ids.text, stream)?;
    Ok(RopeTables {
        img_cos,
        img_sin,
        txt_cos,
        txt_sin,
    })
}
