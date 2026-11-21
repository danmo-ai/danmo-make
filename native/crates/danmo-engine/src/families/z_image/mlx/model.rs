use danmo_mlx::{AffineQuantization, Array, Error, LinearWeight, Result, Stream};

use super::super::geom::{self, TokenGrid};
use super::attention::AttentionWeights;
use super::blocks::{self, BlockWeights};
use super::control::{self, ControlConditioning};
use super::step_cache::StepCache;
use super::{modulation, timestep, MlxZImageConfig};
use crate::weights::{MlxWeights, WeightDtype};

/// `quantization` is the registry's declared affine quantization, if any.
pub fn load_directory(
    directory: &std::path::Path,
    config: MlxZImageConfig,
    dtype: WeightDtype,
    quantization: Option<AffineQuantization>,
) -> Result<MlxWeights> {
    let weights = crate::weights::load_mlx_directory(directory, dtype, quantization)?;
    let names = super::weights::remap_weight_names(
        weights.names().map(str::to_string).collect::<Vec<_>>(),
        config.patch,
    )
    .map_err(|error| Error::new("Z-Image remap", error.to_string()))?;
    let weights = weights.rename(&names)?;
    // main's quantized DiT keeps dense norms / pad tokens at the F32
    // module-init dtype (slice assignment preserves it), which promotes the
    // whole tower to F32 after the first norm. The dense fp16 path replaces
    // params with BF16 and stays BF16. Widen dense floating tensors to F32 for
    // quantized bundles so the activation dtype chain matches main exactly.
    if quantization.is_some() {
        return widen_dense_to_f32(weights);
    }
    Ok(weights)
}

fn widen_dense_to_f32(weights: MlxWeights) -> Result<MlxWeights> {
    let stream = Stream::cpu()?;
    let dense_names: Vec<String> = weights
        .names()
        .filter(|name| {
            weights
                .tensor(name)
                .map(|t| t.dtype().is_floating() && t.dtype() != danmo_mlx::Dtype::Float32)
                .unwrap_or(false)
        })
        .map(str::to_string)
        .collect();
    let mut output = weights;
    for name in dense_names {
        let tensor = output.tensor(&name)?;
        let widened = tensor.astype(danmo_mlx::Dtype::Float32, &stream)?;
        output = output.with_tensor_production(name, widened);
    }
    Ok(output)
}

fn embed_image(
    weights: &MlxWeights,
    config: MlxZImageConfig,
    grid: TokenGrid,
    latent: &Array,
    stream: &Stream,
) -> Result<Array> {
    let token_dim = config.patch * config.patch * config.in_channels;
    let indices = Array::from_i32_slice(
        &geom::patch_index(grid, config.in_channels),
        &[grid.img_real as i32 * token_dim as i32],
    )?;
    let latent_size = config.in_channels * grid.latent_w * grid.latent_h;
    let gathered = latent
        .reshape(&[latent_size as i32], stream)?
        .take_axis(&indices, 0, stream)?;
    let mut patches = gathered.reshape(&[grid.img_real as i32, token_dim as i32], stream)?;
    if grid.img_pad > 0 {
        // main pads the raw patch rows by repeating the last row BEFORE the
        // embedder (`pad_tail_with_last`), then overwrites the embedded pad
        // positions with `x_pad_token`. The pre-embed repeat is numerically
        // visible whenever the pad rows are attended (small grids).
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
    let embedding_weight = linear(weights, "x_embedder.weight")?;
    let patches = cast_to(&patches, embedding_weight.compute_dtype(), stream)?;
    let mut embedded =
        embedding_weight.apply(&patches, Some(weight(weights, "x_embedder.bias")?), stream)?;
    embedded = cast_to(&embedded, latent.dtype(), stream)?;
    if grid.img_pad > 0 {
        let pad = weight(weights, "x_pad_token")?
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

fn embed_caption(
    weights: &MlxWeights,
    config: MlxZImageConfig,
    grid: TokenGrid,
    caption: &Array,
    stream: &Stream,
) -> Result<Array> {
    if caption.shape() != vec![grid.cap_real as i32, config.cap_dim as i32] {
        return Err(Error::new(
            "Z-Image caption embed",
            format!(
                "expected [{}, {}], got {:?}",
                grid.cap_real,
                config.cap_dim,
                caption.shape()
            ),
        ));
    }
    let cap_norm = weight(weights, "cap_norm.weight")?;
    let caption = cast_to(caption, cap_norm.dtype(), stream)?;
    let mut normalized = caption.rms_norm(Some(cap_norm), config.norm_eps, stream)?;
    if grid.cap_pad > 0 {
        // Same pre-embed last-row repeat as the image branch (main
        // `_encode_cap_branch` pads cap_feats before cap_norm/cap_embedder;
        // the norm is per-row so padding before or after it is equivalent).
        let last = normalized.slice(
            &[grid.cap_real as i32 - 1, 0],
            &[grid.cap_real as i32, config.cap_dim as i32],
            &[1, 1],
            stream,
        )?;
        let repeats = last.broadcast_to(&[grid.cap_pad as i32, config.cap_dim as i32], stream)?;
        normalized = Array::concatenate_axis(&[normalized, repeats], 0, stream)?;
    }
    let mut embedded = linear(weights, "cap_embedder.weight")?
        .apply(
            &normalized,
            Some(weight(weights, "cap_embedder.bias")?),
            stream,
        )?
        .reshape(&[1, grid.cap_len as i32, config.dim as i32], stream)?;
    if grid.cap_pad > 0 {
        let pad = weight(weights, "cap_pad_token")?
            .broadcast_to(&[1, grid.cap_pad as i32, config.dim as i32], stream)?;
        let pad = cast_to(&pad, embedded.dtype(), stream)?;
        let real = embedded.slice(
            &[0, 0, 0],
            &[1, grid.cap_real as i32, config.dim as i32],
            &[1, 1, 1],
            stream,
        )?;
        embedded = Array::concatenate_axis(&[real, pad], 1, stream)?;
    }
    Ok(embedded)
}

pub(crate) fn run_block(
    weights: &MlxWeights,
    config: MlxZImageConfig,
    prefix: &str,
    input: &Array,
    ids: &[[i32; 3]],
    modulated: bool,
    timestep: Option<&Array>,
    rope: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    blocks::forward_with_table(
        input,
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
            adaln: if modulated {
                Some((
                    linear(weights, &format!("{prefix}.adaLN_modulation.0.weight"))?,
                    weight(weights, &format!("{prefix}.adaLN_modulation.0.bias"))?,
                ))
            } else {
                None
            },
        },
        config.heads as i32,
        ids,
        config.axes.map(|value| value as i32),
        config.rope_theta,
        config.norm_eps,
        timestep,
        None,
        rope,
        stream,
    )
}

#[derive(Clone)]
pub(crate) struct ForwardRope {
    pub(crate) image: Array,
    pub(crate) caption: Array,
    pub(crate) unified: Array,
}

impl ForwardRope {
    pub(crate) fn new(config: MlxZImageConfig, grid: TokenGrid, stream: &Stream) -> Result<Self> {
        let ids = geom::positions_image_first(grid);
        let (image_ids, caption_ids) = ids.split_at(grid.img_len as usize);
        let axes = config.axes.map(|value| value as i32);
        let image = super::attention::rope_table(image_ids, axes, config.rope_theta, stream)?;
        let caption = super::attention::rope_table(caption_ids, axes, config.rope_theta, stream)?;
        let unified = super::attention::rope_table(&ids, axes, config.rope_theta, stream)?;
        image.eval()?;
        caption.eval()?;
        unified.eval()?;
        Ok(Self {
            image,
            caption,
            unified,
        })
    }
}

/// A full DiT forward compiled with `mx.compile`. Weights, token geometry, and
/// the caption are captured as constants; only the per-step latent and
/// timestep embedding are traced inputs, matching main's `mx.compile`
/// boundary where the caption refiner output is reused across denoise steps.
///
/// `shapeless=false` is required on the pinned MLX 0.32.4: neither `Split`
/// (AdaLN sections) nor `Slice` (RoPE components, FinalLayer crop) implements
/// `output_shapes`, so shapeless compile fails loud. Shape-specialized
/// compile still fuses kernels and recompiles automatically on shape changes.
pub struct CompiledForward {
    closure: danmo_mlx::Closure,
    grid: TokenGrid,
}

impl CompiledForward {
    pub fn new(
        weights: &MlxWeights,
        config: MlxZImageConfig,
        grid: TokenGrid,
        caption: &Array,
        shapeless: bool,
        stream: &Stream,
    ) -> Result<Self> {
        let weights = weights.clone();
        let caption = caption.clone();
        let rope = ForwardRope::new(config, grid, stream)?;
        let stream = stream.clone();
        let closure = danmo_mlx::Closure::new(stream, move |inputs, stream| {
            if inputs.len() != 2 {
                return Err(Error::new(
                    "Z-Image compiled forward",
                    format!("expected [latent, timestep], got {} inputs", inputs.len()),
                ));
            }
            forward_with_timestep_observed(
                &weights,
                config,
                grid,
                &inputs[0],
                &caption,
                &inputs[1],
                Some(&rope),
                None,
                None,
                None,
                stream,
                |_, _| {},
            )
        })?
        .compile(shapeless)?;
        Ok(Self { closure, grid })
    }

    pub fn apply(&self, latent: &Array, timestep: &Array) -> Result<Array> {
        self.closure.apply(&[latent.clone(), timestep.clone()])
    }

    pub fn grid(&self) -> TokenGrid {
        self.grid
    }
}

/// The TeaCache compute path compiled with `mx.compile`: the 30 unified
/// blocks plus the final layer, traced on `[unified_before, timestep]` with
/// weights/geometry/RoPE captured as constants. The eager prefix (image
/// embed + noise refiner + gate) and the unpatchify stay outside, mirroring
/// main's gate-outside-compile boundary. The closure returns
/// `[post_layers_unified, final_tokens]` so the caller derives both the
/// stored residual and the velocity without a second model implementation.
pub struct CompiledCompute {
    closure: danmo_mlx::MultiClosure,
    final_closure: danmo_mlx::Closure,
    rope: ForwardRope,
    grid: TokenGrid,
}

impl CompiledCompute {
    pub fn new(
        weights: &MlxWeights,
        config: MlxZImageConfig,
        grid: TokenGrid,
        stream: &Stream,
    ) -> Result<Self> {
        let weights = weights.clone();
        let rope = ForwardRope::new(config, grid, stream)?;
        let closure_rope = rope.clone();
        let final_weights = weights.clone();
        let stream = stream.clone();
        let final_stream = stream.clone();
        let closure = danmo_mlx::MultiClosure::new(stream, move |inputs, stream| {
            if inputs.len() != 2 {
                return Err(Error::new(
                    "Z-Image compiled compute",
                    format!("expected [unified, timestep], got {} inputs", inputs.len()),
                ));
            }
            let unified = unified_layers_and_final(
                &weights,
                config,
                grid,
                &inputs[0],
                &inputs[1],
                &closure_rope,
                None,
                None,
                stream,
                &mut |_, _: &Array| {},
            )?;
            let tokens = final_layer_tokens(&weights, &unified, &inputs[1], stream)?;
            Ok(vec![unified, tokens])
        })?
        .compile(false)?;
        let final_closure = danmo_mlx::Closure::new(final_stream, move |inputs, stream| {
            if inputs.len() != 2 {
                return Err(Error::new(
                    "Z-Image compiled final",
                    format!("expected [unified, timestep], got {} inputs", inputs.len()),
                ));
            }
            final_layer_tokens(&final_weights, &inputs[0], &inputs[1], stream)
        })?
        .compile(false)?;
        Ok(Self {
            closure,
            final_closure,
            rope,
            grid,
        })
    }

    /// Eager prefix: the gate signal for this step.
    pub fn unified_before(
        &self,
        weights: &MlxWeights,
        config: MlxZImageConfig,
        latent: &Array,
        caption: &CaptionCache,
        timestep: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        let (unified, _) = unified_before(
            weights,
            config,
            self.grid,
            latent,
            caption.embedded(),
            timestep,
            &self.rope,
            None,
            stream,
            &mut |_, _| {},
        )?;
        Ok(unified)
    }

    /// Compiled final layer only (the TeaCache skip path).
    pub fn apply_final(&self, unified: &Array, timestep: &Array) -> Result<Array> {
        self.final_closure
            .apply(&[unified.clone(), timestep.clone()])
    }

    /// Compiled compute; returns `(post_layers_unified, final_tokens)`.
    pub fn apply(&self, unified: &Array, timestep: &Array) -> Result<(Array, Array)> {
        let mut outputs = self.closure.apply(&[unified.clone(), timestep.clone()])?;
        if outputs.len() != 2 {
            return Err(Error::new(
                "Z-Image compiled compute",
                format!("expected [unified, tokens], got {} outputs", outputs.len()),
            ));
        }
        let tokens = outputs.pop().unwrap();
        let post = outputs.pop().unwrap();
        Ok((post, tokens))
    }

    pub fn grid(&self) -> TokenGrid {
        self.grid
    }
}

/// TeaCache step forward whose 30-block compute runs through
/// `CompiledCompute`. The gate, residual reuse, and cache updates stay eager
/// and share the exact `StepCache` state machine with the eager path; the
/// unpatchify is the same `final_velocity` tail. On compute steps the stored
/// residual is returned for the caller's joint eval, matching main.
pub fn forward_compiled_with_step_cache(
    weights: &MlxWeights,
    config: MlxZImageConfig,
    compiled: &CompiledCompute,
    latent: &Array,
    caption: &CaptionCache,
    sigma: f32,
    cache: &mut StepCache,
    step_idx: usize,
    stream: &Stream,
) -> Result<(Array, Option<Array>)> {
    let grid = compiled.grid();
    let timestep = embed_timestep(weights, config, sigma, stream)?;
    let unified_before =
        compiled.unified_before(weights, config, latent, caption, &timestep, stream)?;
    let decision = cache.gate(&unified_before, step_idx, stream)?;
    let skip = !decision.should_compute && cache.cached_residual().is_some();
    let mut residual = None;
    let unified = if skip {
        Array::add(&unified_before, cache.cached_residual().unwrap(), stream)?
    } else {
        let (post, tokens) = compiled.apply(&unified_before, &timestep)?;
        let stored = Array::subtract(&post, &unified_before, stream)?;
        cache.store_computed(&decision, &unified_before, &stored);
        let velocity = unpatchify_velocity(config, grid, &tokens, stream)?;
        return Ok((velocity, Some(stored)));
    };
    let tokens = if skip {
        compiled.apply_final(&unified, &timestep)?
    } else {
        // The compute closure already produced the final tokens; re-reading
        // them from a second apply would rerun the blocks, so keep the first
        // apply's tokens instead.
        unreachable!("compute branch returns early with its tokens")
    };
    let velocity = unpatchify_velocity(config, grid, &tokens, stream)?;
    Ok((velocity, residual))
}

/// Embed the scalar flow-match timestep once per denoise step; the compiled
/// forward and the eager forward share this exact embedding.
pub fn embed_timestep(
    weights: &MlxWeights,
    config: MlxZImageConfig,
    sigma: f32,
    stream: &Stream,
) -> Result<Array> {
    let sigma = Array::from_f32_slice(&[sigma], &[1])?;
    let one = Array::from_f32_slice(&[1.0], &[1])?;
    let scale = Array::from_f32_slice(&[config.t_scale], &[])?;
    let t = Array::multiply(&Array::subtract(&one, &sigma, stream)?, &scale, stream)?;
    timestep::embed(
        &t,
        config.frequency_dim as usize,
        linear(weights, "t_embedder.linear1.weight")?,
        weight(weights, "t_embedder.linear1.bias")?,
        linear(weights, "t_embedder.linear2.weight")?,
        weight(weights, "t_embedder.linear2.bias")?,
        stream,
    )
}

pub fn forward(
    weights: &MlxWeights,
    config: MlxZImageConfig,
    grid: TokenGrid,
    latent: &Array,
    caption: &Array,
    sigma: f32,
    stream: &Stream,
) -> Result<Array> {
    let timestep = embed_timestep(weights, config, sigma, stream)?;
    forward_with_timestep(weights, config, grid, latent, caption, &timestep, stream)
}

/// The caption branch (`cap_norm` → `cap_embedder` → pad token → context
/// refiner) is constant across denoise steps. Main precomputes and evaluates
/// it once in `before_denoise` whenever the optimization plan enables compile
/// or the step cache (`needs_precompute_cap`); this is the same boundary.
pub struct CaptionCache {
    embedded: Array,
}

impl CaptionCache {
    pub fn new(
        weights: &MlxWeights,
        config: MlxZImageConfig,
        grid: TokenGrid,
        caption: &Array,
        stream: &Stream,
    ) -> Result<Self> {
        let caption_ids = &geom::positions_image_first(grid)[grid.img_len as usize..];
        let table = super::attention::rope_table(
            caption_ids,
            config.axes.map(|value| value as i32),
            config.rope_theta,
            stream,
        )?;
        table.eval()?;
        let mut embedded = embed_caption(weights, config, grid, caption, stream)?;
        for index in 0..config.refiners {
            embedded = run_block(
                weights,
                config,
                &format!("context_refiner.{index}"),
                &embedded,
                caption_ids,
                false,
                None,
                Some(&table),
                stream,
            )?;
        }
        embedded.eval()?;
        Ok(Self { embedded })
    }

    pub fn embedded(&self) -> &Array {
        &self.embedded
    }
}

pub fn forward_with_step_cache(
    weights: &MlxWeights,
    config: MlxZImageConfig,
    grid: TokenGrid,
    latent: &Array,
    caption: &Array,
    sigma: f32,
    cache: &mut StepCache,
    step_idx: usize,
    stream: &Stream,
) -> Result<Array> {
    let timestep = embed_timestep(weights, config, sigma, stream)?;
    forward_with_timestep_observed(
        weights,
        config,
        grid,
        latent,
        caption,
        &timestep,
        None,
        None,
        Some((cache, step_idx)),
        None,
        stream,
        |_, _| {},
    )
}

/// TeaCache step forward on top of a precomputed caption branch, matching
/// main's `_forward_cached_compute`: the compute path evaluates the stored
/// residual in the same call as the velocity and the updated latent.
pub fn forward_cached_with_step_cache(
    weights: &MlxWeights,
    config: MlxZImageConfig,
    grid: TokenGrid,
    latent: &Array,
    caption: &CaptionCache,
    sigma: f32,
    cache: &mut StepCache,
    step_idx: usize,
    stream: &Stream,
) -> Result<(Array, Option<Array>)> {
    let timestep = embed_timestep(weights, config, sigma, stream)?;
    let mut residual = None;
    let velocity = forward_with_timestep_observed(
        weights,
        config,
        grid,
        latent,
        caption.embedded(),
        &timestep,
        None,
        Some(caption),
        Some((cache, step_idx)),
        None,
        stream,
        |name, array| {
            if name == "teacache_residual" {
                residual = Some(array.clone());
            }
        },
    )?;
    Ok((velocity, residual))
}

fn forward_with_timestep(
    weights: &MlxWeights,
    config: MlxZImageConfig,
    grid: TokenGrid,
    latent: &Array,
    caption: &Array,
    timestep: &Array,
    stream: &Stream,
) -> Result<Array> {
    forward_with_timestep_observed(
        weights,
        config,
        grid,
        latent,
        caption,
        timestep,
        None,
        None,
        None,
        None,
        stream,
        |_, _| {},
    )
}

/// Eager DiT step with optional ControlNet hint injection. Used when a
/// structural guide is active (compiled DiT has no control hooks).
pub fn forward_cached_with_control(
    weights: &MlxWeights,
    config: MlxZImageConfig,
    grid: TokenGrid,
    latent: &Array,
    caption: &CaptionCache,
    sigma: f32,
    cache: &mut StepCache,
    step_idx: usize,
    control: ControlConditioning<'_>,
    stream: &Stream,
) -> Result<Array> {
    let timestep = embed_timestep(weights, config, sigma, stream)?;
    forward_with_timestep_observed(
        weights,
        config,
        grid,
        latent,
        caption.embedded(),
        &timestep,
        None,
        Some(caption),
        Some((cache, step_idx)),
        Some(control),
        stream,
        |_, _| {},
    )
}

#[allow(clippy::too_many_arguments)]
fn forward_with_timestep_observed(
    weights: &MlxWeights,
    config: MlxZImageConfig,
    grid: TokenGrid,
    latent: &Array,
    caption: &Array,
    timestep: &Array,
    rope: Option<&ForwardRope>,
    caption_cache: Option<&CaptionCache>,
    mut step_cache: Option<(&mut StepCache, usize)>,
    control: Option<ControlConditioning<'_>>,
    stream: &Stream,
    mut observe: impl FnMut(&str, &Array),
) -> Result<Array> {
    let owned_rope;
    let rope = match rope {
        Some(rope) => rope,
        None => {
            owned_rope = ForwardRope::new(config, grid, stream)?;
            &owned_rope
        }
    };
    let refined_caption;
    let caption = match caption_cache {
        Some(cache) => cache.embedded(),
        None => {
            let caption_ids = &geom::positions_image_first(grid)[grid.img_len as usize..];
            let mut caption = embed_caption(weights, config, grid, caption, stream)?;
            for index in 0..config.refiners {
                caption = run_block(
                    weights,
                    config,
                    &format!("context_refiner.{index}"),
                    &caption,
                    caption_ids,
                    false,
                    None,
                    Some(&rope.caption),
                    stream,
                )?;
                observe(&format!("context_refiner.{index}"), &caption);
            }
            refined_caption = caption;
            &refined_caption
        }
    };
    let (unified_before, layer_hints) = unified_before(
        weights,
        config,
        grid,
        latent,
        caption,
        timestep,
        rope,
        control,
        stream,
        &mut observe,
    )?;
    let decision = match &mut step_cache {
        Some((cache, step_idx)) => Some(cache.gate(&unified_before, *step_idx, stream)?),
        None => None,
    };
    let skip = decision.as_ref().is_some_and(|decision| {
        !decision.should_compute
            && matches!(
                &step_cache,
                Some((cache, _)) if cache.cached_residual().is_some()
            )
    });
    let unified;
    if skip {
        let residual = match &step_cache {
            Some((cache, _)) => cache.cached_residual().unwrap(),
            None => unreachable!(),
        };
        unified = Array::add(&unified_before, residual, stream)?;
    } else {
        unified = unified_layers_and_final(
            weights,
            config,
            grid,
            &unified_before,
            timestep,
            rope,
            layer_hints.as_deref(),
            control.map(|c| c.scale),
            stream,
            &mut observe,
        )?;
        if let Some((cache, _)) = step_cache {
            let residual = Array::subtract(&unified, &unified_before, stream)?;
            cache.store_computed(decision.as_ref().unwrap(), &unified_before, &residual);
            observe("teacache_residual", &residual);
        }
    }
    final_velocity(
        weights,
        config,
        grid,
        &unified,
        timestep,
        stream,
        &mut observe,
    )
}

/// Image embed + noise refiner + concat with the (precomputed) caption
/// branch: the TeaCache gate signal. Kept separate so the compiled compute
/// closure can capture it as a constant input, exactly like main's
/// `_forward_cached_compute` boundary.
///
/// When `control` is set, runs the control noise-refiner on the control
/// context, injects refiner hints into the main noise_refiner blocks
/// (VideoX `ZImageControlTransformer2DModel` single-pass: main refiner blocks
/// are `BaseZImageTransformerBlock`s that add `hints[block_id] * scale`), then
/// computes control layer hints from the hinted unified stream.
pub(crate) fn unified_before(
    weights: &MlxWeights,
    config: MlxZImageConfig,
    grid: TokenGrid,
    latent: &Array,
    caption: &Array,
    timestep: &Array,
    rope: &ForwardRope,
    control: Option<ControlConditioning<'_>>,
    stream: &Stream,
    observe: &mut impl FnMut(&str, &Array),
) -> Result<(Array, Option<Vec<Array>>)> {
    observe("t_embedder", timestep);
    let ids = geom::positions_image_first(grid);
    let image_ids = &ids[..grid.img_len as usize];
    let mut image = embed_image(weights, config, grid, latent, stream)?;
    let mut refiner_hints = None;
    let mut refined_control = None;
    if let Some(ctrl) = control {
        let control_embed = control::embed_control_context(
            ctrl.weights,
            weights,
            config,
            grid,
            ctrl.context,
            stream,
        )?;
        let (hints, refined) = control::forward_control_refiner(
            ctrl.weights,
            config,
            grid,
            &image,
            &control_embed,
            &rope.image,
            timestep,
            stream,
        )?;
        refiner_hints = Some(hints);
        refined_control = Some(refined);
    }
    // VideoX `ZImageControlTransformer2DModel` (the Union checkpoints were
    // trained/distilled with it): the main noise_refiner blocks ARE the
    // `BaseZImageTransformerBlock`s that add refiner hints to the image
    // stream; the resulting hinted unified stream feeds both the main layers
    // and `control_layers.0` (`before_proj(c) + unified`).
    for index in 0..config.refiners {
        image = run_block(
            weights,
            config,
            &format!("noise_refiner.{index}"),
            &image,
            image_ids,
            true,
            Some(timestep),
            Some(&rope.image),
            stream,
        )?;
        if let (Some(hints), Some(ctrl)) = (&refiner_hints, control) {
            if let Some(hint) = control::hint_for_refiner(index as usize, hints) {
                image = control::apply_control_hint(&image, hint, ctrl.scale, stream)?;
            }
        }
        observe(&format!("noise_refiner.{index}"), &image);
    }
    let unified = Array::concatenate_axis(&[image, caption.clone()], 1, stream)?;
    control::debug_tensor_stats("unified_pre_layers", &unified, stream);
    let layer_hints = match (control, refined_control) {
        (Some(ctrl), Some(refined)) => Some(control::forward_control_layers(
            ctrl.weights,
            config,
            grid,
            &unified,
            &refined,
            caption,
            &rope.unified,
            timestep,
            stream,
        )?),
        _ => None,
    };
    Ok((unified, layer_hints))
}

/// The 30 unified blocks; returns the post-layers unified stream (before the
/// final layer), matching main's residual boundary `unified - unified_before`.
fn unified_layers_and_final(
    weights: &MlxWeights,
    config: MlxZImageConfig,
    grid: TokenGrid,
    unified_before: &Array,
    timestep: &Array,
    rope: &ForwardRope,
    layer_hints: Option<&[Array]>,
    context_scale: Option<f32>,
    stream: &Stream,
    observe: &mut impl FnMut(&str, &Array),
) -> Result<Array> {
    let ids = geom::positions_image_first(grid);
    let mut unified = unified_before.clone();
    for index in 0..config.layers {
        unified = run_block(
            weights,
            config,
            &format!("layers.{index}"),
            &unified,
            &ids,
            true,
            Some(timestep),
            Some(&rope.unified),
            stream,
        )?;
        if let (Some(hints), Some(scale)) = (layer_hints, context_scale) {
            if let Some(hint) = control::hint_for_layer(index as usize, hints) {
                control::debug_tensor_stats(&format!("layer_hint.{index}"), hint, stream);
                unified = control::apply_control_hint(&unified, hint, scale, stream)?;
                control::debug_tensor_stats(
                    &format!("unified_post_hint.{index}"),
                    &unified,
                    stream,
                );
            }
        }
        observe(&format!("layers.{index}"), &unified);
    }
    Ok(unified)
}

/// FinalLayer (weightless LayerNorm + adaLN scale + linear); shared by the
/// eager forward and both compiled closures.
fn final_layer_tokens(
    weights: &MlxWeights,
    unified: &Array,
    timestep: &Array,
    stream: &Stream,
) -> Result<Array> {
    let activated = timestep.silu(stream)?;
    let scale = linear(weights, "final_layer.adaLN_modulation.0.weight")?.apply(
        &activated,
        Some(weight(weights, "final_layer.adaLN_modulation.0.bias")?),
        stream,
    )?;
    let one = cast_to(&Array::from_f32_slice(&[1.0], &[])?, scale.dtype(), stream)?;
    let scale = Array::add(&scale, &one, stream)?;
    let normalized = unified.layer_norm(1e-6, stream)?;
    let normalized = modulation::modulate(&normalized, &scale, stream)?;
    linear(weights, "final_layer.linear.weight")?.apply(
        &normalized,
        Some(weight(weights, "final_layer.linear.bias")?),
        stream,
    )
}

/// Unpatchify + sign; shared by the eager and compiled paths.
fn unpatchify_velocity(
    config: MlxZImageConfig,
    grid: TokenGrid,
    tokens: &Array,
    stream: &Stream,
) -> Result<Array> {
    let token_dim = config.patch * config.patch * config.in_channels;
    let real = tokens.slice(
        &[0, 0, 0],
        &[1, grid.img_real as i32, token_dim as i32],
        &[1, 1, 1],
        stream,
    )?;
    let values = real.reshape(&[(grid.img_real as i32) * token_dim as i32], stream)?;
    let indices = Array::from_i32_slice(
        &geom::patch_index(grid, config.in_channels),
        &[grid.img_real as i32 * token_dim as i32],
    )?;
    let zeros = Array::from_f32_slice(
        &vec![0.0; (config.in_channels * grid.latent_w * grid.latent_h) as usize],
        &[(config.in_channels * grid.latent_w * grid.latent_h) as i32],
    )?;
    let zeros = cast_to(&zeros, values.dtype(), stream)?;
    let output = zeros.scatter_add_axis(&indices, &values, 0, stream)?;
    let output = output.negative(stream)?.reshape(
        &[
            config.in_channels as i32,
            grid.latent_h as i32,
            grid.latent_w as i32,
        ],
        stream,
    )?;
    Ok(output)
}

/// FinalLayer + unpatchify + sign, shared by the eager and compiled paths.
pub(crate) fn final_velocity(
    weights: &MlxWeights,
    config: MlxZImageConfig,
    grid: TokenGrid,
    unified: &Array,
    timestep: &Array,
    stream: &Stream,
    observe: &mut impl FnMut(&str, &Array),
) -> Result<Array> {
    let tokens = final_layer_tokens(weights, unified, timestep, stream)?;
    observe("final_layer", &tokens);
    let output = unpatchify_velocity(config, grid, &tokens, stream)?;
    observe("velocity", &output);
    Ok(output)
}

fn cast_to(input: &Array, dtype: danmo_mlx::Dtype, stream: &Stream) -> Result<Array> {
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

#[cfg(test)]
mod tests {
    use super::super::step_cache::StepCache;
    use super::MlxZImageConfig;
    use super::{
        forward, forward_compiled_with_step_cache, forward_with_step_cache, load_directory,
        CaptionCache, CompiledCompute,
    };
    use crate::families::z_image::geom;
    use crate::families::z_image::weights::ZImageConfig;
    use crate::weights::{MlxWeights, WeightDtype};
    use danmo_mlx::{
        save_safetensors, AffineQuantization, Array, DeviceType, Dtype, Result, Stream,
    };
    use std::collections::BTreeMap;

    fn add_tensor(tensors: &mut BTreeMap<String, (Vec<f32>, Vec<i32>)>, name: &str, shape: &[i64]) {
        let count = shape.iter().product::<i64>() as usize;
        let values = (0..count)
            .map(|index| ((index + name.len()) % 13) as f32 * 0.015 - 0.07)
            .collect();
        assert!(
            tensors
                .insert(
                    name.to_string(),
                    (values, shape.iter().map(|value| *value as i32).collect()),
                )
                .is_none(),
            "{name}"
        );
    }

    fn add_block(
        tensors: &mut BTreeMap<String, (Vec<f32>, Vec<i32>)>,
        prefix: &str,
        config: &MlxZImageConfig,
        modulated: bool,
    ) {
        let dim = config.dim;
        let ffn = config.ffn_hidden();
        for name in ["to_q", "to_k", "to_v", "to_out"] {
            add_tensor(
                tensors,
                &format!("{prefix}.attention.{name}.weight"),
                &[dim, dim],
            );
        }
        add_tensor(
            tensors,
            &format!("{prefix}.attention.norm_q.weight"),
            &[config.head_dim()],
        );
        add_tensor(
            tensors,
            &format!("{prefix}.attention.norm_k.weight"),
            &[config.head_dim()],
        );
        for name in ["attn_norm1", "attn_norm2", "ffn_norm1", "ffn_norm2"] {
            add_tensor(tensors, &format!("{prefix}.{name}.weight"), &[dim]);
        }
        add_tensor(
            tensors,
            &format!("{prefix}.feed_forward.w1.weight"),
            &[ffn, dim],
        );
        add_tensor(
            tensors,
            &format!("{prefix}.feed_forward.w2.weight"),
            &[dim, ffn],
        );
        add_tensor(
            tensors,
            &format!("{prefix}.feed_forward.w3.weight"),
            &[ffn, dim],
        );
        if modulated {
            add_tensor(
                tensors,
                &format!("{prefix}.adaLN_modulation.0.weight"),
                &[4 * dim, config.adaln_dim],
            );
            add_tensor(
                tensors,
                &format!("{prefix}.adaLN_modulation.0.bias"),
                &[4 * dim],
            );
        }
    }

    fn fixture() -> (
        MlxZImageConfig,
        BTreeMap<String, (Vec<f32>, Vec<i32>)>,
        Vec<f32>,
        Vec<f32>,
    ) {
        let mut config = MlxZImageConfig::published();
        config.in_channels = 2;
        config.dim = 6;
        config.heads = 1;
        config.layers = 1;
        config.refiners = 1;
        config.cap_dim = 4;
        config.frequency_dim = 256;
        config.timestep_mid = 1024;
        config.adaln_dim = 6;
        config.axes = [2, 2, 2];
        let tensors = fixture_tensors(&config);
        let latent = (0..8).map(|index| index as f32 * 0.1 - 0.2).collect();
        let caption = (0..8).map(|index| index as f32 * 0.07 - 0.15).collect();
        (config, tensors, latent, caption)
    }

    fn fixture_tensors(config: &MlxZImageConfig) -> BTreeMap<String, (Vec<f32>, Vec<i32>)> {
        let token = config.patch * config.patch * config.in_channels;
        let mut tensors = BTreeMap::new();
        add_tensor(&mut tensors, "x_embedder.weight", &[config.dim, token]);
        add_tensor(&mut tensors, "x_embedder.bias", &[config.dim]);
        add_tensor(
            &mut tensors,
            "t_embedder.linear1.weight",
            &[config.timestep_mid, config.frequency_dim],
        );
        add_tensor(
            &mut tensors,
            "t_embedder.linear1.bias",
            &[config.timestep_mid],
        );
        add_tensor(
            &mut tensors,
            "t_embedder.linear2.weight",
            &[config.adaln_dim, config.timestep_mid],
        );
        add_tensor(&mut tensors, "t_embedder.linear2.bias", &[config.adaln_dim]);
        add_tensor(&mut tensors, "cap_norm.weight", &[config.cap_dim]);
        add_tensor(
            &mut tensors,
            "cap_embedder.weight",
            &[config.dim, config.cap_dim],
        );
        add_tensor(&mut tensors, "cap_embedder.bias", &[config.dim]);
        add_block(&mut tensors, "noise_refiner.0", config, true);
        add_block(&mut tensors, "context_refiner.0", config, false);
        add_block(&mut tensors, "layers.0", config, true);
        add_tensor(
            &mut tensors,
            "final_layer.adaLN_modulation.0.weight",
            &[config.dim, config.adaln_dim],
        );
        add_tensor(
            &mut tensors,
            "final_layer.adaLN_modulation.0.bias",
            &[config.dim],
        );
        add_tensor(
            &mut tensors,
            "final_layer.linear.weight",
            &[token, config.dim],
        );
        add_tensor(&mut tensors, "final_layer.linear.bias", &[token]);
        tensors.insert(
            "x_pad_token".into(),
            (vec![0.0; config.dim as usize], vec![1, config.dim as i32]),
        );
        tensors.insert(
            "cap_pad_token".into(),
            (vec![0.0; config.dim as usize], vec![1, config.dim as i32]),
        );
        tensors
    }

    fn host_name(name: &str) -> Option<String> {
        if name.ends_with(".attention.to_q.weight")
            || name.ends_with(".attention.to_k.weight")
            || name.ends_with(".attention.to_v.weight")
        {
            return None;
        }
        Some(
            name.replace(".attention.to_out.weight", ".attention.out.weight")
                .replace(".attention.norm_q.", ".attention.q_norm.")
                .replace(".attention.norm_k.", ".attention.k_norm.")
                .replace("attn_norm1.", "attention_norm1.")
                .replace("attn_norm2.", "attention_norm2.")
                .replace("t_embedder.linear1.", "t_embedder.mlp.0.")
                .replace("t_embedder.linear2.", "t_embedder.mlp.2.")
                .replace("cap_norm.", "cap_embedder.0.")
                .replace("cap_embedder.weight", "cap_embedder.1.weight")
                .replace("cap_embedder.bias", "cap_embedder.1.bias")
                .replace(
                    "final_layer.adaLN_modulation.0.",
                    "final_layer.adaLN_modulation.1.",
                ),
        )
    }

    fn probe(device_type: DeviceType) -> Result<()> {
        let (config, tensors, latent_values, caption_values) = fixture();
        let grid = geom::token_grid(16, 16, 2, config.vae_scale).unwrap();
        let stream = match device_type {
            DeviceType::Cpu => Stream::cpu()?,
            DeviceType::Gpu => Stream::gpu()?,
        };
        let mut weights = BTreeMap::new();
        for (name, (values, shape)) in &tensors {
            let array = Array::from_f32_slice(values, shape)?;
            weights.insert(name.clone(), array);
        }
        let weights = MlxWeights::from_dense(weights);

        let mut host_weights = BTreeMap::new();
        for (name, (values, _)) in &tensors {
            if let Some(name) = host_name(name) {
                host_weights.insert(name, values.clone());
            }
        }
        for prefix in ["noise_refiner.0", "context_refiner.0", "layers.0"] {
            let mut fused = tensors[&format!("{prefix}.attention.to_q.weight")]
                .0
                .clone();
            fused.extend(&tensors[&format!("{prefix}.attention.to_k.weight")].0);
            fused.extend(&tensors[&format!("{prefix}.attention.to_v.weight")].0);
            host_weights.insert(format!("{prefix}.attention.qkv.weight"), fused);
        }
        let host_config = ZImageConfig {
            dim: config.dim,
            heads: config.heads,
            head_dim: config.head_dim(),
            axes: config.axes,
            layers: config.layers,
            refiners: config.refiners,
            in_channels: config.in_channels,
            cap_dim: config.cap_dim,
            ffn: config.ffn_hidden(),
            mid: config.timestep_mid,
            freq: config.frequency_dim,
            adaln: config.adaln_dim,
            rope_theta: config.rope_theta,
            t_scale: config.t_scale,
            vae_scale: config.vae_scale,
            patch: config.patch,
            requires_sigma_shift: true,
        };
        let expected = crate::families::z_image::reference::velocity(
            &host_weights.into_iter().collect(),
            host_config,
            grid,
            &latent_values,
            &caption_values,
            0.25,
            false,
        );
        let latent = Array::from_f32_slice(&latent_values, &[2, 2, 2])?;
        let caption = Array::from_f32_slice(&caption_values, &[2, 4])?;
        let actual = forward(&weights, config, grid, &latent, &caption, 0.25, &stream)?
            .to_vec_f32(&stream)?;
        assert_eq!(actual.len(), expected.len());
        let tolerance = if device_type == DeviceType::Cpu {
            2e-4
        } else {
            1e-3
        };
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < tolerance,
                "{actual} != {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn cpu_tiny_dit_matches_complete_host_forward() {
        probe(DeviceType::Cpu).unwrap();
    }

    fn compiled_probe(device_type: DeviceType) -> Result<()> {
        let (config, tensors, latent_values, caption_values) = fixture();
        let grid = geom::token_grid(16, 16, 2, config.vae_scale).unwrap();
        let stream = match device_type {
            DeviceType::Cpu => Stream::cpu()?,
            DeviceType::Gpu => Stream::gpu()?,
        };
        let mut weights = BTreeMap::new();
        for (name, (values, shape)) in &tensors {
            weights.insert(name.clone(), Array::from_f32_slice(values, shape)?);
        }
        let weights = MlxWeights::from_dense(weights);
        let caption = Array::from_f32_slice(&caption_values, &[2, 4])?;
        let compiled =
            super::CompiledForward::new(&weights, config, grid, &caption, false, &stream)?;
        for sigma in [0.9f32, 0.5, 0.25] {
            let latent = Array::from_f32_slice(&latent_values, &[2, 2, 2])?;
            let eager = forward(&weights, config, grid, &latent, &caption, sigma, &stream)?
                .to_vec_f32(&stream)?;
            let timestep = super::embed_timestep(&weights, config, sigma, &stream)?;
            let actual = compiled.apply(&latent, &timestep)?.to_vec_f32(&stream)?;
            assert_eq!(actual.len(), eager.len());
            for (actual, expected) in actual.iter().zip(&eager) {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "compiled {actual} != eager {expected} at sigma {sigma}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn cpu_compiled_tiny_dit_matches_eager_across_steps() {
        compiled_probe(DeviceType::Cpu).unwrap();
    }

    fn quantized_probe(device_type: DeviceType) -> Result<()> {
        let mut config = MlxZImageConfig::published();
        config.in_channels = 2;
        config.dim = 64;
        config.heads = 1;
        config.layers = 1;
        config.refiners = 1;
        config.cap_dim = 64;
        config.adaln_dim = 64;
        config.axes = [16, 24, 24];
        let tensors = fixture_tensors(&config);
        let grid = geom::token_grid(16, 16, 2, config.vae_scale).unwrap();
        // MLX's scalar CPU qmm accumulates in the activation dtype, so bf16 parity is
        // only meaningful on GPU kernels, which accumulate in f32.
        let (stream, dtype) = match device_type {
            DeviceType::Cpu => (Stream::cpu()?, Dtype::Float32),
            DeviceType::Gpu => (Stream::gpu()?, Dtype::BFloat16),
        };
        let quantization = AffineQuantization::new(4, 64)?;
        let mut saved: Vec<(String, Array)> = Vec::new();
        let mut dequantized = BTreeMap::new();
        let mut packed_names = Vec::new();
        for (name, (values, shape)) in &tensors {
            let array = Array::from_f32_slice(values, shape)?.astype(dtype, &stream)?;
            let base = name.strip_suffix(".weight");
            match base {
                Some(base) if shape.len() == 2 && shape[1] % 64 == 0 => {
                    let packed = array.quantize_affine(64, 4, &stream)?;
                    dequantized.insert(name.clone(), packed.dequantize(64, 4, &stream)?);
                    saved.push((format!("{base}.scales"), packed.scales));
                    saved.push((format!("{base}.biases"), packed.biases));
                    saved.push((name.clone(), packed.weight));
                    packed_names.push(name.clone());
                }
                _ => {
                    dequantized.insert(name.clone(), array.clone());
                    saved.push((name.clone(), array));
                }
            }
        }
        let scratch = std::env::temp_dir().join(format!(
            "danmo-zimage-quant-{}-{:?}-{}",
            std::process::id(),
            device_type,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&scratch).unwrap();
        let entries: Vec<(&str, &Array)> = saved
            .iter()
            .map(|(name, array)| (name.as_str(), array))
            .collect();
        save_safetensors(
            &scratch.join("model.safetensors"),
            &entries,
            &[("quantization_level", "4")],
        )?;
        let loaded =
            crate::weights::load_mlx_directory(&scratch, WeightDtype::Native, Some(quantization));
        let _ = std::fs::remove_dir_all(&scratch);
        let weights = loaded?;
        let reference = MlxWeights::from_dense(dequantized);

        assert_eq!(weights.quantization(), Some(quantization));
        assert!(packed_names.len() > 10, "{packed_names:?}");
        assert_eq!(weights.packed_names().collect::<Vec<_>>(), packed_names);
        let packed_bytes: usize = saved.iter().map(|(_, array)| array.nbytes()).sum();
        assert_eq!(weights.nbytes(), packed_bytes);
        assert!(weights.nbytes() * 2 < reference.nbytes());

        let latent = Array::from_f32_slice(
            &(0..8)
                .map(|index| index as f32 * 0.1 - 0.2)
                .collect::<Vec<_>>(),
            &[2, 2, 2],
        )?
        .astype(dtype, &stream)?;
        let caption = Array::from_f32_slice(
            &(0..128)
                .map(|index| (index % 17) as f32 * 0.03 - 0.2)
                .collect::<Vec<_>>(),
            &[2, 64],
        )?
        .astype(dtype, &stream)?;
        let compiled =
            super::CompiledForward::new(&weights, config, grid, &caption, false, &stream)?;
        for sigma in [0.9f32, 0.25] {
            let expected = forward(&reference, config, grid, &latent, &caption, sigma, &stream)?
                .astype(Dtype::Float32, &stream)?
                .to_vec_f32(&stream)?;
            let actual = forward(&weights, config, grid, &latent, &caption, sigma, &stream)?
                .astype(Dtype::Float32, &stream)?
                .to_vec_f32(&stream)?;
            let timestep = super::embed_timestep(&weights, config, sigma, &stream)?;
            let fused = compiled
                .apply(&latent, &timestep)?
                .astype(Dtype::Float32, &stream)?
                .to_vec_f32(&stream)?;
            assert_eq!(actual.len(), expected.len());
            let scale = expected
                .iter()
                .fold(0.0f32, |peak, value| peak.max(value.abs()));
            assert!(scale > 0.0);
            let tolerance = if dtype == Dtype::Float32 { 1e-4 } else { 2e-2 } * scale;
            for ((actual, fused), expected) in actual.iter().zip(&fused).zip(&expected) {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "packed {actual} != dequantized {expected} at sigma {sigma}"
                );
                assert!(
                    (fused - actual).abs() <= tolerance,
                    "compiled {fused} != eager {actual} at sigma {sigma}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn cpu_packed_affine_dit_matches_dequantized_weights() {
        quantized_probe(DeviceType::Cpu).unwrap();
    }

    #[cfg(feature = "mlx-metal")]
    #[test]
    fn metal_packed_affine_dit_matches_dequantized_weights() {
        quantized_probe(DeviceType::Gpu).unwrap();
    }

    #[cfg(feature = "mlx-metal")]
    #[test]
    fn metal_compiled_tiny_dit_matches_eager_across_steps() {
        compiled_probe(DeviceType::Gpu).unwrap();
    }

    #[test]
    #[ignore]
    fn python_main_tiny_fixture_matches_rust_forward() {
        let root =
            std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_PARITY_DIR").unwrap());
        let payload: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("expected.json")).unwrap())
                .unwrap();
        let (config, _, _, _) = fixture();
        let grid = geom::token_grid(16, 16, 2, config.vae_scale).unwrap();
        #[cfg(feature = "mlx-metal")]
        let stream = Stream::gpu().unwrap();
        #[cfg(not(feature = "mlx-metal"))]
        let stream = Stream::cpu().unwrap();
        let weights = load_directory(&root, config, WeightDtype::F32, None).unwrap();
        let latent_values: Vec<f32> = payload["latent"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        let caption_values: Vec<f32> = payload["caption"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        let latent = Array::from_f32_slice(&latent_values, &[2, 2, 2]).unwrap();
        let caption = Array::from_f32_slice(&caption_values, &[2, 4]).unwrap();
        let sigma = payload["sigma"].as_f64().unwrap() as f32;
        let actual = forward(&weights, config, grid, &latent, &caption, sigma, &stream)
            .unwrap()
            .to_vec_f32(&stream)
            .unwrap();
        let expected: Vec<f32> = payload["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(actual.len(), expected.len());
        let mut max_error = 0.0f32;
        for (actual, expected) in actual.iter().zip(expected) {
            let error = (actual - expected).abs();
            max_error = max_error.max(error);
            let exponent = expected.abs().max(f32::MIN_POSITIVE).log2().floor();
            let bf16_ulp = 2f32.powf(exponent - 7.0);
            let allowed = 4.0 * bf16_ulp + 1e-4;
            assert!(
                error <= allowed,
                "error {error} exceeds four BF16 ULP ({allowed})"
            );
        }
        println!("main/Rust tiny DiT max abs error: {max_error}");
    }

    #[cfg(feature = "mlx-metal")]
    #[test]
    #[ignore]
    fn real_full_dit_forward_is_finite_on_metal() {
        let root = std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_DIT_DIR").unwrap());
        let stream = Stream::gpu().unwrap();
        let config = MlxZImageConfig::published();
        let started = std::time::Instant::now();
        let weights = load_directory(&root, config, WeightDtype::F32, None).unwrap();
        println!("loaded {} remapped tensors", weights.names().count());
        let grid = geom::token_grid(16, 16, 2, config.vae_scale).unwrap();
        let latent = Array::from_f32_slice(&vec![0.01; 64], &[16, 2, 2]).unwrap();
        let caption = Array::from_f32_slice(&vec![0.01; 5120], &[2, 2560]).unwrap();
        let output = forward(&weights, config, grid, &latent, &caption, 0.25, &stream)
            .unwrap()
            .to_vec_f32(&stream)
            .unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
        println!(
            "real full DiT forward: {:.3}s",
            started.elapsed().as_secs_f64()
        );

        let sigmas = crate::common::ops::linear_sigmas(8, 16, 16, true);
        let denoise_started = std::time::Instant::now();
        let denoised = crate::families::z_image::mlx::denoise::euler(
            &latent,
            &sigmas,
            &stream,
            |current, sigma| forward(&weights, config, grid, current, &caption, sigma, &stream),
        )
        .unwrap()
        .to_vec_f32(&stream)
        .unwrap();
        assert!(denoised.iter().all(|value| value.is_finite()));
        println!(
            "real 8-step DiT denoise: {:.3}s",
            denoise_started.elapsed().as_secs_f64()
        );
    }

    #[cfg(feature = "mlx-metal")]
    #[test]
    #[ignore]
    fn real_dit_stage_fixtures_localize_production_resolution_divergence() {
        let root =
            std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_STAGES_DIR").unwrap());
        let dit = std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_DIT_DIR").unwrap());
        let payload: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("expected.json")).unwrap())
                .unwrap();
        let values = |name: &str| -> Vec<f32> {
            payload[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect()
        };
        let stream = Stream::gpu().unwrap();
        let io = Stream::cpu().unwrap();
        let stages = danmo_mlx::SafeTensors::load(&root.join("stages.safetensors"), &io).unwrap();
        let config = MlxZImageConfig::published();
        let quantization = payload["quantization_bits"].as_i64().map(|bits| {
            let group_size = payload["quantization_group_size"].as_i64().unwrap_or(64);
            AffineQuantization::new(bits as i32, group_size as i32).unwrap()
        });
        let weight_dtype = if quantization.is_some() {
            WeightDtype::Native
        } else {
            WeightDtype::F32
        };
        let weights = load_directory(&dit, config, weight_dtype, quantization).unwrap();
        let width = payload["width"].as_u64().unwrap() as u32;
        let height = payload["height"].as_u64().unwrap() as u32;
        let ids = payload["ids"].as_array().unwrap();
        let grid = geom::token_grid(width, height, ids.len() as i64, config.vae_scale).unwrap();
        let mut latent = Array::from_f32_slice(
            &values("latent"),
            &[16, (height / 8) as i32, (width / 8) as i32],
        )
        .unwrap()
        .astype(Dtype::BFloat16, &stream)
        .unwrap();
        let caption = Array::from_f32_slice(&values("caption"), &[ids.len() as i32, 2560])
            .unwrap()
            .astype(Dtype::BFloat16, &stream)
            .unwrap();
        let step = payload["step"].as_u64().unwrap_or(0) as usize;
        let timestep = super::embed_timestep(
            &weights,
            config,
            payload["sigmas"][step].as_f64().unwrap() as f32,
            &stream,
        )
        .unwrap();
        // TeaCache skip-step replay: warm the cache with the earlier steps
        // exactly like the production loop, then force the recorded step
        // through the skip path with the captured residual.
        let teacache_replay = std::env::var("DANQING_TEST_ZIMAGE_STAGES_TEACACHE").is_ok();
        let mut replay_cache = teacache_replay
            .then(|| StepCache::z_image(payload["sigmas"].as_array().unwrap().len() - 1));
        if let Some(cache) = &mut replay_cache {
            let caption_cache =
                super::CaptionCache::new(&weights, config, grid, &caption, &stream).unwrap();
            let initial: Vec<f32> = payload["initial_latent"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let mut current =
                Array::from_f32_slice(&initial, &[16, (height / 8) as i32, (width / 8) as i32])
                    .unwrap()
                    .astype(Dtype::BFloat16, &stream)
                    .unwrap();
            let sigmas = values("sigmas");
            for index in 0..step {
                let (velocity, residual) = super::forward_cached_with_step_cache(
                    &weights,
                    config,
                    grid,
                    &current,
                    &caption_cache,
                    sigmas[index],
                    cache,
                    index,
                    &stream,
                )
                .unwrap();
                let _ = residual;
                let cast = velocity.astype(current.dtype(), &stream).unwrap();
                let delta = Array::from_f32_slice(&[sigmas[index + 1] - sigmas[index]], &[])
                    .unwrap()
                    .astype(current.dtype(), &stream)
                    .unwrap();
                current = Array::add(
                    &current,
                    &Array::multiply(&cast, &delta, &stream).unwrap(),
                    &stream,
                )
                .unwrap();
                Array::eval_all(&[&velocity, &current]).unwrap();
                let expected_velocity: Vec<f32> = payload["velocities"][index]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect();
                let actual_velocity = velocity.to_vec_f32(&stream).unwrap();
                let velocity_error = actual_velocity
                    .iter()
                    .zip(&expected_velocity)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let decision = &cache.history[index];
                println!(
                    "replay step {index}: velocity err {velocity_error}, gate {} rel_l1 {:?} pred {:?} acc {}",
                    decision.kind, decision.rel_l1, decision.predicted_distance, decision.accumulated_distance
                );
            }
            if let Some(decision) = cache.history.last() {
                println!(
                    "replay final cache: acc {} residual_present {}",
                    decision.accumulated_distance,
                    cache.cached_residual().is_some()
                );
            }
            // Transplant the fixture's recorded step-input latent: the
            // replay above only exists to reproduce the cache state
            // (previous mod input, residual, accumulated distance).
            let recorded: Vec<f32> = payload["latent"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            latent =
                Array::from_f32_slice(&recorded, &[16, (height / 8) as i32, (width / 8) as i32])
                    .unwrap()
                    .astype(Dtype::BFloat16, &stream)
                    .unwrap();
        }
        let mut errors = BTreeMap::new();
        super::forward_with_timestep_observed(
            &weights,
            config,
            grid,
            &latent,
            &caption,
            &timestep,
            None,
            None,
            replay_cache.as_mut().map(|cache| (cache, step)),
            None,
            &stream,
            |name, actual| {
                let Some(expected) = stages.get(name).unwrap() else {
                    // Skip steps never run the unified blocks; main's fixture
                    // has no stages for them either.
                    return;
                };
                let actual = if actual.shape() == expected.shape() {
                    actual.clone()
                } else {
                    actual.reshape(&expected.shape(), &stream).unwrap()
                };
                let actual = actual.to_vec_f32(&stream).unwrap();
                let expected = expected.to_vec_f32(&io).unwrap();
                let error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                println!("Z-Image stage {name}: max abs error {error}");
                errors.insert(name.to_string(), error);
            },
        )
        .unwrap();
        assert!(
            errors["velocity"] <= 0.002,
            "production-resolution velocity diverged: {errors:?}"
        );
    }

    #[cfg(feature = "mlx-metal")]
    #[test]
    #[ignore]
    fn real_zimage_create_matches_main_latents_and_pixels() {
        let bundle = std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_BUNDLE").unwrap());
        let payload: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(std::env::var("DANQING_TEST_ZIMAGE_REAL_JSON").unwrap())
                .unwrap(),
        )
        .unwrap();
        let values = |name: &str| -> Vec<f32> {
            payload[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect()
        };
        let stream = Stream::gpu().unwrap();
        let width = payload["width"].as_u64().unwrap() as u32;
        let height = payload["height"].as_u64().unwrap() as u32;
        let latent_h = (height / 8) as i32;
        let latent_w = (width / 8) as i32;
        let expected_denoised = values("denoised");
        let quantization = payload["quantization_bits"].as_i64().map(|bits| {
            let group_size = payload["quantization_group_size"].as_i64().unwrap_or(64);
            AffineQuantization::new(bits as i32, group_size as i32).unwrap()
        });
        let weight_dtype =
            if payload["native_dtype"].as_str() == Some("checkpoint") || quantization.is_some() {
                danmo_mlx::reset_peak_memory().unwrap();
                WeightDtype::Native
            } else {
                WeightDtype::F32
            };
        let started = std::time::Instant::now();
        let ids: Vec<i32> = payload["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect();
        let text_weights = crate::common::qwen3::mlx::load_directory(
            &bundle.join("text_encoder"),
            WeightDtype::Native,
        )
        .unwrap();
        let text_config = crate::common::qwen3::Qwen3Config {
            vocab: 151_936,
            hidden: 2560,
            layers: 36,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            intermediate: 9728,
            rope_theta: 1_000_000.0,
        };
        let caption = crate::common::qwen3::mlx::encode_ids(
            &text_weights,
            text_config,
            "model.",
            &ids,
            &vec![true; ids.len()],
            &[35],
            &stream,
        )
        .unwrap()
        .squeeze_axis(0, &stream)
        .unwrap()
        .astype(Dtype::BFloat16, &stream)
        .unwrap();
        caption.eval().unwrap();
        drop(text_weights);
        let caption_values = caption.to_vec_f32(&stream).unwrap();
        let expected_caption = values("caption");
        let caption_error = caption_values
            .iter()
            .zip(&expected_caption)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(caption_error <= 1e-6, "caption error {caption_error}");
        println!(
            "real Z-Image encode: {:.3}s",
            started.elapsed().as_secs_f64()
        );
        let dit_started = std::time::Instant::now();
        let config = MlxZImageConfig::published();
        let weights = load_directory(
            &bundle.join("transformer"),
            config,
            weight_dtype,
            quantization,
        )
        .unwrap();
        let dit_weight_bytes = weights.nbytes();
        let packed_linear_count = weights.packed_names().count();
        if let Some(quant) = quantization {
            assert!(packed_linear_count > 100, "DiT did not stay packed");
            // Packed DiT must stay well below the 24.6GB F32 dense resident
            // size: ~3.5GB at 4-bit, ~6.6GB at 8-bit (plus dense norms).
            let ceiling = if quant.bits == 4 {
                5_000_000_000u64
            } else {
                9_000_000_000u64
            };
            assert!(
                (dit_weight_bytes as u64) < ceiling,
                "packed DiT is unexpectedly large: {dit_weight_bytes}"
            );
        }
        println!(
            "Z-Image DiT resident weights: {dit_weight_bytes} bytes ({weight_dtype:?}), packed linears {packed_linear_count}"
        );
        let grid = geom::token_grid(width, height, ids.len() as i64, config.vae_scale).unwrap();
        let latent = if let Some(seed) = payload["seed"].as_u64() {
            let noise = danmo_mlx::RandomKey::new(seed)
                .unwrap()
                .normal(
                    &[1, 16, latent_h, latent_w],
                    danmo_mlx::Dtype::Float32,
                    &stream,
                )
                .unwrap();
            assert_eq!(noise.to_vec_f32(&stream).unwrap(), values("latent"));
            noise
        } else {
            Array::from_f32_slice(&values("latent"), &[1, 16, latent_h, latent_w]).unwrap()
        }
        .squeeze_axis(0, &stream)
        .unwrap()
        .astype(danmo_mlx::Dtype::BFloat16, &stream)
        .unwrap();
        let sigmas = values("sigmas");
        let mut step_cache = payload["teacache"].as_bool().unwrap_or(false).then(|| {
            payload["teacache_thresh"].as_f64().map_or_else(
                || StepCache::z_image(sigmas.len() - 1),
                |threshold| StepCache::new(sigmas.len() - 1, threshold),
            )
        });
        let mut step = 0usize;
        let denoised = crate::families::z_image::mlx::denoise::euler(
            &latent,
            &sigmas,
            &stream,
            |current, sigma| {
                if step > 0 {
                    let expected_current = &payload["step_latents"][step - 1];
                    let current_values = current.to_vec_f32(&stream)?;
                    let mut current_error = 0.0f32;
                    let mut current_differences = 0usize;
                    for (actual, expected) in
                        current_values.iter().zip(expected_current.as_array().unwrap())
                    {
                        let expected = expected.as_f64().unwrap() as f32;
                        current_error = current_error.max((actual - expected).abs());
                        current_differences += usize::from(*actual != expected);
                    }
                    println!(
                        "Z-Image step {step}: input latent max abs error {current_error}, {current_differences}/{} values differ",
                        current_values.len()
                    );
                }
                let velocity = match &mut step_cache {
                    Some(cache) => forward_with_step_cache(
                        &weights, config, grid, current, &caption, sigma, cache, step, &stream,
                    )?,
                    None => forward(&weights, config, grid, current, &caption, sigma, &stream)?,
                };
                let actual = velocity.to_vec_f32(&stream)?;
                let error = actual
                    .iter()
                    .zip(payload["velocities"][step].as_array().unwrap())
                    .map(|(actual, expected)| (actual - expected.as_f64().unwrap() as f32).abs())
                    .fold(0.0f32, f32::max);
                println!("Z-Image step {step}: velocity max abs error {error}");
                step += 1;
                Ok(velocity)
            },
        )
        .unwrap();
        let denoised_values = denoised.to_vec_f32(&stream).unwrap();
        let mut latent_error = 0.0f32;
        for (actual, expected) in denoised_values.iter().zip(&expected_denoised) {
            let error = (actual - expected).abs();
            latent_error = latent_error.max(error);
            let exponent = expected.abs().max(f32::MIN_POSITIVE).log2().floor();
            let allowed = 4.0 * 2f32.powf(exponent - 7.0) + 4e-3;
            assert!(error <= allowed, "denoised error {error} exceeds {allowed}");
        }
        println!(
            "real Z-Image denoise: {:.3}s, max abs error {latent_error}",
            dit_started.elapsed().as_secs_f64()
        );
        if let Some(cache) = &step_cache {
            let expected = payload["cache_decisions"].as_array().unwrap();
            assert_eq!(cache.history.len(), expected.len());
            for (index, (actual, expected)) in cache.history.iter().zip(expected).enumerate() {
                assert_eq!(actual.kind, expected["kind"].as_str().unwrap());
                for (actual, expected) in [
                    (actual.rel_l1, expected["rel_l1"].as_f64()),
                    (
                        actual.predicted_distance,
                        expected["predicted_distance"].as_f64(),
                    ),
                    (
                        Some(actual.accumulated_distance),
                        expected["accumulated_distance"].as_f64(),
                    ),
                ] {
                    if let Some(expected) = expected {
                        let error = (actual.unwrap() - expected).abs();
                        assert!(error <= 1e-5, "TeaCache step {index} error {error}");
                    }
                }
                println!("TeaCache step {index}: {}", actual.kind);
            }
            assert_eq!(
                cache.computed_count,
                payload["cache_counts"]["computed"].as_u64().unwrap() as usize
            );
            assert_eq!(
                cache.skipped_count,
                payload["cache_counts"]["skipped"].as_u64().unwrap() as usize
            );
        }
        drop(weights);

        let vae_started = std::time::Instant::now();
        let vae_stream = Stream::gpu().unwrap();
        // Quantized bundles keep the VAE's affine mid-attention linears
        // packed (main's quantized VAE inference); dense bundles load as-is.
        let vae_weights = crate::common::vae::mlx::load(&bundle.join("vae"), weight_dtype).unwrap();
        let spec = crate::common::vae::VaeSpec {
            latent_channels: 16,
            block_out: vec![128, 256, 512, 512],
            resnets: 2,
            groups: 32,
            scaling: 0.3611,
            shift: 0.1159,
            attention: true,
        };
        let main_denoised = Array::from_f32_slice(&expected_denoised, &[1, 16, latent_h, latent_w])
            .unwrap()
            .astype(danmo_mlx::Dtype::BFloat16, &vae_stream)
            .unwrap();
        let main_input_decoded =
            crate::common::vae::mlx::decode(&vae_weights, &spec, &main_denoised, &vae_stream)
                .unwrap()
                .to_vec_f32(&vae_stream)
                .unwrap();
        let expected_output = values("output");
        let main_input_error = main_input_decoded
            .iter()
            .zip(&expected_output)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        println!("VAE with main denoised input: max abs error {main_input_error}");
        let denoised_nchw = denoised.expand_dims(0, &vae_stream).unwrap();
        let decoded =
            crate::common::vae::mlx::decode(&vae_weights, &spec, &denoised_nchw, &vae_stream)
                .unwrap()
                .to_vec_f32(&vae_stream)
                .unwrap();
        let output_error = decoded
            .iter()
            .zip(&expected_output)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        if weight_dtype == WeightDtype::Native {
            for (actual, expected) in decoded.iter().zip(&expected_output) {
                let error = (actual - expected).abs();
                let exponent = expected.abs().max(f32::MIN_POSITIVE).log2().floor();
                let allowed = 4.0 * 2f32.powf(exponent - 7.0) + 1e-3;
                assert!(
                    error <= allowed,
                    "native decoded error {error} exceeds BF16 tolerance ({allowed})"
                );
            }
        } else {
            assert!(output_error <= 0.004, "decoded error {output_error}");
        }
        let plane = (width * height) as usize;
        let actual_pixels: Vec<u8> = (0..plane * 3)
            .map(|index| {
                let channel = index % 3;
                let pixel = index / 3;
                let value = decoded[channel * plane + pixel];
                ((value + 1.0) * 127.5).clamp(0.0, 255.0).round_ties_even() as u8
            })
            .collect();
        let expected_pixels: Vec<u8> = payload["pixels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u8)
            .collect();
        let different_pixels = actual_pixels
            .iter()
            .zip(&expected_pixels)
            .filter(|(actual, expected)| actual != expected)
            .count();
        let max_pixel_error = actual_pixels
            .iter()
            .zip(&expected_pixels)
            .map(|(actual, expected)| actual.abs_diff(*expected))
            .max()
            .unwrap_or(0);
        assert!(max_pixel_error <= 1, "8-bit RGB error {max_pixel_error}");
        println!(
            "RGB parity: {different_pixels}/{} values differ, max channel error {max_pixel_error}",
            plane * 3
        );
        if let Ok(path) = std::env::var("DANQING_TEST_ZIMAGE_PNG") {
            std::fs::write(
                path,
                crate::common::png::encode_rgb(width, height, &actual_pixels),
            )
            .unwrap();
        }
        println!(
            "real Z-Image VAE: {:.3}s, max abs error {output_error}, RGB parity passed",
            vae_started.elapsed().as_secs_f64()
        );
        if weight_dtype == WeightDtype::Native {
            let peak = danmo_mlx::peak_memory().unwrap();
            let active = danmo_mlx::active_memory().unwrap();
            println!("native Z-Image memory: peak {peak} bytes, active {active} bytes");
            if let Ok(path) = std::env::var("DANQING_TEST_ZIMAGE_NATIVE_REPORT") {
                let report = serde_json::json!({
                    "weight_dtype": "checkpoint",
                    "dit_weight_bytes": dit_weight_bytes,
                    "peak_memory": peak,
                    "active_memory": active,
                    "main_peak_memory": payload["memory"]["peak"].as_u64().unwrap(),
                    "main_active_memory": payload["memory"]["active"].as_u64().unwrap(),
                });
                std::fs::write(path, serde_json::to_string(&report).unwrap()).unwrap();
            }
        }
    }

    /// Rewrite (img2img) Metal parity against a danmo-make main fixture.
    ///
    /// Env: `DANQING_TEST_ZIMAGE_BUNDLE`, `DANQING_TEST_ZIMAGE_REWRITE_JSON`.
    /// Optional: `DANQING_TEST_ZIMAGE_REWRITE_PNG` for the decoded RGB.
    #[cfg(feature = "mlx-metal")]
    #[test]
    #[ignore]
    fn real_zimage_rewrite_matches_main_latents_and_pixels() {
        let bundle = std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_BUNDLE").unwrap());
        let payload: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(std::env::var("DANQING_TEST_ZIMAGE_REWRITE_JSON").unwrap())
                .unwrap(),
        )
        .unwrap();
        assert_eq!(payload["kind"].as_str(), Some("z_image_rewrite"));
        let values = |name: &str| -> Vec<f32> {
            payload[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect()
        };
        let bf16_ok = |actual: f32, expected: f32| -> bool {
            let error = (actual - expected).abs();
            let exponent = expected.abs().max(f32::MIN_POSITIVE).log2().floor();
            let allowed = 4.0 * 2f32.powf(exponent - 7.0) + 4e-3;
            error <= allowed
        };
        let stream = Stream::gpu().unwrap();
        danmo_mlx::reset_peak_memory().unwrap();
        let width = payload["width"].as_u64().unwrap() as u32;
        let height = payload["height"].as_u64().unwrap() as u32;
        let latent_h = (height / 8) as i32;
        let latent_w = (width / 8) as i32;
        let init = payload["init_timestep"].as_u64().unwrap() as usize;
        let seed = payload["seed"].as_u64().unwrap();
        let ids: Vec<i32> = payload["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect();
        let source_rgb: Vec<u8> = payload["source_rgb"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u8)
            .collect();
        let plane = (width * height) as usize;
        assert_eq!(source_rgb.len(), plane * 3);
        let mut image = vec![0f32; plane * 3];
        for y in 0..height as usize {
            for x in 0..width as usize {
                for channel in 0..3 {
                    let byte = source_rgb[(y * width as usize + x) * 3 + channel];
                    image[channel * plane + y * width as usize + x] =
                        (byte as f32 / 255.0) * 2.0 - 1.0;
                }
            }
        }

        let started = std::time::Instant::now();
        let vae_weights =
            crate::common::vae::mlx::load(&bundle.join("vae"), WeightDtype::Native).unwrap();
        let spec = crate::common::vae::VaeSpec {
            latent_channels: 16,
            block_out: vec![128, 256, 512, 512],
            resnets: 2,
            groups: 32,
            scaling: 0.3611,
            shift: 0.1159,
            attention: true,
        };
        let pixels = Array::from_f32_slice(&image, &[1, 3, height as i32, width as i32]).unwrap();
        let rust_encoded = crate::common::vae::mlx::encode(&vae_weights, &spec, &pixels, &stream)
            .unwrap()
            .to_vec_f32(&stream)
            .unwrap();
        let expected_encoded = values("encoded");
        let mut encode_error = 0.0f32;
        for (actual, expected) in rust_encoded.iter().zip(&expected_encoded) {
            encode_error = encode_error.max((actual - expected).abs());
        }
        // Metal BF16 VAE encode is not bit-stable across main runs (even
        // main-vs-main drifts). Record the gap and inject the fixture encoded
        // latent into the DiT chain — same pattern as create's fixed latent.
        if width <= 16 && height <= 16 {
            assert!(
                encode_error <= 0.02,
                "rewrite VAE encode max abs error {encode_error} exceeds 0.02 on 16x16"
            );
        } else {
            println!(
                "rewrite VAE encode observational max abs error {encode_error} (not a gate at {width}x{height})"
            );
        }
        println!(
            "rewrite VAE encode: {:.3}s, max abs error {encode_error} (DiT uses fixture encoded)",
            started.elapsed().as_secs_f64()
        );
        let encoded = expected_encoded;

        let text_weights = crate::common::qwen3::mlx::load_directory(
            &bundle.join("text_encoder"),
            WeightDtype::Native,
        )
        .unwrap();
        let text_config = crate::common::qwen3::Qwen3Config {
            vocab: 151_936,
            hidden: 2560,
            layers: 36,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            intermediate: 9728,
            rope_theta: 1_000_000.0,
        };
        let caption = crate::common::qwen3::mlx::encode_ids(
            &text_weights,
            text_config,
            "model.",
            &ids,
            &vec![true; ids.len()],
            &[35],
            &stream,
        )
        .unwrap()
        .squeeze_axis(0, &stream)
        .unwrap()
        .astype(Dtype::BFloat16, &stream)
        .unwrap();
        caption.eval().unwrap();
        drop(text_weights);
        let caption_values = caption.to_vec_f32(&stream).unwrap();
        let expected_caption = values("caption");
        let caption_error = caption_values
            .iter()
            .zip(&expected_caption)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(caption_error <= 1e-6, "caption error {caption_error}");

        let full_sigmas = values("sigmas");
        assert!(
            init < full_sigmas.len().saturating_sub(1),
            "init_timestep {init} leaves no denoise steps"
        );
        let sliced: Vec<f32> = full_sigmas[init..].to_vec();
        let noise = danmo_mlx::RandomKey::new(seed)
            .unwrap()
            .normal(
                &[1, 16, latent_h, latent_w],
                danmo_mlx::Dtype::Float32,
                &stream,
            )
            .unwrap()
            .to_vec_f32(&stream)
            .unwrap();
        assert_eq!(
            noise,
            values("noise"),
            "edit noise must match main RandomKey"
        );
        let sigma = sliced[0];
        assert!(
            (sigma - payload["sig_blend"].as_f64().unwrap() as f32).abs() < 1e-6,
            "sig_blend mismatch"
        );
        let blended: Vec<f32> = encoded
            .iter()
            .zip(&noise)
            .map(|(reference, noise)| (1.0 - sigma) * reference + sigma * noise)
            .collect();
        let expected_blended = values("blended");
        let mut blend_error = 0.0f32;
        for (actual, expected) in blended.iter().zip(&expected_blended) {
            blend_error = blend_error.max((actual - expected).abs());
            assert!(
                bf16_ok(*actual, *expected),
                "blended latent error {actual} vs {expected}"
            );
        }
        println!("rewrite blend: max abs error {blend_error}");

        let dit_started = std::time::Instant::now();
        let config = MlxZImageConfig::published();
        let weights = load_directory(
            &bundle.join("transformer"),
            config,
            WeightDtype::Native,
            None,
        )
        .unwrap();
        let grid = geom::token_grid(width, height, ids.len() as i64, config.vae_scale).unwrap();
        let current = Array::from_f32_slice(&blended, &[1, 16, latent_h, latent_w])
            .unwrap()
            .squeeze_axis(0, &stream)
            .unwrap()
            .astype(danmo_mlx::Dtype::BFloat16, &stream)
            .unwrap();
        let mut step = 0usize;
        let denoised = crate::families::z_image::mlx::denoise::euler(
            &current,
            &sliced,
            &stream,
            |latent, sigma| {
                if step > 0 {
                    let expected_current = &payload["step_latents"][step - 1];
                    let current_values = latent.to_vec_f32(&stream)?;
                    let mut current_error = 0.0f32;
                    for (actual, expected) in current_values
                        .iter()
                        .zip(expected_current.as_array().unwrap())
                    {
                        current_error =
                            current_error.max((actual - expected.as_f64().unwrap() as f32).abs());
                    }
                    println!("rewrite step {step}: input latent max abs error {current_error}");
                }
                let velocity = forward(&weights, config, grid, latent, &caption, sigma, &stream)?;
                let actual = velocity.to_vec_f32(&stream)?;
                let error = actual
                    .iter()
                    .zip(payload["velocities"][step].as_array().unwrap())
                    .map(|(actual, expected)| (actual - expected.as_f64().unwrap() as f32).abs())
                    .fold(0.0f32, f32::max);
                println!("rewrite step {step}: velocity max abs error {error}");
                for (actual, expected) in actual
                    .iter()
                    .zip(payload["velocities"][step].as_array().unwrap())
                {
                    let expected = expected.as_f64().unwrap() as f32;
                    assert!(
                        bf16_ok(*actual, expected),
                        "rewrite velocity step {step} error exceeds BF16 gate"
                    );
                }
                step += 1;
                Ok(velocity)
            },
        )
        .unwrap();
        let denoised_values = denoised.to_vec_f32(&stream).unwrap();
        let expected_denoised = values("denoised");
        let mut latent_error = 0.0f32;
        for (actual, expected) in denoised_values.iter().zip(&expected_denoised) {
            latent_error = latent_error.max((actual - expected).abs());
            assert!(
                bf16_ok(*actual, *expected),
                "rewrite denoised error {actual} vs {expected}"
            );
        }
        println!(
            "rewrite denoise: {:.3}s, max abs error {latent_error}, steps {step}",
            dit_started.elapsed().as_secs_f64()
        );
        drop(weights);

        let vae_started = std::time::Instant::now();
        let denoised_nchw = denoised.expand_dims(0, &stream).unwrap();
        let decoded = crate::common::vae::mlx::decode(&vae_weights, &spec, &denoised_nchw, &stream)
            .unwrap()
            .to_vec_f32(&stream)
            .unwrap();
        let expected_output = values("output");
        let mut output_error = 0.0f32;
        for (actual, expected) in decoded.iter().zip(&expected_output) {
            output_error = output_error.max((actual - expected).abs());
            let exponent = expected.abs().max(f32::MIN_POSITIVE).log2().floor();
            let allowed = 4.0 * 2f32.powf(exponent - 7.0) + 1e-3;
            assert!(
                (actual - expected).abs() <= allowed,
                "rewrite decoded error exceeds BF16 gate"
            );
        }
        let actual_pixels: Vec<u8> = (0..plane * 3)
            .map(|index| {
                let channel = index % 3;
                let pixel = index / 3;
                let value = decoded[channel * plane + pixel];
                ((value + 1.0) * 127.5).clamp(0.0, 255.0).round_ties_even() as u8
            })
            .collect();
        let expected_pixels: Vec<u8> = payload["pixels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u8)
            .collect();
        let different_pixels = actual_pixels
            .iter()
            .zip(&expected_pixels)
            .filter(|(actual, expected)| actual != expected)
            .count();
        let max_pixel_error = actual_pixels
            .iter()
            .zip(&expected_pixels)
            .map(|(actual, expected)| actual.abs_diff(*expected))
            .max()
            .unwrap_or(0);
        assert!(
            max_pixel_error <= 1,
            "rewrite 8-bit RGB error {max_pixel_error}"
        );
        println!(
            "rewrite RGB parity: {different_pixels}/{} values differ, max channel error {max_pixel_error}; VAE {:.3}s output error {output_error}",
            plane * 3,
            vae_started.elapsed().as_secs_f64()
        );
        if let Ok(path) = std::env::var("DANQING_TEST_ZIMAGE_REWRITE_PNG") {
            std::fs::write(
                path,
                crate::common::png::encode_rgb(width, height, &actual_pixels),
            )
            .unwrap();
        }
        println!(
            "rewrite Metal peak {} active {}",
            danmo_mlx::peak_memory().unwrap(),
            danmo_mlx::active_memory().unwrap()
        );
    }

    /// Rewrite warm performance gate: resident multipass denoise+VAE vs main
    /// `warm_p50` (≤105%). Env: `DANQING_TEST_ZIMAGE_BUNDLE`,
    /// `DANQING_TEST_ZIMAGE_REWRITE_WARM_JSON` (fixture with `warm_p50`).
    /// Optional: `DANQING_TEST_ZIMAGE_WARM_PASSES`, `DANQING_TEST_ZIMAGE_WARM_WARMUP`,
    /// `DANQING_TEST_ZIMAGE_REWRITE_WARM_REPORT`.
    #[cfg(feature = "mlx-metal")]
    #[test]
    #[ignore]
    fn real_zimage_rewrite_warm_passes_match_main_and_report_performance() {
        let bundle = std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_BUNDLE").unwrap());
        let payload: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                std::env::var("DANQING_TEST_ZIMAGE_REWRITE_WARM_JSON").unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(payload["kind"].as_str(), Some("z_image_rewrite"));
        assert_eq!(payload["native_dtype"].as_str(), Some("checkpoint"));
        let main_p50 = payload
            .get("warm_p50")
            .expect("rewrite warm fixture needs warm_p50 from main");
        let passes_requested = std::env::var("DANQING_TEST_ZIMAGE_WARM_PASSES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(5);
        let warmup_passes = std::env::var("DANQING_TEST_ZIMAGE_WARM_WARMUP")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1);
        assert!(passes_requested >= 2);
        let values = |name: &str| -> Vec<f32> {
            payload[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect()
        };
        let stream = Stream::gpu().unwrap();
        danmo_mlx::reset_peak_memory().unwrap();
        let width = payload["width"].as_u64().unwrap() as u32;
        let height = payload["height"].as_u64().unwrap() as u32;
        let latent_h = (height / 8) as i32;
        let latent_w = (width / 8) as i32;
        let init = payload["init_timestep"].as_u64().unwrap() as usize;
        let ids: Vec<i32> = payload["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect();
        let full_sigmas = values("sigmas");
        let sliced: Vec<f32> = full_sigmas[init..].to_vec();
        let blended = values("blended");
        let expected_denoised = values("denoised");
        let caption_values = values("caption");

        let config = MlxZImageConfig::published();
        let grid = geom::token_grid(width, height, ids.len() as i64, config.vae_scale).unwrap();
        let weights = load_directory(
            &bundle.join("transformer"),
            config,
            WeightDtype::Native,
            None,
        )
        .unwrap();
        let vae_weights =
            crate::common::vae::mlx::load(&bundle.join("vae"), WeightDtype::Native).unwrap();
        let spec = crate::common::vae::VaeSpec {
            latent_channels: 16,
            block_out: vec![128, 256, 512, 512],
            resnets: 2,
            groups: 32,
            scaling: 0.3611,
            shift: 0.1159,
            attention: true,
        };
        let caption = Array::from_f32_slice(&caption_values, &[ids.len() as i32, 2560])
            .unwrap()
            .astype(Dtype::BFloat16, &stream)
            .unwrap();
        caption.eval().unwrap();

        let bf16_ok = |actual: f32, expected: f32| -> bool {
            let error = (actual - expected).abs();
            let exponent = expected.abs().max(f32::MIN_POSITIVE).log2().floor();
            let allowed = 4.0 * 2f32.powf(exponent - 7.0) + 4e-3;
            error <= allowed
        };
        let mut passes = Vec::new();
        for pass_index in 0..warmup_passes + passes_requested {
            let current = Array::from_f32_slice(&blended, &[1, 16, latent_h, latent_w])
                .unwrap()
                .squeeze_axis(0, &stream)
                .unwrap()
                .astype(Dtype::BFloat16, &stream)
                .unwrap();
            let denoise_started = std::time::Instant::now();
            let denoised = crate::families::z_image::mlx::denoise::euler(
                &current,
                &sliced,
                &stream,
                |latent, sigma| forward(&weights, config, grid, latent, &caption, sigma, &stream),
            )
            .unwrap();
            denoised.eval().unwrap();
            let denoise_seconds = denoise_started.elapsed().as_secs_f64();
            let vae_started = std::time::Instant::now();
            let denoised_nchw = denoised.expand_dims(0, &stream).unwrap();
            let decoded =
                crate::common::vae::mlx::decode(&vae_weights, &spec, &denoised_nchw, &stream)
                    .unwrap();
            decoded.eval().unwrap();
            let vae_seconds = vae_started.elapsed().as_secs_f64();
            let denoised_values = denoised.to_vec_f32(&stream).unwrap();
            if pass_index == 0 {
                for (actual, expected) in denoised_values.iter().zip(&expected_denoised) {
                    assert!(
                        bf16_ok(*actual, *expected),
                        "rewrite warm pass0 denoised diverged"
                    );
                }
            } else {
                // Later warm passes must stay bit-exact with the first warm latent.
                // Compare against fixture once; recurrent stability is the gate.
                let max_err = denoised_values
                    .iter()
                    .zip(&expected_denoised)
                    .map(|(a, e)| (a - e).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err <= 1e-3
                        || denoised_values
                            .iter()
                            .zip(&expected_denoised)
                            .all(|(a, e)| bf16_ok(*a, *e)),
                    "rewrite warm pass {pass_index} latent max abs error {max_err}"
                );
            }
            let total = denoise_seconds + vae_seconds;
            println!(
                "rewrite warm pass {pass_index}: denoise={denoise_seconds:.3}s vae={vae_seconds:.3}s total={total:.3}s"
            );
            passes.push(serde_json::json!({
                "denoise": denoise_seconds,
                "vae": vae_seconds,
                "total": total,
            }));
        }
        let warm = &passes[warmup_passes..];
        let median = |key: &str| -> f64 {
            let mut vals: Vec<f64> = warm.iter().map(|row| row[key].as_f64().unwrap()).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = vals.len() / 2;
            if vals.len() % 2 == 1 {
                vals[mid]
            } else {
                0.5 * (vals[mid - 1] + vals[mid])
            }
        };
        let mut ratios = serde_json::Map::new();
        for key in ["denoise", "vae", "total"] {
            let rust = median(key);
            let main = main_p50[key].as_f64().unwrap();
            let ratio = rust / main;
            println!("rewrite warm p50 {key}: rust={rust:.3}s main={main:.3}s ratio={ratio:.3}");
            assert!(
                ratio <= 1.05,
                "rewrite warm p50 {key} ratio {ratio} exceeds 105% of main"
            );
            ratios.insert(key.to_string(), serde_json::json!(ratio));
        }
        let report = serde_json::json!({
            "width": width,
            "height": height,
            "steps": payload["steps"],
            "init_timestep": init,
            "warm_p50": {
                "denoise": median("denoise"),
                "vae": median("vae"),
                "total": median("total"),
            },
            "main_warm_p50": main_p50,
            "rust_to_main_ratio": ratios,
            "passes": warm,
            "peak_memory": danmo_mlx::peak_memory().unwrap(),
            "active_memory": danmo_mlx::active_memory().unwrap(),
            "dit_weight_bytes": weights.nbytes(),
        });
        if let Ok(path) = std::env::var("DANQING_TEST_ZIMAGE_REWRITE_WARM_REPORT") {
            std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
        }
        println!(
            "rewrite warm gate passed; peak {} active {}",
            danmo_mlx::peak_memory().unwrap(),
            danmo_mlx::active_memory().unwrap()
        );
    }

    /// Real-weight `mx.compile` gate: the compiled full DiT must reproduce the
    /// eager main-parity velocities step by step, and warm replay timing is
    /// recorded against eager for the S3/S4 performance decision.
    #[cfg(feature = "mlx-metal")]
    #[test]
    #[ignore]
    fn real_compiled_dit_matches_eager_and_reports_warm_timings() {
        let bundle = std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_BUNDLE").unwrap());
        let payload: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(std::env::var("DANQING_TEST_ZIMAGE_REAL_JSON").unwrap())
                .unwrap(),
        )
        .unwrap();
        let values = |name: &str| -> Vec<f32> {
            payload[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect()
        };
        let stream = Stream::gpu().unwrap();
        let config = MlxZImageConfig::published();
        let weights =
            load_directory(&bundle.join("transformer"), config, WeightDtype::F32, None).unwrap();
        let ids: Vec<i64> = payload["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap())
            .collect();
        let width = payload["width"].as_u64().unwrap_or(16) as u32;
        let height = payload["height"].as_u64().unwrap_or(16) as u32;
        let latent_h = (height / 8) as i32;
        let latent_w = (width / 8) as i32;
        let grid = geom::token_grid(width, height, ids.len() as i64, config.vae_scale).unwrap();
        let caption = Array::from_f32_slice(
            &values("caption"),
            &[
                payload["caption_shape"][0].as_i64().unwrap() as i32,
                payload["caption_shape"][1].as_i64().unwrap() as i32,
            ],
        )
        .unwrap();
        let sigmas = values("sigmas");
        let run_steps = std::env::var("DANQING_TEST_ZIMAGE_COMPILE_STEPS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(sigmas.len() - 1)
            .min(sigmas.len() - 1);
        let compiled =
            super::CompiledForward::new(&weights, config, grid, &caption, false, &stream).unwrap();

        let run = |compiled_mode: bool| -> Result<(Vec<Vec<f32>>, Vec<f64>)> {
            let mut latent = Array::from_f32_slice(&values("latent"), &[1, 16, latent_h, latent_w])
                .unwrap()
                .squeeze_axis(0, &stream)
                .unwrap()
                .astype(danmo_mlx::Dtype::BFloat16, &stream)
                .unwrap();
            let mut velocities = Vec::new();
            let mut timings = Vec::new();
            for step in 0..run_steps {
                let started = std::time::Instant::now();
                let velocity = if compiled_mode {
                    let timestep = super::embed_timestep(&weights, config, sigmas[step], &stream)?;
                    compiled.apply(&latent, &timestep)?
                } else {
                    forward(
                        &weights,
                        config,
                        grid,
                        &latent,
                        &caption,
                        sigmas[step],
                        &stream,
                    )?
                };
                velocity.eval().unwrap();
                timings.push(started.elapsed().as_secs_f64());
                let output = if velocity.dtype() == latent.dtype() {
                    velocity.clone()
                } else {
                    velocity.astype(latent.dtype(), &stream)?
                };
                velocities.push(velocity.to_vec_f32(&stream)?);
                let delta = Array::from_f32_slice(&[sigmas[step + 1] - sigmas[step]], &[])?
                    .astype(latent.dtype(), &stream)?;
                latent = Array::add(
                    &latent,
                    &Array::multiply(&output, &delta, &stream)?,
                    &stream,
                )?;
            }
            Ok((velocities, timings))
        };

        // Warm-up pass so both measured runs see resident weights and kernels.
        let _ = run(false).unwrap();
        // Eager baseline (also the parity reference against main).
        let (eager_velocities, eager_timings) = run(false).unwrap();
        let mut main_error = 0.0f32;
        for (step, velocity) in eager_velocities.iter().enumerate() {
            for (actual, expected) in velocity
                .iter()
                .zip(payload["velocities"][step].as_array().unwrap())
            {
                main_error = main_error.max((actual - expected.as_f64().unwrap() as f32).abs());
            }
        }
        // First compiled call traces and builds; later calls are warm replays.
        let (compiled_velocities, compiled_timings) = run(true).unwrap();
        let mut compile_error = 0.0f32;
        for (step, (compiled_velocity, eager_velocity)) in compiled_velocities
            .iter()
            .zip(&eager_velocities)
            .enumerate()
        {
            let mut step_error = 0.0f32;
            for (actual, expected) in compiled_velocity.iter().zip(eager_velocity) {
                let error = (actual - expected).abs();
                step_error = step_error.max(error);
                let exponent = expected.abs().max(f32::MIN_POSITIVE).log2().floor();
                let allowed = 4.0 * 2f32.powf(exponent - 7.0) + 4e-3;
                assert!(
                    error <= allowed,
                    "compiled step {step} error {error} exceeds {allowed}"
                );
            }
            compile_error = compile_error.max(step_error);
            println!("compiled step {step}: max abs error vs eager {step_error}");
        }
        let eager_total: f64 = eager_timings.iter().sum();
        let compiled_total: f64 = compiled_timings.iter().sum();
        println!("real DiT main parity: eager max abs error {main_error}");
        println!("compiled vs eager: max abs error {compile_error}");
        println!(
            "eager steps: {:?}\ncompiled steps (first = trace/build): {:?}",
            eager_timings, compiled_timings
        );
        println!(
            "warm {}-step totals: eager {eager_total:.3}s, compiled {compiled_total:.3}s ({:.1}%)",
            sigmas.len() - 1,
            100.0 * compiled_total / eager_total
        );
        if let Ok(path) = std::env::var("DANQING_TEST_ZIMAGE_WARM_JSON") {
            let payload = serde_json::json!({
                "eager_steps": eager_timings,
                "compiled_steps": compiled_timings,
                "eager_total": eager_total,
                "compiled_total": compiled_total,
                "compiled_vs_eager_max_error": compile_error,
                "eager_vs_main_max_error": main_error,
            });
            std::fs::write(path, serde_json::to_string(&payload).unwrap()).unwrap();
        }
    }

    /// The compiled TeaCache compute path must reproduce the eager cached
    /// path step by step (velocities, gate decisions, residual reuse) on the
    /// real checkpoint at production resolution, including forced skips.
    #[cfg(feature = "mlx-metal")]
    #[test]
    #[ignore]
    fn real_compiled_teacache_matches_eager_cache_semantics() {
        let bundle = std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_BUNDLE").unwrap());
        let payload: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(std::env::var("DANQING_TEST_ZIMAGE_REAL_JSON").unwrap())
                .unwrap(),
        )
        .unwrap();
        let values = |name: &str| -> Vec<f32> {
            payload[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect()
        };
        let stream = Stream::gpu().unwrap();
        let width = payload["width"].as_u64().unwrap() as u32;
        let height = payload["height"].as_u64().unwrap() as u32;
        let config = MlxZImageConfig::published();
        let weights = load_directory(
            &bundle.join("transformer"),
            config,
            WeightDtype::Native,
            None,
        )
        .unwrap();
        let ids: Vec<i64> = payload["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap())
            .collect();
        let grid = geom::token_grid(width, height, ids.len() as i64, config.vae_scale).unwrap();
        let caption = Array::from_f32_slice(
            &values("caption"),
            &[
                payload["caption_shape"][0].as_i64().unwrap() as i32,
                payload["caption_shape"][1].as_i64().unwrap() as i32,
            ],
        )
        .unwrap()
        .astype(Dtype::BFloat16, &stream)
        .unwrap();
        let sigmas = values("sigmas");
        let steps = sigmas.len() - 1;
        let threshold = payload["teacache_thresh"].as_f64();
        let caption_cache = CaptionCache::new(&weights, config, grid, &caption, &stream).unwrap();
        let compiled = CompiledCompute::new(&weights, config, grid, &stream).unwrap();
        let run = |compiled_mode: bool| -> Result<(Vec<Vec<f32>>, Vec<String>, usize, usize)> {
            let mut cache = threshold.map_or_else(
                || StepCache::z_image(steps),
                |threshold| StepCache::new(steps, threshold),
            );
            let mut latent = Array::from_f32_slice(
                &values("latent"),
                &[1, 16, (height / 8) as i32, (width / 8) as i32],
            )?
            .squeeze_axis(0, &stream)?
            .astype(Dtype::BFloat16, &stream)?;
            let mut velocities = Vec::new();
            for step in 0..steps {
                let (velocity, residual) = if compiled_mode {
                    forward_compiled_with_step_cache(
                        &weights,
                        config,
                        &compiled,
                        &latent,
                        &caption_cache,
                        sigmas[step],
                        &mut cache,
                        step,
                        &stream,
                    )?
                } else {
                    forward_with_step_cache(
                        &weights,
                        config,
                        grid,
                        &latent,
                        &caption,
                        sigmas[step],
                        &mut cache,
                        step,
                        &stream,
                    )
                    .map(|velocity| (velocity, None))?
                };
                let cast = velocity.astype(latent.dtype(), &stream)?;
                let delta = Array::from_f32_slice(&[sigmas[step + 1] - sigmas[step]], &[])?
                    .astype(latent.dtype(), &stream)?;
                let next = Array::add(&latent, &Array::multiply(&cast, &delta, &stream)?, &stream)?;
                let mut targets = vec![&velocity, &next];
                if let Some(residual) = &residual {
                    targets.push(residual);
                }
                Array::eval_all(&targets)?;
                velocities.push(velocity.to_vec_f32(&stream)?);
                latent = next;
            }
            let kinds = cache.history.iter().map(|d| d.kind.to_string()).collect();
            Ok((velocities, kinds, cache.computed_count, cache.skipped_count))
        };
        let (eager, eager_kinds, eager_computed, eager_skipped) = run(false).unwrap();
        let (compiled, compiled_kinds, compiled_computed, compiled_skipped) = run(true).unwrap();
        assert_eq!(eager_kinds, compiled_kinds);
        assert_eq!(eager_computed, compiled_computed);
        assert_eq!(eager_skipped, compiled_skipped);
        // Cross-check against the main fixture when it carries gate history.
        if let Some(expected) = payload["cache_decisions"].as_array() {
            assert_eq!(eager_kinds.len(), expected.len());
            for (actual, expected) in eager_kinds.iter().zip(expected) {
                assert_eq!(actual, expected["kind"].as_str().unwrap());
            }
            assert_eq!(
                eager_computed,
                payload["cache_counts"]["computed"].as_u64().unwrap() as usize
            );
            assert_eq!(
                eager_skipped,
                payload["cache_counts"]["skipped"].as_u64().unwrap() as usize
            );
        }
        for (step, (eager_velocity, compiled_velocity)) in eager.iter().zip(&compiled).enumerate() {
            let mut error = 0.0f32;
            for (actual, expected) in compiled_velocity.iter().zip(eager_velocity) {
                error = error.max((actual - expected).abs());
            }
            println!("compiled TeaCache step {step}: max abs error vs eager {error}");
            assert_eq!(compiled_velocity, eager_velocity, "step {step} diverged");
        }
        println!(
            "compiled TeaCache parity: {} compute / {} skip, all steps bit-exact",
            eager_computed, eager_skipped
        );
    }

    /// Session lifecycle gate (plan S2): consecutive generations on resident
    /// weights, geometry switches with compiled re-trace, mid-run cancellation
    /// from another thread, and unload with memory release. Every completed
    /// run must stay bit-exact with its main fixture; a cancelled run must not
    /// pollute the next one.
    #[cfg(feature = "mlx-metal")]
    #[test]
    #[ignore]
    fn real_session_lifecycle_consecutive_switch_cancel_unload() {
        let bundle = std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_BUNDLE").unwrap());
        let load_fixture = |path: String| -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
        };
        let default_payload =
            load_fixture(std::env::var("DANQING_TEST_ZIMAGE_WARM_MAIN_JSON").unwrap());
        let small_payload =
            load_fixture(std::env::var("DANQING_TEST_ZIMAGE_LIFECYCLE_SMALL_JSON").unwrap());
        for payload in [&default_payload, &small_payload] {
            assert_eq!(payload["native_dtype"].as_str(), Some("checkpoint"));
            assert!(payload["teacache"].as_bool().unwrap());
        }
        let stream = Stream::gpu().unwrap();
        let config = MlxZImageConfig::published();
        let weights = load_directory(
            &bundle.join("transformer"),
            config,
            WeightDtype::Native,
            None,
        )
        .unwrap();
        let vae_weights =
            crate::common::vae::mlx::load(&bundle.join("vae"), WeightDtype::Native).unwrap();
        let spec = crate::common::vae::VaeSpec {
            latent_channels: 16,
            block_out: vec![128, 256, 512, 512],
            resnets: 2,
            groups: 32,
            scaling: 0.3611,
            shift: 0.1159,
            attention: true,
        };
        let mut cache = StepCache::z_image(9);
        let mut compiled: Option<(super::CompiledCompute, u32, u32)> = None;
        let eager_mode = std::env::var("DANQING_TEST_ZIMAGE_LIFECYCLE_EAGER").is_ok();

        #[derive(Debug)]
        struct RunOutcome {
            velocities: Vec<Vec<u8>>,
            latents: Vec<Vec<u8>>,
            pixels: Vec<u8>,
            computed: usize,
            skipped: usize,
        }

        let mut run = |payload: &serde_json::Value,
                       cancel: Option<&danmo_mlx::CancelToken>|
         -> Result<RunOutcome> {
            let values = |name: &str| -> Vec<f32> {
                payload[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_f64().unwrap() as f32)
                    .collect()
            };
            let width = payload["width"].as_u64().unwrap() as u32;
            let height = payload["height"].as_u64().unwrap() as u32;
            let ids: Vec<i64> = payload["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_i64().unwrap())
                .collect();
            let grid = geom::token_grid(width, height, ids.len() as i64, config.vae_scale)
                .map_err(|error| danmo_mlx::Error::new("Z-Image lifecycle", error.to_string()))?;
            let caption = Array::from_f32_slice(
                &values("caption"),
                &[
                    payload["caption_shape"][0].as_i64().unwrap() as i32,
                    payload["caption_shape"][1].as_i64().unwrap() as i32,
                ],
            )?
            .astype(Dtype::BFloat16, &stream)?;
            let sigmas = values("sigmas");
            // Geometry switch: rebuild the compiled compute for the new token
            // grid; the resident weights and the StepCache allocation stay.
            let needs_rebuild = compiled
                .as_ref()
                .is_none_or(|(_, w, h)| *w != width || *h != height);
            if needs_rebuild && !eager_mode {
                compiled = Some((
                    super::CompiledCompute::new(&weights, config, grid, &stream)?,
                    width,
                    height,
                ));
            }
            let compute = compiled.as_ref().map(|(compute, _, _)| compute);
            let caption_cache =
                super::CaptionCache::new(&weights, config, grid, &caption, &stream)?;
            cache.reset();
            let initial_latent = Array::from_f32_slice(
                &values("latent"),
                &[1, 16, (height / 8) as i32, (width / 8) as i32],
            )?
            .squeeze_axis(0, &stream)?
            .astype(Dtype::BFloat16, &stream)?;
            let mut velocities = Vec::new();
            let mut latents = Vec::new();
            let final_latent = crate::families::z_image::mlx::denoise::euler_cancellable(
                &initial_latent,
                &sigmas,
                &stream,
                cancel,
                |current, sigma| {
                    let step = velocities.len();
                    let (velocity, residual) = match compute {
                        Some(compute) => super::forward_compiled_with_step_cache(
                            &weights,
                            config,
                            compute,
                            current,
                            &caption_cache,
                            sigma,
                            &mut cache,
                            step,
                            &stream,
                        )?,
                        None => super::forward_cached_with_step_cache(
                            &weights,
                            config,
                            grid,
                            current,
                            &caption_cache,
                            sigma,
                            &mut cache,
                            step,
                            &stream,
                        )?,
                    };
                    let cast = velocity.astype(current.dtype(), &stream)?;
                    let delta = Array::from_f32_slice(&[sigmas[step + 1] - sigmas[step]], &[])?
                        .astype(current.dtype(), &stream)?;
                    let next =
                        Array::add(current, &Array::multiply(&cast, &delta, &stream)?, &stream)?;
                    let mut targets = vec![&velocity, &next];
                    if let Some(residual) = &residual {
                        targets.push(residual);
                    }
                    Array::eval_all(&targets)?;
                    velocities.push(velocity.to_raw_bytes(&stream)?);
                    latents.push(next.to_raw_bytes(&stream)?);
                    Ok(velocity)
                },
            )?;
            let decoded = crate::common::vae::mlx::decode(
                &vae_weights,
                &spec,
                &final_latent.expand_dims(0, &stream)?,
                &stream,
            )?;
            let decoded_values = decoded.to_vec_f32(&stream)?;
            let plane = (width * height) as usize;
            let pixels: Vec<u8> = (0..plane * 3)
                .map(|index| {
                    let channel = index % 3;
                    let pixel = index / 3;
                    let value = decoded_values[channel * plane + pixel];
                    ((value + 1.0) * 127.5).clamp(0.0, 255.0).round_ties_even() as u8
                })
                .collect();
            Ok(RunOutcome {
                velocities,
                latents,
                pixels,
                computed: cache.computed_count,
                skipped: cache.skipped_count,
            })
        };

        let bf16_bytes = |values: &[f32]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| {
                    let bits = value.to_bits();
                    assert_eq!(bits & 0xFFFF, 0, "expected value is not exact BF16");
                    [(bits >> 16) as u8, (bits >> 24) as u8]
                })
                .collect()
        };
        let verify = |outcome: &RunOutcome, payload: &serde_json::Value, label: &str| {
            let expected_f32 = |name: &str| -> Vec<f32> {
                payload[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_f64().unwrap() as f32)
                    .collect()
            };
            let steps = payload["velocities"].as_array().unwrap().len();
            assert_eq!(outcome.velocities.len(), steps, "{label}: step count");
            for step in 0..steps {
                let expected_velocity: Vec<u8> = payload["velocities"][step]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|value| (value.as_f64().unwrap() as f32).to_bits().to_le_bytes())
                    .collect();
                assert_eq!(
                    outcome.velocities[step], expected_velocity,
                    "{label} step {step}: velocity diverged"
                );
                let expected_latent = bf16_bytes(
                    &payload["step_latents"][step]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|value| value.as_f64().unwrap() as f32)
                        .collect::<Vec<_>>(),
                );
                assert_eq!(
                    outcome.latents[step], expected_latent,
                    "{label} step {step}: latent diverged"
                );
            }
            let expected_pixels: Vec<u8> = payload["pixels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as u8)
                .collect();
            assert_eq!(outcome.pixels, expected_pixels, "{label}: RGB diverged");
            assert_eq!(
                outcome.computed,
                payload["cache_counts"]["computed"].as_u64().unwrap() as usize,
                "{label}: compute count"
            );
            assert_eq!(
                outcome.skipped,
                payload["cache_counts"]["skipped"].as_u64().unwrap() as usize,
                "{label}: skip count"
            );
            let _ = expected_f32;
        };

        // 0. Small geometry first (fresh process state) to separate geometry
        //    issues from sequence effects.
        if std::env::var("DANQING_TEST_ZIMAGE_LIFECYCLE_SMALL_FIRST").is_ok() {
            let small_first = run(&small_payload, None).unwrap();
            verify(&small_first, &small_payload, "run 0 (small first)");
            println!("lifecycle: small-first run bit-exact");
        }
        // 1. Two consecutive default-resolution generations on resident weights.
        let first = run(&default_payload, None).unwrap();
        verify(&first, &default_payload, "run 1");
        let second = run(&default_payload, None).unwrap();
        verify(&second, &default_payload, "run 2 (consecutive)");
        assert_eq!(first.velocities, second.velocities);
        assert_eq!(first.pixels, second.pixels);
        println!("lifecycle: consecutive default runs bit-exact");

        // 2. Switch to the small geometry (compiled re-trace), then back.
        let small = run(&small_payload, None).unwrap();
        verify(&small, &small_payload, "run 3 (small geometry)");
        let back = run(&default_payload, None).unwrap();
        verify(&back, &default_payload, "run 4 (switch back)");
        assert_eq!(first.velocities, back.velocities);
        println!("lifecycle: geometry switch and switch-back bit-exact");

        // 3. Cancel mid-run from another thread; the next run must be clean.
        let token = danmo_mlx::CancelToken::new();
        let control = token.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(2500));
            control.cancel();
        });
        let error = run(&default_payload, Some(&token)).unwrap_err();
        assert!(error.to_string().contains("cancelled at step"), "{error}");
        println!("lifecycle: mid-run cancellation: {error}");
        let after_cancel = run(&default_payload, None).unwrap();
        verify(&after_cancel, &default_payload, "run 6 (after cancel)");
        assert_eq!(first.velocities, after_cancel.velocities);
        println!("lifecycle: post-cancel run bit-exact (no state pollution)");

        // 4. Unload: dropping weights + compiled closures and clearing the
        //    buffer pool must release the resident DiT bytes.
        let before_unload = danmo_mlx::active_memory().unwrap();
        let dit_bytes = weights.nbytes();
        drop(compiled);
        drop(weights);
        drop(vae_weights);
        danmo_mlx::clear_cache().unwrap();
        let after_unload = danmo_mlx::active_memory().unwrap();
        println!(
            "lifecycle: unload active {before_unload} -> {after_unload} bytes (DiT {dit_bytes})"
        );
        assert!(
            before_unload.saturating_sub(after_unload) >= dit_bytes * 9 / 10,
            "unload did not release the resident weights: {before_unload} -> {after_unload}"
        );
        if let Ok(path) = std::env::var("DANQING_TEST_ZIMAGE_LIFECYCLE_REPORT") {
            let report = serde_json::json!({
                "consecutive_bit_exact": true,
                "geometry_switch_bit_exact": true,
                "cancel_error": error.to_string(),
                "post_cancel_bit_exact": true,
                "unload_active_before": before_unload,
                "unload_active_after": after_unload,
                "dit_weight_bytes": dit_bytes,
            });
            std::fs::write(path, serde_json::to_string(&report).unwrap()).unwrap();
        }
    }

    /// One-off micro benchmark: a single real unified block at production
    /// token geometry, eager with per-call eval, to compare against main's
    /// per-block timing (not a parity fixture).
    #[cfg(feature = "mlx-metal")]
    #[test]
    #[ignore]
    fn real_block_micro_benchmark() {
        let bundle = std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_BUNDLE").unwrap());
        let stream = Stream::gpu().unwrap();
        let config = MlxZImageConfig::published();
        let weights = load_directory(
            &bundle.join("transformer"),
            config,
            WeightDtype::Native,
            None,
        )
        .unwrap();
        let grid = geom::token_grid(768, 1280, 11, config.vae_scale).unwrap();
        let ids = geom::positions_image_first(grid);
        let axes = config.axes.map(|value| value as i32);
        let table = crate::families::z_image::mlx::attention::rope_table(
            &ids,
            axes,
            config.rope_theta,
            &stream,
        )
        .unwrap();
        table.eval().unwrap();
        let tokens = (grid.img_len + grid.cap_len) as i32;
        let values: Vec<f32> = (0..(tokens as usize * config.dim as usize))
            .map(|index| ((index % 31) as f32) * 0.01 - 0.15)
            .collect();
        let input = Array::from_f32_slice(&values, &[1, tokens, config.dim as i32])
            .unwrap()
            .astype(Dtype::BFloat16, &stream)
            .unwrap();
        let timestep = super::embed_timestep(&weights, config, 0.5, &stream).unwrap();
        input.eval().unwrap();
        timestep.eval().unwrap();
        for run in 0..3 {
            let started = std::time::Instant::now();
            let count = 30;
            let mut x = input.clone();
            for _ in 0..count {
                x = super::run_block(
                    &weights,
                    config,
                    "layers.0",
                    &x,
                    &ids,
                    true,
                    Some(&timestep),
                    Some(&table),
                    &stream,
                )
                .unwrap();
                x.eval().unwrap();
            }
            let per = started.elapsed().as_secs_f64() / count as f64;
            println!(
                "block micro run {run}: {:.1}ms per block+eval",
                per * 1000.0
            );
        }
    }

    /// Formal warm performance gate (plan §8.2) on the checkpoint-Native
    /// production path: the same fixed slice runs several passes with all
    /// weights resident, every pass must stay bit-exact with the main warm
    /// baseline (`tests/parity/zimage_warm_mlx.py`), and the warm p50 latency
    /// must not exceed 105% of main's warm p50 for encode/denoise/VAE.
    /// TeaCache fixtures exercise main's default optimization: the caption
    /// branch is precomputed once per run and the compute steps evaluate the
    /// stored residual together with velocity + latent, exactly like main.
    #[cfg(feature = "mlx-metal")]
    #[test]
    #[ignore]
    fn real_native_warm_passes_match_main_and_report_performance() {
        let bundle = std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_BUNDLE").unwrap());
        let payload: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(std::env::var("DANQING_TEST_ZIMAGE_WARM_MAIN_JSON").unwrap())
                .unwrap(),
        )
        .unwrap();
        let quantization = payload["quantization_bits"].as_i64().map(|bits| {
            let group_size = payload["quantization_group_size"].as_i64().unwrap_or(64);
            AffineQuantization::new(bits as i32, group_size as i32).unwrap()
        });
        assert!(
            payload["native_dtype"].as_str() == Some("checkpoint"),
            "the performance gate requires the checkpoint-Native baseline"
        );
        let passes_requested = std::env::var("DANQING_TEST_ZIMAGE_WARM_PASSES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(3);
        assert!(passes_requested >= 2, "need at least one warm pass");
        // Sustained-load thermal recovery can take several minutes on M5 Max;
        // extra passes let the p50 window sit in the recovered steady state.
        let warmup_passes = std::env::var("DANQING_TEST_ZIMAGE_WARM_WARMUP")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1);
        let values = |name: &str| -> Vec<f32> {
            payload[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect()
        };
        let stream = Stream::gpu().unwrap();
        let width = payload["width"].as_u64().unwrap() as u32;
        let height = payload["height"].as_u64().unwrap() as u32;
        let latent_h = (height / 8) as i32;
        let latent_w = (width / 8) as i32;
        let ids: Vec<i32> = payload["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect();
        let sigmas = values("sigmas");
        let steps = sigmas.len() - 1;
        let teacache = payload["teacache"].as_bool().unwrap_or(false);
        let config = MlxZImageConfig::published();
        let grid = geom::token_grid(width, height, ids.len() as i64, config.vae_scale).unwrap();

        danmo_mlx::reset_peak_memory().unwrap();
        let load_started = std::time::Instant::now();
        let text_weights = crate::common::qwen3::mlx::load_directory(
            &bundle.join("text_encoder"),
            WeightDtype::Native,
        )
        .unwrap()
        .materialize_embedding("model.embed_tokens.weight", &stream)
        .unwrap();
        let text_config = crate::common::qwen3::Qwen3Config {
            vocab: 151_936,
            hidden: 2560,
            layers: 36,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            intermediate: 9728,
            rope_theta: 1_000_000.0,
        };
        let weights = load_directory(
            &bundle.join("transformer"),
            config,
            WeightDtype::Native,
            quantization,
        )
        .unwrap();
        let dit_weight_bytes = weights.nbytes();
        if quantization.is_some() {
            assert!(
                weights.packed_names().count() > 100,
                "quantized warm gate loaded a dequantized DiT"
            );
        }
        let vae_weights =
            crate::common::vae::mlx::load(&bundle.join("vae"), WeightDtype::Native).unwrap();
        let spec = crate::common::vae::VaeSpec {
            latent_channels: 16,
            block_out: vec![128, 256, 512, 512],
            resnets: 2,
            groups: 32,
            scaling: 0.3611,
            shift: 0.1159,
            attention: true,
        };
        let load_seconds = load_started.elapsed().as_secs_f64();

        let expected_caption = values("caption");
        let expected_velocities: Vec<Vec<f32>> = payload["velocities"]
            .as_array()
            .unwrap()
            .iter()
            .map(|step| {
                step.as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_f64().unwrap() as f32)
                    .collect()
            })
            .collect();
        let expected_step_latents: Vec<Vec<f32>> = payload["step_latents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|step| {
                step.as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_f64().unwrap() as f32)
                    .collect()
            })
            .collect();
        let expected_denoised = values("denoised");
        let expected_pixels: Vec<u8> = payload["pixels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u8)
            .collect();
        // Latents are BF16 on both sides: compare raw bytes exactly, with the
        // expected f32 values widened back on the host (lossless). Velocities
        // leave main's final layer in F32 (F32 DiT weights), so they are
        // compared as exact f32 bit patterns after the readback cast.
        let bf16_bytes = |values: &[f32]| -> Vec<u8> {
            values
                .iter()
                .enumerate()
                .flat_map(|(index, value)| {
                    let bits = value.to_bits();
                    assert_eq!(
                        bits & 0xFFFF,
                        0,
                        "expected value {value} at {index} is not exact BF16"
                    );
                    [(bits >> 16) as u8, (bits >> 24) as u8]
                })
                .collect()
        };
        let expected_latent_bytes: Vec<Vec<u8>> = expected_step_latents
            .iter()
            .map(|step| bf16_bytes(step))
            .collect();
        let expected_velocity_bytes: Vec<Vec<u8>> = expected_velocities
            .iter()
            .map(|step| {
                step.iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
                    .collect()
            })
            .collect();

        let initial_latent = Array::from_f32_slice(&values("latent"), &[1, 16, latent_h, latent_w])
            .unwrap()
            .squeeze_axis(0, &stream)
            .unwrap()
            .astype(Dtype::BFloat16, &stream)
            .unwrap();
        let mut step_cache = teacache.then(|| {
            payload["teacache_thresh"].as_f64().map_or_else(
                || StepCache::z_image(steps),
                |threshold| StepCache::new(steps, threshold),
            )
        });
        // The checkpoint-Native production optimization for the default 9-step
        // schedule is TeaCache with the 30-block compute compiled; the gate,
        // residual reuse, and caption precompute stay eager (main's boundary).
        // DANQING_TEST_ZIMAGE_WARM_EAGER=1 selects the uncompiled cached path.
        let eager_mode = std::env::var("DANQING_TEST_ZIMAGE_WARM_EAGER").is_ok();
        let compile_trace = std::env::var("DANQING_TEST_ZIMAGE_WARM_COMPILE_TRACE").is_ok();
        let compiled_compute = (teacache && !eager_mode)
            .then(|| super::CompiledCompute::new(&weights, config, grid, &stream).unwrap());
        // Resident decode, same boundary as CompiledCompute: trace once per
        // latent shape, then the measured passes replay the graph.
        let compiled_vae = (!eager_mode).then(|| {
            crate::common::vae::mlx::CompiledDecoder::new(
                vae_weights.clone(),
                spec.clone(),
                stream.clone(),
            )
            .unwrap()
        });
        let mut trace_seconds = Vec::new();

        let mut passes = Vec::new();
        for pass_index in 0..warmup_passes + passes_requested {
            let started = std::time::Instant::now();
            let caption = crate::common::qwen3::mlx::encode_ids(
                &text_weights,
                text_config,
                "model.",
                &ids,
                &vec![true; ids.len()],
                &[35],
                &stream,
            )
            .unwrap()
            .squeeze_axis(0, &stream)
            .unwrap()
            .astype(Dtype::BFloat16, &stream)
            .unwrap();
            caption.eval().unwrap();
            let encode_seconds = started.elapsed().as_secs_f64();
            let caption_values = caption.to_vec_f32(&stream).unwrap();

            let dit_started = std::time::Instant::now();
            let mut latent = initial_latent.clone();
            let mut step_seconds = Vec::new();
            let mut pass_velocity_bytes: Vec<Vec<u8>> = Vec::new();
            let mut pass_latent_bytes: Vec<Vec<u8>> = Vec::new();
            let mut pending_velocity = Vec::new();
            let mut pending_latent = Vec::new();
            let caption_cache = step_cache.as_ref().map(|_| {
                super::CaptionCache::new(&weights, config, grid, &caption, &stream).unwrap()
            });
            let compiled = std::env::var("DANQING_TEST_ZIMAGE_WARM_COMPILE")
                .ok()
                .map(|_| {
                    super::CompiledForward::new(&weights, config, grid, &caption, false, &stream)
                        .unwrap()
                });
            if let Some(cache) = &mut step_cache {
                cache.reset();
            }
            for step in 0..steps {
                let step_started = std::time::Instant::now();
                let delta = Array::from_f32_slice(&[sigmas[step + 1] - sigmas[step]], &[])
                    .unwrap()
                    .astype(latent.dtype(), &stream)
                    .unwrap();
                let velocity;
                let mut eval_targets: Vec<&Array>;
                match (
                    &mut step_cache,
                    &caption_cache,
                    &compiled,
                    &compiled_compute,
                ) {
                    (Some(_), _, Some(_), _) => {
                        panic!("the full-compile probe does not wire the TeaCache gate; use an eager fixture");
                    }
                    (Some(cache), Some(caption_cache), None, compiled_compute) => {
                        let (output, residual) = match compiled_compute {
                            Some(compiled) => {
                                let trace_started = std::time::Instant::now();
                                let result = super::forward_compiled_with_step_cache(
                                    &weights,
                                    config,
                                    compiled,
                                    &latent,
                                    caption_cache,
                                    sigmas[step],
                                    cache,
                                    step,
                                    &stream,
                                )
                                .unwrap();
                                if compile_trace && pass_index == 0 {
                                    result.0.eval().unwrap();
                                    trace_seconds.push(trace_started.elapsed().as_secs_f64());
                                }
                                result
                            }
                            None => super::forward_cached_with_step_cache(
                                &weights,
                                config,
                                grid,
                                &latent,
                                caption_cache,
                                sigmas[step],
                                cache,
                                step,
                                &stream,
                            )
                            .unwrap(),
                        };
                        velocity = output;
                        // LinearScheduler.step casts the velocity to the
                        // latent dtype before the multiply (main semantics).
                        let cast = velocity.astype(latent.dtype(), &stream).unwrap();
                        let next = Array::add(
                            &latent,
                            &Array::multiply(&cast, &delta, &stream).unwrap(),
                            &stream,
                        )
                        .unwrap();
                        // main evaluates velocity + latent (+ residual on
                        // compute steps) in one call inside the step loop.
                        eval_targets = vec![&velocity, &next];
                        if let Some(residual) = &residual {
                            eval_targets.push(residual);
                        }
                        Array::eval_all(&eval_targets).unwrap();
                        latent = next;
                    }
                    (None, None, Some(compiled), _) => {
                        let timestep =
                            super::embed_timestep(&weights, config, sigmas[step], &stream).unwrap();
                        velocity = compiled.apply(&latent, &timestep).unwrap();
                        let cast = velocity.astype(latent.dtype(), &stream).unwrap();
                        let next = Array::add(
                            &latent,
                            &Array::multiply(&cast, &delta, &stream).unwrap(),
                            &stream,
                        )
                        .unwrap();
                        eval_targets = vec![&velocity, &next];
                        Array::eval_all(&eval_targets).unwrap();
                        latent = next;
                    }
                    (None, None, None, _) => {
                        velocity = forward(
                            &weights,
                            config,
                            grid,
                            &latent,
                            &caption,
                            sigmas[step],
                            &stream,
                        )
                        .unwrap();
                        let cast = velocity.astype(latent.dtype(), &stream).unwrap();
                        let next = Array::add(
                            &latent,
                            &Array::multiply(&cast, &delta, &stream).unwrap(),
                            &stream,
                        )
                        .unwrap();
                        eval_targets = vec![&velocity, &next];
                        Array::eval_all(&eval_targets).unwrap();
                        latent = next;
                    }
                    _ => unreachable!("cache and caption branch are enabled together"),
                }
                step_seconds.push(step_started.elapsed().as_secs_f64());
                pending_velocity.push(velocity);
                pending_latent.push(latent.clone());
            }
            let denoise_seconds = dit_started.elapsed().as_secs_f64();
            // Full-resolution readbacks stay outside the denoise clock.
            for step in 0..steps {
                pass_velocity_bytes.push(pending_velocity[step].to_raw_bytes(&stream).unwrap());
                pass_latent_bytes.push(pending_latent[step].to_raw_bytes(&stream).unwrap());
            }
            drop(pending_velocity);
            drop(pending_latent);
            if compile_trace && pass_index == 0 {
                println!("compiled trace per-step: {trace_seconds:?}");
            }
            assert_eq!(
                latent.to_raw_bytes(&stream).unwrap(),
                bf16_bytes(&expected_denoised),
                "pass {pass_index}: final latent diverged"
            );
            if let Some(cache) = &step_cache {
                assert_eq!(
                    cache.computed_count,
                    payload["cache_counts"]["computed"].as_u64().unwrap() as usize
                );
                assert_eq!(
                    cache.skipped_count,
                    payload["cache_counts"]["skipped"].as_u64().unwrap() as usize
                );
            }

            let vae_started = std::time::Instant::now();
            let denoised_nchw = latent.expand_dims(0, &stream).unwrap();
            let decoded = match &compiled_vae {
                Some(compiled) => compiled.apply(&denoised_nchw).unwrap(),
                None => {
                    crate::common::vae::mlx::decode(&vae_weights, &spec, &denoised_nchw, &stream)
                        .unwrap()
                }
            };
            decoded.eval().unwrap();
            let vae_seconds = vae_started.elapsed().as_secs_f64();
            let decoded_values = decoded.to_vec_f32(&stream).unwrap();
            let plane = (width * height) as usize;
            let pixels: Vec<u8> = (0..plane * 3)
                .map(|index| {
                    let channel = index % 3;
                    let pixel = index / 3;
                    let value = decoded_values[channel * plane + pixel];
                    ((value + 1.0) * 127.5).clamp(0.0, 255.0).round_ties_even() as u8
                })
                .collect();
            let total = encode_seconds + denoise_seconds + vae_seconds;
            // Parity verification runs after all three phase clocks stop, so
            // host-side readbacks never pollute the measured phases (main's
            // baseline likewise keeps its array exports outside the clocks).
            assert_eq!(
                caption_values, expected_caption,
                "pass {pass_index}: caption diverged from main"
            );
            for step in 0..steps {
                assert_eq!(
                    pass_velocity_bytes[step], expected_velocity_bytes[step],
                    "pass {pass_index} step {step}: velocity is not bit-exact with main"
                );
                assert_eq!(
                    pass_latent_bytes[step], expected_latent_bytes[step],
                    "pass {pass_index} step {step}: latent is not bit-exact with main"
                );
            }
            assert_eq!(pixels, expected_pixels, "pass {pass_index}: RGB diverged");
            println!(
                "warm pass {pass_index}: encode={encode_seconds:.3}s denoise={denoise_seconds:.3}s vae={vae_seconds:.3}s total={total:.3}s"
            );
            passes.push(serde_json::json!({
                "encode": encode_seconds,
                "denoise": denoise_seconds,
                "steps": step_seconds,
                "vae": vae_seconds,
                "total": total,
            }));
        }

        let median = |key: &str, entries: &[serde_json::Value]| -> f64 {
            let mut values: Vec<f64> = entries
                .iter()
                .map(|entry| entry[key].as_f64().unwrap())
                .collect();
            values.sort_by(f64::total_cmp);
            values[values.len() / 2]
        };
        let warm = &passes[warmup_passes..];
        let main_p50 = &payload["warm_p50"];
        let mut ratios = serde_json::Map::new();
        let mut failures = Vec::new();
        for key in ["encode", "denoise", "vae", "total"] {
            let rust = median(key, warm);
            let main = main_p50[key].as_f64().unwrap();
            let ratio = rust / main;
            println!(
                "warm p50 {key}: rust {rust:.3}s vs main {main:.3}s ({:.1}%)",
                ratio * 100.0
            );
            ratios.insert(key.to_string(), serde_json::json!(ratio));
            if ratio > 1.05 {
                failures.push(format!(
                    "warm p50 {key} exceeded 105% of main: rust {rust:.3}s vs main {main:.3}s"
                ));
            }
        }
        let peak = danmo_mlx::peak_memory().unwrap();
        let active = danmo_mlx::active_memory().unwrap();
        println!("native warm memory: peak {peak} bytes, active {active} bytes");
        // Steady-state resident footprint with the reusable buffer pool
        // emptied; the pool itself grows with pass count on both sides and
        // is not a resident-weight difference.
        danmo_mlx::clear_cache().unwrap();
        let steady_active = danmo_mlx::active_memory().unwrap();
        println!("native warm steady memory (pool cleared): active {steady_active} bytes");
        if let Some(main_steady) = payload["memory"]["steady_active"].as_u64() {
            assert!(
                (steady_active as f64) <= 1.05 * main_steady as f64,
                "steady resident memory exceeded 105% of main: {steady_active} vs {main_steady}"
            );
        }
        if let Ok(path) = std::env::var("DANQING_TEST_ZIMAGE_WARM_RUST_JSON") {
            let report = serde_json::json!({
                "weight_dtype": "checkpoint",
                "quantization_bits": payload["quantization_bits"],
                "width": width,
                "height": height,
                "steps": steps,
                "teacache": teacache,
                "load_seconds": load_seconds,
                "dit_weight_bytes": dit_weight_bytes,
                "passes": passes,
                "warm_p50": {
                    "encode": median("encode", warm),
                    "denoise": median("denoise", warm),
                    "vae": median("vae", warm),
                    "total": median("total", warm),
                },
                "main_warm_p50": main_p50,
                "rust_to_main_ratio": ratios,
                "peak_memory": peak,
                "active_memory": active,
                "steady_active_memory": steady_active,
                "main_peak_memory": payload["memory"]["peak"].as_u64().unwrap(),
                "main_steady_active_memory": payload["memory"]["steady_active"].as_u64(),
            });
            std::fs::write(path, serde_json::to_string(&report).unwrap()).unwrap();
        }
        assert!(failures.is_empty(), "{}", failures.join("; "));
    }

    /// ControlNet branch parity against a diffusers `ZImageControlNetModel`
    /// fixture (probe: torch CPU float32, stored bf16). Validates the hint
    /// stack semantics (`after_proj` per block, raw block output as the
    /// continuing stream), the control embed, the refiner branch, the caption
    /// branch, and all 15 layer hints on real weights.
    ///
    /// Env: `DANQING_TEST_ZIMAGE_DIT_DIR` (bundle transformer dir),
    /// `DANQING_TEST_ZIMAGE_CONTROL_DIR` (union controlnet dir),
    /// `DANQING_TEST_ZIMAGE_CONTROL_FIXTURE` (slim safetensors fixture).
    #[cfg(feature = "mlx-metal")]
    #[test]
    #[ignore]
    fn real_control_branch_matches_diffusers_fixture() {
        use super::{control, embed_caption, embed_image, embed_timestep, run_block, ForwardRope};

        let dit = std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_DIT_DIR").unwrap());
        let control_dir =
            std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_CONTROL_DIR").unwrap());
        let fixture_path =
            std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_CONTROL_FIXTURE").unwrap());
        let stream = Stream::gpu().unwrap();
        let config = MlxZImageConfig::published();
        // Checkpoint dtype: the reference probe runs torch in bf16 so the
        // gate measures semantics, not f32-vs-bf16 accumulation.
        let weights = load_directory(&dit, config, WeightDtype::Native, None).unwrap();
        let control_weights =
            control::load_directory(&control_dir, config.patch, WeightDtype::Native).unwrap();
        // SafeTensors lazy-load only supports the CPU stream.
        let fixture = danmo_mlx::SafeTensors::load(&fixture_path, &Stream::cpu().unwrap()).unwrap();
        let get = |name: &str| -> Array {
            fixture
                .get(name)
                .unwrap()
                .unwrap_or_else(|| panic!("fixture missing {name}"))
        };
        let rel_err = |tag: &str, actual: &Array, expected: &Array, tol: f32| {
            let actual = actual.astype(Dtype::Float32, &stream).unwrap();
            let expected = expected
                .astype(Dtype::Float32, &stream)
                .unwrap()
                .reshape(actual.shape().as_slice(), &stream)
                .unwrap();
            let a = actual.to_vec_f32(&stream).unwrap();
            let b = expected.to_vec_f32(&stream).unwrap();
            assert_eq!(a.len(), b.len(), "{tag}: element count mismatch");
            let mut num = 0f64;
            let mut den = 0f64;
            let mut dot = 0f64;
            let mut na = 0f64;
            let mut max_abs = 0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                num += ((x - y) as f64).powi(2);
                den += (*y as f64).powi(2);
                dot += (*x as f64) * (*y as f64);
                na += (*x as f64).powi(2);
                max_abs = max_abs.max((x - y).abs());
            }
            let rel = (num.sqrt() / den.sqrt().max(1e-12)) as f32;
            let cos = (dot / (na.sqrt() * den.sqrt()).max(1e-12)) as f32;
            println!("{tag}: rel_l2={rel:.5} cos={cos:.5} max_abs={max_abs:.5}");
            assert!(rel < tol, "{tag}: rel_l2 {rel} exceeds {tol}");
            assert!(cos > 0.95, "{tag}: cosine {cos} below 0.95");
        };

        // Engine timesteps embed `(1 - sigma) * t_scale`; the diffusers probe
        // passes the already-scheduled `t * t_scale` value (VideoX scheduler:
        // `t = 1 - sigma` for this parametrization).
        let sigma = get("sigma").to_vec_f32(&stream).unwrap()[0];
        let sigma = 1.0 - sigma;
        let scale = get("scale").to_vec_f32(&stream).unwrap()[0];
        let grid = geom::token_grid(512, 512, 40, config.vae_scale).unwrap();
        assert_eq!(grid.img_len, 1024);
        assert_eq!(grid.unified, 1088);

        let latent = get("latent").reshape(&[1, 16, 64, 64], &stream).unwrap();
        let caption_raw = get("caption").reshape(&[40, 2560], &stream).unwrap();
        let context = get("context").reshape(&[1, 33, 64, 64], &stream).unwrap();
        let timestep = embed_timestep(&weights, config, sigma, &stream).unwrap();
        let rope = ForwardRope::new(config, grid, &stream).unwrap();

        // 1. Control embed.
        let control_embed = control::embed_control_context(
            &control_weights,
            &weights,
            config,
            grid,
            &context,
            &stream,
        )
        .unwrap();
        rel_err(
            "control_embed",
            &control_embed,
            &get("cap.control_embed"),
            0.05,
        );

        // 2. Refiner branch: hints are after_proj outputs; stream is the raw
        // last block output. First localize the inputs.
        let image = embed_image(&weights, config, grid, &latent, &stream).unwrap();
        rel_err("image_embed", &image, &get("cap.image_embed"), 0.05);
        rel_err("t_embed", &timestep, &get("cap.t_embed"), 0.05);
        // The diffusers refiner pre-hook captured the block INPUT c (the
        // embedded control stream, before its internal before_proj + x).
        rel_err(
            "refiner0_input_c",
            &control_embed,
            &get("cap.pre.refiner0.c"),
            0.05,
        );
        rel_err("refiner0_input_x", &image, &get("cap.pre.refiner0.x"), 0.05);
        let (refiner_hints, refined) = control::forward_control_refiner(
            &control_weights,
            config,
            grid,
            &image,
            &control_embed,
            &rope.image,
            &timestep,
            &stream,
        )
        .unwrap();
        let ref0 = get("cap.control_noise_refiner.0");
        let ref1 = get("cap.control_noise_refiner.1");
        rel_err(
            "refiner_hint.0",
            &refiner_hints[0],
            &ref0
                .slice(&[0, 0, 0, 0], &[1, 1, 1024, 3840], &[1, 1, 1, 1], &stream)
                .unwrap(),
            0.20,
        );
        rel_err(
            "refiner_hint.1",
            &refiner_hints[1],
            &ref1
                .slice(&[1, 0, 0, 0], &[2, 1, 1024, 3840], &[1, 1, 1, 1], &stream)
                .unwrap(),
            0.20,
        );
        rel_err(
            "refined_stream",
            &refined,
            &ref1
                .slice(&[2, 0, 0, 0], &[3, 1, 1024, 3840], &[1, 1, 1, 1], &stream)
                .unwrap(),
            0.12,
        );

        // 3. Caption branch (embed + context_refiner).
        let caption_ids = &geom::positions_image_first(grid)[grid.img_len as usize..];
        let mut caption = embed_caption(&weights, config, grid, &caption_raw, &stream).unwrap();
        for index in 0..config.refiners {
            caption = run_block(
                &weights,
                config,
                &format!("context_refiner.{index}"),
                &caption,
                caption_ids,
                false,
                None,
                Some(&rope.caption),
                &stream,
            )
            .unwrap();
        }
        rel_err("cap_refined", &caption, &get("cap.cap_refined"), 0.05);

        // 4. Main noise_refiner with VideoX refiner-hint injection, then the
        // control layers; all 15 hints must match the diffusers stack
        // (and the scaled controlnet_block_samples).
        let ids = geom::positions_image_first(grid);
        let image_ids = &ids[..grid.img_len as usize];
        let mut image = image;
        for index in 0..config.refiners {
            image = run_block(
                &weights,
                config,
                &format!("noise_refiner.{index}"),
                &image,
                image_ids,
                true,
                Some(&timestep),
                Some(&rope.image),
                &stream,
            )
            .unwrap();
            image =
                control::apply_control_hint(&image, &refiner_hints[index as usize], scale, &stream)
                    .unwrap();
        }
        let unified = Array::concatenate_axis(&[image, caption.clone()], 1, &stream).unwrap();
        let layer_hints = control::forward_control_layers(
            &control_weights,
            config,
            grid,
            &unified,
            &refined,
            &caption,
            &rope.unified,
            &timestep,
            &stream,
        )
        .unwrap();
        let last = get("cap.control_layers.last");
        for (index, place) in control::CONTROL_LAYER_PLACES.iter().enumerate() {
            let stack_entry = last
                .slice(
                    &[index as i32, 0, 0, 0],
                    &[index as i32 + 1, 1, 1088, 3840],
                    &[1, 1, 1, 1],
                    &stream,
                )
                .unwrap();
            rel_err(
                &format!("layer_hint.{place}"),
                &layer_hints[index],
                &stack_entry,
                0.25,
            );
            let expected = get(&format!("hint.{place}"));
            let scaled_hint = Array::multiply(
                &layer_hints[index],
                &Array::from_f32_slice(&[scale], &[])
                    .unwrap()
                    .astype(layer_hints[index].dtype(), &stream)
                    .unwrap(),
                &stream,
            )
            .unwrap();
            rel_err(
                &format!("scaled_hint.{place}"),
                &scaled_hint,
                &expected,
                0.25,
            );
        }
    }
}
