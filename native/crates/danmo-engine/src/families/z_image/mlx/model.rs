use std::collections::BTreeMap;

use danmo_mlx::{Array, Error, Result, Stream};

use super::super::geom::{self, TokenGrid};
use super::attention::AttentionWeights;
use super::blocks::{self, BlockWeights};
use super::{modulation, timestep, MlxZImageConfig};

pub fn load_directory(
    directory: &std::path::Path,
    config: MlxZImageConfig,
    _stream: &Stream,
) -> Result<BTreeMap<String, Array>> {
    let io = Stream::cpu()?;
    let tensors = crate::weights::load_mlx_directory(directory)?
        .into_iter()
        .map(|(name, tensor)| Ok((name, tensor.astype(danmo_mlx::Dtype::Float32, &io)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let names = super::weights::remap_weight_names(
        tensors.keys().cloned().collect::<Vec<_>>(),
        config.patch,
    )
    .map_err(|error| Error::new("Z-Image remap", error.to_string()))?;
    let mut output = BTreeMap::new();
    for (source, target) in names {
        let tensor = tensors
            .get(&source)
            .cloned()
            .ok_or_else(|| Error::new("Z-Image safetensors", format!("missing {source}")))?;
        if output.insert(target.clone(), tensor).is_some() {
            return Err(Error::new(
                "Z-Image safetensors",
                format!("duplicate remapped tensor {target}"),
            ));
        }
    }
    Ok(output)
}

fn embed_image(
    weights: &BTreeMap<String, Array>,
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
    let patches = gathered.reshape(&[1, grid.img_real as i32, token_dim as i32], stream)?;
    let embedding_weight = weight(weights, "x_embedder.weight")?;
    let patches = if patches.dtype() == embedding_weight.dtype() {
        patches
    } else {
        patches.astype(embedding_weight.dtype(), stream)?
    };
    let mut embedded = Array::linear(
        &patches,
        embedding_weight,
        Some(weight(weights, "x_embedder.bias")?),
        stream,
    )?;
    if grid.img_pad > 0 {
        let pad = weight(weights, "x_pad_token")?
            .broadcast_to(&[1, grid.img_pad as i32, config.dim as i32], stream)?;
        let pad = cast_to(&pad, embedded.dtype(), stream)?;
        embedded = Array::concatenate_axis(&[embedded, pad], 1, stream)?;
    }
    Ok(embedded)
}

fn embed_caption(
    weights: &BTreeMap<String, Array>,
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
    let normalized = caption.rms_norm(Some(cap_norm), config.norm_eps, stream)?;
    let mut embedded = Array::linear(
        &normalized,
        weight(weights, "cap_embedder.weight")?,
        Some(weight(weights, "cap_embedder.bias")?),
        stream,
    )?
    .reshape(&[1, grid.cap_real as i32, config.dim as i32], stream)?;
    if grid.cap_pad > 0 {
        let pad = weight(weights, "cap_pad_token")?
            .broadcast_to(&[1, grid.cap_pad as i32, config.dim as i32], stream)?;
        let pad = cast_to(&pad, embedded.dtype(), stream)?;
        embedded = Array::concatenate_axis(&[embedded, pad], 1, stream)?;
    }
    Ok(embedded)
}

fn run_block(
    weights: &BTreeMap<String, Array>,
    config: MlxZImageConfig,
    prefix: &str,
    input: &Array,
    ids: &[[i32; 3]],
    modulated: bool,
    timestep: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    blocks::forward(
        input,
        BlockWeights {
            attention: AttentionWeights {
                to_q: weight(weights, &format!("{prefix}.attention.to_q.weight"))?,
                to_k: weight(weights, &format!("{prefix}.attention.to_k.weight"))?,
                to_v: weight(weights, &format!("{prefix}.attention.to_v.weight"))?,
                to_out: weight(weights, &format!("{prefix}.attention.to_out.weight"))?,
                norm_q: weight(weights, &format!("{prefix}.attention.norm_q.weight"))?,
                norm_k: weight(weights, &format!("{prefix}.attention.norm_k.weight"))?,
            },
            attention_norm1: weight(weights, &format!("{prefix}.attn_norm1.weight"))?,
            attention_norm2: weight(weights, &format!("{prefix}.attn_norm2.weight"))?,
            ffn_norm1: weight(weights, &format!("{prefix}.ffn_norm1.weight"))?,
            ffn_norm2: weight(weights, &format!("{prefix}.ffn_norm2.weight"))?,
            w1: weight(weights, &format!("{prefix}.feed_forward.w1.weight"))?,
            w2: weight(weights, &format!("{prefix}.feed_forward.w2.weight"))?,
            w3: weight(weights, &format!("{prefix}.feed_forward.w3.weight"))?,
            adaln: if modulated {
                Some((
                    weight(weights, &format!("{prefix}.adaLN_modulation.0.weight"))?,
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
        stream,
    )
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
        weights: &BTreeMap<String, Array>,
        config: MlxZImageConfig,
        grid: TokenGrid,
        caption: &Array,
        shapeless: bool,
        stream: &Stream,
    ) -> Result<Self> {
        let weights = weights.clone();
        let caption = caption.clone();
        let stream = stream.clone();
        let closure = danmo_mlx::Closure::new(stream, move |inputs, stream| {
            if inputs.len() != 2 {
                return Err(Error::new(
                    "Z-Image compiled forward",
                    format!("expected [latent, timestep], got {} inputs", inputs.len()),
                ));
            }
            forward_with_timestep(
                &weights, config, grid, &inputs[0], &caption, &inputs[1], stream,
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

/// Embed the scalar flow-match timestep once per denoise step; the compiled
/// forward and the eager forward share this exact embedding.
pub fn embed_timestep(
    weights: &BTreeMap<String, Array>,
    config: MlxZImageConfig,
    sigma: f32,
    stream: &Stream,
) -> Result<Array> {
    let t = (1.0 - sigma) * config.t_scale;
    timestep::embed(
        t,
        config.frequency_dim as usize,
        weight(weights, "t_embedder.linear1.weight")?,
        weight(weights, "t_embedder.linear1.bias")?,
        weight(weights, "t_embedder.linear2.weight")?,
        weight(weights, "t_embedder.linear2.bias")?,
        stream,
    )
}

pub fn forward(
    weights: &BTreeMap<String, Array>,
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

fn forward_with_timestep(
    weights: &BTreeMap<String, Array>,
    config: MlxZImageConfig,
    grid: TokenGrid,
    latent: &Array,
    caption: &Array,
    timestep: &Array,
    stream: &Stream,
) -> Result<Array> {
    let ids = geom::positions_image_first(grid);
    let (image_ids, caption_ids) = ids.split_at(grid.img_len as usize);
    let mut image = embed_image(weights, config, grid, latent, stream)?;
    let mut caption = embed_caption(weights, config, grid, caption, stream)?;
    for index in 0..config.refiners {
        image = run_block(
            weights,
            config,
            &format!("noise_refiner.{index}"),
            &image,
            image_ids,
            true,
            Some(timestep),
            stream,
        )?;
        caption = run_block(
            weights,
            config,
            &format!("context_refiner.{index}"),
            &caption,
            caption_ids,
            false,
            None,
            stream,
        )?;
    }
    let mut unified = Array::concatenate_axis(&[image, caption], 1, stream)?;
    for index in 0..config.layers {
        unified = run_block(
            weights,
            config,
            &format!("layers.{index}"),
            &unified,
            &ids,
            true,
            Some(timestep),
            stream,
        )?;
    }
    let activated = timestep.silu(stream)?;
    let scale = Array::linear(
        &activated,
        weight(weights, "final_layer.adaLN_modulation.0.weight")?,
        Some(weight(weights, "final_layer.adaLN_modulation.0.bias")?),
        stream,
    )?;
    let one = cast_to(&Array::from_f32_slice(&[1.0], &[])?, scale.dtype(), stream)?;
    let scale = Array::add(&scale, &one, stream)?;
    let normalized = unified.layer_norm(1e-6, stream)?;
    let normalized = modulation::modulate(&normalized, &scale, stream)?;
    let tokens = Array::linear(
        &normalized,
        weight(weights, "final_layer.linear.weight")?,
        Some(weight(weights, "final_layer.linear.bias")?),
        stream,
    )?;
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
    output.negative(stream)?.reshape(
        &[
            config.in_channels as i32,
            grid.latent_h as i32,
            grid.latent_w as i32,
        ],
        stream,
    )
}

fn cast_to(input: &Array, dtype: danmo_mlx::Dtype, stream: &Stream) -> Result<Array> {
    if input.dtype() == dtype {
        Ok(input.clone())
    } else {
        input.astype(dtype, stream)
    }
}

fn weight<'a>(weights: &'a BTreeMap<String, Array>, name: &str) -> Result<&'a Array> {
    weights
        .get(name)
        .ok_or_else(|| Error::new("Z-Image weights", format!("missing tensor {name}")))
}

#[cfg(test)]
mod tests {
    use super::MlxZImageConfig;
    use super::{forward, load_directory};
    use crate::families::z_image::geom;
    use crate::families::z_image::weights::ZImageConfig;
    use danmo_mlx::{Array, DeviceType, Result, Stream};
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
        add_block(&mut tensors, "noise_refiner.0", &config, true);
        add_block(&mut tensors, "context_refiner.0", &config, false);
        add_block(&mut tensors, "layers.0", &config, true);
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
        let latent = (0..8).map(|index| index as f32 * 0.1 - 0.2).collect();
        let caption = (0..8).map(|index| index as f32 * 0.07 - 0.15).collect();
        (config, tensors, latent, caption)
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
        let caption = Array::from_f32_slice(&caption_values, &[2, 4])?;
        let compiled =
            super::CompiledForward::new(&weights, config, grid, &caption, false, &stream)?;
        for sigma in [0.9f32, 0.5, 0.25] {
            let latent = Array::from_f32_slice(&latent_values, &[2, 2, 2])?;
            let eager = forward(&weights, config, grid, &latent, &caption, sigma, &stream)?
                .to_vec_f32(&stream)?;
            let timestep = super::embed_timestep(&weights, config, sigma, &stream)?;
            let actual = compiled
                .apply(&latent, &timestep)?
                .to_vec_f32(&stream)?;
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
        let weights = load_directory(&root, config, &stream).unwrap();
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
        let weights = load_directory(&root, config, &stream).unwrap();
        println!("loaded {} remapped tensors", weights.len());
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
        let started = std::time::Instant::now();
        let ids: Vec<i32> = payload["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect();
        let text_weights =
            crate::common::qwen3::mlx::load_directory(&bundle.join("text_encoder"), &stream)
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
        let weights = load_directory(&bundle.join("transformer"), config, &stream).unwrap();
        let grid = geom::token_grid(width, height, ids.len() as i64, config.vae_scale).unwrap();
        let latent = if let Some(seed) = payload["seed"].as_u64() {
            let noise = danmo_mlx::RandomKey::new(seed)
                .unwrap()
                .normal(&[1, 16, latent_h, latent_w], danmo_mlx::Dtype::Float32, &stream)
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
        let mut step = 0usize;
        let denoised = crate::families::z_image::mlx::denoise::euler(
            &latent,
            &sigmas,
            &stream,
            |current, sigma| {
                let velocity = forward(&weights, config, grid, current, &caption, sigma, &stream)?;
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
        drop(weights);

        let vae_started = std::time::Instant::now();
        let vae_stream = Stream::gpu().unwrap();
        let vae_weights = crate::common::vae::mlx::load_dense(&bundle.join("vae")).unwrap();
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
        assert!(output_error <= 0.004, "decoded error {output_error}");
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
        let weights = load_directory(&bundle.join("transformer"), config, &stream).unwrap();
        let ids: Vec<i64> = payload["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap())
            .collect();
        let grid = geom::token_grid(16, 16, ids.len() as i64, config.vae_scale).unwrap();
        let caption = Array::from_f32_slice(
            &values("caption"),
            &[
                payload["caption_shape"][0].as_i64().unwrap() as i32,
                payload["caption_shape"][1].as_i64().unwrap() as i32,
            ],
        )
        .unwrap();
        let sigmas = values("sigmas");
        let compiled =
            super::CompiledForward::new(&weights, config, grid, &caption, false, &stream).unwrap();

        let run = |compiled_mode: bool| -> Result<(Vec<Vec<f32>>, Vec<f64>)> {
            let mut latent = Array::from_f32_slice(&values("latent"), &[1, 16, 2, 2])
                .unwrap()
                .squeeze_axis(0, &stream)
                .unwrap()
                .astype(danmo_mlx::Dtype::BFloat16, &stream)
                .unwrap();
            let mut velocities = Vec::new();
            let mut timings = Vec::new();
            for step in 0..sigmas.len() - 1 {
                let started = std::time::Instant::now();
                let velocity = if compiled_mode {
                    let timestep = super::embed_timestep(&weights, config, sigmas[step], &stream)?;
                    compiled.apply(&latent, &timestep)?
                } else {
                    forward(&weights, config, grid, &latent, &caption, sigmas[step], &stream)?
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
                main_error =
                    main_error.max((actual - expected.as_f64().unwrap() as f32).abs());
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
            "warm 8-step totals: eager {eager_total:.3}s, compiled {compiled_total:.3}s ({:.1}%)",
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
}
