use danmo_mlx::{Array, Closure, Dtype, Error, Result, Stream};

use super::config::ErnieConfig;
use crate::weights::MlxWeights;

#[derive(Clone)]
pub struct Prepared {
    cos: Array,
    sin: Array,
    mask: Array,
    image_tokens: i32,
}

impl Prepared {
    pub fn new(
        config: ErnieConfig,
        batch: i32,
        grid_h: i32,
        grid_w: i32,
        text_lens: &[i32],
        stream: &Stream,
    ) -> Result<Self> {
        if batch < 1 || grid_h < 1 || grid_w < 1 || text_lens.len() != batch as usize {
            return Err(Error::new(
                "ERNIE DiT",
                "invalid batch, grid, or text lengths",
            ));
        }
        let image_tokens = grid_h * grid_w;
        let text_tokens = text_lens.iter().copied().max().unwrap_or(0);
        let sequence = image_tokens + text_tokens;
        let mut positions = vec![0.0f32; (batch * sequence * 3) as usize];
        let mut mask = vec![0.0f32; (batch * sequence) as usize];
        for batch_index in 0..batch {
            let text_len = text_lens[batch_index as usize];
            if text_len < 0 || text_len > text_tokens {
                return Err(Error::new(
                    "ERNIE DiT",
                    "text length is outside the padded sequence",
                ));
            }
            for token in 0..sequence {
                let index = (batch_index * sequence + token) as usize;
                if token < image_tokens {
                    positions[index * 3] = text_len as f32;
                    positions[index * 3 + 1] = (token / grid_w) as f32;
                    positions[index * 3 + 2] = (token % grid_w) as f32;
                } else {
                    positions[index * 3] = (token - image_tokens) as f32;
                    if token - image_tokens >= text_len {
                        mask[index] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        let mut axes_cos = Vec::new();
        let mut axes_sin = Vec::new();
        for axis in 0..3 {
            let values: Vec<f32> = positions.chunks_exact(3).map(|ids| ids[axis]).collect();
            let (cos, sin) = axis_rope(&values, config.rope_axes[axis], config.rope_theta, stream)?;
            axes_cos.push(cos.reshape(&[batch, sequence, config.rope_axes[axis]], stream)?);
            axes_sin.push(sin.reshape(&[batch, sequence, config.rope_axes[axis]], stream)?);
        }
        let cos = Array::concatenate_axis(&axes_cos, -1, stream)?.expand_dims(2, stream)?;
        let sin = Array::concatenate_axis(&axes_sin, -1, stream)?.expand_dims(2, stream)?;
        let mask = Array::from_f32_slice(&mask, &[batch, 1, 1, sequence])?;
        Ok(Self {
            cos,
            sin,
            mask,
            image_tokens,
        })
    }
}

pub fn forward(
    weights: &MlxWeights,
    config: ErnieConfig,
    latent: &Array,
    timestep: &Array,
    text: &Array,
    prepared: &Prepared,
    stream: &Stream,
) -> Result<Array> {
    let shape = latent.shape();
    if shape.len() != 4 || shape[1] != config.in_channels || timestep.shape() != vec![shape[0]] {
        return Err(Error::new(
            "ERNIE DiT",
            format!(
                "latent/timestep shapes {shape:?} / {:?} are invalid",
                timestep.shape()
            ),
        ));
    }
    let batch = shape[0];
    let grid_h = shape[2] / config.patch;
    let grid_w = shape[3] / config.patch;
    if prepared.image_tokens != grid_h * grid_w {
        return Err(Error::new(
            "ERNIE DiT",
            "latent grid does not match prepared RoPE",
        ));
    }
    let text_shape = text.shape();
    if text_shape.len() != 3 || text_shape[0] != batch || text_shape[2] != config.text_dim {
        return Err(Error::new(
            "ERNIE DiT",
            format!("text shape {text_shape:?} is invalid"),
        ));
    }
    let kernel = weights
        .tensor("x_embedder.proj.weight")?
        .transpose_axes(&[0, 2, 3, 1], stream)?;
    let image = latent.transpose_axes(&[0, 2, 3, 1], stream)?;
    let image = Array::conv2d(
        &image,
        &kernel,
        [config.patch, config.patch],
        [0, 0],
        [1, 1],
        1,
        stream,
    )?;
    let image = Array::add(&image, weights.tensor("x_embedder.proj.bias")?, stream)?
        .reshape(&[batch, prepared.image_tokens, config.hidden], stream)?;
    let text = if config.text_dim == config.hidden {
        text.clone()
    } else {
        weights
            .linear("text_proj.weight")?
            .apply(text, None, stream)?
    };
    let mut hidden = Array::concatenate_axis(&[image, text], 1, stream)?;
    let features =
        timestep_features(timestep, config.hidden, stream)?.astype(latent.dtype(), stream)?;
    let conditioning =
        linear(weights, "time_embedding.linear_1", &features, stream)?.silu(stream)?;
    let conditioning = linear(weights, "time_embedding.linear_2", &conditioning, stream)?;
    let modulation = linear(
        weights,
        "adaLN_modulation.1",
        &conditioning.silu(stream)?,
        stream,
    )?;
    let mods = modulation_chunks(&modulation, config.hidden, stream)?;
    for layer in 0..config.layers {
        hidden = block(
            weights,
            config,
            layer,
            &hidden,
            &conditioning,
            &mods,
            prepared,
            stream,
        )?;
    }
    let normed = hidden.layer_norm(config.eps, stream)?;
    let output_mod = linear(weights, "final_norm.linear", &conditioning, stream)?;
    let scale = channel(&output_mod, 0, config.hidden, stream)?.expand_dims(1, stream)?;
    let shift =
        channel(&output_mod, config.hidden, config.hidden, stream)?.expand_dims(1, stream)?;
    let one = Array::from_f32_slice(&[1.0], &[])?.astype(normed.dtype(), stream)?;
    let normed = Array::add(
        &Array::multiply(&normed, &Array::add(&one, &scale, stream)?, stream)?,
        &shift,
        stream,
    )?;
    let output = linear(weights, "final_linear", &normed, stream)?;
    let output = output.slice(
        &[0, 0, 0],
        &[
            batch,
            prepared.image_tokens,
            config.patch * config.patch * config.out_channels,
        ],
        &[1, 1, 1],
        stream,
    )?;
    output
        .reshape(&[batch, grid_h, grid_w, config.out_channels], stream)?
        .transpose_axes(&[0, 3, 1, 2], stream)
}

fn block(
    weights: &MlxWeights,
    config: ErnieConfig,
    layer: i32,
    input: &Array,
    _conditioning: &Array,
    mods: &[Array; 6],
    prepared: &Prepared,
    stream: &Stream,
) -> Result<Array> {
    let prefix = format!("layers.{layer}");
    let normed = rms_norm(
        input,
        weights.tensor(&format!("{prefix}.adaLN_sa_ln.weight"))?,
        config.eps,
        stream,
    )?;
    let normed = modulate_f32(&normed, &mods[0], &mods[1], stream)?;
    let shape = normed.shape();
    let (batch, sequence) = (shape[0], shape[1]);
    let project = |name: &str| {
        weights
            .linear(&format!("{prefix}.self_attention.{name}.weight"))?
            .apply(&normed, None, stream)
    };
    let q = project("to_q")?.reshape(&[batch, sequence, config.heads, config.head_dim], stream)?;
    let k = project("to_k")?.reshape(&[batch, sequence, config.heads, config.head_dim], stream)?;
    let v = project("to_v")?.reshape(&[batch, sequence, config.heads, config.head_dim], stream)?;
    let q = rms_norm(
        &q,
        weights.tensor(&format!("{prefix}.self_attention.norm_q.weight"))?,
        config.eps,
        stream,
    )?;
    let k = rms_norm(
        &k,
        weights.tensor(&format!("{prefix}.self_attention.norm_k.weight"))?,
        config.eps,
        stream,
    )?;
    let q = apply_rope(&q, &prepared.cos, &prepared.sin, stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let k = apply_rope(&k, &prepared.cos, &prepared.sin, stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let v = v.transpose_axes(&[0, 2, 1, 3], stream)?;
    let mask = prepared.mask.astype(q.dtype(), stream)?;
    let attended = Array::scaled_dot_product_attention(
        &q,
        &k,
        &v,
        1.0 / (config.head_dim as f32).sqrt(),
        "",
        Some(&mask),
        stream,
    )?
    .transpose_axes(&[0, 2, 1, 3], stream)?
    .reshape(&[batch, sequence, config.hidden], stream)?;
    let attended = weights
        .linear(&format!("{prefix}.self_attention.to_out.0.weight"))?
        .apply(&attended, None, stream)?;
    let hidden = gated_residual(input, &attended, &mods[2], stream)?;
    let normed = rms_norm(
        &hidden,
        weights.tensor(&format!("{prefix}.adaLN_mlp_ln.weight"))?,
        config.eps,
        stream,
    )?;
    let normed = modulate_f32(&normed, &mods[3], &mods[4], stream)?;
    let gate = weights
        .linear(&format!("{prefix}.mlp.gate_proj.weight"))?
        .apply(&normed, None, stream)?;
    let up = weights
        .linear(&format!("{prefix}.mlp.up_proj.weight"))?
        .apply(&normed, None, stream)?;
    let mid = Array::multiply(&gelu(&gate, stream)?, &up, stream)?;
    let mlp = weights
        .linear(&format!("{prefix}.mlp.linear_fc2.weight"))?
        .apply(&mid, None, stream)?;
    gated_residual(&hidden, &mlp, &mods[5], stream)
}

pub struct CompiledForward {
    closure: Closure,
}

impl CompiledForward {
    pub fn new(
        weights: MlxWeights,
        config: ErnieConfig,
        text: Array,
        prepared: Prepared,
        stream: &Stream,
    ) -> Result<Self> {
        let closure = Closure::new(stream.clone(), move |inputs, stream| {
            if inputs.len() != 2 {
                return Err(Error::new(
                    "ERNIE compiled DiT",
                    "expected latent and timestep inputs",
                ));
            }
            forward(
                &weights, config, &inputs[0], &inputs[1], &text, &prepared, stream,
            )
        })?
        .compile(false)?;
        Ok(Self { closure })
    }

    pub fn apply(&self, latent: &Array, timestep: &Array) -> Result<Array> {
        self.closure.apply(&[latent.clone(), timestep.clone()])
    }
}

fn timestep_features(timestep: &Array, dim: i32, stream: &Stream) -> Result<Array> {
    let half = dim / 2;
    let indices = Array::arange(0.0, half as f64, 1.0, Dtype::Float32, stream)?;
    let exponent = Array::multiply(
        &indices,
        &Array::from_f32_slice(&[-10000f32.ln()], &[])?,
        stream,
    )?;
    let exponent = Array::divide(
        &exponent,
        &Array::from_f32_slice(&[half as f32], &[])?,
        stream,
    )?;
    let frequencies = exponent.exp(stream)?.expand_dims(0, stream)?;
    let timestep = timestep
        .astype(Dtype::Float32, stream)?
        .expand_dims(1, stream)?;
    let angles = Array::multiply(&timestep, &frequencies, stream)?;
    Array::concatenate_axis(&[angles.sin(stream)?, angles.cos(stream)?], -1, stream)
}

fn modulation_chunks(input: &Array, dim: i32, stream: &Stream) -> Result<[Array; 6]> {
    Ok([
        channel(input, 0, dim, stream)?.expand_dims(1, stream)?,
        channel(input, dim, dim, stream)?.expand_dims(1, stream)?,
        channel(input, dim * 2, dim, stream)?.expand_dims(1, stream)?,
        channel(input, dim * 3, dim, stream)?.expand_dims(1, stream)?,
        channel(input, dim * 4, dim, stream)?.expand_dims(1, stream)?,
        channel(input, dim * 5, dim, stream)?.expand_dims(1, stream)?,
    ])
}

fn modulate_f32(input: &Array, shift: &Array, scale: &Array, stream: &Stream) -> Result<Array> {
    let dtype = input.dtype();
    let input = input.astype(Dtype::Float32, stream)?;
    let shift = shift.astype(Dtype::Float32, stream)?;
    let scale = scale.astype(Dtype::Float32, stream)?;
    let one = Array::from_f32_slice(&[1.0], &[])?;
    Array::add(
        &Array::multiply(&input, &Array::add(&one, &scale, stream)?, stream)?,
        &shift,
        stream,
    )?
    .astype(dtype, stream)
}

fn gated_residual(input: &Array, delta: &Array, gate: &Array, stream: &Stream) -> Result<Array> {
    let dtype = input.dtype();
    let delta = delta.astype(Dtype::Float32, stream)?;
    let gate = gate.astype(Dtype::Float32, stream)?;
    let delta = Array::multiply(&delta, &gate, stream)?.astype(dtype, stream)?;
    Array::add(input, &delta, stream)
}

fn rms_norm(input: &Array, weight: &Array, eps: f32, stream: &Stream) -> Result<Array> {
    let dtype = input.dtype();
    let input = input.astype(Dtype::Float32, stream)?;
    let weight = weight.astype(Dtype::Float32, stream)?;
    input
        .rms_norm(Some(&weight), eps, stream)?
        .astype(dtype, stream)
}

fn apply_rope(input: &Array, cos: &Array, sin: &Array, stream: &Stream) -> Result<Array> {
    let dim = input.shape()[3];
    let parts = input.split_sections(&[dim / 2], -1, stream)?;
    let rotated =
        Array::concatenate_axis(&[parts[1].negative(stream)?, parts[0].clone()], -1, stream)?;
    Array::add(
        &Array::multiply(input, cos, stream)?,
        &Array::multiply(&rotated, sin, stream)?,
        stream,
    )
}

fn axis_rope(positions: &[f32], dim: i32, theta: f32, stream: &Stream) -> Result<(Array, Array)> {
    let pos = Array::from_f32_slice(positions, &[positions.len() as i32, 1])?;
    let arange = Array::arange(0.0, dim as f64, 2.0, Dtype::Float32, stream)?;
    let exponent = Array::divide(&arange, &Array::from_f32_slice(&[dim as f32], &[])?, stream)?;
    let frequency = Array::divide(
        &Array::from_f32_slice(&[1.0], &[])?,
        &Array::power(&Array::from_f32_slice(&[theta], &[])?, &exponent, stream)?,
        stream,
    )?;
    let angles = Array::multiply(&pos, &frequency.expand_dims(0, stream)?, stream)?
        .repeat_axis(2, 1, stream)?;
    Ok((angles.cos(stream)?, angles.sin(stream)?))
}

fn gelu(input: &Array, stream: &Stream) -> Result<Array> {
    let half = Array::from_f32_slice(&[0.5], &[])?.astype(input.dtype(), stream)?;
    let one = Array::from_f32_slice(&[1.0], &[])?.astype(input.dtype(), stream)?;
    let scale = Array::from_f32_slice(&[std::f32::consts::FRAC_1_SQRT_2], &[])?
        .astype(input.dtype(), stream)?;
    let erf = Array::multiply(input, &scale, stream)?.erf(stream)?;
    Array::multiply(
        &Array::multiply(input, &half, stream)?,
        &Array::add(&one, &erf, stream)?,
        stream,
    )
}

fn linear(weights: &MlxWeights, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
    weights.linear(&format!("{prefix}.weight"))?.apply(
        input,
        Some(weights.tensor(&format!("{prefix}.bias"))?),
        stream,
    )
}

fn channel(input: &Array, start: i32, length: i32, stream: &Stream) -> Result<Array> {
    let shape = input.shape();
    let axis = shape.len() - 1;
    let mut begin = vec![0; shape.len()];
    let mut end = shape;
    begin[axis] = start;
    end[axis] = start + length;
    input.slice(&begin, &end, &vec![1; begin.len()], stream)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use danmo_mlx::{Array, Stream};

    use super::{forward, Prepared};
    use crate::families::ernie_image::mlx::config::ErnieConfig;
    use crate::weights::MlxWeights;

    fn matrix(tensors: &mut BTreeMap<String, Array>, name: String, rows: i32, cols: i32) {
        let values: Vec<f32> = (0..rows * cols)
            .map(|index| ((index % 13) as f32 - 6.0) * 0.005)
            .collect();
        tensors.insert(name, Array::from_f32_slice(&values, &[rows, cols]).unwrap());
    }

    fn vector(tensors: &mut BTreeMap<String, Array>, name: String, len: i32, value: f32) {
        tensors.insert(
            name,
            Array::from_f32_slice(&vec![value; len as usize], &[len]).unwrap(),
        );
    }

    fn weights(config: ErnieConfig) -> MlxWeights {
        let mut tensors = BTreeMap::new();
        let conv: Vec<f32> = (0..config.hidden * config.in_channels)
            .map(|index| ((index % 7) as f32 - 3.0) * 0.01)
            .collect();
        tensors.insert(
            "x_embedder.proj.weight".into(),
            Array::from_f32_slice(&conv, &[config.hidden, config.in_channels, 1, 1]).unwrap(),
        );
        vector(
            &mut tensors,
            "x_embedder.proj.bias".into(),
            config.hidden,
            0.0,
        );
        matrix(
            &mut tensors,
            "text_proj.weight".into(),
            config.hidden,
            config.text_dim,
        );
        for name in ["time_embedding.linear_1", "time_embedding.linear_2"] {
            matrix(
                &mut tensors,
                format!("{name}.weight"),
                config.hidden,
                config.hidden,
            );
            vector(&mut tensors, format!("{name}.bias"), config.hidden, 0.0);
        }
        matrix(
            &mut tensors,
            "adaLN_modulation.1.weight".into(),
            config.hidden * 6,
            config.hidden,
        );
        vector(
            &mut tensors,
            "adaLN_modulation.1.bias".into(),
            config.hidden * 6,
            0.0,
        );
        for layer in 0..config.layers {
            let prefix = format!("layers.{layer}");
            for norm in ["adaLN_sa_ln.weight", "adaLN_mlp_ln.weight"] {
                vector(&mut tensors, format!("{prefix}.{norm}"), config.hidden, 1.0);
            }
            for name in ["to_q", "to_k", "to_v"] {
                matrix(
                    &mut tensors,
                    format!("{prefix}.self_attention.{name}.weight"),
                    config.hidden,
                    config.hidden,
                );
            }
            vector(
                &mut tensors,
                format!("{prefix}.self_attention.norm_q.weight"),
                config.head_dim,
                1.0,
            );
            vector(
                &mut tensors,
                format!("{prefix}.self_attention.norm_k.weight"),
                config.head_dim,
                1.0,
            );
            matrix(
                &mut tensors,
                format!("{prefix}.self_attention.to_out.0.weight"),
                config.hidden,
                config.hidden,
            );
            matrix(
                &mut tensors,
                format!("{prefix}.mlp.gate_proj.weight"),
                config.intermediate,
                config.hidden,
            );
            matrix(
                &mut tensors,
                format!("{prefix}.mlp.up_proj.weight"),
                config.intermediate,
                config.hidden,
            );
            matrix(
                &mut tensors,
                format!("{prefix}.mlp.linear_fc2.weight"),
                config.hidden,
                config.intermediate,
            );
        }
        matrix(
            &mut tensors,
            "final_norm.linear.weight".into(),
            config.hidden * 2,
            config.hidden,
        );
        vector(
            &mut tensors,
            "final_norm.linear.bias".into(),
            config.hidden * 2,
            0.0,
        );
        matrix(
            &mut tensors,
            "final_linear.weight".into(),
            config.out_channels,
            config.hidden,
        );
        vector(
            &mut tensors,
            "final_linear.bias".into(),
            config.out_channels,
            0.0,
        );
        MlxWeights::from_dense(tensors)
    }

    #[test]
    fn joint_forward_has_the_packed_nchw_output_shape() {
        let config = ErnieConfig::smoke();
        let weights = weights(config);
        let stream = Stream::cpu().unwrap();
        let latent = Array::from_f32_slice(
            &(0..16).map(|value| value as f32 * 0.02).collect::<Vec<_>>(),
            &[1, 4, 2, 2],
        )
        .unwrap();
        let text = Array::from_f32_slice(&vec![0.1; 18], &[1, 3, 6]).unwrap();
        let timestep = Array::from_f32_slice(&[500.0], &[1]).unwrap();
        let prepared = Prepared::new(config, 1, 2, 2, &[3], &stream).unwrap();
        let output = forward(
            &weights, config, &latent, &timestep, &text, &prepared, &stream,
        )
        .unwrap();
        assert_eq!(output.shape(), vec![1, 4, 2, 2]);
        assert!(output
            .to_vec_f32(&stream)
            .unwrap()
            .into_iter()
            .all(|value| value.is_finite()));
    }
}

#[cfg(test)]
mod parity_tests {
    use std::collections::BTreeMap;

    use danmo_mlx::{Array, SafeTensors, Stream};

    use super::{forward, CompiledForward, Prepared};
    use crate::families::ernie_image::mlx::config::ErnieConfig;
    use crate::weights::MlxWeights;

    #[test]
    #[ignore = "requires the official Diffusers miniature fixture"]
    fn official_diffusers_dit_fixture_matches() {
        let root =
            std::path::PathBuf::from(std::env::var("DANQING_TEST_ERNIE_DIT_FIXTURE").unwrap());
        let expected: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("expected.json")).unwrap())
                .unwrap();
        let stream = Stream::cpu().unwrap();
        let tensors = SafeTensors::load(&root.join("fixture.safetensors"), &stream).unwrap();
        let mut dense = BTreeMap::new();
        for name in tensors.names().unwrap() {
            dense.insert(name.clone(), tensors.get(&name).unwrap().unwrap());
        }
        let weights = MlxWeights::from_dense(dense);
        let values = |key: &str| {
            expected[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let latent = Array::from_f32_slice(&values("latent"), &[1, 4, 2, 2]).unwrap();
        let text = Array::from_f32_slice(&values("text"), &[1, 3, 6]).unwrap();
        let timestep = Array::from_f32_slice(&values("timestep"), &[1]).unwrap();
        let config = ErnieConfig::smoke();
        let prepared = Prepared::new(config, 1, 2, 2, &[3], &stream).unwrap();
        let actual = forward(
            &weights, config, &latent, &timestep, &text, &prepared, &stream,
        )
        .unwrap()
        .to_vec_f32(&stream)
        .unwrap();
        let expected = values("output");
        assert_eq!(actual.len(), expected.len());
        let mut max_abs = 0.0f32;
        let mut error = 0.0f32;
        let mut norm = 0.0f32;
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            max_abs = max_abs.max((actual - expected).abs());
            error += (actual - expected).powi(2);
            norm += expected.powi(2);
        }
        let relative = (error / norm.max(1e-20)).sqrt();
        println!("ERNIE DiT parity max_abs={max_abs} relative_l2={relative}");
        assert!(relative < 2e-4, "relative L2 {relative}, max abs {max_abs}");
        let compiled = CompiledForward::new(weights, config, text, prepared, &stream).unwrap();
        let compiled = compiled
            .apply(&latent, &timestep)
            .unwrap()
            .to_vec_f32(&stream)
            .unwrap();
        for (compiled, expected) in compiled.into_iter().zip(expected) {
            assert!((compiled - expected).abs() < 2e-4);
        }
    }
}
