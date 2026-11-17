use std::collections::BTreeMap;

use danmo_mlx::{Array, Error, Result, Stream};

use super::VaeSpec;

pub fn load_dense(
    directory: &std::path::Path,
    dtype: crate::weights::WeightDtype,
) -> Result<BTreeMap<String, Array>> {
    crate::weights::load_mlx_directory(directory, dtype, None)?.into_dense()
}

pub fn decode(
    weights: &BTreeMap<String, Array>,
    spec: &VaeSpec,
    latents: &Array,
    stream: &Stream,
) -> Result<Array> {
    decode_impl(weights, spec, latents, stream, None)
}

fn decode_impl(
    weights: &BTreeMap<String, Array>,
    spec: &VaeSpec,
    latents: &Array,
    stream: &Stream,
    mut trace: Option<&mut BTreeMap<String, Array>>,
) -> Result<Array> {
    if spec.latent_channels < 1 || spec.groups < 1 || spec.resnets < 1 {
        return Err(Error::new("VAE MLX decode", "invalid VaeSpec"));
    }
    let mut x = if spec.scaling != 1.0 || spec.shift != 0.0 {
        let scaling = scalar(spec.scaling, latents.dtype(), stream)?;
        let shift = scalar(spec.shift, latents.dtype(), stream)?;
        Array::add(&Array::divide(latents, &scaling, stream)?, &shift, stream)?
    } else {
        latents.clone()
    };
    if x.shape().len() == 5 {
        x = x
            .slice(
                &[0, 0, 0, 0, 0],
                &[x.shape()[0], x.shape()[1], 1, x.shape()[3], x.shape()[4]],
                &[1, 1, 1, 1, 1],
                stream,
            )?
            .squeeze_axis(2, stream)?;
    }
    if x.shape().len() != 4 || x.shape()[1] != spec.latent_channels as i32 {
        return Err(Error::new("VAE MLX decode", "expected NCHW latent input"));
    }
    let mut x = x.transpose_axes(&[0, 2, 3, 1], stream)?;
    x = conv(weights, "decoder.conv_in", &x, stream)?;
    record_stage(&mut trace, "conv_in", &x, stream)?;
    x = resnet(weights, spec, "decoder.mid_block.resnets.0", &x, stream)?;
    record_stage(&mut trace, "mid_resnet1", &x, stream)?;
    if spec.attention {
        x = spatial_attention(weights, spec, "decoder.mid_block.attentions.0", &x, stream)?;
        record_stage(&mut trace, "mid_attention", &x, stream)?;
    }
    x = resnet(weights, spec, "decoder.mid_block.resnets.1", &x, stream)?;
    record_stage(&mut trace, "mid_resnet2", &x, stream)?;
    for (index, channels) in spec.block_out.iter().rev().enumerate() {
        for block in 0..spec.resnets {
            x = resnet(
                weights,
                spec,
                &format!("decoder.up_blocks.{index}.resnets.{block}"),
                &x,
                stream,
            )?;
        }
        if index + 1 < spec.block_out.len() {
            x = upsample(
                weights,
                &format!("decoder.up_blocks.{index}.upsamplers.0.conv"),
                &x,
                stream,
            )?;
        }
        if x.shape()[3] != *channels as i32 {
            return Err(Error::new("VAE MLX decode", "up block channel mismatch"));
        }
        record_stage(&mut trace, &format!("up_block_{index}"), &x, stream)?;
    }
    x = group_norm(weights, spec, "decoder.conv_norm_out", &x, stream)?;
    record_stage(&mut trace, "conv_norm_out", &x, stream)?;
    x = x.silu(stream)?;
    x = conv(weights, "decoder.conv_out", &x, stream)?;
    x.transpose_axes(&[0, 3, 1, 2], stream)
}

fn record_stage(
    trace: &mut Option<&mut BTreeMap<String, Array>>,
    name: &str,
    value: &Array,
    stream: &Stream,
) -> Result<()> {
    if let Some(trace) = trace.as_deref_mut() {
        trace.insert(
            name.to_owned(),
            value.transpose_axes(&[0, 3, 1, 2], stream)?,
        );
    }
    Ok(())
}

fn resnet(
    weights: &BTreeMap<String, Array>,
    spec: &VaeSpec,
    prefix: &str,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let mut x = group_norm(weights, spec, &format!("{prefix}.norm1"), input, stream)?;
    x = x.silu(stream)?;
    x = conv(weights, &format!("{prefix}.conv1"), &x, stream)?;
    x = group_norm(weights, spec, &format!("{prefix}.norm2"), &x, stream)?;
    x = x.silu(stream)?;
    x = conv(weights, &format!("{prefix}.conv2"), &x, stream)?;
    let shortcut = format!("{prefix}.conv_shortcut");
    if weights.contains_key(&format!("{shortcut}.weight")) {
        let shortcut = conv(weights, &shortcut, input, stream)?;
        Array::add(&shortcut, &x, stream)
    } else {
        Array::add(input, &x, stream)
    }
}

fn spatial_attention(
    weights: &BTreeMap<String, Array>,
    spec: &VaeSpec,
    prefix: &str,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (batch, height, width, channels) = (shape[0], shape[1], shape[2], shape[3]);
    let norm_name = if weights.contains_key(&format!("{prefix}.group_norm.weight")) {
        format!("{prefix}.group_norm")
    } else {
        format!("{prefix}.norm")
    };
    let x = group_norm(weights, spec, &norm_name, input, stream)?;
    let q = Array::linear(
        &x,
        weight(weights, &format!("{prefix}.to_q.weight"))?,
        Some(weight(weights, &format!("{prefix}.to_q.bias"))?),
        stream,
    )?
    .reshape(&[batch, height * width, 1, channels], stream)?
    .transpose_axes(&[0, 2, 1, 3], stream)?;
    let k = Array::linear(
        &x,
        weight(weights, &format!("{prefix}.to_k.weight"))?,
        Some(weight(weights, &format!("{prefix}.to_k.bias"))?),
        stream,
    )?
    .reshape(&[batch, height * width, 1, channels], stream)?
    .transpose_axes(&[0, 2, 1, 3], stream)?;
    let v = Array::linear(
        &x,
        weight(weights, &format!("{prefix}.to_v.weight"))?,
        Some(weight(weights, &format!("{prefix}.to_v.bias"))?),
        stream,
    )?
    .reshape(&[batch, height * width, 1, channels], stream)?
    .transpose_axes(&[0, 2, 1, 3], stream)?;
    let output = Array::scaled_dot_product_attention(
        &q,
        &k,
        &v,
        1.0 / (channels as f32).sqrt(),
        "",
        None,
        stream,
    )?
    .transpose_axes(&[0, 2, 1, 3], stream)?
    .reshape(&[batch, height, width, channels], stream)?;
    let output = Array::linear(
        &output,
        weight_alias(
            weights,
            &format!("{prefix}.to_out.weight"),
            &format!("{prefix}.to_out.0.weight"),
        )?,
        Some(weight_alias(
            weights,
            &format!("{prefix}.to_out.bias"),
            &format!("{prefix}.to_out.0.bias"),
        )?),
        stream,
    )?;
    Array::add(input, &output, stream)
}

fn upsample(
    weights: &BTreeMap<String, Array>,
    prefix: &str,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (batch, height, width, channels) = (shape[0], shape[1], shape[2], shape[3]);
    let x = input.reshape(&[batch, height, 1, width, channels], stream)?;
    let x = Array::concatenate_axis(&[x.clone(), x], 2, stream)?
        .reshape(&[batch, height * 2, width, channels], stream)?;
    let x = x.reshape(&[batch, height * 2, width, 1, channels], stream)?;
    let x = Array::concatenate_axis(&[x.clone(), x], 3, stream)?
        .reshape(&[batch, height * 2, width * 2, channels], stream)?;
    conv(weights, prefix, &x, stream)
}

fn group_norm(
    weights: &BTreeMap<String, Array>,
    spec: &VaeSpec,
    prefix: &str,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (batch, height, width, channels) = (shape[0], shape[1], shape[2], shape[3]);
    let groups = spec.groups;
    if channels % groups != 0 {
        return Err(Error::new(
            "VAE MLX group norm",
            "channels are not divisible by groups",
        ));
    }
    let grouped = input.reshape(&[batch, height, width, groups, channels / groups], stream)?;
    let mean = grouped.mean_axes(&[1, 2, 4], true, stream)?;
    let centered = Array::subtract(&grouped, &mean, stream)?;
    let squared = Array::multiply(&centered, &centered, stream)?;
    let variance = squared.mean_axes(&[1, 2, 4], true, stream)?;
    let eps = scalar(1e-6, input.dtype(), stream)?;
    let denominator = Array::add(&variance, &eps, stream)?.sqrt(stream)?;
    let normalized = Array::divide(&centered, &denominator, stream)?.reshape(&shape, stream)?;
    let normalized = Array::multiply(
        &normalized,
        weight(weights, &format!("{prefix}.weight"))?,
        stream,
    )?;
    Array::add(
        &normalized,
        weight(weights, &format!("{prefix}.bias"))?,
        stream,
    )
}

fn conv(
    weights: &BTreeMap<String, Array>,
    prefix: &str,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let weight = weight(weights, &format!("{prefix}.weight"))?;
    let shape = weight.shape();
    let padding = if shape.len() == 4 && shape[1] == 1 && shape[2] == 1 {
        0
    } else {
        1
    };
    if prefix.ends_with("conv_shortcut") {
        let weight = weight.reshape(&[shape[0], shape[1]], stream)?;
        let output = Array::linear(
            input,
            &weight,
            weights.get(&format!("{prefix}.bias")),
            stream,
        )?;
        return Ok(output);
    }
    let kernel = weight.transpose_axes(&[0, 2, 3, 1], stream)?;
    let mut output = Array::conv2d(
        input,
        &kernel,
        [1, 1],
        [padding, padding],
        [1, 1],
        1,
        stream,
    )?;
    if let Some(bias) = weights.get(&format!("{prefix}.bias")) {
        output = Array::add(&output, bias, stream)?;
    }
    Ok(output)
}

fn scalar(value: f32, dtype: danmo_mlx::Dtype, stream: &Stream) -> Result<Array> {
    Array::from_f32_slice(&[value], &[])?.astype(dtype, stream)
}

fn weight_alias<'a>(
    weights: &'a BTreeMap<String, Array>,
    primary: &str,
    alias: &str,
) -> Result<&'a Array> {
    weights
        .get(primary)
        .or_else(|| weights.get(alias))
        .ok_or_else(|| Error::new("VAE MLX weights", format!("missing {primary} or {alias}")))
}

fn weight<'a>(weights: &'a BTreeMap<String, Array>, name: &str) -> Result<&'a Array> {
    weights
        .get(name)
        .ok_or_else(|| Error::new("VAE MLX weights", format!("missing {name}")))
}

#[cfg(test)]
mod tests {
    use super::decode;
    use super::VaeSpec;
    use danmo_mlx::{Array, Stream};

    #[test]
    #[ignore]
    fn python_main_vae_fixture_matches_rust() {
        let root = std::path::PathBuf::from(std::env::var("DANQING_TEST_VAE_PARITY_DIR").unwrap());
        let payload: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("expected.json")).unwrap())
                .unwrap();
        #[cfg(feature = "mlx-metal")]
        let stream = Stream::gpu().unwrap();
        #[cfg(not(feature = "mlx-metal"))]
        let stream = Stream::cpu().unwrap();
        let weights = super::load_dense(&root, crate::weights::WeightDtype::F32).unwrap();
        let spec = VaeSpec {
            latent_channels: payload["latent_channels"].as_i64().unwrap(),
            block_out: payload["block_out"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_i64().unwrap())
                .collect(),
            resnets: payload["resnets"].as_u64().unwrap() as usize,
            groups: payload["groups"].as_i64().unwrap() as i32,
            scaling: payload["scaling"].as_f64().unwrap() as f32,
            shift: payload["shift"].as_f64().unwrap() as f32,
            attention: payload["attention"].as_bool().unwrap(),
        };
        let latent: Vec<f32> = payload["latent"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        let latent = Array::from_f32_slice(&latent, &[1, 2, 2, 2])
            .unwrap()
            .astype(danmo_mlx::Dtype::BFloat16, &stream)
            .unwrap();
        let actual = decode(&weights, &spec, &latent, &stream)
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
            let allowed = 4.0 * 2f32.powf(exponent - 7.0) + 1e-3;
            assert!(
                error <= allowed,
                "error {error} exceeds four BF16 ULP ({allowed})"
            );
        }
        println!("main/Rust VAE max abs error: {max_error}");
    }

    #[test]
    #[ignore]
    fn real_vae_decoder_matches_main_pixels() {
        let vae = std::path::PathBuf::from(std::env::var("DANQING_TEST_VAE_DIR").unwrap());
        let payload: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(std::env::var("DANQING_TEST_VAE_REAL_JSON").unwrap()).unwrap(),
        )
        .unwrap();
        #[cfg(feature = "mlx-metal")]
        let stream = Stream::gpu().unwrap();
        #[cfg(not(feature = "mlx-metal"))]
        let stream = Stream::cpu().unwrap();
        let started = std::time::Instant::now();
        let weights = super::load_dense(&vae, crate::weights::WeightDtype::F32).unwrap();
        let spec = VaeSpec {
            latent_channels: payload["latent_channels"].as_i64().unwrap(),
            block_out: payload["block_out"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_i64().unwrap())
                .collect(),
            resnets: payload["resnets"].as_u64().unwrap() as usize,
            groups: payload["groups"].as_i64().unwrap() as i32,
            scaling: payload["scaling"].as_f64().unwrap() as f32,
            shift: payload["shift"].as_f64().unwrap() as f32,
            attention: payload["attention"].as_bool().unwrap(),
        };
        let latent: Vec<f32> = payload["latent"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        let latent = Array::from_f32_slice(&latent, &[1, 16, 2, 2])
            .unwrap()
            .astype(danmo_mlx::Dtype::BFloat16, &stream)
            .unwrap();
        let mut trace = std::collections::BTreeMap::new();
        let actual = super::decode_impl(&weights, &spec, &latent, &stream, Some(&mut trace))
            .unwrap()
            .to_vec_f32(&stream)
            .unwrap();
        for (name, expected) in payload["stages"].as_object().unwrap() {
            let expected: Vec<f32> = expected
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect();
            let actual = trace[name].to_vec_f32(&stream).unwrap();
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            println!("VAE stage {name}: max abs error {max_error}");
        }
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
            let allowed = 4.0 * 2f32.powf(exponent - 7.0) + 1e-3;
            assert!(
                error <= allowed,
                "error {error} exceeds BF16 tolerance ({allowed})"
            );
        }
        println!(
            "real VAE decode: {:.3}s, max abs error {max_error}",
            started.elapsed().as_secs_f64()
        );
    }
}
