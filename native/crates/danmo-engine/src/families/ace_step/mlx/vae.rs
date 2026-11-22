//! Oobleck VAE decode. Weight-norm is fused at use, then PyTorch conv
//! layouts are transposed into MLX channels-last kernels.
//!
//! Decoder block 0 is the widest: published `downsampling_ratios` run in
//! reverse, so `[2, 4, 4, 6, 10]` becomes strides `10, 6, 4, 4, 2`.

use danmo_mlx::{Array, Dtype, Result, Stream};

use crate::weights::MlxWeights;

#[derive(Clone, Debug)]
pub struct VaeSpec {
    pub channels: i32,
    pub latent: i32,
    pub audio: i32,
    pub encoder_hidden: i32,
    pub ratios: Vec<i32>,
    pub multiples: Vec<i32>,
}

impl VaeSpec {
    pub fn from_json(text: &str) -> std::result::Result<Self, String> {
        let value: serde_json::Value = serde_json::from_str(text)
            .map_err(|error| format!("ACE-Step VAE config.json is not json: {error}"))?;
        let ratios = value
            .get("downsampling_ratios")
            .and_then(|item| item.as_array())
            .ok_or("ACE-Step VAE config is missing downsampling_ratios")?
            .iter()
            .map(|item| {
                item.as_i64()
                    .map(|value| value as i32)
                    .ok_or("ACE-Step VAE downsampling ratio is not an integer")
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let multiples = value
            .get("channel_multiples")
            .and_then(|item| item.as_array())
            .ok_or("ACE-Step VAE config is missing channel_multiples")?
            .iter()
            .map(|item| {
                item.as_i64()
                    .map(|value| value as i32)
                    .ok_or("ACE-Step VAE channel multiple is not an integer")
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if ratios.len() != multiples.len() {
            return Err(format!(
                "ACE-Step VAE has {} ratios and {} channel multiples",
                ratios.len(),
                multiples.len()
            ));
        }
        Ok(Self {
            channels: value
                .get("decoder_channels")
                .and_then(|item| item.as_i64())
                .ok_or("ACE-Step VAE config is missing decoder_channels")?
                as i32,
            latent: value
                .get("decoder_input_channels")
                .and_then(|item| item.as_i64())
                .ok_or("ACE-Step VAE config is missing decoder_input_channels")?
                as i32,
            audio: value
                .get("audio_channels")
                .and_then(|item| item.as_i64())
                .unwrap_or(2) as i32,
            encoder_hidden: value
                .get("encoder_hidden_size")
                .and_then(|item| item.as_i64())
                .ok_or("ACE-Step VAE config is missing encoder_hidden_size")?
                as i32,
            ratios,
            multiples,
        })
    }

    pub fn hop(&self) -> i32 {
        self.ratios.iter().product()
    }
}

/// Decode `[1, T, 64]` latents to planar stereo `[left | right]`.
pub fn decode_planar(
    weights: &MlxWeights,
    spec: &VaeSpec,
    latents: &Array,
    stream: &Stream,
) -> Result<Vec<f32>> {
    let latents = latents.astype(Dtype::Float32, stream)?;
    let mut hidden = conv(weights, "decoder.conv1", &latents, 1, 1, 3, false, stream)?;
    let mut cm = vec![1];
    cm.extend_from_slice(&spec.multiples);
    let steps = spec.ratios.len();
    for (index, stride) in spec.ratios.iter().rev().enumerate() {
        let input = spec.channels * cm[steps - index];
        let output = spec.channels * cm[steps - index - 1];
        hidden = decoder_block(
            weights,
            &format!("decoder.block.{index}"),
            &hidden,
            input,
            output,
            *stride,
            stream,
        )?;
    }
    hidden = snake(weights, "decoder.snake1", &hidden, stream)?;
    hidden = conv(weights, "decoder.conv2", &hidden, 1, 1, 3, false, stream)?;
    hidden.eval()?;
    let flat = hidden.to_vec_f32(stream)?;
    let channels = spec.audio as usize;
    if channels != 2 || flat.len() % channels != 0 {
        return Err(danmo_mlx::Error::new(
            "ACE-Step VAE",
            format!(
                "decode produced {} samples for {channels} channels",
                flat.len()
            ),
        ));
    }
    let frames = flat.len() / channels;
    let mut planar = vec![0.0; flat.len()];
    for frame in 0..frames {
        planar[frame] = flat[frame * channels];
        planar[frames + frame] = flat[frame * channels + 1];
    }
    Ok(planar)
}

/// Encode interleaved-free NLC audio `[1, samples, 2]` to the Gaussian mean `[T, 64]`.
pub fn encode_mean(
    weights: &MlxWeights,
    spec: &VaeSpec,
    audio: &Array,
    stream: &Stream,
) -> Result<Vec<f32>> {
    if spec.encoder_hidden <= 0 {
        return Err(danmo_mlx::Error::new(
            "ACE-Step VAE",
            "encoder_hidden_size must be positive",
        ));
    }
    let audio = audio.astype(Dtype::Float32, stream)?;
    let mut hidden = conv(weights, "encoder.conv1", &audio, 1, 1, 3, false, stream)?;
    let mut cm = vec![1];
    cm.extend_from_slice(&spec.multiples);
    for (index, stride) in spec.ratios.iter().enumerate() {
        hidden = encoder_block(
            weights,
            &format!("encoder.block.{index}"),
            &hidden,
            *stride,
            stream,
        )?;
    }
    hidden = snake(weights, "encoder.snake1", &hidden, stream)?;
    hidden = conv(weights, "encoder.conv2", &hidden, 1, 1, 1, false, stream)?;
    let shape = hidden.shape();
    if shape.len() != 3 || shape[2] != spec.latent * 2 {
        return Err(danmo_mlx::Error::new(
            "ACE-Step VAE",
            format!(
                "encoder output {:?} is not [1, T, {}]",
                shape,
                spec.latent * 2
            ),
        ));
    }
    let mean = hidden.slice(
        &[0, 0, 0],
        &[shape[0], shape[1], spec.latent],
        &[1, 1, 1],
        stream,
    )?;
    mean.eval()?;
    Ok(mean.to_vec_f32(stream)?)
}

fn encoder_block(
    weights: &MlxWeights,
    prefix: &str,
    hidden: &Array,
    stride: i32,
    stream: &Stream,
) -> Result<Array> {
    let hidden = residual(weights, &format!("{prefix}.res_unit1"), hidden, 1, stream)?;
    let hidden = residual(weights, &format!("{prefix}.res_unit2"), &hidden, 3, stream)?;
    let hidden = residual(weights, &format!("{prefix}.res_unit3"), &hidden, 9, stream)?;
    let hidden = snake(weights, &format!("{prefix}.snake1"), &hidden, stream)?;
    let padding = (stride + 1) / 2;
    conv(
        weights,
        &format!("{prefix}.conv1"),
        &hidden,
        stride,
        1,
        padding,
        false,
        stream,
    )
}

fn decoder_block(
    weights: &MlxWeights,
    prefix: &str,
    hidden: &Array,
    _input: i32,
    _output: i32,
    stride: i32,
    stream: &Stream,
) -> Result<Array> {
    let hidden = snake(weights, &format!("{prefix}.snake1"), hidden, stream)?;
    let kernel = stride * 2;
    let padding = (stride + 1) / 2;
    let hidden = conv_transpose(
        weights,
        &format!("{prefix}.conv_t1"),
        &hidden,
        kernel,
        stride,
        padding,
        stream,
    )?;
    let hidden = residual(weights, &format!("{prefix}.res_unit1"), &hidden, 1, stream)?;
    let hidden = residual(weights, &format!("{prefix}.res_unit2"), &hidden, 3, stream)?;
    residual(weights, &format!("{prefix}.res_unit3"), &hidden, 9, stream)
}

fn residual(
    weights: &MlxWeights,
    prefix: &str,
    hidden: &Array,
    dilation: i32,
    stream: &Stream,
) -> Result<Array> {
    let pad = ((7 - 1) * dilation) / 2;
    let mut output = snake(weights, &format!("{prefix}.snake1"), hidden, stream)?;
    output = conv(
        weights,
        &format!("{prefix}.conv1"),
        &output,
        1,
        dilation,
        pad,
        false,
        stream,
    )?;
    output = snake(weights, &format!("{prefix}.snake2"), &output, stream)?;
    output = conv(
        weights,
        &format!("{prefix}.conv2"),
        &output,
        1,
        1,
        0,
        false,
        stream,
    )?;
    let skip = match_length(hidden, &output, stream)?;
    Array::add(&skip, &output, stream)
}

fn match_length(skip: &Array, output: &Array, stream: &Stream) -> Result<Array> {
    let input_len = skip.shape()[1];
    let output_len = output.shape()[1];
    if output_len == input_len {
        return Ok(skip.clone());
    }
    if output_len > input_len || (input_len - output_len) % 2 != 0 {
        return Err(danmo_mlx::Error::new(
            "ACE-Step VAE",
            format!("residual lengths {input_len} and {output_len} do not line up"),
        ));
    }
    let pad = (input_len - output_len) / 2;
    let shape = skip.shape();
    skip.slice(
        &[0, pad, 0],
        &[shape[0], input_len - pad, shape[2]],
        &[1, 1, 1],
        stream,
    )
}

fn snake(weights: &MlxWeights, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
    let alpha =
        squeeze_vector(weights.tensor(&format!("{prefix}.alpha"))?.clone(), stream)?.exp(stream)?;
    let beta =
        squeeze_vector(weights.tensor(&format!("{prefix}.beta"))?.clone(), stream)?.exp(stream)?;
    let eps = Array::from_f32_slice(&[1.0e-9], &[])?;
    let sine = Array::multiply(&alpha, input, stream)?.sin(stream)?;
    let square = Array::multiply(&sine, &sine, stream)?;
    let scale = Array::divide(
        &Array::from_f32_slice(&[1.0], &[])?,
        &Array::add(&beta, &eps, stream)?,
        stream,
    )?;
    Array::add(input, &Array::multiply(&scale, &square, stream)?, stream)
}

fn conv(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    stride: i32,
    dilation: i32,
    padding: i32,
    transpose: bool,
    stream: &Stream,
) -> Result<Array> {
    let kernel = fuse(weights, prefix, transpose, stream)?;
    let output = if transpose {
        Array::conv_transpose1d(input, &kernel, dilation, padding, 1, 0, 1, stream)?
    } else {
        Array::conv1d(input, &kernel, stride, padding, dilation, 1, stream)?
    };
    let bias = format!("{prefix}.bias");
    if weights.has(&bias) {
        Array::add(&output, weights.tensor(&bias)?, stream)
    } else {
        Ok(output)
    }
}

fn conv_transpose(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    _kernel: i32,
    stride: i32,
    padding: i32,
    stream: &Stream,
) -> Result<Array> {
    let kernel = fuse(weights, prefix, true, stream)?;
    let output = Array::conv_transpose1d(input, &kernel, stride, padding, 1, 0, 1, stream)?;
    let bias = format!("{prefix}.bias");
    if weights.has(&bias) {
        Array::add(&output, weights.tensor(&bias)?, stream)
    } else {
        Ok(output)
    }
}

fn fuse(weights: &MlxWeights, prefix: &str, transpose: bool, stream: &Stream) -> Result<Array> {
    let direction = weights
        .tensor(&format!("{prefix}.weight_v"))?
        .astype(Dtype::Float32, stream)?;
    let gain = weights
        .tensor(&format!("{prefix}.weight_g"))?
        .astype(Dtype::Float32, stream)?;
    let shape = direction.shape();
    if shape.len() != 3 {
        return Err(danmo_mlx::Error::new(
            "ACE-Step VAE",
            format!("{prefix}.weight_v rank {} is not 3", shape.len()),
        ));
    }
    let flat = direction.reshape(&[shape[0], shape[1] * shape[2]], stream)?;
    let square = Array::multiply(&flat, &flat, stream)?;
    let mean = square.mean_axes(&[1], true, stream)?;
    let count = Array::from_f32_slice(&[(shape[1] * shape[2]) as f32], &[1, 1])?;
    let norm = Array::multiply(&mean, &count, stream)?.sqrt(stream)?;
    let eps = Array::from_f32_slice(&[1.0e-9], &[])?;
    let norm = norm.reshape(&[shape[0], 1, 1], stream)?;
    let fused = Array::divide(
        &Array::multiply(&gain, &direction, stream)?,
        &Array::add(&norm, &eps, stream)?,
        stream,
    )?;
    if transpose {
        fused.transpose_axes(&[1, 2, 0], stream)
    } else {
        fused.transpose_axes(&[0, 2, 1], stream)
    }
}

fn squeeze_vector(tensor: Array, stream: &Stream) -> Result<Array> {
    let mut tensor = tensor;
    while tensor.shape().len() > 1 {
        let shape = tensor.shape();
        let Some(axis) = shape.iter().position(|dim| *dim == 1) else {
            break;
        };
        tensor = tensor.squeeze_axis(axis as i32, stream)?;
    }
    Ok(tensor)
}
