use std::collections::BTreeMap;

use danmo_mlx::{Array, Error, Result, Stream};

use super::VaeSpec;
use crate::weights::MlxWeights;

/// Dense-only load (fails loud on affine-quantized tensors).
pub fn load_dense(
    directory: &std::path::Path,
    dtype: crate::weights::WeightDtype,
) -> Result<BTreeMap<String, Array>> {
    crate::weights::load_mlx_directory(directory, dtype, None)?.into_dense()
}

/// Load keeping affine linears packed; the mid-attention projections run
/// through QMM exactly like main's quantized VAE inference.
///
/// mflux-serialized bundles wrap conv/norm modules one level deeper
/// (`decoder.conv_in.conv.weight`, `decoder.conv_norm_out.norm.weight`) and
/// store conv kernels in MLX NHWC `[O, kH, kW, I]`. Keys are unwrapped to the
/// diffusers names; the kernels stay NHWC so each conv reads them directly.
/// Diffusers checkpoints keep `[O, I, kH, kW]` and are transposed at the conv.
/// Detection is by key shape, never by family id.
pub fn load(directory: &std::path::Path, dtype: crate::weights::WeightDtype) -> Result<MlxWeights> {
    let weights = crate::weights::load_mlx_directory(directory, dtype, None)?;
    let weights = normalize_mflux_layout(weights)?;
    // main's quantized VAE keeps dense conv/norm params at the F32 module-init
    // dtype (slice assignment preserves it); the dense fp16 path replaces them
    // with BF16. Widen dense floating tensors for quantized bundles so the
    // activation dtype chain matches main exactly.
    if weights.quantization().is_some() {
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

fn normalize_mflux_layout(weights: MlxWeights) -> Result<MlxWeights> {
    let mflux = weights.names().any(|name| name.contains(".conv."));
    if !mflux {
        return Ok(weights);
    }
    let mut names = BTreeMap::new();
    for name in weights.names() {
        names.insert(name.to_string(), unwrap_mflux_key(name));
    }
    let mut weights = weights.rename(&names)?;
    // mflux conv_in is [O, kH, kW, I] with I = 3, the same rank as diffusers
    // [O, I, kH, kW]. Fold those kernels to OIHW here so the conv reader can
    // treat every ambiguous 3×3 as diffusers. Wider kernels stay NHWC.
    let stream = Stream::cpu()?;
    let ambiguous: Vec<String> = weights
        .names()
        .filter(|name| {
            weights
                .tensor(name)
                .ok()
                .is_some_and(|tensor| ambiguous_square_kernel(&tensor.shape()))
        })
        .map(str::to_string)
        .collect();
    for name in ambiguous {
        let tensor = weights.tensor(&name)?;
        let oihw = tensor.transpose_axes(&[0, 3, 1, 2], &stream)?;
        weights = weights.with_tensor_production(name, oihw);
    }
    Ok(weights)
}

fn unwrap_mflux_key(name: &str) -> String {
    let Some((prefix, rest)) = name
        .strip_prefix("decoder.")
        .map(|rest| ("decoder.", rest))
        .or_else(|| name.strip_prefix("encoder.").map(|rest| ("encoder.", rest)))
    else {
        return name.to_string();
    };
    // conv_in / conv_out wrap the Conv2d one level deeper in mflux. The
    // encoder file spells that wrapper conv2d; the decoder spells it conv.
    // Upsamplers and downsamplers keep their .conv submodule in both layouts.
    let rest = rest
        .replacen("conv_in.conv2d.", "conv_in.", 1)
        .replacen("conv_in.conv.", "conv_in.", 1)
        .replacen("conv_out.conv2d.", "conv_out.", 1)
        .replacen("conv_out.conv.", "conv_out.", 1)
        .replacen("conv_norm_out.norm.", "conv_norm_out.", 1);
    format!("{prefix}{rest}")
}

fn ambiguous_square_kernel(shape: &[i32]) -> bool {
    shape.len() == 4 && shape[1] <= 7 && shape[2] <= 7 && shape[3] <= 7 && shape[2] == shape[3]
}

/// MLX conv2d wants `[O, kH, kW, I]`. mflux already stores that. Diffusers
/// stores `[O, I, kH, kW]`; transpose those once per call. A square kernel
/// with a small input channel is treated as diffusers OIHW; mflux copies of
/// that shape are folded to OIHW when the bundle is normalized.
fn mlx_conv_kernel(kernel: &Array, stream: &Stream) -> Result<Array> {
    let shape = kernel.shape();
    if shape.len() != 4 {
        return Err(Error::new("VAE MLX conv", "conv kernel rank is not 4"));
    }
    let nhwc = shape[1] <= 7 && shape[2] <= 7 && shape[3] > 7;
    let oihw = shape[1] > 7 && shape[2] <= 7 && shape[3] <= 7;
    if nhwc {
        Ok(kernel.clone())
    } else if oihw || ambiguous_square_kernel(&shape) {
        // A square kernel whose input channel is also small (conv_in is
        // [128, 3, 3, 3]) cannot be told from NHWC by rank. mflux copies of
        // those kernels are folded to OIHW while the bundle is normalized.
        kernel.transpose_axes(&[0, 2, 3, 1], stream)
    } else {
        Err(Error::new(
            "VAE MLX conv",
            "conv kernel layout is ambiguous",
        ))
    }
}

/// Encode an NCHW image in `[-1, 1]` to the scaled latent `[1, C, H/f, W/f]`.
///
/// `spec.resnets` is `layers_per_block`: each down block runs that many
/// resnets, then a stride-2 conv on every block but the last. The mean half
/// of `conv_out` is `(mean - shift) * scaling`. Z-Image has no quant conv.
pub fn encode(
    weights: &MlxWeights,
    spec: &VaeSpec,
    image: &Array,
    stream: &Stream,
) -> Result<Array> {
    if spec.latent_channels < 1 || spec.groups < 1 || spec.resnets < 1 || spec.block_out.len() < 2
    {
        return Err(Error::new("VAE MLX encode", "invalid VaeSpec"));
    }
    if image.shape().len() != 4 || image.shape()[0] != 1 || image.shape()[1] != 3 {
        return Err(Error::new(
            "VAE MLX encode",
            "expected NCHW image [1, 3, H, W]",
        ));
    }
    let factor = 1i32 << (spec.block_out.len() as i32 - 1);
    let height = image.shape()[2];
    let width = image.shape()[3];
    if height < factor || width < factor || height % factor != 0 || width % factor != 0 {
        return Err(Error::new(
            "VAE MLX encode",
            format!("image {width}x{height} is not divisible by scale {factor}"),
        ));
    }
    let mut x = image.transpose_axes(&[0, 2, 3, 1], stream)?;
    x = conv(weights, "encoder.conv_in", &x, stream)?;
    for (index, channels) in spec.block_out.iter().enumerate() {
        for block in 0..spec.resnets {
            x = resnet(
                weights,
                spec,
                &format!("encoder.down_blocks.{index}.resnets.{block}"),
                &x,
                stream,
                &mut None,
            )?;
        }
        if index + 1 < spec.block_out.len() {
            x = downsample(
                weights,
                &format!("encoder.down_blocks.{index}.downsamplers.0.conv"),
                &x,
                stream,
            )?;
        }
        if x.shape()[3] != *channels as i32 {
            return Err(Error::new("VAE MLX encode", "down block channel mismatch"));
        }
    }
    x = resnet(
        weights,
        spec,
        "encoder.mid_block.resnets.0",
        &x,
        stream,
        &mut None,
    )?;
    if spec.attention {
        x = spatial_attention(weights, spec, "encoder.mid_block.attentions.0", &x, stream)?;
    }
    x = resnet(
        weights,
        spec,
        "encoder.mid_block.resnets.1",
        &x,
        stream,
        &mut None,
    )?;
    let norm_name = if weights.has("encoder.conv_norm_out.weight") {
        "encoder.conv_norm_out"
    } else {
        "encoder.norm_out"
    };
    x = group_norm(weights, spec, norm_name, &x, stream)?;
    x = x.silu(stream)?;
    x = conv(weights, "encoder.conv_out", &x, stream)?;
    let x = x.transpose_axes(&[0, 3, 1, 2], stream)?;
    let channels = spec.latent_channels as i32;
    if x.shape()[1] != channels * 2 {
        return Err(Error::new(
            "VAE MLX encode",
            "conv_out is not mean and logvar",
        ));
    }
    let mean = x.slice(
        &[0, 0, 0, 0],
        &[x.shape()[0], channels, x.shape()[2], x.shape()[3]],
        &[1, 1, 1, 1],
        stream,
    )?;
    let scaling = scalar(spec.scaling, mean.dtype(), stream)?;
    let shift = scalar(spec.shift, mean.dtype(), stream)?;
    Array::multiply(&Array::subtract(&mean, &shift, stream)?, &scaling, stream)
}

pub fn decode(
    weights: &MlxWeights,
    spec: &VaeSpec,
    latents: &Array,
    stream: &Stream,
) -> Result<Array> {
    decode_impl(weights, spec, latents, stream, None)
}

/// Shape-specialized compiled decode. The first `apply` traces the decoder
/// for that latent shape; later calls with the same shape replay it. A new
/// shape fails in MLX rather than silently retracing into a different graph.
pub struct CompiledDecoder {
    closure: danmo_mlx::UnaryClosure,
}

impl CompiledDecoder {
    pub fn new(weights: MlxWeights, spec: VaeSpec, stream: Stream) -> Result<Self> {
        let closure = danmo_mlx::UnaryClosure::new(stream, move |latent, stream| {
            decode_impl(&weights, &spec, latent, stream, None)
        })?
        .compile(false)?;
        Ok(Self { closure })
    }

    pub fn apply(&self, latent: &Array) -> Result<Array> {
        self.closure.apply(latent)
    }
}

fn decode_impl(
    weights: &MlxWeights,
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
    x = resnet(
        weights,
        spec,
        "decoder.mid_block.resnets.0",
        &x,
        stream,
        &mut trace,
    )?;
    record_stage(&mut trace, "mid_resnet1", &x, stream)?;
    if spec.attention {
        x = spatial_attention(weights, spec, "decoder.mid_block.attentions.0", &x, stream)?;
        record_stage(&mut trace, "mid_attention", &x, stream)?;
    }
    x = resnet(
        weights,
        spec,
        "decoder.mid_block.resnets.1",
        &x,
        stream,
        &mut trace,
    )?;
    record_stage(&mut trace, "mid_resnet2", &x, stream)?;
    for (index, channels) in spec.block_out.iter().rev().enumerate() {
        for block in 0..spec.resnets {
            x = resnet(
                weights,
                spec,
                &format!("decoder.up_blocks.{index}.resnets.{block}"),
                &x,
                stream,
                &mut trace,
            )?;
            record_stage(
                &mut trace,
                &format!("up_{index}_resnet_{block}"),
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
            record_stage(&mut trace, &format!("up_{index}_upsample"), &x, stream)?;
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
    weights: &MlxWeights,
    spec: &VaeSpec,
    prefix: &str,
    input: &Array,
    stream: &Stream,
    trace: &mut Option<&mut BTreeMap<String, Array>>,
) -> Result<Array> {
    let mut x = group_norm(weights, spec, &format!("{prefix}.norm1"), input, stream)?;
    record_stage(trace, &format!("{prefix}.norm1"), &x, stream)?;
    x = x.silu(stream)?;
    x = conv(weights, &format!("{prefix}.conv1"), &x, stream)?;
    record_stage(trace, &format!("{prefix}.conv1"), &x, stream)?;
    x = group_norm(weights, spec, &format!("{prefix}.norm2"), &x, stream)?;
    record_stage(trace, &format!("{prefix}.norm2"), &x, stream)?;
    x = x.silu(stream)?;
    x = conv(weights, &format!("{prefix}.conv2"), &x, stream)?;
    record_stage(trace, &format!("{prefix}.conv2"), &x, stream)?;
    let shortcut = format!("{prefix}.conv_shortcut");
    if weights.has(&format!("{shortcut}.weight")) {
        let shortcut = conv(weights, &shortcut, input, stream)?;
        record_stage(trace, &format!("{prefix}.shortcut"), &shortcut, stream)?;
        Array::add(&shortcut, &x, stream)
    } else {
        Array::add(input, &x, stream)
    }
}

fn spatial_attention(
    weights: &MlxWeights,
    spec: &VaeSpec,
    prefix: &str,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (batch, height, width, channels) = (shape[0], shape[1], shape[2], shape[3]);
    let norm_name = if weights.has(&format!("{prefix}.group_norm.weight")) {
        format!("{prefix}.group_norm")
    } else {
        format!("{prefix}.norm")
    };
    let x = group_norm(weights, spec, &norm_name, input, stream)?;
    let q = weights
        .linear(&format!("{prefix}.to_q.weight"))?
        .apply(
            &x,
            Some(weight(weights, &format!("{prefix}.to_q.bias"))?),
            stream,
        )?
        .reshape(&[batch, height * width, 1, channels], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let k = weights
        .linear(&format!("{prefix}.to_k.weight"))?
        .apply(
            &x,
            Some(weight(weights, &format!("{prefix}.to_k.bias"))?),
            stream,
        )?
        .reshape(&[batch, height * width, 1, channels], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let v = weights
        .linear(&format!("{prefix}.to_v.weight"))?
        .apply(
            &x,
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
    let output = linear_alias(
        weights,
        &format!("{prefix}.to_out.weight"),
        &format!("{prefix}.to_out.0.weight"),
    )?
    .apply(
        &output,
        Some(weight_alias(
            weights,
            &format!("{prefix}.to_out.bias"),
            &format!("{prefix}.to_out.0.bias"),
        )?),
        stream,
    )?;
    Array::add(input, &output, stream)
}

/// Pad the bottom and right edges by one, then a 3×3 stride-2 conv with no
/// extra padding. That is diffusers `Downsample2D` and main's VAE encoder.
fn downsample(weights: &MlxWeights, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
    let shape = input.shape();
    let (batch, height, width, channels) = (shape[0], shape[1], shape[2], shape[3]);
    let row = zeros(&[batch, 1, width, channels], input.dtype(), stream)?;
    let x = Array::concatenate_axis(&[input.clone(), row], 1, stream)?;
    let column = zeros(&[batch, height + 1, 1, channels], input.dtype(), stream)?;
    let x = Array::concatenate_axis(&[x, column], 2, stream)?;
    conv_sized(weights, prefix, &x, [2, 2], [0, 0], stream)
}

fn zeros(shape: &[i32], dtype: danmo_mlx::Dtype, stream: &Stream) -> Result<Array> {
    Array::from_f32_slice(&[0.0], &[])?
        .astype(dtype, stream)?
        .broadcast_to(shape, stream)
}

fn upsample(weights: &MlxWeights, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
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
    weights: &MlxWeights,
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
    let group_size = channels / groups;
    let grouped = input.reshape(&[batch, height * width, groups, group_size], stream)?;
    let grouped = grouped
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, groups, height * width * group_size], stream)?;
    let normalized = grouped.layer_norm(1e-6, stream)?;
    let normalized = normalized
        .reshape(&[batch, groups, height * width, group_size], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&shape, stream)?;
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

fn linear_alias<'a>(
    weights: &'a MlxWeights,
    primary: &str,
    alias: &str,
) -> Result<&'a danmo_mlx::LinearWeight> {
    if weights.has(primary) {
        weights.linear(primary)
    } else {
        weights.linear(alias)
    }
}

fn conv(weights: &MlxWeights, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
    let kernel_weight = weight(weights, &format!("{prefix}.weight"))?;
    let kernel = mlx_conv_kernel(kernel_weight, stream)?;
    let shape = kernel.shape();
    let padding = if shape.len() == 4 && shape[1] == 1 && shape[2] == 1 {
        0
    } else {
        1
    };
    conv_kernel(weights, prefix, input, &kernel, [1, 1], [padding, padding], stream)
}

fn conv_sized(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    stride: [i32; 2],
    padding: [i32; 2],
    stream: &Stream,
) -> Result<Array> {
    let kernel_weight = weight(weights, &format!("{prefix}.weight"))?;
    let kernel = mlx_conv_kernel(kernel_weight, stream)?;
    conv_kernel(weights, prefix, input, &kernel, stride, padding, stream)
}

fn conv_kernel(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    kernel: &Array,
    stride: [i32; 2],
    padding: [i32; 2],
    stream: &Stream,
) -> Result<Array> {
    let mut output = Array::conv2d(input, kernel, stride, padding, [1, 1], 1, stream)?;
    if weights.has(&format!("{prefix}.bias")) {
        output = Array::add(&output, weight(weights, &format!("{prefix}.bias"))?, stream)?;
    }
    Ok(output)
}

fn scalar(value: f32, dtype: danmo_mlx::Dtype, stream: &Stream) -> Result<Array> {
    Array::from_f32_slice(&[value], &[])?.astype(dtype, stream)
}

fn weight_alias<'a>(weights: &'a MlxWeights, primary: &str, alias: &str) -> Result<&'a Array> {
    if weights.has(primary) {
        weights.tensor(primary)
    } else {
        weights.tensor(alias)
    }
}

fn weight<'a>(weights: &'a MlxWeights, name: &str) -> Result<&'a Array> {
    weights.tensor(name)
}

#[cfg(test)]
mod tests {
    use super::decode;
    use super::VaeSpec;
    use crate::weights::MlxWeights;
    use danmo_mlx::{Array, Stream};
    use std::collections::BTreeMap;

    /// The quantized VAE path runs the mid-attention projections through
    /// packed QMM; against the dequantized dense reference the outputs must
    /// agree within the affine quantization error (same contract as the DiT
    /// packed probe).
    fn quantized_vae_probe(device_type: danmo_mlx::DeviceType) -> danmo_mlx::Result<()> {
        use danmo_mlx::{LinearWeight, QuantizedArray};
        let stream = match device_type {
            danmo_mlx::DeviceType::Cpu => Stream::cpu()?,
            danmo_mlx::DeviceType::Gpu => Stream::gpu()?,
        };
        let channels = 64i32;
        let spec = VaeSpec {
            latent_channels: 2,
            block_out: vec![4],
            resnets: 1,
            groups: 2,
            scaling: 1.0,
            shift: 0.0,
            attention: true,
        };
        let attention = "decoder.mid_block.attentions.0";
        fn make(name: &str, shape: &[i32], seed: usize) -> Array {
            let count: i32 = shape.iter().product();
            let values: Vec<f32> = (0..count)
                .map(|index| (((index as usize + seed) % 13) as f32) * 0.02 - 0.12)
                .collect();
            Array::from_f32_slice(&values, shape).unwrap()
        }
        let mut dense: BTreeMap<String, Array> = BTreeMap::new();
        dense.insert(
            format!("{attention}.group_norm.weight"),
            make("norm", &[channels], 17),
        );
        dense.insert(
            format!("{attention}.group_norm.bias"),
            make("norm", &[channels], 18),
        );
        let mut packed_weights = MlxWeights::from_dense(dense.clone());
        for (index, projection) in ["to_q", "to_k", "to_v", "to_out"].iter().enumerate() {
            let base = format!("{attention}.{projection}");
            let weight = make("weight", &[channels, channels], 19 + index);
            let bias = make("bias", &[channels], 23 + index);
            dense.insert(format!("{base}.bias"), bias);
            packed_weights = packed_weights.with_tensor(
                format!("{base}.bias"),
                dense[&format!("{base}.bias")].clone(),
            );
            let quantized = weight.quantize_affine(32, 4, &stream)?;
            dense.insert(
                format!("{base}.weight"),
                quantized.dequantize(32, 4, &stream)?,
            );
            let layer = LinearWeight::affine(
                QuantizedArray {
                    weight: quantized.weight,
                    scales: quantized.scales,
                    biases: quantized.biases,
                },
                danmo_mlx::AffineQuantization::new(4, 32)?,
            )?;
            packed_weights = packed_weights.with_linear(format!("{base}.weight"), layer);
        }
        let reference = MlxWeights::from_dense(dense);
        let input_values: Vec<f32> = (0..(2 * 2 * channels))
            .map(|index| ((index % 11) as f32) * 0.05 - 0.25)
            .collect();
        let input = Array::from_f32_slice(&input_values, &[1, 2, 2, channels])?;
        let actual = super::spatial_attention(&packed_weights, &spec, attention, &input, &stream)?
            .to_vec_f32(&stream)?;
        let expected = super::spatial_attention(&reference, &spec, attention, &input, &stream)?
            .to_vec_f32(&stream)?;
        assert_eq!(actual.len(), expected.len());
        let scale = expected.iter().fold(0.0f32, |peak, v| peak.max(v.abs()));
        let mut max_error = 0.0f32;
        for (actual, expected) in actual.iter().zip(&expected) {
            let error = (actual - expected).abs();
            max_error = max_error.max(error);
            assert!(
                error <= 2e-2 * scale + 1e-3,
                "packed {actual} != dequantized {expected}"
            );
        }
        println!("quantized VAE attention max abs error {max_error} (scale {scale})");
        Ok(())
    }

    #[test]
    fn cpu_quantized_vae_attention_matches_dequantized_reference() {
        quantized_vae_probe(danmo_mlx::DeviceType::Cpu).unwrap();
    }

    #[cfg(feature = "mlx-metal")]
    #[test]
    fn metal_quantized_vae_attention_matches_dequantized_reference() {
        quantized_vae_probe(danmo_mlx::DeviceType::Gpu).unwrap();
    }

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
        let weights = MlxWeights::from_dense(
            super::load_dense(&root, crate::weights::WeightDtype::F32).unwrap(),
        );
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
        let weight_dtype = if payload["native_dtype"].as_str() == Some("checkpoint") {
            crate::weights::WeightDtype::Native
        } else {
            crate::weights::WeightDtype::F32
        };
        let weights = MlxWeights::from_dense(super::load_dense(&vae, weight_dtype).unwrap());
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
        let latent_shape: Vec<i32> = payload["latent_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect();
        let latent = Array::from_f32_slice(&latent, &latent_shape)
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

    fn fill_const(
        tensors: &mut BTreeMap<String, Array>,
        prefix: &str,
        oc: i32,
        ic: i32,
        value: f32,
    ) {
        let weight = oc as usize * ic as usize * 9;
        tensors.insert(
            format!("{prefix}.weight"),
            Array::from_f32_slice(&vec![value; weight], &[oc, ic, 3, 3]).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.bias"),
            Array::from_f32_slice(&vec![value; oc as usize], &[oc]).unwrap(),
        );
    }

    fn fill_affine(tensors: &mut BTreeMap<String, Array>, prefix: &str, scale: f32) {
        tensors.insert(
            format!("{prefix}.weight"),
            Array::from_f32_slice(&vec![scale; 8], &[8]).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.bias"),
            Array::from_f32_slice(&vec![0.0; 8], &[8]).unwrap(),
        );
    }

    #[test]
    fn encoder_downsamples_once_and_returns_the_scaled_mean() {
        let stream = Stream::cpu().unwrap();
        let spec = VaeSpec {
            latent_channels: 4,
            block_out: vec![8, 8],
            resnets: 1,
            groups: 8,
            scaling: 2.0,
            shift: 0.5,
            attention: false,
        };
        let mut tensors = BTreeMap::new();
        fill_const(&mut tensors, "encoder.conv_in", 8, 3, 0.0);
        for prefix in [
            "encoder.down_blocks.0.resnets.0",
            "encoder.down_blocks.1.resnets.0",
            "encoder.mid_block.resnets.0",
            "encoder.mid_block.resnets.1",
        ] {
            fill_affine(&mut tensors, &format!("{prefix}.norm1"), 1.0);
            fill_const(&mut tensors, &format!("{prefix}.conv1"), 8, 8, 0.0);
            fill_affine(&mut tensors, &format!("{prefix}.norm2"), 1.0);
            fill_const(&mut tensors, &format!("{prefix}.conv2"), 8, 8, 0.0);
        }
        fill_const(
            &mut tensors,
            "encoder.down_blocks.0.downsamplers.0.conv",
            8,
            8,
            0.0,
        );
        fill_affine(&mut tensors, "encoder.conv_norm_out", 1.0);
        // conv_out bias is the mean. shift 0.5 and scaling 2 turn it into 1.
        fill_const(&mut tensors, "encoder.conv_out", 8, 8, 0.0);
        let bias = Array::from_f32_slice(&vec![1.0; 8], &[8]).unwrap();
        tensors.insert("encoder.conv_out.bias".into(), bias);
        let weights = MlxWeights::from_dense(tensors);
        let image = Array::from_f32_slice(&vec![0.0; 8 * 8 * 3], &[1, 3, 8, 8]).unwrap();
        let latent = super::encode(&weights, &spec, &image, &stream).unwrap();
        assert_eq!(latent.shape(), vec![1, 4, 4, 4]);
        let values = latent.to_vec_f32(&stream).unwrap();
        assert!(values.iter().all(|value| (value - 1.0).abs() < 1e-4), "{values:?}");
    }
}
