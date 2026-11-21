//! LTX 2.5 convolution/padding and depth-space layout policies.
//! These are checkpoint semantics shared by this family's upsampler and VAE,
//! not a generic pipeline abstraction.

use danmo_mlx::{Array, Result, Stream};

use crate::weights::MlxWeights;

pub fn conv3d(
    weights: &MlxWeights,
    name: &str,
    input: &Array,
    kernel: [i32; 3],
    stride: [i32; 3],
    causal: bool,
    spatial_padding_mode: &str,
    stream: &Stream,
) -> Result<Array> {
    conv3d_with_temporal(
        weights,
        name,
        input,
        kernel,
        stride,
        causal,
        spatial_padding_mode,
        "edge",
        stream,
    )
}

pub fn conv3d_zero_temporal(
    weights: &MlxWeights,
    name: &str,
    input: &Array,
    kernel: [i32; 3],
    stride: [i32; 3],
    spatial_padding_mode: &str,
    stream: &Stream,
) -> Result<Array> {
    conv3d_with_temporal(
        weights,
        name,
        input,
        kernel,
        stride,
        false,
        spatial_padding_mode,
        "constant",
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn conv3d_with_temporal(
    weights: &MlxWeights,
    name: &str,
    input: &Array,
    kernel: [i32; 3],
    stride: [i32; 3],
    causal: bool,
    spatial_padding_mode: &str,
    temporal_padding_mode: &str,
    stream: &Stream,
) -> Result<Array> {
    let mut x = input.clone();
    let temporal = kernel[0] - 1;
    let (before, after) = if causal {
        (temporal, 0)
    } else {
        (temporal / 2, temporal / 2)
    };
    if before > 0 || after > 0 {
        x = x.pad(
            &[1],
            &[before],
            &[after],
            0.0,
            temporal_padding_mode,
            stream,
        )?;
    }
    let height = (kernel[1] - 1) / 2;
    let width = (kernel[2] - 1) / 2;
    if height > 0 || width > 0 {
        let spatial_padding_mode = if spatial_padding_mode == "zeros" {
            "constant"
        } else {
            spatial_padding_mode
        };
        x = x.pad(
            &[2, 3],
            &[height, width],
            &[height, width],
            0.0,
            spatial_padding_mode,
            stream,
        )?;
    }
    let mut output = Array::conv3d(
        &x,
        weights.tensor(&format!("{name}.weight"))?,
        stride,
        [0, 0, 0],
        [1, 1, 1],
        1,
        stream,
    )?;
    let bias_name = format!("{name}.bias");
    if weights.has(&bias_name) {
        output = Array::add(&output, weights.tensor(&bias_name)?, stream)?;
    }
    Ok(output)
}

pub fn group_norm(
    weights: &MlxWeights,
    name: &str,
    input: &Array,
    groups: i32,
    eps: f32,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (b, d, h, w, c) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let dtype = input.dtype();
    let grouped = input
        .reshape(&[b, d, h, w, groups, c / groups], stream)?
        .astype(danmo_mlx::Dtype::Float32, stream)?;
    let mean = grouped.mean_axes(&[1, 2, 3, 5], true, stream)?;
    let centered = Array::subtract(&grouped, &mean, stream)?;
    let variance =
        Array::multiply(&centered, &centered, stream)?.mean_axes(&[1, 2, 3, 5], true, stream)?;
    let denominator = Array::add(&variance, &Array::scalar_f32(eps)?, stream)?.sqrt(stream)?;
    let inverse = Array::divide(&Array::scalar_f32(1.0)?, &denominator, stream)?;
    let normalized = Array::multiply(&centered, &inverse, stream)?
        .reshape(&[b, d, h, w, c], stream)?
        .astype(dtype, stream)?;
    let scaled = Array::multiply(
        &normalized,
        weights.tensor(&format!("{name}.weight"))?,
        stream,
    )?;
    Array::add(&scaled, weights.tensor(&format!("{name}.bias"))?, stream)
}

pub fn depth_to_space_3d(
    input: &Array,
    spatial_factor: i32,
    temporal_factor: i32,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (b, d, h, w, total) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let channels = total / (spatial_factor * spatial_factor * temporal_factor);
    input
        .reshape(
            &[
                b,
                d,
                h,
                w,
                channels,
                temporal_factor,
                spatial_factor,
                spatial_factor,
            ],
            stream,
        )?
        .transpose_axes(&[0, 1, 5, 2, 6, 3, 7, 4], stream)?
        .reshape(
            &[
                b,
                d * temporal_factor,
                h * spatial_factor,
                w * spatial_factor,
                channels,
            ],
            stream,
        )
}

pub fn unpatchify_spatial(input: &Array, patch: i32, stream: &Stream) -> Result<Array> {
    let shape = input.shape();
    let (b, f, h, w, total) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let channels = total / (patch * patch);
    input
        .reshape(&[b, f, h, w, channels, patch, patch], stream)?
        .transpose_axes(&[0, 1, 2, 6, 3, 5, 4], stream)?
        .reshape(&[b, f, h * patch, w * patch, channels], stream)
}

pub fn patchify_spatial(input: &Array, patch: i32, stream: &Stream) -> Result<Array> {
    let shape = input.shape();
    let (b, f, h, w, channels) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    input
        .reshape(
            &[b, f, h / patch, patch, w / patch, patch, channels],
            stream,
        )?
        .transpose_axes(&[0, 1, 2, 4, 6, 5, 3], stream)?
        .reshape(
            &[b, f, h / patch, w / patch, channels * patch * patch],
            stream,
        )
}

/// Inverse of [`depth_to_space_3d`]: pack temporal/spatial neighbors into channels.
pub fn space_to_depth_3d(
    input: &Array,
    spatial_factor: i32,
    temporal_factor: i32,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (b, d_full, h_full, w_full, channels) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    if d_full % temporal_factor != 0 || h_full % spatial_factor != 0 || w_full % spatial_factor != 0
    {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 space_to_depth",
            format!(
                "shape [{d_full},{h_full},{w_full}] is not divisible by stride [{temporal_factor},{spatial_factor},{spatial_factor}]"
            ),
        ));
    }
    let d = d_full / temporal_factor;
    let h = h_full / spatial_factor;
    let w = w_full / spatial_factor;
    input
        .reshape(
            &[
                b,
                d,
                temporal_factor,
                h,
                spatial_factor,
                w,
                spatial_factor,
                channels,
            ],
            stream,
        )?
        .transpose_axes(&[0, 1, 3, 5, 7, 2, 4, 6], stream)?
        .reshape(
            &[
                b,
                d,
                h,
                w,
                channels * temporal_factor * spatial_factor * spatial_factor,
            ],
            stream,
        )
}

/// LTX `SpaceToDepthDownsample`: causal first-frame pad, averaged skip, then
/// stride-1 conv packed by the same space-to-depth layout.
pub fn space_to_depth_downsample(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    in_channels: i32,
    out_channels: i32,
    stride: [i32; 3],
    spatial_padding_mode: &str,
    stream: &Stream,
) -> Result<Array> {
    let prod = stride[0] * stride[1] * stride[2];
    if prod <= 0 || out_channels % prod != 0 {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 space_to_depth downsample",
            format!("out_channels {out_channels} must be divisible by stride product {prod}"),
        ));
    }
    if in_channels * prod % out_channels != 0 {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 space_to_depth downsample",
            format!(
                "group size is not integral for in={in_channels} out={out_channels} stride={stride:?}"
            ),
        ));
    }
    let group_size = in_channels * prod / out_channels;
    let mut x = input.clone();
    if stride[0] == 2 {
        let first = x.slice(
            &[0, 0, 0, 0, 0],
            &[x.shape()[0], 1, x.shape()[2], x.shape()[3], x.shape()[4]],
            &[1, 1, 1, 1, 1],
            stream,
        )?;
        x = Array::concatenate_axis(&[first, x], 1, stream)?;
    }
    let skip = space_to_depth_3d(&x, stride[1], stride[0], stream)?;
    let skip_shape = skip.shape();
    let skip = skip
        .reshape(
            &[
                skip_shape[0],
                skip_shape[1],
                skip_shape[2],
                skip_shape[3],
                out_channels,
                group_size,
            ],
            stream,
        )?
        .mean_axes(&[5], false, stream)?;
    let conv_out = out_channels / prod;
    let _ = conv_out;
    let packed = conv3d(
        weights,
        &format!("{prefix}.conv"),
        &x,
        [3, 3, 3],
        [1, 1, 1],
        true,
        spatial_padding_mode,
        stream,
    )?;
    let packed = space_to_depth_3d(&packed, stride[1], stride[0], stream)?;
    Array::add(&packed, &skip, stream)
}

#[cfg(test)]
mod tests {
    use super::{depth_to_space_3d, space_to_depth_3d};

    #[test]
    fn space_and_depth_round_trip_preserves_shape() {
        let stream = danmo_mlx::Stream::cpu().unwrap();
        let input = danmo_mlx::Array::from_f32_slice(
            &(0..2 * 4 * 4 * 4 * 3).map(|v| v as f32).collect::<Vec<_>>(),
            &[2, 4, 4, 4, 3],
        )
        .unwrap();
        let packed = space_to_depth_3d(&input, 2, 2, &stream).unwrap();
        assert_eq!(packed.shape(), &[2, 2, 2, 2, 24]);
        let restored = depth_to_space_3d(&packed, 2, 2, &stream).unwrap();
        assert_eq!(restored.shape(), input.shape());
        let left = input.to_vec_f32(&stream).unwrap();
        let right = restored.to_vec_f32(&stream).unwrap();
        assert_eq!(left, right);
    }
}
