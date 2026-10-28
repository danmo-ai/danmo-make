//! Qwen-Image 3D causal decoder, first frame only.
//!
//! Names are the diffusers checkpoint names (`decoder.conv_in.weight`,
//! `*.norm1.gamma`, `upsamplers.0.resample.1.weight`). Main transposes those
//! into MLX `conv3d` / `norm.weight`. This graph reads the file names.
//!
//! Decode of one image is diffusers `_decode` frame 0: every causal cache slot
//! starts empty, so each conv uses a left time pad of `2 * padding` and no
//! cached frames. `upsample3d` on that frame writes `"Rep"` and does not run
//! `time_conv`. The tensor is still required. A later frame, which would run
//! `time_conv` and double time, is not built. Main's MLX resample never calls
//! `time_conv`; that omission is not copied.

use crate::runtime::{flash_attend, shape, GraphBuilder, RuntimeError, Tensor};
use std::collections::HashMap;

pub const LATENTS_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
pub const LATENTS_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.916,
];

const NORM_EPS: f32 = 1e-12;

#[derive(Clone, Copy, PartialEq, Eq)]
enum UpMode {
    Spatial,
    /// First frame keeps time and only doubles width and height.
    SpatialAfterTime,
}

#[derive(Debug, Clone, Copy)]
pub struct DecoderSpec {
    pub z_dim: i64,
    pub base_dim: i64,
    pub dim_mult: [i64; 4],
    pub num_res_blocks: i64,
    pub temperal_upsample: [bool; 3],
}

impl DecoderSpec {
    /// `base_dim=96`, `z_dim=16`, `dim_mult=[1,2,4,4]`, two residual blocks, temporal upsample reversed.
    pub fn published() -> Self {
        Self {
            z_dim: 16,
            base_dim: 96,
            dim_mult: [1, 2, 4, 4],
            num_res_blocks: 2,
            temperal_upsample: [true, true, false],
        }
    }

    /// Same block counts as the published decoder, with channels cut down for a CPU graph.
    pub fn smoke() -> Self {
        Self {
            z_dim: 4,
            base_dim: 4,
            dim_mult: [1, 2, 4, 4],
            num_res_blocks: 2,
            temperal_upsample: [true, true, false],
        }
    }
}

#[derive(Clone)]
pub struct NamedShape {
    pub name: String,
    pub ne: [i64; 4],
    pub rank: u8,
}

/// `latents * std + mean` on `[W, H, T, 16]`.
pub fn scale_latents(
    latent: &[f32],
    width: usize,
    height: usize,
    frames: usize,
) -> Result<Vec<f32>, RuntimeError> {
    let channels = LATENTS_MEAN.len();
    if width == 0
        || height == 0
        || frames == 0
        || latent.len() != width * height * frames * channels
    {
        return Err(RuntimeError::Message(format!(
            "qwen latent scale expected {width}x{height}x{frames}x{channels}, got {} values",
            latent.len()
        )));
    }
    let mut out = latent.to_vec();
    for frame in 0..frames {
        for y in 0..height {
            for x in 0..width {
                for channel in 0..channels {
                    let index = x + width * (y + height * (frame + frames * channel));
                    out[index] = latent[index] * LATENTS_STD[channel] + LATENTS_MEAN[channel];
                }
            }
        }
    }
    Ok(out)
}

pub fn decoder_dims(base: i64, mult: [i64; 4]) -> [i64; 5] {
    let rev = [mult[3], mult[2], mult[1], mult[0]];
    [
        base * mult[3],
        base * rev[0],
        base * rev[1],
        base * rev[2],
        base * rev[3],
    ]
}

/// Checkpoint names, including `time_conv`, which frame 0 does not read.
pub fn required_names(spec: DecoderSpec) -> Vec<String> {
    tensors(spec, true)
        .into_iter()
        .map(|row| row.name)
        .collect()
}

fn frame0_tensors(spec: DecoderSpec) -> Vec<NamedShape> {
    tensors(spec, false)
}

fn tensors(spec: DecoderSpec, include_time: bool) -> Vec<NamedShape> {
    let mut out = Vec::new();
    let dims = decoder_dims(spec.base_dim, spec.dim_mult);
    push_conv3(&mut out, "post_quant_conv", 1, 1, 1, spec.z_dim, spec.z_dim);
    push_conv3(&mut out, "decoder.conv_in", 3, 3, 3, spec.z_dim, dims[0]);
    for index in 0..2 {
        push_res(
            &mut out,
            &format!("decoder.mid_block.resnets.{index}"),
            dims[0],
            dims[0],
        );
    }
    push_gamma(
        &mut out,
        "decoder.mid_block.attentions.0.norm.gamma",
        dims[0],
    );
    push_conv2(
        &mut out,
        "decoder.mid_block.attentions.0.to_qkv",
        1,
        1,
        dims[0],
        dims[0] * 3,
    );
    push_conv2(
        &mut out,
        "decoder.mid_block.attentions.0.proj",
        1,
        1,
        dims[0],
        dims[0],
    );
    let blocks = up_blocks(spec);
    for (block, (in_dim, out_dim, mode)) in blocks.iter().copied().enumerate() {
        let mut current = in_dim;
        for res in 0..(spec.num_res_blocks + 1) {
            push_res(
                &mut out,
                &format!("decoder.up_blocks.{block}.resnets.{res}"),
                current,
                out_dim,
            );
            current = out_dim;
        }
        if let Some(mode) = mode {
            let prefix = format!("decoder.up_blocks.{block}.upsamplers.0");
            push_conv2(
                &mut out,
                &format!("{prefix}.resample.1"),
                3,
                3,
                out_dim,
                out_dim / 2,
            );
            if include_time && mode == UpMode::SpatialAfterTime {
                push_conv3(
                    &mut out,
                    &format!("{prefix}.time_conv"),
                    1,
                    1,
                    3,
                    out_dim,
                    out_dim * 2,
                );
            }
        }
    }
    push_gamma(&mut out, "decoder.norm_out.gamma", dims[4]);
    push_conv3(&mut out, "decoder.conv_out", 3, 3, 3, dims[4], 3);
    out
}

fn up_blocks(spec: DecoderSpec) -> [(i64, i64, Option<UpMode>); 4] {
    let dims = decoder_dims(spec.base_dim, spec.dim_mult);
    let mut rows = [(0, 0, None); 4];
    for i in 0..4 {
        let mut in_dim = dims[i];
        if i > 0 {
            in_dim /= 2;
        }
        let mode = if i == 3 {
            None
        } else if spec.temperal_upsample[i] {
            Some(UpMode::SpatialAfterTime)
        } else {
            Some(UpMode::Spatial)
        };
        rows[i] = (in_dim, dims[i + 1], mode);
    }
    rows
}

fn push_conv3(
    out: &mut Vec<NamedShape>,
    prefix: &str,
    kw: i64,
    kh: i64,
    kd: i64,
    ic: i64,
    oc: i64,
) {
    out.push(NamedShape {
        name: format!("{prefix}.weight"),
        ne: [kw, kh, kd, ic * oc],
        rank: 4,
    });
    out.push(NamedShape {
        name: format!("{prefix}.bias"),
        ne: [oc, 1, 1, 1],
        rank: 1,
    });
}

fn push_conv2(out: &mut Vec<NamedShape>, prefix: &str, kw: i64, kh: i64, ic: i64, oc: i64) {
    out.push(NamedShape {
        name: format!("{prefix}.weight"),
        ne: [kw, kh, ic, oc],
        rank: 4,
    });
    out.push(NamedShape {
        name: format!("{prefix}.bias"),
        ne: [oc, 1, 1, 1],
        rank: 1,
    });
}

fn push_res(out: &mut Vec<NamedShape>, prefix: &str, in_dim: i64, out_dim: i64) {
    push_gamma(out, &format!("{prefix}.norm1.gamma"), in_dim);
    push_conv3(out, &format!("{prefix}.conv1"), 3, 3, 3, in_dim, out_dim);
    push_gamma(out, &format!("{prefix}.norm2.gamma"), out_dim);
    push_conv3(out, &format!("{prefix}.conv2"), 3, 3, 3, out_dim, out_dim);
    if in_dim != out_dim {
        push_conv3(
            out,
            &format!("{prefix}.conv_shortcut"),
            1,
            1,
            1,
            in_dim,
            out_dim,
        );
    }
}

fn push_gamma(out: &mut Vec<NamedShape>, name: &str, channels: i64) {
    out.push(NamedShape {
        name: name.to_string(),
        ne: [channels, 1, 1, 1],
        rank: 1,
    });
}

pub struct VaeDecode {
    graph: crate::runtime::ReusedGraph,
    width: i64,
    height: i64,
    z_dim: i64,
}

impl VaeDecode {
    pub fn build(
        ctx: &crate::runtime::GgmlContext,
        spec: DecoderSpec,
        width: i64,
        height: i64,
    ) -> Result<Self, RuntimeError> {
        validate(spec, width, height)?;
        let mut builder = ctx.builder(64 * 1024 * 1024)?;
        let mut weights = HashMap::new();
        for row in frame0_tensors(spec) {
            let tensor = match row.rank {
                1 => builder.f32_1d(&row.name, row.ne[0], true),
                4 => builder.f32_4d(&row.name, row.ne[0], row.ne[1], row.ne[2], row.ne[3], true),
                _ => {
                    return Err(RuntimeError::Message(format!(
                        "vae tensor {} has rank {}",
                        row.name, row.rank
                    )))
                }
            };
            if tensor.is_null() {
                return Err(RuntimeError::Init);
            }
            weights.insert(row.name, tensor);
        }
        let latent = nn(builder.f32_4d("latent", width, height, 1, spec.z_dim, true))?;
        let output = decode_graph(&builder, &weights, spec, latent)?;
        let graph = builder.finish(output)?;
        Ok(Self {
            graph,
            width,
            height,
            z_dim: spec.z_dim,
        })
    }

    pub fn load_f32(&mut self, name: &str, data: &[f32]) -> Result<(), RuntimeError> {
        self.graph.set_f32(name, data)
    }

    pub fn decode(&mut self, latent: &[f32]) -> Result<Vec<f32>, RuntimeError> {
        let expect = (self.width * self.height * self.z_dim) as usize;
        if latent.len() != expect {
            return Err(RuntimeError::Message(format!(
                "qwen vae latent has {} values, graph expects {expect}",
                latent.len()
            )));
        }
        self.graph.set_f32("latent", latent)?;
        self.graph.compute()
    }
}

fn validate(spec: DecoderSpec, width: i64, height: i64) -> Result<(), RuntimeError> {
    if width < 1 || height < 1 || spec.z_dim < 1 || spec.base_dim < 1 || spec.num_res_blocks < 0 {
        return Err(RuntimeError::Message(
            "qwen vae width, height, or channel count is empty".into(),
        ));
    }
    let dims = decoder_dims(spec.base_dim, spec.dim_mult);
    if dims.iter().any(|dim| *dim < 2 || dim % 2 != 0) {
        return Err(RuntimeError::Message(format!(
            "qwen vae channels {dims:?} are not even"
        )));
    }
    for (in_dim, out_dim, mode) in up_blocks(spec) {
        if in_dim < 1 || out_dim < 2 || out_dim % 2 != 0 {
            return Err(RuntimeError::Message(format!(
                "qwen vae up block {in_dim}->{out_dim} cannot halve channels"
            )));
        }
        let _ = mode;
    }
    Ok(())
}

fn decode_graph(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    spec: DecoderSpec,
    latent: Tensor,
) -> Result<Tensor, RuntimeError> {
    let dims = decoder_dims(spec.base_dim, spec.dim_mult);
    let mut x = causal(
        b,
        weights,
        latent,
        "post_quant_conv",
        spec.z_dim,
        spec.z_dim,
        1,
        1,
        1,
        0,
        0,
        0,
    )?;
    x = causal(
        b,
        weights,
        x,
        "decoder.conv_in",
        spec.z_dim,
        dims[0],
        3,
        3,
        3,
        1,
        1,
        1,
    )?;
    x = res_block(
        b,
        weights,
        x,
        "decoder.mid_block.resnets.0",
        dims[0],
        dims[0],
    )?;
    x = attention(b, weights, x, dims[0])?;
    x = res_block(
        b,
        weights,
        x,
        "decoder.mid_block.resnets.1",
        dims[0],
        dims[0],
    )?;
    for (block, (in_dim, out_dim, mode)) in up_blocks(spec).iter().copied().enumerate() {
        let mut current = in_dim;
        for res in 0..(spec.num_res_blocks + 1) {
            let prefix = format!("decoder.up_blocks.{block}.resnets.{res}");
            x = res_block(b, weights, x, &prefix, current, out_dim)?;
            current = out_dim;
        }
        if mode.is_some() {
            let prefix = format!("decoder.up_blocks.{block}.upsamplers.0");
            x = upsample(b, weights, x, &prefix, out_dim)?;
        }
    }
    x = channel_norm(b, x, wt(weights, "decoder.norm_out.gamma")?, dims[4])?;
    x = nn(b.silu(x))?;
    causal(
        b,
        weights,
        x,
        "decoder.conv_out",
        dims[4],
        3,
        3,
        3,
        3,
        1,
        1,
        1,
    )
}

fn res_block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    in_dim: i64,
    out_dim: i64,
) -> Result<Tensor, RuntimeError> {
    let mut h = channel_norm(b, x, wt(weights, &format!("{prefix}.norm1.gamma"))?, in_dim)?;
    h = nn(b.silu(h))?;
    h = causal(
        b,
        weights,
        h,
        &format!("{prefix}.conv1"),
        in_dim,
        out_dim,
        3,
        3,
        3,
        1,
        1,
        1,
    )?;
    h = channel_norm(
        b,
        h,
        wt(weights, &format!("{prefix}.norm2.gamma"))?,
        out_dim,
    )?;
    h = nn(b.silu(h))?;
    h = causal(
        b,
        weights,
        h,
        &format!("{prefix}.conv2"),
        out_dim,
        out_dim,
        3,
        3,
        3,
        1,
        1,
        1,
    )?;
    let skip = if in_dim != out_dim {
        causal(
            b,
            weights,
            x,
            &format!("{prefix}.conv_shortcut"),
            in_dim,
            out_dim,
            1,
            1,
            1,
            0,
            0,
            0,
        )?
    } else {
        x
    };
    nn(b.add(h, skip))
}

fn attention(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    dim: i64,
) -> Result<Tensor, RuntimeError> {
    let identity = x;
    let dims = shape(x);
    if dims[2] != 1 {
        return Err(RuntimeError::Message(
            "qwen vae attention got more than one frame".into(),
        ));
    }
    let x = channel_norm(
        b,
        x,
        wt(weights, "decoder.mid_block.attentions.0.norm.gamma")?,
        dim,
    )?;
    let nchw = nn(b.cont(b.permute(x, 0, 1, 3, 2)))?;
    let qkv = nn(b.conv2d_direct(
        wt(weights, "decoder.mid_block.attentions.0.to_qkv.weight")?,
        nchw,
        1,
        1,
        0,
        0,
        1,
        1,
    ))?;
    let qkv = add_bias_ne2(
        b,
        qkv,
        wt(weights, "decoder.mid_block.attentions.0.to_qkv.bias")?,
        dim * 3,
    )?;
    // ggml_permute stores source axis i at destination slot axis_i.
    // [W, H, 3C, T] → [3C, W, H, T] so one token is a contiguous Q|K|V row.
    let flat = nn(b.cont(b.permute(qkv, 1, 2, 0, 3)))?;
    let seq = dims[0] * dims[1];
    let flat = nn(b.reshape_2d(flat, dim * 3, seq))?;
    let elem = std::mem::size_of::<f32>();
    let stride = (dim * 3) as usize * elem;
    let q = nn(b.cont(b.view_2d(flat, dim, seq, stride, 0)))?;
    let k = nn(b.cont(b.view_2d(flat, dim, seq, stride, dim as usize * elem)))?;
    let v = nn(b.cont(b.view_2d(flat, dim, seq, stride, (dim * 2) as usize * elem)))?;
    let q = nn(b.reshape_3d(q, dim, 1, seq))?;
    let k = nn(b.reshape_3d(k, dim, 1, seq))?;
    let v = nn(b.reshape_3d(v, dim, 1, seq))?;
    let mixed = flash_attend(b, q, k, v, std::ptr::null_mut())?;
    let mixed = nn(b.reshape_3d(mixed, dim, dims[0], dims[1]))?;
    // [C, W, H, 1] → [W, H, C, 1] for conv2d_direct.
    let mixed = nn(b.cont(b.permute(mixed, 2, 0, 1, 3)))?;
    let proj = nn(b.conv2d_direct(
        wt(weights, "decoder.mid_block.attentions.0.proj.weight")?,
        mixed,
        1,
        1,
        0,
        0,
        1,
        1,
    ))?;
    let proj = add_bias_ne2(
        b,
        proj,
        wt(weights, "decoder.mid_block.attentions.0.proj.bias")?,
        dim,
    )?;
    let proj = nn(b.cont(b.permute(proj, 0, 1, 3, 2)))?;
    nn(b.add(proj, identity))
}

fn upsample(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    channels: i64,
) -> Result<Tensor, RuntimeError> {
    if shape(x)[2] != 1 {
        return Err(RuntimeError::Message(
            "multi-frame causal cache is not built".into(),
        ));
    }
    let nchw = nn(b.cont(b.permute(x, 0, 1, 3, 2)))?;
    let up = nn(b.upscale_nearest(nchw))?;
    let y = nn(b.conv2d_direct(
        wt(weights, &format!("{prefix}.resample.1.weight"))?,
        up,
        1,
        1,
        1,
        1,
        1,
        1,
    ))?;
    let y = add_bias_ne2(
        b,
        y,
        wt(weights, &format!("{prefix}.resample.1.bias"))?,
        channels / 2,
    )?;
    nn(b.cont(b.permute(y, 0, 1, 3, 2)))
}

fn causal(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    ic: i64,
    oc: i64,
    kw: i64,
    kh: i64,
    kd: i64,
    pad_w: i32,
    pad_h: i32,
    pad_t: i32,
) -> Result<Tensor, RuntimeError> {
    let _ = (kw, kh, kd);
    let padded = nn(b.pad_ext(x, pad_w, pad_w, pad_h, pad_h, 2 * pad_t, 0, 0, 0))?;
    let y = nn(b.conv3d_direct(
        wt(weights, &format!("{prefix}.weight"))?,
        padded,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        ic as i32,
        1,
        oc as i32,
    ))?;
    add_bias_ne3(b, y, wt(weights, &format!("{prefix}.bias"))?, oc)
}

fn channel_norm(
    b: &GraphBuilder,
    x: Tensor,
    gamma: Tensor,
    channels: i64,
) -> Result<Tensor, RuntimeError> {
    // [W, H, T, C] → [C, W, H, T]. sum_rows then reduces the channel axis.
    let x = nn(b.cont(b.permute(x, 1, 2, 3, 0)))?;
    let sum_sq = nn(b.sum_rows(nn(b.sqr(x))?))?;
    let denom = nn(b.clamp(nn(b.sqrt(sum_sq))?, NORM_EPS, 1.0e30))?;
    let normed = nn(b.div(x, denom))?;
    let scaled = nn(b.scale(normed, (channels as f32).sqrt()))?;
    let gamma = nn(b.reshape_4d(gamma, channels, 1, 1, 1))?;
    let y = nn(b.mul(scaled, gamma))?;
    // [C, W, H, T] → [W, H, T, C].
    nn(b.cont(b.permute(y, 3, 0, 1, 2)))
}

fn add_bias_ne3(b: &GraphBuilder, y: Tensor, bias: Tensor, n: i64) -> Result<Tensor, RuntimeError> {
    let bias = nn(b.reshape_4d(bias, 1, 1, 1, n))?;
    nn(b.add(y, nn(b.repeat(bias, y))?))
}

fn add_bias_ne2(b: &GraphBuilder, y: Tensor, bias: Tensor, n: i64) -> Result<Tensor, RuntimeError> {
    let bias = nn(b.reshape_4d(bias, 1, 1, n, 1))?;
    nn(b.add(y, nn(b.repeat(bias, y))?))
}

fn wt(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor, RuntimeError> {
    weights
        .get(name)
        .copied()
        .filter(|tensor| !tensor.is_null())
        .ok_or_else(|| RuntimeError::Message(format!("graph is missing {name}")))
}

fn nn(tensor: Tensor) -> Result<Tensor, RuntimeError> {
    if tensor.is_null() {
        Err(RuntimeError::Init)
    } else {
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        causal, channel_norm, decoder_dims, frame0_tensors, required_names, scale_latents,
        DecoderSpec, VaeDecode, LATENTS_MEAN, LATENTS_STD,
    };
    use crate::runtime::{BackendKind, GgmlContext, RuntimeError};
    use std::collections::HashMap;

    fn tap(name: &str, index: usize) -> f32 {
        let mut hash = 2166136261u32;
        for byte in name.bytes() {
            hash = hash.wrapping_mul(16777619) ^ u32::from(byte);
        }
        hash = hash.wrapping_mul(16777619) ^ index as u32;
        ((hash % 5) as f32 - 2.0) * 0.02
    }

    #[test]
    fn published_schedule_requires_time_conv_on_the_first_two_blocks() {
        let spec = DecoderSpec::published();
        assert_eq!(
            decoder_dims(spec.base_dim, spec.dim_mult),
            [384, 384, 384, 192, 96]
        );
        let names = required_names(spec);
        assert!(names
            .iter()
            .any(|name| name == "decoder.up_blocks.0.resnets.2.conv1.weight"));
        assert!(names
            .iter()
            .any(|name| name == "decoder.up_blocks.0.upsamplers.0.time_conv.weight"));
        assert!(names
            .iter()
            .any(|name| name == "decoder.up_blocks.1.upsamplers.0.time_conv.bias"));
        assert!(names
            .iter()
            .any(|name| name == "decoder.up_blocks.1.resnets.0.conv_shortcut.weight"));
        assert!(names
            .iter()
            .any(|name| name == "decoder.mid_block.attentions.0.norm.gamma"));
        assert!(names.iter().any(|name| name == "post_quant_conv.weight"));
        assert!(names
            .iter()
            .all(|name| !name.contains("up_blocks.3.upsamplers")));
        assert!(names
            .iter()
            .all(|name| !name.contains("up_blocks.0.resnets.0.conv_shortcut")));
        assert!(names
            .iter()
            .all(|name| !name.contains("up_blocks.2.upsamplers.0.time_conv")));
        assert_eq!(LATENTS_MEAN.len(), 16);
        assert_eq!(LATENTS_STD.len(), 16);
    }

    #[test]
    fn scale_is_std_then_mean_on_the_channel_axis() {
        let mut latent = vec![0f32; 16];
        latent[0] = 1.0;
        let scaled = scale_latents(&latent, 1, 1, 1).unwrap();
        assert!((scaled[0] - (LATENTS_STD[0] + LATENTS_MEAN[0])).abs() < 1e-6);
        assert!((scaled[15] - LATENTS_MEAN[15]).abs() < 1e-6);
    }

    #[test]
    fn channel_norm_uses_the_vector_length_floor() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut builder = ctx.builder(4 * 1024 * 1024).unwrap();
        let x = builder.f32_4d("x", 1, 1, 1, 2, true);
        let gamma = builder.f32_1d("gamma", 2, true);
        let y = channel_norm(&builder, x, gamma, 2).unwrap();
        let mut graph = builder.finish(y).unwrap();
        graph.set_f32("x", &[3.0, 4.0]).unwrap();
        graph.set_f32("gamma", &[1.0, 1.0]).unwrap();
        let got = graph.compute().unwrap();
        let scale = 2f32.sqrt();
        assert!((got[0] - 0.6 * scale).abs() < 1e-5, "{}", got[0]);
        assert!((got[1] - 0.8 * scale).abs() < 1e-5, "{}", got[1]);
    }

    #[test]
    fn causal_conv_reads_the_current_frame_with_the_last_time_tap() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut builder = ctx.builder(8 * 1024 * 1024).unwrap();
        let x = builder.f32_4d("x", 1, 1, 1, 1, true);
        let weight = builder.f32_4d("decoder.conv_in.weight", 3, 3, 3, 1, true);
        let bias = builder.f32_1d("decoder.conv_in.bias", 1, true);
        let mut weights = HashMap::new();
        weights.insert("decoder.conv_in.weight".into(), weight);
        weights.insert("decoder.conv_in.bias".into(), bias);
        let y = causal(
            &builder,
            &weights,
            x,
            "decoder.conv_in",
            1,
            1,
            3,
            3,
            3,
            1,
            1,
            1,
        )
        .unwrap();
        let mut graph = builder.finish(y).unwrap();
        let mut kernel = vec![0f32; 27];
        kernel[22] = 5.0;
        graph.set_f32("x", &[4.0]).unwrap();
        graph.set_f32("decoder.conv_in.weight", &kernel).unwrap();
        graph.set_f32("decoder.conv_in.bias", &[0.0]).unwrap();
        let got = graph.compute().unwrap();
        assert_eq!(got.len(), 1);
        assert!((got[0] - 20.0).abs() < 1e-4, "{}", got[0]);
    }

    #[test]
    fn frame0_graph_matches_the_host_and_does_not_read_time_conv() {
        let spec = DecoderSpec::smoke();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut decode = VaeDecode::build(&ctx, spec, 2, 2).unwrap();
        let missing = decode.load_f32("decoder.up_blocks.0.upsamplers.0.time_conv.weight", &[0.0]);
        assert!(
            matches!(missing, Err(RuntimeError::MissingInput(_))),
            "{missing:?}"
        );
        let mut weights = HashMap::new();
        for row in frame0_tensors(spec) {
            let count = if row.rank == 1 {
                row.ne[0] as usize
            } else {
                row.ne.iter().product::<i64>() as usize
            };
            let data: Vec<f32> = (0..count)
                .map(|index| {
                    if row.name.ends_with(".gamma") {
                        1.0
                    } else {
                        tap(&row.name, index)
                    }
                })
                .collect();
            decode.load_f32(&row.name, &data).unwrap();
            weights.insert(row.name, data);
        }
        let latent: Vec<f32> = (0..2 * 2 * spec.z_dim as usize)
            .map(|index| ((index % 5) as f32 - 2.0) * 0.1)
            .collect();
        let got = decode.decode(&latent).unwrap();
        let expect = host_decode(&weights, spec, &latent, 2, 2);
        assert_eq!(got.len(), expect.len());
        assert_eq!(got.len(), 16 * 16 * 3);
        let mut worst = 0f32;
        for (got, want) in got.iter().zip(&expect) {
            worst = worst.max((got - want).abs());
        }
        assert!(worst < 2e-3, "vae decode max abs {worst}");
    }

    #[derive(Clone)]
    struct Vol {
        w: i64,
        h: i64,
        t: i64,
        c: i64,
        data: Vec<f32>,
    }

    impl Vol {
        fn new(w: i64, h: i64, t: i64, c: i64, data: Vec<f32>) -> Self {
            Self { w, h, t, c, data }
        }

        fn zeros(w: i64, h: i64, t: i64, c: i64) -> Self {
            Self::new(w, h, t, c, vec![0.0; (w * h * t * c) as usize])
        }

        fn get(&self, x: i64, y: i64, z: i64, c: i64) -> f32 {
            self.data[(x + self.w * (y + self.h * (z + self.t * c))) as usize]
        }

        fn set(&mut self, x: i64, y: i64, z: i64, c: i64, value: f32) {
            let index = (x + self.w * (y + self.h * (z + self.t * c))) as usize;
            self.data[index] = value;
        }
    }

    fn host_decode(
        weights: &HashMap<String, Vec<f32>>,
        spec: DecoderSpec,
        latent: &[f32],
        width: i64,
        height: i64,
    ) -> Vec<f32> {
        let dims = decoder_dims(spec.base_dim, spec.dim_mult);
        let mut x = Vol::new(width, height, 1, spec.z_dim, latent.to_vec());
        x = host_conv3(weights, &x, "post_quant_conv", spec.z_dim, 1, 1, 1, 0, 0, 0);
        x = host_conv3(weights, &x, "decoder.conv_in", dims[0], 3, 3, 3, 1, 1, 1);
        x = host_res(weights, &x, "decoder.mid_block.resnets.0", dims[0], dims[0]);
        x = host_attn(weights, &x, dims[0]);
        x = host_res(weights, &x, "decoder.mid_block.resnets.1", dims[0], dims[0]);
        let blocks = [
            (dims[0], dims[1], true),
            (dims[1] / 2, dims[2], true),
            (dims[2] / 2, dims[3], false),
            (dims[3] / 2, dims[4], false),
        ];
        for (block, (in_dim, out_dim, upsample)) in blocks.into_iter().enumerate() {
            let mut current = in_dim;
            for res in 0..(spec.num_res_blocks + 1) {
                x = host_res(
                    weights,
                    &x,
                    &format!("decoder.up_blocks.{block}.resnets.{res}"),
                    current,
                    out_dim,
                );
                current = out_dim;
            }
            if block < 3 {
                x = host_up(
                    weights,
                    &x,
                    &format!("decoder.up_blocks.{block}.upsamplers.0"),
                    out_dim,
                );
                let _ = upsample;
            }
        }
        x = host_norm(&x, &weights["decoder.norm_out.gamma"]);
        for value in &mut x.data {
            *value = silu(*value);
        }
        x = host_conv3(weights, &x, "decoder.conv_out", 3, 3, 3, 3, 1, 1, 1);
        x.data
    }

    fn host_res(
        weights: &HashMap<String, Vec<f32>>,
        x: &Vol,
        prefix: &str,
        in_dim: i64,
        out_dim: i64,
    ) -> Vol {
        let mut h = host_norm(x, &weights[&format!("{prefix}.norm1.gamma")]);
        for value in &mut h.data {
            *value = silu(*value);
        }
        h = host_conv3(
            weights,
            &h,
            &format!("{prefix}.conv1"),
            out_dim,
            3,
            3,
            3,
            1,
            1,
            1,
        );
        h = host_norm(&h, &weights[&format!("{prefix}.norm2.gamma")]);
        for value in &mut h.data {
            *value = silu(*value);
        }
        h = host_conv3(
            weights,
            &h,
            &format!("{prefix}.conv2"),
            out_dim,
            3,
            3,
            3,
            1,
            1,
            1,
        );
        let skip = if in_dim != out_dim {
            host_conv3(
                weights,
                x,
                &format!("{prefix}.conv_shortcut"),
                out_dim,
                1,
                1,
                1,
                0,
                0,
                0,
            )
        } else {
            x.clone()
        };
        for i in 0..h.data.len() {
            h.data[i] += skip.data[i];
        }
        h
    }

    fn host_attn(weights: &HashMap<String, Vec<f32>>, x: &Vol, dim: i64) -> Vol {
        let identity = x.clone();
        let normed = host_norm(x, &weights["decoder.mid_block.attentions.0.norm.gamma"]);
        let qkv = host_conv2(
            weights,
            &normed,
            "decoder.mid_block.attentions.0.to_qkv",
            dim * 3,
            1,
            1,
            0,
        );
        let seq = (x.w * x.h) as usize;
        let dim_us = dim as usize;
        let scale = 1.0 / (dim as f32).sqrt();
        let mut mixed = Vol::zeros(x.w, x.h, 1, dim);
        for y in 0..x.h {
            for ox in 0..x.w {
                let q_tok = (ox + x.w * y) as usize;
                let mut scores = vec![0f32; seq];
                let mut max_score = f32::NEG_INFINITY;
                for k_tok in 0..seq {
                    let mut dot = 0f32;
                    for d in 0..dim_us {
                        let q = qkv.get(
                            (q_tok % x.w as usize) as i64,
                            (q_tok / x.w as usize) as i64,
                            0,
                            d as i64,
                        );
                        let k = qkv.get(
                            (k_tok % x.w as usize) as i64,
                            (k_tok / x.w as usize) as i64,
                            0,
                            (dim_us + d) as i64,
                        );
                        let _ = q_tok;
                        dot += q * k;
                    }
                    scores[k_tok] = dot * scale;
                    max_score = max_score.max(scores[k_tok]);
                }
                let mut denom = 0f32;
                for score in &mut scores {
                    *score = (*score - max_score).exp();
                    denom += *score;
                }
                for d in 0..dim_us {
                    let mut acc = 0f32;
                    for k_tok in 0..seq {
                        let v = qkv.get(
                            (k_tok % x.w as usize) as i64,
                            (k_tok / x.w as usize) as i64,
                            0,
                            (2 * dim_us + d) as i64,
                        );
                        acc += scores[k_tok] / denom * v;
                    }
                    mixed.set(ox, y, 0, d as i64, acc);
                }
            }
        }
        let proj = host_conv2(
            weights,
            &mixed,
            "decoder.mid_block.attentions.0.proj",
            dim,
            1,
            1,
            0,
        );
        let mut out = proj;
        for i in 0..out.data.len() {
            out.data[i] += identity.data[i];
        }
        out
    }

    fn host_up(weights: &HashMap<String, Vec<f32>>, x: &Vol, prefix: &str, channels: i64) -> Vol {
        let mut up = Vol::zeros(x.w * 2, x.h * 2, x.t, x.c);
        for z in 0..x.t {
            for y in 0..up.h {
                for ox in 0..up.w {
                    for c in 0..x.c {
                        up.set(ox, y, z, c, x.get(ox / 2, y / 2, z, c));
                    }
                }
            }
        }
        host_conv2(
            weights,
            &up,
            &format!("{prefix}.resample.1"),
            channels / 2,
            3,
            3,
            1,
        )
    }

    fn host_norm(x: &Vol, gamma: &[f32]) -> Vol {
        let mut out = x.clone();
        let scale = (x.c as f32).sqrt();
        for z in 0..x.t {
            for y in 0..x.h {
                for ox in 0..x.w {
                    let mut sum = 0f32;
                    for c in 0..x.c {
                        let value = x.get(ox, y, z, c);
                        sum += value * value;
                    }
                    let denom = sum.sqrt().max(1e-12);
                    for c in 0..x.c {
                        out.set(
                            ox,
                            y,
                            z,
                            c,
                            x.get(ox, y, z, c) / denom * scale * gamma[c as usize],
                        );
                    }
                }
            }
        }
        out
    }

    fn host_conv3(
        weights: &HashMap<String, Vec<f32>>,
        x: &Vol,
        prefix: &str,
        oc: i64,
        kw: i64,
        kh: i64,
        kd: i64,
        pad_w: i64,
        pad_h: i64,
        pad_t: i64,
    ) -> Vol {
        let kernel = &weights[&format!("{prefix}.weight")];
        let bias = &weights[&format!("{prefix}.bias")];
        let ow = x.w + 2 * pad_w - kw + 1;
        let oh = x.h + 2 * pad_h - kh + 1;
        let ot = x.t + 2 * pad_t - kd + 1;
        let mut out = Vol::zeros(ow, oh, ot, oc);
        for o in 0..oc {
            for z in 0..ot {
                for y in 0..oh {
                    for ox in 0..ow {
                        let mut acc = bias[o as usize];
                        for ic in 0..x.c {
                            for kz in 0..kd {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let ix = ox + kx - pad_w;
                                        let iy = y + ky - pad_h;
                                        let iz = z + kz - 2 * pad_t;
                                        if ix < 0
                                            || iy < 0
                                            || iz < 0
                                            || ix >= x.w
                                            || iy >= x.h
                                            || iz >= x.t
                                        {
                                            continue;
                                        }
                                        let kindex = (kx
                                            + kw * (ky + kh * (kz + kd * (ic + x.c * o))))
                                            as usize;
                                        acc += x.get(ix, iy, iz, ic) * kernel[kindex];
                                    }
                                }
                            }
                        }
                        out.set(ox, y, z, o, acc);
                    }
                }
            }
        }
        out
    }

    fn host_conv2(
        weights: &HashMap<String, Vec<f32>>,
        x: &Vol,
        prefix: &str,
        oc: i64,
        kw: i64,
        kh: i64,
        pad: i64,
    ) -> Vol {
        let kernel = &weights[&format!("{prefix}.weight")];
        let bias = &weights[&format!("{prefix}.bias")];
        let ow = x.w + 2 * pad - kw + 1;
        let oh = x.h + 2 * pad - kh + 1;
        let mut out = Vol::zeros(ow, oh, x.t, oc);
        for frame in 0..x.t {
            for o in 0..oc {
                for y in 0..oh {
                    for ox in 0..ow {
                        let mut acc = bias[o as usize];
                        for ic in 0..x.c {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let ix = ox + kx - pad;
                                    let iy = y + ky - pad;
                                    if ix < 0 || iy < 0 || ix >= x.w || iy >= x.h {
                                        continue;
                                    }
                                    let kindex = (kx + kw * (ky + kh * (ic + x.c * o))) as usize;
                                    acc += x.get(ix, iy, frame, ic) * kernel[kindex];
                                }
                            }
                        }
                        out.set(ox, y, frame, o, acc);
                    }
                }
            }
        }
        out
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }
}
