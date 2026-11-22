//! AutoencoderKL decoder.
//!
//! The caller supplies [`VaeSpec`]: latent channels, block widths, resnets per
//! block, group count, scaling, shift, and whether the mid block has attention.
//! Tensor names are `decoder.*`. Conv kernels are ggml `[KW, KH, IC, OC]`, the
//! same bytes as PyTorch contiguous `(OC, IC, KH, KW)`. Attention projections
//! are ggml `[in, out]` and run as a 1×1 conv. Biases and group-norm affine
//! vectors are `[channels]`. Activations stay `[W, H, C, 1]`. Group norm has
//! no affine; the graph multiplies and adds the per-channel vectors afterwards.
//!
//! Output samples are the decoder tensor. The caller maps them to pixels.

use crate::runtime::{flash_attend, shape, GgmlContext, GraphBuilder, RuntimeError, Tensor};

#[cfg(feature = "mlx")]
pub mod mlx;
#[cfg(feature = "mlx")]
pub mod taesd;
use crate::weights::{GgufWeights, WeightError};
use std::collections::{BTreeSet, HashMap};

const NORM_EPS: f32 = 1e-6;
const GRAPH_MEM: usize = 64 * 1024 * 1024;

#[derive(Clone)]
pub struct VaeSpec {
    pub latent_channels: i64,
    pub block_out: Vec<i64>,
    pub resnets: usize,
    pub groups: i32,
    pub scaling: f32,
    pub shift: f32,
    pub attention: bool,
}

enum WeightLayout {
    Kernel,
    Linear,
}

struct Conv {
    weight: String,
    bias: String,
    kw: i64,
    kh: i64,
    ic: i64,
    oc: i64,
    pad: i32,
    layout: WeightLayout,
}

struct Affine {
    weight: String,
    bias: String,
    channels: i64,
}

struct Resnet {
    norm1: Affine,
    conv1: Conv,
    norm2: Affine,
    conv2: Conv,
    shortcut: Option<Conv>,
}

struct Attn {
    norm: Affine,
    q: Conv,
    k: Conv,
    v: Conv,
    out: Conv,
}

struct Up {
    resnets: Vec<Resnet>,
    upsample: Option<Conv>,
}

struct Topology {
    conv_in: Conv,
    mid: [Resnet; 2],
    attn: Option<Attn>,
    ups: Vec<Up>,
    norm_out: Affine,
    conv_out: Conv,
}

enum Kind {
    Vector(i64),
    Linear { inp: i64, out: i64 },
    Kernel { kw: i64, kh: i64, ic: i64, oc: i64 },
}

pub(crate) struct Param {
    pub(crate) name: String,
    pub(crate) shape: Vec<i64>,
    kind: Kind,
}

pub(crate) fn decode_samples(
    ctx: &GgmlContext,
    vae: &GgufWeights,
    latent: &[f32],
    spec: &VaeSpec,
    in_ch: i64,
    latent_w: i64,
    latent_h: i64,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, WeightError> {
    let topo = topology(spec)?;
    let factor = scale_factor(spec.block_out.len())?;
    if latent_w < 1 || latent_h < 1 {
        return Err(WeightError::NotBuilt(format!(
            "vae latent grid {latent_w}x{latent_h} is empty"
        )));
    }
    let out_w = latent_w * factor;
    let out_h = latent_h * factor;
    if width as i64 != out_w || height as i64 != out_h {
        return Err(WeightError::NotBuilt(format!(
            "vae decode is {out_w}x{out_h}, image is {width}x{height}"
        )));
    }
    if in_ch != spec.latent_channels {
        return Err(WeightError::NotBuilt(format!(
            "vae latent has {in_ch} channels, decoder conv_in expects {}",
            spec.latent_channels
        )));
    }
    let pixels = (latent_w * latent_h * in_ch) as usize;
    if latent.len() != pixels {
        return Err(WeightError::Shape {
            name: "latent".into(),
            expected: vec![latent_w, latent_h, in_ch],
            got: vec![latent.len() as i64],
        });
    }
    let params = parameter_list(spec)?;
    let mut loaded = Vec::with_capacity(params.len());
    for param in &params {
        loaded.push((
            param.name.clone(),
            vae.f32_widened(&param.name, &param.shape)?,
        ));
    }
    let mut builder = ctx.builder(GRAPH_MEM)?;
    let declared = declare(&mut builder, &params);
    let output = forward(
        &mut builder,
        spec,
        &topo,
        &declared,
        latent_w,
        latent_h,
        width,
        height,
    )?;
    let mut graph = builder.finish(output)?;
    graph.set_f32("latent", latent)?;
    graph.set_f32("vae_inv_scale", &[1.0 / spec.scaling])?;
    graph.set_f32("vae_shift", &[spec.shift])?;
    for (name, data) in &loaded {
        graph.set_f32(name, data)?;
    }
    let samples = graph.compute()?;
    let expect = (out_w * out_h * 3) as usize;
    if samples.len() != expect {
        return Err(WeightError::Shape {
            name: "vae decode".into(),
            expected: vec![out_w, out_h, 3],
            got: vec![samples.len() as i64],
        });
    }
    if samples.iter().any(|value| !value.is_finite()) {
        return Err(WeightError::NotBuilt(
            "vae decode produced a non-finite sample".into(),
        ));
    }
    Ok(samples)
}

pub(crate) fn rgb_bytes(samples: &[f32], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let plane = w * h;
    let mut rgb = vec![0u8; plane * 3];
    for y in 0..h {
        for x in 0..w {
            for channel in 0..3 {
                let v = samples[x + y * w + channel * plane];
                let byte = ((v + 1.0) * 0.5 * 255.0).round().clamp(0.0, 255.0);
                rgb[(y * w + x) * 3 + channel] = byte as u8;
            }
        }
    }
    rgb
}

fn scale_factor(levels: usize) -> Result<i64, RuntimeError> {
    if !(2..=8).contains(&levels) {
        return Err(RuntimeError::Message(format!(
            "vae block_out_channels needs 2 to 8 entries, got {levels}"
        )));
    }
    Ok(1i64 << (levels - 1))
}

fn topology(spec: &VaeSpec) -> Result<Topology, RuntimeError> {
    scale_factor(spec.block_out.len())?;
    if spec.resnets < 1 {
        return Err(RuntimeError::Message(
            "vae decoder needs at least one resnet per up block".into(),
        ));
    }
    if spec.groups < 1 {
        return Err(RuntimeError::Message(
            "vae group count must be positive".into(),
        ));
    }
    if spec.latent_channels < 1 {
        return Err(RuntimeError::Message(
            "vae latent channels must be positive".into(),
        ));
    }
    if !spec.scaling.is_finite() || spec.scaling == 0.0 {
        return Err(RuntimeError::Message(format!(
            "vae scaling {} is not a finite non-zero value",
            spec.scaling
        )));
    }
    if !spec.shift.is_finite() {
        return Err(RuntimeError::Message(format!(
            "vae shift {} is not finite",
            spec.shift
        )));
    }
    let groups = spec.groups as i64;
    for &channels in &spec.block_out {
        if channels < 1 || channels % groups != 0 {
            return Err(RuntimeError::Message(format!(
                "vae channels {channels} are not divisible by {} groups",
                spec.groups
            )));
        }
    }
    let top = *spec.block_out.last().expect("block list");
    let rev: Vec<i64> = spec.block_out.iter().copied().rev().collect();
    let mut ups = Vec::with_capacity(rev.len());
    for i in 0..rev.len() {
        let out_ch = rev[i];
        let in_ch = if i == 0 { rev[0] } else { rev[i - 1] };
        let mut resnets = Vec::with_capacity(spec.resnets);
        for j in 0..spec.resnets {
            let (rin, rout, shortcut) = if j == 0 {
                (in_ch, out_ch, in_ch != out_ch)
            } else {
                (out_ch, out_ch, false)
            };
            resnets.push(resnet_block(
                &format!("decoder.up_blocks.{i}.resnets.{j}"),
                rin,
                rout,
                shortcut,
            ));
        }
        let upsample = if i + 1 == rev.len() {
            None
        } else {
            Some(conv(
                &format!("decoder.up_blocks.{i}.upsamplers.0.conv"),
                3,
                3,
                out_ch,
                out_ch,
                1,
                WeightLayout::Kernel,
            ))
        };
        ups.push(Up { resnets, upsample });
    }
    Ok(Topology {
        conv_in: conv(
            "decoder.conv_in",
            3,
            3,
            spec.latent_channels,
            top,
            1,
            WeightLayout::Kernel,
        ),
        mid: [
            resnet_block("decoder.mid_block.resnets.0", top, top, false),
            resnet_block("decoder.mid_block.resnets.1", top, top, false),
        ],
        attn: spec
            .attention
            .then(|| attention_block("decoder.mid_block.attentions.0", top)),
        ups,
        norm_out: affine("decoder.conv_norm_out", spec.block_out[0]),
        conv_out: conv(
            "decoder.conv_out",
            3,
            3,
            spec.block_out[0],
            3,
            1,
            WeightLayout::Kernel,
        ),
    })
}

pub(crate) fn parameter_list(spec: &VaeSpec) -> Result<Vec<Param>, RuntimeError> {
    let topo = topology(spec)?;
    let mut params = Vec::new();
    push_conv(&mut params, &topo.conv_in);
    push_resnet(&mut params, &topo.mid[0]);
    if let Some(attn) = &topo.attn {
        push_affine(&mut params, &attn.norm);
        push_conv(&mut params, &attn.q);
        push_conv(&mut params, &attn.k);
        push_conv(&mut params, &attn.v);
        push_conv(&mut params, &attn.out);
    }
    push_resnet(&mut params, &topo.mid[1]);
    for up in &topo.ups {
        for block in &up.resnets {
            push_resnet(&mut params, block);
        }
        if let Some(upsample) = &up.upsample {
            push_conv(&mut params, upsample);
        }
    }
    push_affine(&mut params, &topo.norm_out);
    push_conv(&mut params, &topo.conv_out);
    let mut seen = BTreeSet::new();
    for param in &params {
        if !seen.insert(param.name.clone()) {
            return Err(RuntimeError::Message(format!(
                "vae decoder parameter {} is declared twice",
                param.name
            )));
        }
    }
    Ok(params)
}

// ===== Encoder =====
//
// The mirror of the decoder: `encoder.conv_in`, down blocks with
// `spec.resnets - 1` resnets each and a stride-2 downsampler on every block
// but the last, the mid block, `encoder.conv_norm_out` (some files spell it
// `encoder.norm_out`), and `encoder.conv_out` producing mean and logvar
// channels. The downsampler pads the right and bottom edge by one and then
// runs a 3×3 stride-2 zero-pad conv, matching diffusers `Downsample2D` and
// `main`'s reference encoder. Activations stay `[W, H, C, 1]`.

struct Down {
    resnets: Vec<Resnet>,
    downsample: Option<Conv>,
}

struct EncoderTopology {
    conv_in: Conv,
    downs: Vec<Down>,
    mid: [Resnet; 2],
    attn: Option<Attn>,
    norm_out: Affine,
    conv_out: Conv,
}

fn validate_spec(spec: &VaeSpec) -> Result<(), RuntimeError> {
    scale_factor(spec.block_out.len())?;
    if spec.groups < 1 {
        return Err(RuntimeError::Message(
            "vae group count must be positive".into(),
        ));
    }
    if spec.latent_channels < 1 {
        return Err(RuntimeError::Message(
            "vae latent channels must be positive".into(),
        ));
    }
    if !spec.scaling.is_finite() || spec.scaling == 0.0 {
        return Err(RuntimeError::Message(format!(
            "vae scaling {} is not a finite non-zero value",
            spec.scaling
        )));
    }
    if !spec.shift.is_finite() {
        return Err(RuntimeError::Message(format!(
            "vae shift {} is not finite",
            spec.shift
        )));
    }
    let groups = spec.groups as i64;
    for &channels in &spec.block_out {
        if channels < 1 || channels % groups != 0 {
            return Err(RuntimeError::Message(format!(
                "vae channels {channels} are not divisible by {} groups",
                spec.groups
            )));
        }
    }
    Ok(())
}

fn encoder_norm_out_name(vae: &GgufWeights) -> &'static str {
    if vae.contains("encoder.conv_norm_out.weight") {
        "encoder.conv_norm_out"
    } else {
        "encoder.norm_out"
    }
}

fn encoder_topology(spec: &VaeSpec, norm_out: &str) -> Result<EncoderTopology, RuntimeError> {
    validate_spec(spec)?;
    if spec.resnets < 2 {
        return Err(RuntimeError::Message(
            "vae encoder needs a spec with at least two resnets per block".into(),
        ));
    }
    let enc_resnets = spec.resnets - 1;
    let levels = spec.block_out.len();
    let top = *spec.block_out.last().expect("block list");
    let mut downs = Vec::with_capacity(levels);
    for i in 0..levels {
        let out_ch = spec.block_out[i];
        let in_ch = if i == 0 {
            out_ch
        } else {
            spec.block_out[i - 1]
        };
        let mut resnets = Vec::with_capacity(enc_resnets);
        for j in 0..enc_resnets {
            let (rin, rout, shortcut) = if j == 0 {
                (in_ch, out_ch, in_ch != out_ch)
            } else {
                (out_ch, out_ch, false)
            };
            resnets.push(resnet_block(
                &format!("encoder.down_blocks.{i}.resnets.{j}"),
                rin,
                rout,
                shortcut,
            ));
        }
        let downsample = if i + 1 == levels {
            None
        } else {
            Some(conv(
                &format!("encoder.down_blocks.{i}.downsamplers.0.conv"),
                3,
                3,
                out_ch,
                out_ch,
                0,
                WeightLayout::Kernel,
            ))
        };
        downs.push(Down {
            resnets,
            downsample,
        });
    }
    Ok(EncoderTopology {
        conv_in: conv(
            "encoder.conv_in",
            3,
            3,
            3,
            spec.block_out[0],
            1,
            WeightLayout::Kernel,
        ),
        downs,
        mid: [
            resnet_block("encoder.mid_block.resnets.0", top, top, false),
            resnet_block("encoder.mid_block.resnets.1", top, top, false),
        ],
        attn: spec
            .attention
            .then(|| attention_block("encoder.mid_block.attentions.0", top)),
        norm_out: affine(norm_out, top),
        conv_out: conv(
            "encoder.conv_out",
            3,
            3,
            top,
            spec.latent_channels * 2,
            1,
            WeightLayout::Kernel,
        ),
    })
}

pub(crate) fn encoder_parameter_list(
    spec: &VaeSpec,
    norm_out: &str,
) -> Result<Vec<Param>, RuntimeError> {
    let topo = encoder_topology(spec, norm_out)?;
    let mut params = Vec::new();
    push_conv(&mut params, &topo.conv_in);
    for down in &topo.downs {
        for block in &down.resnets {
            push_resnet(&mut params, block);
        }
        if let Some(downsample) = &down.downsample {
            push_conv(&mut params, downsample);
        }
    }
    push_resnet(&mut params, &topo.mid[0]);
    if let Some(attn) = &topo.attn {
        push_affine(&mut params, &attn.norm);
        push_conv(&mut params, &attn.q);
        push_conv(&mut params, &attn.k);
        push_conv(&mut params, &attn.v);
        push_conv(&mut params, &attn.out);
    }
    push_resnet(&mut params, &topo.mid[1]);
    push_affine(&mut params, &topo.norm_out);
    push_conv(&mut params, &topo.conv_out);
    let mut seen = BTreeSet::new();
    for param in &params {
        if !seen.insert(param.name.clone()) {
            return Err(RuntimeError::Message(format!(
                "vae encoder parameter {} is declared twice",
                param.name
            )));
        }
    }
    Ok(params)
}

/// Encoder `conv_out` output: `[W/f, H/f, 2*latent]` planar, unscaled. The
/// caller slices the mean half and applies its own quant conv and scaling.
pub(crate) fn encode_conv_out(
    ctx: &GgmlContext,
    vae: &GgufWeights,
    image: &[f32],
    spec: &VaeSpec,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, WeightError> {
    let factor = scale_factor(spec.block_out.len())?;
    if width < 1 || height < 1 || width as i64 % factor != 0 || height as i64 % factor != 0 {
        return Err(WeightError::NotBuilt(format!(
            "vae encode image {width}x{height} is not divisible by the {factor}x scale"
        )));
    }
    let pixels = (width as i64 * height as i64 * 3) as usize;
    if image.len() != pixels {
        return Err(WeightError::Shape {
            name: "pixels".into(),
            expected: vec![width as i64, height as i64, 3],
            got: vec![image.len() as i64],
        });
    }
    if !vae.contains("encoder.conv_in.weight") {
        return Err(WeightError::NotBuilt(
            "vae encoder graph is not built".into(),
        ));
    }
    let norm_out = encoder_norm_out_name(vae);
    let topo = encoder_topology(spec, norm_out)?;
    let params = encoder_parameter_list(spec, norm_out)?;
    let mut loaded = Vec::with_capacity(params.len());
    for param in &params {
        loaded.push((
            param.name.clone(),
            vae.f32_widened(&param.name, &param.shape)?,
        ));
    }
    let mut builder = ctx.builder(GRAPH_MEM)?;
    let declared = declare(&mut builder, &params);
    let output = forward_encoder(&mut builder, spec, &topo, &declared, width, height)?;
    let mut graph = builder.finish(output)?;
    graph.set_f32("pixels", image)?;
    for (name, data) in &loaded {
        graph.set_f32(name, data)?;
    }
    let encoded = graph.compute()?;
    let out_w = width as i64 / factor;
    let out_h = height as i64 / factor;
    let expect = (out_w * out_h * spec.latent_channels * 2) as usize;
    if encoded.len() != expect {
        return Err(WeightError::Shape {
            name: "vae encode".into(),
            expected: vec![out_w, out_h, spec.latent_channels * 2],
            got: vec![encoded.len() as i64],
        });
    }
    if encoded.iter().any(|value| !value.is_finite()) {
        return Err(WeightError::NotBuilt(
            "vae encode produced a non-finite value".into(),
        ));
    }
    Ok(encoded)
}

/// Full generic encode: mean half of `conv_out`, `z = (mean - shift) * scaling`,
/// `[W/f, H/f, latent]` planar.
pub(crate) fn encode_samples(
    ctx: &GgmlContext,
    vae: &GgufWeights,
    image: &[f32],
    spec: &VaeSpec,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, WeightError> {
    let conv_out = encode_conv_out(ctx, vae, image, spec, width, height)?;
    let factor = scale_factor(spec.block_out.len())?;
    let out_w = (width as i64 / factor) as usize;
    let out_h = (height as i64 / factor) as usize;
    let channels = spec.latent_channels as usize;
    let plane = out_w * out_h;
    let mut out = vec![0f32; plane * channels];
    for c in 0..channels {
        for i in 0..plane {
            out[i + c * plane] = (conv_out[i + c * plane] - spec.shift) * spec.scaling;
        }
    }
    Ok(out)
}

fn forward_encoder(
    b: &mut GraphBuilder,
    spec: &VaeSpec,
    topo: &EncoderTopology,
    weights: &HashMap<String, Tensor>,
    width: u32,
    height: u32,
) -> Result<Tensor, RuntimeError> {
    let pixels = b.f32_3d("pixels", width as i64, height as i64, 3, true);
    let x = nn(b.cont(b.reshape_4d(pixels, width as i64, height as i64, 3, 1)))?;
    let mut w = width as i64;
    let mut h = height as i64;
    let mut x = apply_conv(b, weights, x, &topo.conv_in, w, h)?;
    for down in &topo.downs {
        for block in &down.resnets {
            x = apply_resnet(b, weights, x, block, spec.groups, w, h)?;
        }
        if let Some(downsample) = &down.downsample {
            x = nn(b.cont(x))?;
            let padded = nn(b.pad_ext(x, 0, 1, 0, 1, 0, 0, 0, 0))?;
            let padded = shaped(
                padded,
                [w + 1, h + 1, downsample.ic, 1],
                "vae downsample pad",
            )?;
            let y =
                nn(b.conv2d_direct(wt(weights, &downsample.weight)?, padded, 2, 2, 0, 0, 1, 1))?;
            w = (w + 1 - 3) / 2 + 1;
            h = (h + 1 - 3) / 2 + 1;
            let y = nn(b.cont(y))?;
            let y = shaped(y, [w, h, downsample.oc, 1], &downsample.weight)?;
            x = add_bias(b, y, wt(weights, &downsample.bias)?, downsample.oc)?;
        }
    }
    x = apply_resnet(b, weights, x, &topo.mid[0], spec.groups, w, h)?;
    if let Some(attn) = &topo.attn {
        x = apply_attention(b, weights, x, attn, spec.groups, w, h)?;
    }
    x = apply_resnet(b, weights, x, &topo.mid[1], spec.groups, w, h)?;
    x = apply_norm(b, weights, x, &topo.norm_out, spec.groups, w, h)?;
    x = nn(b.silu(x))?;
    let x = apply_conv(b, weights, x, &topo.conv_out, w, h)?;
    shaped(x, [w, h, spec.latent_channels * 2, 1], "encoder.conv_out")
}

/// Derive the decoder spec from tensor shapes. Falls back to `defaults` when
/// the file does not carry the full decoder block list, so partial files keep
/// their original fail-loud messages.
pub(crate) fn decoder_spec_from_weights(
    vae: &GgufWeights,
    defaults: &VaeSpec,
) -> Result<VaeSpec, WeightError> {
    let conv_in = vae.shape("decoder.conv_in.weight")?;
    let mut levels = 0i64;
    while vae.contains(&format!(
        "decoder.up_blocks.{levels}.resnets.0.conv2.weight"
    )) {
        levels += 1;
    }
    if levels < 2 {
        return Ok(defaults.clone());
    }
    let mut block_out = Vec::with_capacity(levels as usize);
    for i in 0..levels {
        let shape = vae.shape(&format!("decoder.up_blocks.{i}.resnets.0.conv2.weight"))?;
        block_out.push(shape[3]);
    }
    block_out.reverse();
    let mut resnets = 0usize;
    for i in 0..levels {
        let mut count = 0usize;
        while vae.contains(&format!(
            "decoder.up_blocks.{i}.resnets.{count}.conv1.weight"
        )) {
            count += 1;
        }
        resnets = resnets.max(count);
    }
    Ok(VaeSpec {
        latent_channels: conv_in[2],
        block_out,
        resnets,
        groups: defaults.groups,
        scaling: defaults.scaling,
        shift: defaults.shift,
        attention: vae.contains("decoder.mid_block.attentions.0.to_q.weight"),
    })
}

/// Derive the encoder spec from tensor shapes. `spec.resnets` is the decoder
/// convention, so an encoder with N resnets per down block reports N + 1.
pub(crate) fn encoder_spec_from_weights(
    vae: &GgufWeights,
    defaults: &VaeSpec,
) -> Result<VaeSpec, WeightError> {
    let conv_in = vae.shape("encoder.conv_in.weight")?;
    let mut levels = 0i64;
    while vae.contains(&format!(
        "encoder.down_blocks.{levels}.resnets.0.conv2.weight"
    )) {
        levels += 1;
    }
    if levels < 2 {
        return Ok(defaults.clone());
    }
    let mut block_out = Vec::with_capacity(levels as usize);
    for i in 0..levels {
        let shape = vae.shape(&format!("encoder.down_blocks.{i}.resnets.0.conv2.weight"))?;
        block_out.push(shape[3]);
    }
    let conv_out = vae.shape("encoder.conv_out.weight")?;
    if conv_out[3] < 2 || conv_out[3] % 2 != 0 {
        return Err(WeightError::Shape {
            name: "encoder.conv_out.weight".into(),
            expected: vec![3, 3, conv_out[2], conv_out[3]],
            got: conv_out.to_vec(),
        });
    }
    let mut count = 0usize;
    while vae.contains(&format!(
        "encoder.down_blocks.0.resnets.{count}.conv1.weight"
    )) {
        count += 1;
    }
    Ok(VaeSpec {
        latent_channels: conv_out[3] / 2,
        block_out,
        resnets: count + 1,
        groups: defaults.groups,
        scaling: defaults.scaling,
        shift: defaults.shift,
        attention: vae.contains("encoder.mid_block.attentions.0.to_q.weight"),
    })
}

fn conv(prefix: &str, kw: i64, kh: i64, ic: i64, oc: i64, pad: i32, layout: WeightLayout) -> Conv {
    Conv {
        weight: format!("{prefix}.weight"),
        bias: format!("{prefix}.bias"),
        kw,
        kh,
        ic,
        oc,
        pad,
        layout,
    }
}

fn affine(prefix: &str, channels: i64) -> Affine {
    Affine {
        weight: format!("{prefix}.weight"),
        bias: format!("{prefix}.bias"),
        channels,
    }
}

fn resnet_block(prefix: &str, in_ch: i64, out_ch: i64, shortcut: bool) -> Resnet {
    Resnet {
        norm1: affine(&format!("{prefix}.norm1"), in_ch),
        conv1: conv(
            &format!("{prefix}.conv1"),
            3,
            3,
            in_ch,
            out_ch,
            1,
            WeightLayout::Kernel,
        ),
        norm2: affine(&format!("{prefix}.norm2"), out_ch),
        conv2: conv(
            &format!("{prefix}.conv2"),
            3,
            3,
            out_ch,
            out_ch,
            1,
            WeightLayout::Kernel,
        ),
        shortcut: shortcut.then(|| {
            conv(
                &format!("{prefix}.conv_shortcut"),
                1,
                1,
                in_ch,
                out_ch,
                0,
                WeightLayout::Kernel,
            )
        }),
    }
}

fn attention_block(prefix: &str, channels: i64) -> Attn {
    Attn {
        norm: affine(&format!("{prefix}.group_norm"), channels),
        q: conv(
            &format!("{prefix}.to_q"),
            1,
            1,
            channels,
            channels,
            0,
            WeightLayout::Linear,
        ),
        k: conv(
            &format!("{prefix}.to_k"),
            1,
            1,
            channels,
            channels,
            0,
            WeightLayout::Linear,
        ),
        v: conv(
            &format!("{prefix}.to_v"),
            1,
            1,
            channels,
            channels,
            0,
            WeightLayout::Linear,
        ),
        out: conv(
            &format!("{prefix}.to_out.0"),
            1,
            1,
            channels,
            channels,
            0,
            WeightLayout::Linear,
        ),
    }
}

fn push_conv(out: &mut Vec<Param>, conv: &Conv) {
    let (shape, kind) = match conv.layout {
        WeightLayout::Kernel => (
            vec![conv.kw, conv.kh, conv.ic, conv.oc],
            Kind::Kernel {
                kw: conv.kw,
                kh: conv.kh,
                ic: conv.ic,
                oc: conv.oc,
            },
        ),
        WeightLayout::Linear => (
            vec![conv.ic, conv.oc],
            Kind::Linear {
                inp: conv.ic,
                out: conv.oc,
            },
        ),
    };
    out.push(Param {
        name: conv.weight.clone(),
        shape,
        kind,
    });
    out.push(Param {
        name: conv.bias.clone(),
        shape: vec![conv.oc],
        kind: Kind::Vector(conv.oc),
    });
}

fn push_affine(out: &mut Vec<Param>, norm: &Affine) {
    out.push(Param {
        name: norm.weight.clone(),
        shape: vec![norm.channels],
        kind: Kind::Vector(norm.channels),
    });
    out.push(Param {
        name: norm.bias.clone(),
        shape: vec![norm.channels],
        kind: Kind::Vector(norm.channels),
    });
}

fn push_resnet(out: &mut Vec<Param>, block: &Resnet) {
    push_affine(out, &block.norm1);
    push_conv(out, &block.conv1);
    push_affine(out, &block.norm2);
    push_conv(out, &block.conv2);
    if let Some(shortcut) = &block.shortcut {
        push_conv(out, shortcut);
    }
}

fn declare(b: &mut GraphBuilder, params: &[Param]) -> HashMap<String, Tensor> {
    let mut map = HashMap::new();
    for param in params {
        let tensor = match param.kind {
            Kind::Vector(n) => b.f32_1d(&param.name, n, true),
            Kind::Linear { inp, out } => b.f32_2d(&param.name, inp, out, true),
            Kind::Kernel { kw, kh, ic, oc } => b.f32_4d(&param.name, kw, kh, ic, oc, true),
        };
        map.insert(param.name.clone(), tensor);
    }
    map
}

fn forward(
    b: &mut GraphBuilder,
    spec: &VaeSpec,
    topo: &Topology,
    weights: &HashMap<String, Tensor>,
    latent_w: i64,
    latent_h: i64,
    width: u32,
    height: u32,
) -> Result<Tensor, RuntimeError> {
    let latent = b.f32_3d("latent", latent_w, latent_h, spec.latent_channels, true);
    let inv = b.f32_1d("vae_inv_scale", 1, true);
    let shift = b.f32_1d("vae_shift", 1, true);
    let mut x = scale_shift(
        b,
        latent,
        inv,
        shift,
        latent_w,
        latent_h,
        spec.latent_channels,
    )?;
    x = apply_conv(b, weights, x, &topo.conv_in, latent_w, latent_h)?;
    let mut w = latent_w;
    let mut h = latent_h;
    x = apply_resnet(b, weights, x, &topo.mid[0], spec.groups, w, h)?;
    if let Some(attn) = &topo.attn {
        x = apply_attention(b, weights, x, attn, spec.groups, w, h)?;
    }
    x = apply_resnet(b, weights, x, &topo.mid[1], spec.groups, w, h)?;
    for up in &topo.ups {
        for block in &up.resnets {
            x = apply_resnet(b, weights, x, block, spec.groups, w, h)?;
        }
        if let Some(upsample) = &up.upsample {
            let upsampled = nn(b.upscale_nearest(nn(b.cont(x))?))?;
            let channels = upsample.ic;
            x = shaped(upsampled, [w * 2, h * 2, channels, 1], "vae upsample")?;
            w *= 2;
            h *= 2;
            x = apply_conv(b, weights, x, upsample, w, h)?;
        }
    }
    if w != width as i64 || h != height as i64 {
        return Err(RuntimeError::Message(format!(
            "vae decode is {w}x{h}, image is {width}x{height}"
        )));
    }
    x = apply_norm(b, weights, x, &topo.norm_out, spec.groups, w, h)?;
    x = nn(b.silu(x))?;
    x = apply_conv(b, weights, x, &topo.conv_out, w, h)?;
    shaped(x, [w, h, 3, 1], "decoder.conv_out")
}

fn scale_shift(
    b: &GraphBuilder,
    latent: Tensor,
    inv: Tensor,
    shift: Tensor,
    w: i64,
    h: i64,
    c: i64,
) -> Result<Tensor, RuntimeError> {
    let x = nn(b.cont(b.reshape_4d(latent, w, h, c, 1)))?;
    let inv = nn(b.reshape_4d(inv, 1, 1, 1, 1))?;
    let x = nn(b.mul(x, nn(b.repeat(inv, x))?))?;
    let shift = nn(b.reshape_4d(shift, 1, 1, 1, 1))?;
    let x = nn(b.add(x, nn(b.repeat(shift, x))?))?;
    shaped(x, [w, h, c, 1], "vae latent")
}

fn apply_resnet(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    block: &Resnet,
    groups: i32,
    w: i64,
    h: i64,
) -> Result<Tensor, RuntimeError> {
    let hidden = apply_norm(b, weights, x, &block.norm1, groups, w, h)?;
    let hidden = nn(b.silu(hidden))?;
    let hidden = apply_conv(b, weights, hidden, &block.conv1, w, h)?;
    let hidden = apply_norm(b, weights, hidden, &block.norm2, groups, w, h)?;
    let hidden = nn(b.silu(hidden))?;
    let hidden = apply_conv(b, weights, hidden, &block.conv2, w, h)?;
    let skip = if let Some(shortcut) = &block.shortcut {
        apply_conv(b, weights, x, shortcut, w, h)?
    } else {
        x
    };
    let y = nn(b.add(skip, hidden))?;
    shaped(y, [w, h, block.conv2.oc, 1], "vae resnet")
}

fn apply_attention(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    attn: &Attn,
    groups: i32,
    w: i64,
    h: i64,
) -> Result<Tensor, RuntimeError> {
    let channels = attn.norm.channels;
    let hidden = apply_norm(b, weights, x, &attn.norm, groups, w, h)?;
    let q = apply_conv(b, weights, hidden, &attn.q, w, h)?;
    let k = apply_conv(b, weights, hidden, &attn.k, w, h)?;
    let v = apply_conv(b, weights, hidden, &attn.v, w, h)?;
    let q = to_tokens(b, q, w, h, channels)?;
    let k = to_tokens(b, k, w, h, channels)?;
    let v = to_tokens(b, v, w, h, channels)?;
    let mixed = flash_attend(b, q, k, v, std::ptr::null_mut())?;
    shaped(mixed, [channels, 1, w * h, 1], "vae attention")?;
    let mixed = from_tokens(b, mixed, w, h, channels)?;
    let mixed = apply_conv(b, weights, mixed, &attn.out, w, h)?;
    let y = nn(b.add(x, mixed))?;
    shaped(y, [w, h, channels, 1], "vae attention")
}

fn apply_norm(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    norm: &Affine,
    groups: i32,
    w: i64,
    h: i64,
) -> Result<Tensor, RuntimeError> {
    let x = nn(b.cont(x))?;
    let y = nn(b.group_norm(x, groups, NORM_EPS))?;
    let y = shaped(y, [w, h, norm.channels, 1], &norm.weight)?;
    let scale = broadcast(b, y, wt(weights, &norm.weight)?, norm.channels)?;
    let y = nn(b.mul(y, scale))?;
    add_bias(b, y, wt(weights, &norm.bias)?, norm.channels)
}

fn apply_conv(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    conv: &Conv,
    w: i64,
    h: i64,
) -> Result<Tensor, RuntimeError> {
    let kernel = match conv.layout {
        WeightLayout::Kernel => wt(weights, &conv.weight)?,
        WeightLayout::Linear => {
            nn(b.cont(b.reshape_4d(wt(weights, &conv.weight)?, 1, 1, conv.ic, conv.oc)))?
        }
    };
    let y = nn(b.conv2d_direct(kernel, x, 1, 1, conv.pad, conv.pad, 1, 1))?;
    let y = nn(b.cont(y))?;
    let y = shaped(y, [w, h, conv.oc, 1], &conv.weight)?;
    add_bias(b, y, wt(weights, &conv.bias)?, conv.oc)
}

fn add_bias(
    b: &GraphBuilder,
    x: Tensor,
    bias: Tensor,
    channels: i64,
) -> Result<Tensor, RuntimeError> {
    let bias = broadcast(b, x, bias, channels)?;
    nn(b.add(x, bias))
}

fn broadcast(
    b: &GraphBuilder,
    like: Tensor,
    vector: Tensor,
    channels: i64,
) -> Result<Tensor, RuntimeError> {
    let vector = nn(b.cont(b.reshape_4d(vector, 1, 1, channels, 1)))?;
    nn(b.repeat(vector, like))
}

/// `[W, H, C, 1]` to `[C, 1, W*H]`. `permute(1, 2, 0, 3)` sends C to the fastest axis.
fn to_tokens(b: &GraphBuilder, x: Tensor, w: i64, h: i64, c: i64) -> Result<Tensor, RuntimeError> {
    let p = nn(b.cont(b.permute(x, 1, 2, 0, 3)))?;
    let tokens = nn(b.reshape_3d(p, c, 1, w * h))?;
    shaped(tokens, [c, 1, w * h, 1], "vae attention tokens")
}

/// `[C, 1, W*H]` back to `[W, H, C, 1]`.
fn from_tokens(
    b: &GraphBuilder,
    tokens: Tensor,
    w: i64,
    h: i64,
    c: i64,
) -> Result<Tensor, RuntimeError> {
    let flat = nn(b.cont(tokens))?;
    let s = nn(b.reshape_4d(flat, c, w, h, 1))?;
    let y = nn(b.cont(b.permute(s, 2, 0, 1, 3)))?;
    shaped(y, [w, h, c, 1], "vae attention tokens")
}

fn shaped(tensor: Tensor, ne: [i64; 4], what: &str) -> Result<Tensor, RuntimeError> {
    let got = shape(tensor);
    if got != ne {
        return Err(RuntimeError::Message(format!(
            "{what} is {got:?}, the decoder expects {ne:?}"
        )));
    }
    Ok(tensor)
}

fn wt(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor, RuntimeError> {
    weights
        .get(name)
        .copied()
        .ok_or_else(|| RuntimeError::Message(format!("vae graph is missing {name}")))
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
        decode_samples, parameter_list, rgb_bytes, topology, Affine, Conv, VaeSpec, WeightLayout,
    };
    use crate::runtime::{BackendKind, GgmlContext};
    use crate::weights::{write_f32_nd, F32Nd, GgufWeights};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn tiny() -> VaeSpec {
        VaeSpec {
            latent_channels: 2,
            block_out: vec![2, 4],
            resnets: 2,
            groups: 2,
            scaling: 2.0,
            shift: 0.25,
            attention: true,
        }
    }

    #[test]
    fn groups_must_divide_every_block() {
        let mut spec = tiny();
        spec.groups = 3;
        let err = match parameter_list(&spec) {
            Err(err) => err,
            Ok(_) => panic!("channels that are not divisible by the group count were accepted"),
        };
        assert!(err.to_string().contains("not divisible"), "{err}");
    }

    #[test]
    fn tiny_decoder_matches_the_host_reference() {
        let spec = tiny();
        let dir = scratch("vae-tiny");
        let params = parameter_list(&spec).unwrap();
        let owned: Vec<(String, [i64; 4], Vec<f32>)> = params
            .iter()
            .map(|param| {
                let mut ne = [1i64; 4];
                for (index, dim) in param.shape.iter().enumerate() {
                    ne[index] = *dim;
                }
                let data = fill(&param.name, elems(&param.shape));
                (param.name.clone(), ne, data)
            })
            .collect();
        let tensors: Vec<F32Nd<'_>> = owned
            .iter()
            .map(|(name, ne, data)| F32Nd {
                name,
                ne: *ne,
                data,
            })
            .collect();
        let path = dir.join("vae.gguf");
        write_f32_nd(&path, &tensors).unwrap();
        let map: HashMap<String, Vec<f32>> = owned
            .iter()
            .map(|(name, _, data)| (name.clone(), data.clone()))
            .collect();
        let latent: Vec<f32> = (0..2 * 2 * 2)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.3)
            .collect();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let vae = GgufWeights::open(&path).unwrap();
        let got = decode_samples(&ctx, &vae, &latent, &spec, 2, 2, 2, 4, 4).unwrap();
        let expect = host_reference(&spec, &map, &latent, 2, 2);
        let max = got
            .iter()
            .zip(&expect)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 1e-3,
            "max abs {max} got {:?} expect {:?}",
            &got[..8.min(got.len())],
            &expect[..8.min(expect.len())]
        );
        let again = decode_samples(&ctx, &vae, &latent, &spec, 2, 2, 2, 4, 4).unwrap();
        assert_eq!(got, again);
        let other = vec![0.2f32; latent.len()];
        let changed = decode_samples(&ctx, &vae, &other, &spec, 2, 2, 2, 4, 4).unwrap();
        assert!(got.iter().zip(&changed).any(|(a, b)| (a - b).abs() > 1e-3));
        let rgb = rgb_bytes(&got, 4, 4);
        let png = crate::common::png::encode_rgb(4, 4, &rgb);
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn host_reference(
        spec: &VaeSpec,
        weights: &HashMap<String, Vec<f32>>,
        latent: &[f32],
        w: usize,
        h: usize,
    ) -> Vec<f32> {
        let topo = topology(spec).unwrap();
        let mut x: Vec<f32> = latent
            .iter()
            .map(|value| value / spec.scaling + spec.shift)
            .collect();
        x = conv2d(&x, w, h, &topo.conv_in, weights);
        x = resnet(&x, w, h, &topo.mid[0], spec, weights);
        if let Some(attn) = &topo.attn {
            x = attend(&x, w, h, attn, spec, weights);
        }
        x = resnet(&x, w, h, &topo.mid[1], spec, weights);
        let mut width = w;
        let mut height = h;
        for up in &topo.ups {
            for block in &up.resnets {
                x = resnet(&x, width, height, block, spec, weights);
            }
            if let Some(upsample) = &up.upsample {
                x = nearest2(&x, width, height, upsample.ic as usize);
                width *= 2;
                height *= 2;
                x = conv2d(&x, width, height, upsample, weights);
            }
        }
        x = silu(&group_norm(
            &x,
            width,
            height,
            &topo.norm_out,
            spec,
            weights,
        ));
        conv2d(&x, width, height, &topo.conv_out, weights)
    }

    fn resnet(
        x: &[f32],
        w: usize,
        h: usize,
        block: &super::Resnet,
        spec: &VaeSpec,
        weights: &HashMap<String, Vec<f32>>,
    ) -> Vec<f32> {
        let hidden = conv2d(
            &silu(&group_norm(x, w, h, &block.norm1, spec, weights)),
            w,
            h,
            &block.conv1,
            weights,
        );
        let hidden = conv2d(
            &silu(&group_norm(&hidden, w, h, &block.norm2, spec, weights)),
            w,
            h,
            &block.conv2,
            weights,
        );
        let skip = if let Some(shortcut) = &block.shortcut {
            conv2d(x, w, h, shortcut, weights)
        } else {
            x.to_vec()
        };
        skip.iter().zip(hidden).map(|(a, b)| a + b).collect()
    }

    fn attend(
        x: &[f32],
        w: usize,
        h: usize,
        attn: &super::Attn,
        spec: &VaeSpec,
        weights: &HashMap<String, Vec<f32>>,
    ) -> Vec<f32> {
        let channels = attn.norm.channels as usize;
        let hidden = group_norm(x, w, h, &attn.norm, spec, weights);
        let q = conv2d(&hidden, w, h, &attn.q, weights);
        let k = conv2d(&hidden, w, h, &attn.k, weights);
        let v = conv2d(&hidden, w, h, &attn.v, weights);
        let seq = w * h;
        let scale = 1.0 / (channels as f32).sqrt();
        let mut mixed = vec![0.0f32; seq * channels];
        for i in 0..seq {
            let mut scores = vec![0.0f32; seq];
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..seq {
                let mut dot = 0.0f32;
                for channel in 0..channels {
                    dot += q[i + channel * seq] * k[j + channel * seq];
                }
                scores[j] = dot * scale;
                max_score = max_score.max(scores[j]);
            }
            let mut sum = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            for channel in 0..channels {
                let mut acc = 0.0f32;
                for j in 0..seq {
                    acc += scores[j] / sum * v[j + channel * seq];
                }
                mixed[i + channel * seq] = acc;
            }
        }
        let mixed = conv2d(&mixed, w, h, &attn.out, weights);
        x.iter().zip(mixed).map(|(a, b)| a + b).collect()
    }

    fn group_norm(
        x: &[f32],
        w: usize,
        h: usize,
        norm: &Affine,
        spec: &VaeSpec,
        weights: &HashMap<String, Vec<f32>>,
    ) -> Vec<f32> {
        let channels = norm.channels as usize;
        let groups = spec.groups as usize;
        let per = channels / groups;
        let spatial = w * h;
        let scale = tensor(weights, &norm.weight);
        let shift = tensor(weights, &norm.bias);
        let mut y = vec![0.0f32; x.len()];
        for group in 0..groups {
            let start = group * per;
            let count = (spatial * per) as f32;
            let mut sum = 0.0f32;
            for channel in start..start + per {
                for index in 0..spatial {
                    sum += x[index + channel * spatial];
                }
            }
            let mean = sum / count;
            let mut var = 0.0f32;
            for channel in start..start + per {
                for index in 0..spatial {
                    let delta = x[index + channel * spatial] - mean;
                    var += delta * delta;
                }
            }
            var /= count;
            let inv = 1.0 / (var + 1e-6).sqrt();
            for channel in start..start + per {
                for index in 0..spatial {
                    let value = (x[index + channel * spatial] - mean) * inv;
                    y[index + channel * spatial] = value * scale[channel] + shift[channel];
                }
            }
        }
        y
    }

    fn conv2d(
        x: &[f32],
        w: usize,
        h: usize,
        conv: &Conv,
        weights: &HashMap<String, Vec<f32>>,
    ) -> Vec<f32> {
        let kw = conv.kw as usize;
        let kh = conv.kh as usize;
        let ic = conv.ic as usize;
        let oc = conv.oc as usize;
        let pad = conv.pad as usize;
        let kernel = tensor(weights, &conv.weight);
        let bias = tensor(weights, &conv.bias);
        let ow = w + 2 * pad + 1 - kw;
        let oh = h + 2 * pad + 1 - kh;
        let mut out = vec![0.0f32; ow * oh * oc];
        for oc_i in 0..oc {
            for y in 0..oh {
                for x0 in 0..ow {
                    let mut acc = bias[oc_i];
                    for ic_i in 0..ic {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let ix = x0 as isize + kx as isize - pad as isize;
                                let iy = y as isize + ky as isize - pad as isize;
                                if ix < 0 || iy < 0 || ix >= w as isize || iy >= h as isize {
                                    continue;
                                }
                                let xv = x[ix as usize + iy as usize * w + ic_i * w * h];
                                let kv = match conv.layout {
                                    WeightLayout::Kernel => {
                                        kernel[kx + ky * kw + ic_i * kw * kh + oc_i * kw * kh * ic]
                                    }
                                    WeightLayout::Linear => kernel[ic_i + oc_i * ic],
                                };
                                acc += xv * kv;
                            }
                        }
                    }
                    out[x0 + y * ow + oc_i * ow * oh] = acc;
                }
            }
        }
        out
    }

    fn nearest2(x: &[f32], w: usize, h: usize, channels: usize) -> Vec<f32> {
        let ow = w * 2;
        let oh = h * 2;
        let mut y = vec![0.0f32; ow * oh * channels];
        for channel in 0..channels {
            for y0 in 0..oh {
                for x0 in 0..ow {
                    let sx = x0 / 2;
                    let sy = y0 / 2;
                    y[x0 + y0 * ow + channel * ow * oh] = x[sx + sy * w + channel * w * h];
                }
            }
        }
        y
    }

    fn silu(x: &[f32]) -> Vec<f32> {
        x.iter()
            .map(|value| value / (1.0 + (-value).exp()))
            .collect()
    }

    fn tensor<'a>(weights: &'a HashMap<String, Vec<f32>>, name: &str) -> &'a [f32] {
        weights
            .get(name)
            .map(Vec::as_slice)
            .unwrap_or_else(|| panic!("missing {name}"))
    }

    fn fill(name: &str, n: usize) -> Vec<f32> {
        if name.ends_with(".weight") && name.contains("norm") {
            vec![1.0; n]
        } else if name.ends_with(".bias") {
            (0..n).map(|i| (i as f32 + 1.0) * 0.01).collect()
        } else {
            (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect()
        }
    }

    fn elems(shape: &[i64]) -> usize {
        shape.iter().map(|dim| *dim as usize).product()
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn tiny_encoder_matches_the_host_reference() {
        let spec = tiny();
        let dir = scratch("vae-enc-tiny");
        let params = super::encoder_parameter_list(&spec, "encoder.conv_norm_out").unwrap();
        let owned: Vec<(String, [i64; 4], Vec<f32>)> = params
            .iter()
            .map(|param| {
                let mut ne = [1i64; 4];
                for (index, dim) in param.shape.iter().enumerate() {
                    ne[index] = *dim;
                }
                let data = fill(&param.name, elems(&param.shape));
                (param.name.clone(), ne, data)
            })
            .collect();
        let tensors: Vec<F32Nd<'_>> = owned
            .iter()
            .map(|(name, ne, data)| F32Nd {
                name,
                ne: *ne,
                data,
            })
            .collect();
        let path = dir.join("vae.gguf");
        write_f32_nd(&path, &tensors).unwrap();
        let map: HashMap<String, Vec<f32>> = owned
            .iter()
            .map(|(name, _, data)| (name.clone(), data.clone()))
            .collect();
        let image: Vec<f32> = (0..4 * 4 * 3)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.2)
            .collect();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let vae = GgufWeights::open(&path).unwrap();
        let got = super::encode_conv_out(&ctx, &vae, &image, &spec, 4, 4).unwrap();
        let expect = host_encoder_reference(&spec, &map, &image, 4, 4);
        assert_eq!(got.len(), expect.len());
        let max = got
            .iter()
            .zip(&expect)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max < 1e-3,
            "max abs {max} got {:?} expect {:?}",
            &got[..8.min(got.len())],
            &expect[..8.min(expect.len())]
        );
        let samples = super::encode_samples(&ctx, &vae, &image, &spec, 4, 4).unwrap();
        let plane = 2 * 2;
        for c in 0..spec.latent_channels as usize {
            for i in 0..plane {
                let want = (expect[i + c * plane] - spec.shift) * spec.scaling;
                assert!(
                    (samples[i + c * plane] - want).abs() < 1e-6,
                    "channel {c} index {i}"
                );
            }
        }
        let inferred = super::encoder_spec_from_weights(&vae, &tiny()).unwrap();
        assert_eq!(inferred.latent_channels, spec.latent_channels);
        assert_eq!(inferred.block_out, spec.block_out);
        assert_eq!(inferred.resnets, spec.resnets);
        assert!(inferred.attention);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn encoder_spec_falls_back_without_blocks() {
        let dir = scratch("vae-enc-fallback");
        let path = dir.join("vae.gguf");
        let conv_in = vec![0.0f32; 3 * 3 * 3 * 2];
        write_f32_nd(
            &path,
            &[
                F32Nd {
                    name: "encoder.conv_in.weight",
                    ne: [3, 3, 3, 2],
                    data: &conv_in,
                },
                F32Nd {
                    name: "encoder.conv_in.bias",
                    ne: [2, 1, 1, 1],
                    data: &[0.0, 0.0],
                },
            ],
        )
        .unwrap();
        let vae = GgufWeights::open(&path).unwrap();
        let defaults = tiny();
        let inferred = super::encoder_spec_from_weights(&vae, &defaults).unwrap();
        assert_eq!(inferred.block_out, defaults.block_out);
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let err =
            super::encode_samples(&ctx, &vae, &[0.0; 4 * 4 * 3], &inferred, 4, 4).unwrap_err();
        assert!(err.to_string().contains("encoder.down_blocks"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn host_encoder_reference(
        spec: &VaeSpec,
        weights: &HashMap<String, Vec<f32>>,
        image: &[f32],
        w: usize,
        h: usize,
    ) -> Vec<f32> {
        let topo = super::encoder_topology(spec, "encoder.conv_norm_out").unwrap();
        let mut x = conv2d(image, w, h, &topo.conv_in, weights);
        let mut width = w;
        let mut height = h;
        for down in &topo.downs {
            for block in &down.resnets {
                x = resnet(&x, width, height, block, spec, weights);
            }
            if let Some(downsample) = &down.downsample {
                let (y, ow, oh) = host_downsample(&x, width, height, downsample, weights);
                x = y;
                width = ow;
                height = oh;
            }
        }
        x = resnet(&x, width, height, &topo.mid[0], spec, weights);
        if let Some(attn) = &topo.attn {
            x = attend(&x, width, height, attn, spec, weights);
        }
        x = resnet(&x, width, height, &topo.mid[1], spec, weights);
        x = silu(&group_norm(
            &x,
            width,
            height,
            &topo.norm_out,
            spec,
            weights,
        ));
        conv2d(&x, width, height, &topo.conv_out, weights)
    }

    /// Pad the right and bottom edge by one, then a 3×3 stride-2 zero-pad conv.
    fn host_downsample(
        x: &[f32],
        w: usize,
        h: usize,
        conv: &Conv,
        weights: &HashMap<String, Vec<f32>>,
    ) -> (Vec<f32>, usize, usize) {
        let channels = conv.ic as usize;
        let pw = w + 1;
        let ph = h + 1;
        let mut padded = vec![0.0f32; pw * ph * channels];
        for channel in 0..channels {
            for y in 0..h {
                for x0 in 0..w {
                    padded[x0 + y * pw + channel * pw * ph] = x[x0 + y * w + channel * w * h];
                }
            }
        }
        let ow = (pw - 3) / 2 + 1;
        let oh = (ph - 3) / 2 + 1;
        let kernel = tensor(weights, &conv.weight);
        let bias = tensor(weights, &conv.bias);
        let oc = conv.oc as usize;
        let mut out = vec![0.0f32; ow * oh * oc];
        for oc_i in 0..oc {
            for y in 0..oh {
                for x0 in 0..ow {
                    let mut acc = bias[oc_i];
                    for ic_i in 0..channels {
                        for ky in 0..3usize {
                            for kx in 0..3usize {
                                let ix = x0 * 2 + kx;
                                let iy = y * 2 + ky;
                                let xv = padded[ix + iy * pw + ic_i * pw * ph];
                                let kv = kernel[kx + ky * 3 + ic_i * 9 + oc_i * 9 * channels];
                                acc += xv * kv;
                            }
                        }
                    }
                    out[x0 + y * ow + oc_i * ow * oh] = acc;
                }
            }
        }
        (out, ow, oh)
    }
}
