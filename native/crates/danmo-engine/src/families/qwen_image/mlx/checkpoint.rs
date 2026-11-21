//! Official diffusers bundles (`Qwen/Qwen-Image-2512`) use HF module names.
//! mlx-community bundles are already the flat layout this graph reads.
//! A diffusers directory is renamed and transposed here; an mlx-community
//! directory is left untouched so conv kernels are not transposed twice.

use std::collections::BTreeMap;

use danmo_mlx::{Array, Error, Result, Stream};

use crate::weights::MlxWeights;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Xform {
    None,
    Conv3d,
    Conv2d,
    Gamma,
}

pub(super) fn adapt_dit(weights: MlxWeights, stream: &Stream) -> Result<MlxWeights> {
    if weights.has("transformer_blocks.0.img_mod_linear.weight") {
        return Ok(weights);
    }
    if !weights.has("transformer_blocks.0.img_mod.1.weight") {
        return Err(Error::new(
            "Qwen-Image DiT",
            "checkpoint is neither the mlx-community layout nor a diffusers Qwen-Image transformer",
        ));
    }
    rewrite(weights, dit_target, stream)
}

pub(super) fn adapt_text(weights: MlxWeights, stream: &Stream) -> Result<MlxWeights> {
    if weights.has("encoder.embed_tokens.weight") {
        return Ok(weights);
    }
    if !weights.has("model.embed_tokens.weight") {
        return Err(Error::new(
            "Qwen-Image text",
            "checkpoint is neither the mlx-community layout nor a diffusers Qwen2.5-VL encoder",
        ));
    }
    rewrite(weights, text_target, stream)
}

pub(super) fn adapt_vae(weights: MlxWeights, stream: &Stream) -> Result<MlxWeights> {
    if weights.has("decoder.conv_in.conv3d.weight") {
        return Ok(weights);
    }
    if !weights.has("decoder.conv_in.weight") {
        return Err(Error::new(
            "Qwen-Image VAE",
            "checkpoint is neither the mlx-community layout nor a diffusers AutoencoderKLQwenImage",
        ));
    }
    rewrite(weights, vae_target, stream)
}

fn rewrite(
    weights: MlxWeights,
    target_of: fn(&str) -> std::result::Result<(String, Xform), String>,
    stream: &Stream,
) -> Result<MlxWeights> {
    let names: Vec<String> = weights.names().map(str::to_string).collect();
    let mut rename = BTreeMap::new();
    let mut xforms = Vec::new();
    for name in &names {
        let (target, xform) =
            target_of(name).map_err(|error| Error::new("Qwen-Image remap", error))?;
        if rename.insert(name.clone(), target.clone()).is_some() {
            return Err(Error::new(
                "Qwen-Image remap",
                format!("duplicate source {name}"),
            ));
        }
        if xform != Xform::None {
            xforms.push((target, xform));
        }
    }
    let mut weights = weights.rename(&rename)?;
    for (target, xform) in xforms {
        let tensor = apply_xform(weights.tensor(&target)?, xform, stream)?;
        tensor.eval()?;
        weights = weights.with_tensor_production(target, tensor);
    }
    Ok(weights)
}

fn apply_xform(tensor: &Array, xform: Xform, stream: &Stream) -> Result<Array> {
    match xform {
        Xform::None => Ok(tensor.clone()),
        Xform::Conv3d => {
            let shape = tensor.shape();
            if shape.len() != 5 {
                return Err(Error::new(
                    "Qwen-Image remap",
                    format!("conv3d weight {shape:?} is not rank 5"),
                ));
            }
            tensor.transpose_axes(&[0, 2, 3, 4, 1], stream)
        }
        Xform::Conv2d => {
            let shape = tensor.shape();
            if shape.len() != 4 {
                return Err(Error::new(
                    "Qwen-Image remap",
                    format!("conv2d weight {shape:?} is not rank 4"),
                ));
            }
            tensor.transpose_axes(&[0, 2, 3, 1], stream)
        }
        Xform::Gamma => {
            let shape = tensor.shape();
            if shape.len() == 1 {
                return Ok(tensor.clone());
            }
            if shape.is_empty() || shape[1..].iter().any(|dim| *dim != 1) {
                return Err(Error::new(
                    "Qwen-Image remap",
                    format!("norm gamma {shape:?} is not [C] or [C, 1, ...]"),
                ));
            }
            tensor.reshape(&[shape[0]], stream)
        }
    }
}

fn dit_target(name: &str) -> std::result::Result<(String, Xform), String> {
    let mut out = name.to_string();
    for (from, to) in [
        (".attn.to_out.0.", ".attn.attn_to_out.0."),
        (".img_mod.1.", ".img_mod_linear."),
        (".txt_mod.1.", ".txt_mod_linear."),
        (".img_mlp.net.0.proj.", ".img_ff.mlp_in."),
        (".img_mlp.net.2.", ".img_ff.mlp_out."),
        (".txt_mlp.net.0.proj.", ".txt_ff.mlp_in."),
        (".txt_mlp.net.2.", ".txt_ff.mlp_out."),
    ] {
        if out.contains(from) {
            out = out.replace(from, to);
        }
    }
    if out.contains(".img_mlp.")
        || out.contains(".txt_mlp.")
        || out.contains(".img_mod.")
        || out.contains(".txt_mod.")
        || out.contains(".to_out.")
    {
        return Err(format!("unmapped diffusers DiT key {name}"));
    }
    Ok((out, Xform::None))
}

fn text_target(name: &str) -> std::result::Result<(String, Xform), String> {
    if name == "lm_head.weight" {
        return Ok((name.to_string(), Xform::None));
    }
    if let Some(rest) = name.strip_prefix("model.") {
        return Ok((format!("encoder.{rest}"), Xform::None));
    }
    if name == "visual.patch_embed.proj.weight" {
        return Ok((
            "encoder.visual.patch_embed.proj.weight".to_string(),
            Xform::Conv3d,
        ));
    }
    if let Some(rest) = name.strip_prefix("visual.merger.mlp.0.") {
        return Ok((format!("encoder.visual.merger.mlp_0.{rest}"), Xform::None));
    }
    if let Some(rest) = name.strip_prefix("visual.merger.mlp.2.") {
        return Ok((format!("encoder.visual.merger.mlp_1.{rest}"), Xform::None));
    }
    if let Some(rest) = name.strip_prefix("visual.") {
        return Ok((format!("encoder.visual.{rest}"), Xform::None));
    }
    Err(format!("unmapped diffusers text key {name}"))
}

fn vae_target(name: &str) -> std::result::Result<(String, Xform), String> {
    if let Some(rest) = name.strip_prefix("encoder.down_blocks.") {
        return encoder_down(rest);
    }
    if let Some(rest) = name.strip_prefix("decoder.up_blocks.") {
        return decoder_up(rest);
    }
    for stem in [
        "encoder.conv_in",
        "encoder.conv_out",
        "decoder.conv_in",
        "decoder.conv_out",
        "quant_conv",
        "post_quant_conv",
    ] {
        if let Some(leaf) = name.strip_prefix(&format!("{stem}.")) {
            return stem_conv(stem, leaf);
        }
    }
    for stem in ["encoder.norm_out", "decoder.norm_out"] {
        if name == format!("{stem}.gamma") {
            return Ok((format!("{stem}.weight"), Xform::Gamma));
        }
    }
    for stem in [
        "encoder.mid_block.attentions.0",
        "decoder.mid_block.attentions.0",
    ] {
        if let Some(leaf) = name.strip_prefix(&format!("{stem}.")) {
            return attention(stem, leaf);
        }
    }
    for stem in [
        "encoder.mid_block.resnets.0",
        "encoder.mid_block.resnets.1",
        "decoder.mid_block.resnets.0",
        "decoder.mid_block.resnets.1",
    ] {
        if let Some(leaf) = name.strip_prefix(&format!("{stem}.")) {
            return resnet(stem, leaf);
        }
    }
    Err(format!("unmapped diffusers VAE key {name}"))
}

fn encoder_down(rest: &str) -> std::result::Result<(String, Xform), String> {
    let (index, leaf) = split_index(rest)?;
    let (stage, slot, down) = match index {
        0 => (0, 0, false),
        1 => (0, 1, false),
        2 => (0, 0, true),
        3 => (1, 0, false),
        4 => (1, 1, false),
        5 => (1, 0, true),
        6 => (2, 0, false),
        7 => (2, 1, false),
        8 => (2, 0, true),
        9 => (3, 0, false),
        10 => (3, 1, false),
        other => return Err(format!("encoder down block {other} is not in the Qwen VAE")),
    };
    if down {
        sample(&format!("encoder.down_blocks.{stage}.downsamplers.0"), leaf)
    } else {
        resnet(&format!("encoder.down_blocks.{stage}.resnets.{slot}"), leaf)
    }
}

fn decoder_up(rest: &str) -> std::result::Result<(String, Xform), String> {
    let (block, leaf) = split_index(rest)?;
    if block > 3 {
        return Err(format!("decoder up block {block} is not in the Qwen VAE"));
    }
    let prefix = format!("decoder.up_block{block}");
    if let Some(leaf) = leaf.strip_prefix("resnets.") {
        let (slot, leaf) = split_index(leaf)?;
        return resnet(&format!("{prefix}.resnets.{slot}"), leaf);
    }
    if let Some(leaf) = leaf.strip_prefix("upsamplers.0.") {
        return sample(&format!("{prefix}.upsamplers.0"), leaf);
    }
    Err(format!("unmapped decoder up key {rest}"))
}

fn split_index(rest: &str) -> std::result::Result<(u32, &str), String> {
    let (index, leaf) = rest
        .split_once('.')
        .ok_or_else(|| format!("weight key {rest} has no module index"))?;
    let index = index
        .parse()
        .map_err(|_| format!("weight key index {index} is not a number"))?;
    Ok((index, leaf))
}

fn stem_conv(prefix: &str, leaf: &str) -> std::result::Result<(String, Xform), String> {
    match leaf {
        "weight" => Ok((format!("{prefix}.conv3d.weight"), Xform::Conv3d)),
        "bias" => Ok((format!("{prefix}.conv3d.bias"), Xform::None)),
        other => Err(format!("unmapped conv leaf {prefix}.{other}")),
    }
}

fn resnet(prefix: &str, leaf: &str) -> std::result::Result<(String, Xform), String> {
    match leaf {
        "conv1.weight" => Ok((format!("{prefix}.conv1.conv3d.weight"), Xform::Conv3d)),
        "conv1.bias" => Ok((format!("{prefix}.conv1.conv3d.bias"), Xform::None)),
        "conv2.weight" => Ok((format!("{prefix}.conv2.conv3d.weight"), Xform::Conv3d)),
        "conv2.bias" => Ok((format!("{prefix}.conv2.conv3d.bias"), Xform::None)),
        "conv_shortcut.weight" => Ok((format!("{prefix}.skip_conv.conv3d.weight"), Xform::Conv3d)),
        "conv_shortcut.bias" => Ok((format!("{prefix}.skip_conv.conv3d.bias"), Xform::None)),
        "norm1.gamma" => Ok((format!("{prefix}.norm1.weight"), Xform::Gamma)),
        "norm2.gamma" => Ok((format!("{prefix}.norm2.weight"), Xform::Gamma)),
        other => Err(format!("unmapped resnet leaf {prefix}.{other}")),
    }
}

fn sample(prefix: &str, leaf: &str) -> std::result::Result<(String, Xform), String> {
    match leaf {
        "resample.1.weight" => Ok((format!("{prefix}.resample_conv.weight"), Xform::Conv2d)),
        "resample.1.bias" => Ok((format!("{prefix}.resample_conv.bias"), Xform::None)),
        "time_conv.weight" => Ok((format!("{prefix}.time_conv.conv3d.weight"), Xform::Conv3d)),
        "time_conv.bias" => Ok((format!("{prefix}.time_conv.conv3d.bias"), Xform::None)),
        other => Err(format!("unmapped sampler leaf {prefix}.{other}")),
    }
}

fn attention(prefix: &str, leaf: &str) -> std::result::Result<(String, Xform), String> {
    match leaf {
        "norm.gamma" => Ok((format!("{prefix}.norm.weight"), Xform::Gamma)),
        "to_qkv.weight" => Ok((format!("{prefix}.to_qkv.weight"), Xform::Conv2d)),
        "to_qkv.bias" => Ok((format!("{prefix}.to_qkv.bias"), Xform::None)),
        "proj.weight" => Ok((format!("{prefix}.proj.weight"), Xform::Conv2d)),
        "proj.bias" => Ok((format!("{prefix}.proj.bias"), Xform::None)),
        other => Err(format!("unmapped attention leaf {prefix}.{other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::{dit_target, text_target, vae_target, Xform};

    #[test]
    fn diffusers_dit_names_match_the_flat_layout() {
        let cases = [
            (
                "transformer_blocks.3.attn.to_out.0.weight",
                "transformer_blocks.3.attn.attn_to_out.0.weight",
            ),
            (
                "transformer_blocks.3.img_mod.1.weight",
                "transformer_blocks.3.img_mod_linear.weight",
            ),
            (
                "transformer_blocks.3.txt_mod.1.bias",
                "transformer_blocks.3.txt_mod_linear.bias",
            ),
            (
                "transformer_blocks.3.img_mlp.net.0.proj.weight",
                "transformer_blocks.3.img_ff.mlp_in.weight",
            ),
            (
                "transformer_blocks.3.img_mlp.net.2.weight",
                "transformer_blocks.3.img_ff.mlp_out.weight",
            ),
            (
                "transformer_blocks.3.txt_mlp.net.0.proj.bias",
                "transformer_blocks.3.txt_ff.mlp_in.bias",
            ),
            (
                "transformer_blocks.3.txt_mlp.net.2.bias",
                "transformer_blocks.3.txt_ff.mlp_out.bias",
            ),
            ("img_in.weight", "img_in.weight"),
            ("txt_norm.weight", "txt_norm.weight"),
        ];
        for (source, expected) in cases {
            let (target, xform) = dit_target(source).unwrap();
            assert_eq!(target, expected, "{source}");
            assert_eq!(xform, Xform::None);
        }
    }

    #[test]
    fn diffusers_text_names_match_the_encoder_prefix() {
        let (embed, xform) = text_target("model.embed_tokens.weight").unwrap();
        assert_eq!(embed, "encoder.embed_tokens.weight");
        assert_eq!(xform, Xform::None);
        let (merger, _) = text_target("visual.merger.mlp.2.weight").unwrap();
        assert_eq!(merger, "encoder.visual.merger.mlp_1.weight");
        let (patch, xform) = text_target("visual.patch_embed.proj.weight").unwrap();
        assert_eq!(patch, "encoder.visual.patch_embed.proj.weight");
        assert_eq!(xform, Xform::Conv3d);
        let (head, _) = text_target("lm_head.weight").unwrap();
        assert_eq!(head, "lm_head.weight");
    }

    #[test]
    fn diffusers_vae_names_match_the_grouped_conv3d_layout() {
        let cases = [
            (
                "decoder.conv_in.weight",
                "decoder.conv_in.conv3d.weight",
                Xform::Conv3d,
            ),
            (
                "encoder.down_blocks.6.conv_shortcut.weight",
                "encoder.down_blocks.2.resnets.0.skip_conv.conv3d.weight",
                Xform::Conv3d,
            ),
            (
                "encoder.down_blocks.8.resample.1.weight",
                "encoder.down_blocks.2.downsamplers.0.resample_conv.weight",
                Xform::Conv2d,
            ),
            (
                "encoder.down_blocks.0.norm1.gamma",
                "encoder.down_blocks.0.resnets.0.norm1.weight",
                Xform::Gamma,
            ),
            (
                "decoder.up_blocks.1.resnets.0.conv_shortcut.bias",
                "decoder.up_block1.resnets.0.skip_conv.conv3d.bias",
                Xform::None,
            ),
            (
                "decoder.up_blocks.0.upsamplers.0.resample.1.weight",
                "decoder.up_block0.upsamplers.0.resample_conv.weight",
                Xform::Conv2d,
            ),
            (
                "decoder.mid_block.attentions.0.norm.gamma",
                "decoder.mid_block.attentions.0.norm.weight",
                Xform::Gamma,
            ),
            (
                "quant_conv.weight",
                "quant_conv.conv3d.weight",
                Xform::Conv3d,
            ),
        ];
        for (source, expected, kind) in cases {
            let (target, xform) = vae_target(source).unwrap();
            assert_eq!(target, expected, "{source}");
            assert_eq!(xform, kind);
        }
    }
}
