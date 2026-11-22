//! Diffusers / mflux Flux2 Klein tensor names.
//!
//! Published safetensors keep image Q, K, and V separate. A fused
//! `double_blocks.*.img_attn.qkv` name is not a Klein key.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitTensor {
    TimeIn,
    TimeOut,
    GuidanceIn,
    GuidanceOut,
    ImgIn,
    TxtIn,
    DoubleModImg,
    DoubleModTxt,
    SingleMod,
    DoubleImgQ(u32),
    DoubleImgK(u32),
    DoubleImgV(u32),
    DoubleImgOut(u32),
    DoubleImgNormQ(u32),
    DoubleImgNormK(u32),
    DoubleTxtQ(u32),
    DoubleTxtK(u32),
    DoubleTxtV(u32),
    DoubleTxtOut(u32),
    DoubleTxtNormQ(u32),
    DoubleTxtNormK(u32),
    DoubleImgMlpIn(u32),
    DoubleImgMlpOut(u32),
    DoubleTxtMlpIn(u32),
    DoubleTxtMlpOut(u32),
    SingleFused(u32),
    SingleOut(u32),
    SingleNormQ(u32),
    SingleNormK(u32),
    FinalNorm,
    ProjOut,
}

/// Diffusers wraps `to_out` in a Sequential and nests the timestep MLP under
/// `timestep_embedder`. mflux checkpoints are already flat. `to_add_out.0` is
/// the same Sequential wrapper when a file still has it.
pub fn remap_diffusers_key(name: &str) -> String {
    name.replace(".to_out.0.", ".to_out.")
        .replace(".to_add_out.0.", ".to_add_out.")
        .replace(
            "time_guidance_embed.timestep_embedder.",
            "time_guidance_embed.",
        )
}

/// `source -> target` for every name. A collision is a broken checkpoint.
pub fn remap_diffusers_names<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> Result<BTreeMap<String, String>, String> {
    let mut map = BTreeMap::new();
    let mut targets: BTreeMap<String, String> = BTreeMap::new();
    for source in names {
        let target = remap_diffusers_key(source);
        if let Some(previous) = targets.insert(target.clone(), source.to_string()) {
            return Err(format!(
                "Flux2 remap collision: {previous} and {source} both map to {target}"
            ));
        }
        map.insert(source.to_string(), target);
    }
    Ok(map)
}

/// Classify one safetensors name, including packed `.scales` / `.biases`.
pub fn classify_dit_tensor(name: &str) -> Result<DitTensor, String> {
    let stem = name
        .strip_suffix(".weight")
        .or_else(|| name.strip_suffix(".scales"))
        .or_else(|| name.strip_suffix(".biases"))
        .ok_or_else(|| format!("Flux2 tensor {name} is not a weight, scale, or bias"))?;
    classify_stem(stem).ok_or_else(|| format!("Flux2 tensor {name} is not a Klein DiT key"))
}

fn classify_stem(stem: &str) -> Option<DitTensor> {
    match stem {
        "time_guidance_embed.linear_1" => return Some(DitTensor::TimeIn),
        "time_guidance_embed.linear_2" => return Some(DitTensor::TimeOut),
        "time_guidance_embed.guidance_linear_1" => return Some(DitTensor::GuidanceIn),
        "time_guidance_embed.guidance_linear_2" => return Some(DitTensor::GuidanceOut),
        "x_embedder" => return Some(DitTensor::ImgIn),
        "context_embedder" => return Some(DitTensor::TxtIn),
        "double_stream_modulation_img.linear" => return Some(DitTensor::DoubleModImg),
        "double_stream_modulation_txt.linear" => return Some(DitTensor::DoubleModTxt),
        "single_stream_modulation.linear" => return Some(DitTensor::SingleMod),
        "norm_out.linear" => return Some(DitTensor::FinalNorm),
        "proj_out" => return Some(DitTensor::ProjOut),
        _ => {}
    }
    if let Some(rest) = stem.strip_prefix("transformer_blocks.") {
        let (index, tail) = split_index(rest)?;
        return Some(match tail {
            "attn.to_q" => DitTensor::DoubleImgQ(index),
            "attn.to_k" => DitTensor::DoubleImgK(index),
            "attn.to_v" => DitTensor::DoubleImgV(index),
            "attn.to_out" => DitTensor::DoubleImgOut(index),
            "attn.norm_q" => DitTensor::DoubleImgNormQ(index),
            "attn.norm_k" => DitTensor::DoubleImgNormK(index),
            "attn.add_q_proj" => DitTensor::DoubleTxtQ(index),
            "attn.add_k_proj" => DitTensor::DoubleTxtK(index),
            "attn.add_v_proj" => DitTensor::DoubleTxtV(index),
            "attn.to_add_out" => DitTensor::DoubleTxtOut(index),
            "attn.norm_added_q" => DitTensor::DoubleTxtNormQ(index),
            "attn.norm_added_k" => DitTensor::DoubleTxtNormK(index),
            "ff.linear_in" => DitTensor::DoubleImgMlpIn(index),
            "ff.linear_out" => DitTensor::DoubleImgMlpOut(index),
            "ff_context.linear_in" => DitTensor::DoubleTxtMlpIn(index),
            "ff_context.linear_out" => DitTensor::DoubleTxtMlpOut(index),
            _ => return None,
        });
    }
    if let Some(rest) = stem.strip_prefix("single_transformer_blocks.") {
        let (index, tail) = split_index(rest)?;
        return Some(match tail {
            "attn.to_qkv_mlp_proj" => DitTensor::SingleFused(index),
            "attn.to_out" => DitTensor::SingleOut(index),
            "attn.norm_q" => DitTensor::SingleNormQ(index),
            "attn.norm_k" => DitTensor::SingleNormK(index),
            _ => return None,
        });
    }
    None
}

fn split_index(rest: &str) -> Option<(u32, &str)> {
    let (index, tail) = rest.split_once('.')?;
    let index = index.parse().ok()?;
    Some((index, tail))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KleinLayout {
    pub double_blocks: u32,
    pub single_blocks: u32,
}

/// Every name must classify. Block indices must be contiguous from 0.
/// Depth is whatever the checkpoint contains (Klein 4B is 5 + 20, 9B is 8 + 24).
pub fn klein_layout<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<KleinLayout, String> {
    let mut double = BTreeSet::new();
    let mut single = BTreeSet::new();
    let mut seen = HashSet::new();
    for name in names {
        let tensor = classify_dit_tensor(name)?;
        seen.insert(std::mem::discriminant(&tensor));
        match tensor {
            DitTensor::DoubleImgQ(i)
            | DitTensor::DoubleImgK(i)
            | DitTensor::DoubleImgV(i)
            | DitTensor::DoubleImgOut(i)
            | DitTensor::DoubleImgNormQ(i)
            | DitTensor::DoubleImgNormK(i)
            | DitTensor::DoubleTxtQ(i)
            | DitTensor::DoubleTxtK(i)
            | DitTensor::DoubleTxtV(i)
            | DitTensor::DoubleTxtOut(i)
            | DitTensor::DoubleTxtNormQ(i)
            | DitTensor::DoubleTxtNormK(i)
            | DitTensor::DoubleImgMlpIn(i)
            | DitTensor::DoubleImgMlpOut(i)
            | DitTensor::DoubleTxtMlpIn(i)
            | DitTensor::DoubleTxtMlpOut(i) => {
                double.insert(i);
            }
            DitTensor::SingleFused(i)
            | DitTensor::SingleOut(i)
            | DitTensor::SingleNormQ(i)
            | DitTensor::SingleNormK(i) => {
                single.insert(i);
            }
            _ => {}
        }
    }
    if double.is_empty() || single.is_empty() {
        return Err("Flux2 Klein needs at least one double block and one single block".into());
    }
    for (label, index) in [("double", &double), ("single", &single)] {
        let expected = index.len() as u32;
        if index.iter().copied().ne(0..expected) {
            return Err(format!(
                "Flux2 Klein {label} block indices are not 0..{expected}"
            ));
        }
    }
    for required in [
        std::mem::discriminant(&DitTensor::TimeIn),
        std::mem::discriminant(&DitTensor::TimeOut),
        std::mem::discriminant(&DitTensor::ImgIn),
        std::mem::discriminant(&DitTensor::TxtIn),
        std::mem::discriminant(&DitTensor::DoubleModImg),
        std::mem::discriminant(&DitTensor::DoubleModTxt),
        std::mem::discriminant(&DitTensor::SingleMod),
        std::mem::discriminant(&DitTensor::FinalNorm),
        std::mem::discriminant(&DitTensor::ProjOut),
        std::mem::discriminant(&DitTensor::DoubleImgQ(0)),
        std::mem::discriminant(&DitTensor::SingleFused(0)),
    ] {
        if !seen.contains(&required) {
            return Err("Flux2 Klein bundle is missing a required DiT tensor".into());
        }
    }
    Ok(KleinLayout {
        double_blocks: double.len() as u32,
        single_blocks: single.len() as u32,
    })
}

/// Klein 4B is the 5 double + 20 single checkpoint.
pub fn klein_4b_layout<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> Result<KleinLayout, String> {
    let layout = klein_layout(names)?;
    if layout.double_blocks != 5 || layout.single_blocks != 20 {
        return Err(format!(
            "Flux2 Klein 4B needs 5 double blocks and 20 single blocks, got {} and {}",
            layout.double_blocks, layout.single_blocks
        ));
    }
    Ok(layout)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn klein_4b_names() -> Vec<String> {
        let mut names = vec![
            "time_guidance_embed.linear_1.weight".into(),
            "time_guidance_embed.linear_2.weight".into(),
            "x_embedder.weight".into(),
            "context_embedder.weight".into(),
            "double_stream_modulation_img.linear.weight".into(),
            "double_stream_modulation_txt.linear.weight".into(),
            "single_stream_modulation.linear.weight".into(),
            "norm_out.linear.weight".into(),
            "proj_out.weight".into(),
        ];
        for i in 0..5 {
            for tail in [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out",
                "attn.norm_q",
                "attn.norm_k",
                "attn.add_q_proj",
                "attn.add_k_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "attn.norm_added_q",
                "attn.norm_added_k",
                "ff.linear_in",
                "ff.linear_out",
                "ff_context.linear_in",
                "ff_context.linear_out",
            ] {
                names.push(format!("transformer_blocks.{i}.{tail}.weight"));
            }
        }
        for i in 0..20 {
            for tail in [
                "attn.to_qkv_mlp_proj",
                "attn.to_out",
                "attn.norm_q",
                "attn.norm_k",
            ] {
                names.push(format!("single_transformer_blocks.{i}.{tail}.scales"));
            }
        }
        names
    }

    #[test]
    fn klein_4b_names_match_the_published_layout() {
        let names = klein_4b_names();
        let layout = klein_4b_layout(names.iter().map(String::as_str)).unwrap();
        assert_eq!(
            layout,
            KleinLayout {
                double_blocks: 5,
                single_blocks: 20
            }
        );
    }

    #[test]
    fn nine_billion_depth_and_guidance_embedder_classify() {
        let mut names = klein_4b_names();
        for i in 5..8 {
            names.push(format!("transformer_blocks.{i}.attn.to_q.weight"));
        }
        for i in 20..24 {
            names.push(format!(
                "single_transformer_blocks.{i}.attn.to_qkv_mlp_proj.weight"
            ));
        }
        names.push("time_guidance_embed.guidance_linear_1.weight".into());
        names.push("time_guidance_embed.guidance_linear_2.biases".into());
        let layout = klein_layout(names.iter().map(String::as_str)).unwrap();
        assert_eq!(layout.double_blocks, 8);
        assert_eq!(layout.single_blocks, 24);
        assert!(klein_4b_layout(names.iter().map(String::as_str)).is_err());
        assert_eq!(
            classify_dit_tensor("time_guidance_embed.guidance_linear_1.weight").unwrap(),
            DitTensor::GuidanceIn
        );
    }

    #[test]
    fn diffusers_wrappers_flatten_onto_klein_names() {
        assert_eq!(
            remap_diffusers_key("transformer_blocks.0.attn.to_out.0.weight"),
            "transformer_blocks.0.attn.to_out.weight"
        );
        assert_eq!(
            remap_diffusers_key("transformer_blocks.0.attn.to_add_out.0.weight"),
            "transformer_blocks.0.attn.to_add_out.weight"
        );
        assert_eq!(
            remap_diffusers_key("time_guidance_embed.timestep_embedder.linear_1.weight"),
            "time_guidance_embed.linear_1.weight"
        );
        assert_eq!(
            remap_diffusers_key("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight"),
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight"
        );
        let err = remap_diffusers_names([
            "time_guidance_embed.timestep_embedder.linear_1.weight",
            "time_guidance_embed.linear_1.weight",
        ])
        .unwrap_err();
        assert!(err.contains("collision"), "{err}");
    }

    #[test]
    fn fused_gguf_qkv_is_not_a_klein_key() {
        let err = classify_dit_tensor("double_blocks.0.img_attn.qkv.weight").unwrap_err();
        assert!(err.contains("not a Klein DiT key"), "{err}");
    }

    #[test]
    fn installed_klein_4b_q4_index_matches_when_present() {
        let path = Path::new(
            "/Users/nil.luo/Workspace/studio-workspace/models/Image/flux2-klein-4b-fp16/transformer/model.safetensors.index.json",
        );
        let Some(text) = std::fs::read_to_string(&path).ok() else {
            return;
        };
        let index: serde_json::Value = serde_json::from_str(&text).unwrap();
        let map = index
            .get("weight_map")
            .and_then(|value| value.as_object())
            .expect("weight_map");
        let names: Vec<&str> = map.keys().map(String::as_str).collect();
        let layout = klein_4b_layout(names).unwrap();
        assert_eq!(layout.double_blocks, 5);
        assert_eq!(layout.single_blocks, 20);
    }
}
