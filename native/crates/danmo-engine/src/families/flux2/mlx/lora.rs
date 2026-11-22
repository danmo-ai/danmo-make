//! Flux2 LoRA names, matching danmo-make `families/flux2/weights.py`
//! `remap_flux2_lora_keys`.
//!
//! BFL `double_blocks.*.img_attn.qkv` and `txt_attn.qkv` are split into the
//! separate projections this graph stores. `single_blocks.*.linear2` is
//! `attn.to_out`. RMS norms are not linears, so those keys are dropped; a
//! file that then has no linear factors fails.

use crate::weights::{LoraRoute, WeightError};

pub(super) fn route_flux2(module: &str) -> Result<LoraRoute, WeightError> {
    let mut module = module.trim().trim_matches('.').to_string();
    module = module.replace(".to_out.0", ".to_out");
    module = module.replace(".to_add_out.0", ".to_add_out");
    module = module.replace(
        "time_guidance_embed.timestep_embedder.",
        "time_guidance_embed.",
    );
    if let Some(block) = block_tail(&module, "double_blocks", "img_attn.qkv") {
        return Ok(split_attn(block, &["to_q", "to_k", "to_v"]));
    }
    if let Some(block) = block_tail(&module, "double_blocks", "txt_attn.qkv") {
        return Ok(split_attn(
            block,
            &["add_q_proj", "add_k_proj", "add_v_proj"],
        ));
    }
    if let Some(block) = block_tail(&module, "double_blocks", "img_attn.proj") {
        module = format!("transformer_blocks.{block}.attn.to_out");
    } else if let Some(block) = block_tail(&module, "double_blocks", "txt_attn.proj") {
        module = format!("transformer_blocks.{block}.attn.to_add_out");
    } else if let Some(block) = block_tail(&module, "double_blocks", "img_mlp.0") {
        module = format!("transformer_blocks.{block}.ff.linear_in");
    } else if let Some(block) = block_tail(&module, "double_blocks", "img_mlp.2") {
        module = format!("transformer_blocks.{block}.ff.linear_out");
    } else if let Some(block) = block_tail(&module, "double_blocks", "txt_mlp.0") {
        module = format!("transformer_blocks.{block}.ff_context.linear_in");
    } else if let Some(block) = block_tail(&module, "double_blocks", "txt_mlp.2") {
        module = format!("transformer_blocks.{block}.ff_context.linear_out");
    } else if let Some(block) = block_tail(&module, "single_blocks", "linear1") {
        module = format!("single_transformer_blocks.{block}.attn.to_qkv_mlp_proj");
    } else if let Some(block) = block_tail(&module, "single_blocks", "linear2") {
        module = format!("single_transformer_blocks.{block}.attn.to_out");
    } else if module == "img_in" {
        module = "x_embedder".to_string();
    } else if module == "txt_in" {
        module = "context_embedder".to_string();
    } else if module == "time_in.in_layer" {
        module = "time_guidance_embed.linear_1".to_string();
    } else if module == "time_in.out_layer" {
        module = "time_guidance_embed.linear_2".to_string();
    } else if module == "final_layer.linear" {
        module = "proj_out".to_string();
    } else if module == "double_stream_modulation_img.lin"
        || module == "double_stream_modulation_txt.lin"
        || module == "single_stream_modulation.lin"
    {
        module = format!(
            "{}.linear",
            module.strip_suffix(".lin").unwrap_or(module.as_str())
        );
    } else if module.contains("modulation") && module.contains("single") {
        module = "single_stream_modulation.linear".to_string();
    } else if module.contains("img_mod") {
        module = "double_stream_modulation_img.linear".to_string();
    } else if module.contains("txt_mod") {
        module = "double_stream_modulation_txt.linear".to_string();
    }
    if module.contains("norm") && !module.contains("modulation") {
        return Ok(LoraRoute::Skip);
    }
    Ok(LoraRoute::One(module))
}

fn split_attn(block: u32, suffixes: &[&str]) -> LoraRoute {
    LoraRoute::Split(
        suffixes
            .iter()
            .map(|suffix| format!("transformer_blocks.{block}.attn.{suffix}"))
            .collect(),
    )
}

fn block_tail(module: &str, head: &str, tail: &str) -> Option<u32> {
    let rest = module.strip_prefix(head)?.strip_prefix('.')?;
    let (index, rest) = rest.split_once('.')?;
    if rest != tail {
        return None;
    }
    index.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bfl_names_land_on_the_klein_parameters() {
        assert_eq!(
            route_flux2("double_blocks.0.img_attn.proj").unwrap(),
            LoraRoute::One("transformer_blocks.0.attn.to_out".into())
        );
        assert_eq!(
            route_flux2("double_blocks.2.img_mlp.0").unwrap(),
            LoraRoute::One("transformer_blocks.2.ff.linear_in".into())
        );
        assert_eq!(
            route_flux2("double_blocks.2.txt_mlp.2").unwrap(),
            LoraRoute::One("transformer_blocks.2.ff_context.linear_out".into())
        );
        assert_eq!(
            route_flux2("single_blocks.4.linear1").unwrap(),
            LoraRoute::One("single_transformer_blocks.4.attn.to_qkv_mlp_proj".into())
        );
        assert_eq!(
            route_flux2("single_blocks.4.linear2").unwrap(),
            LoraRoute::One("single_transformer_blocks.4.attn.to_out".into())
        );
        assert_eq!(
            route_flux2("transformer_blocks.0.attn.to_q").unwrap(),
            LoraRoute::One("transformer_blocks.0.attn.to_q".into())
        );
        assert_eq!(
            route_flux2("double_blocks.1.img_mod.lin").unwrap(),
            LoraRoute::One("double_stream_modulation_img.linear".into())
        );
        assert_eq!(
            route_flux2("img_in").unwrap(),
            LoraRoute::One("x_embedder".into())
        );
        assert_eq!(
            route_flux2("txt_in").unwrap(),
            LoraRoute::One("context_embedder".into())
        );
        assert_eq!(
            route_flux2("time_in.in_layer").unwrap(),
            LoraRoute::One("time_guidance_embed.linear_1".into())
        );
        assert_eq!(
            route_flux2("time_in.out_layer").unwrap(),
            LoraRoute::One("time_guidance_embed.linear_2".into())
        );
        assert_eq!(
            route_flux2("final_layer.linear").unwrap(),
            LoraRoute::One("proj_out".into())
        );
        assert_eq!(
            route_flux2("double_stream_modulation_img.lin").unwrap(),
            LoraRoute::One("double_stream_modulation_img.linear".into())
        );
        assert_eq!(
            route_flux2("single_stream_modulation.lin").unwrap(),
            LoraRoute::One("single_stream_modulation.linear".into())
        );
        assert_eq!(
            route_flux2("single_blocks.0.modulation.lin").unwrap(),
            LoraRoute::One("single_stream_modulation.linear".into())
        );
        assert_eq!(
            route_flux2("double_blocks.0.img_attn.norm.query_norm").unwrap(),
            LoraRoute::Skip
        );
        assert_eq!(
            route_flux2("double_blocks.3.img_attn.qkv").unwrap(),
            LoraRoute::Split(vec![
                "transformer_blocks.3.attn.to_q".into(),
                "transformer_blocks.3.attn.to_k".into(),
                "transformer_blocks.3.attn.to_v".into(),
            ])
        );
        assert_eq!(
            route_flux2("double_blocks.1.txt_attn.qkv").unwrap(),
            LoraRoute::Split(vec![
                "transformer_blocks.1.attn.add_q_proj".into(),
                "transformer_blocks.1.attn.add_k_proj".into(),
                "transformer_blocks.1.attn.add_v_proj".into(),
            ])
        );
    }
}
