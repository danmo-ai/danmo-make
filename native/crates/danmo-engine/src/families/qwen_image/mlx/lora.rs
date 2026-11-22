//! Qwen-Image LoRA names, matching danmo-make
//! `families/qwen/weights_mlx.py::remap_qwen_lora_module_prefix`.
//!
//! HF/diffusers module paths map onto the flat DiT layout this graph stores.
//! RMS norms are not linears, so those keys are skipped; a file that then has
//! no linear factors fails in the shared weight layer.

use crate::weights::{LoraRoute, WeightError};

pub(super) fn route_qwen_image(module: &str) -> Result<LoraRoute, WeightError> {
    let mut module = module.trim().trim_matches('.').to_string();
    while module.contains("..") {
        module = module.replace("..", ".");
    }
    for prefix in [
        "transformer.",
        "qwen_image_transformer.",
        "transformer.transformer.",
        "diffusion_model.",
        "pipe.transformer.",
        "model.",
    ] {
        if let Some(rest) = module.strip_prefix(prefix) {
            module = rest.to_string();
            break;
        }
    }
    let module = module.replace(".base_model.model.", ".");
    // Strip the tensor suffix first so `.net.2.weight` still matches the
    // module rewrite (main applies the regexes before suffix removal).
    let module = module
        .strip_suffix(".weight")
        .or_else(|| module.strip_suffix(".bias"))
        .unwrap_or(&module)
        .to_string();
    let module = replace_suffix(&module, ".img_mlp.net.0.proj", ".img_ff.mlp_in");
    let module = replace_suffix(&module, ".img_mlp.net.2", ".img_ff.mlp_out");
    let module = replace_suffix(&module, ".txt_mlp.net.0.proj", ".txt_ff.mlp_in");
    let module = replace_suffix(&module, ".txt_mlp.net.2", ".txt_ff.mlp_out");
    let module = replace_suffix(&module, ".img_mod.1", ".img_mod_linear");
    let module = replace_suffix(&module, ".txt_mod.1", ".txt_mod_linear");
    let module = replace_suffix(&module, ".attn.to_out.0", ".attn.attn_to_out.0");
    let module = replace_suffix(&module, ".attn.to_out", ".attn.attn_to_out.0");
    let module = replace_suffix(&module, ".attn.attn_to_out", ".attn.attn_to_out.0");
    let module = module.trim_matches('.').to_string();
    if module.contains("norm") && !module.contains("mod") {
        return Ok(LoraRoute::Skip);
    }
    Ok(LoraRoute::One(module))
}

fn replace_suffix(module: &str, from: &str, to: &str) -> String {
    match module.strip_suffix(from) {
        Some(stem) => format!("{stem}{to}"),
        None => module.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::route_qwen_image;
    use crate::weights::LoraRoute;

    #[test]
    fn diffusers_paths_map_to_the_flat_dit_layout() {
        let cases = [
            (
                "transformer.transformer_blocks.3.img_mlp.net.0.proj",
                "transformer_blocks.3.img_ff.mlp_in",
            ),
            (
                "diffusion_model.transformer_blocks.7.attn.to_out.0",
                "transformer_blocks.7.attn.attn_to_out.0",
            ),
            (
                "transformer.transformer_blocks.1.img_mod.1",
                "transformer_blocks.1.img_mod_linear",
            ),
            (
                "transformer_blocks.2.txt_mlp.net.2.weight",
                "transformer_blocks.2.txt_ff.mlp_out",
            ),
        ];
        for (input, expected) in cases {
            match route_qwen_image(input).unwrap() {
                LoraRoute::One(module) => assert_eq!(module, expected, "{input}"),
                other => panic!("{input} routed to {other:?}"),
            }
        }
        assert!(matches!(
            route_qwen_image("transformer_blocks.0.attn.norm_q").unwrap(),
            LoraRoute::Skip
        ));
    }
}
