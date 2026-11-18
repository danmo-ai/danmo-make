use crate::weights::WeightError;
use std::collections::BTreeMap;

pub fn remap_weight_names<I, S>(
    names: I,
    patch_size: i64,
) -> Result<BTreeMap<String, String>, WeightError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    if patch_size <= 0 {
        return Err(WeightError::NotBuilt(format!(
            "Z-Image patch size must be positive, got {patch_size}"
        )));
    }
    let prefix = format!("{patch_size}-1");
    let mut output = BTreeMap::new();
    let mut targets = BTreeMap::new();
    for source in names {
        let source = source.as_ref();
        let target = remap_name(source, &prefix);
        if let Some(previous) = targets.insert(target.clone(), source.to_string()) {
            return Err(WeightError::NotBuilt(format!(
                "Z-Image remap collision: {previous} and {source} both map to {target}"
            )));
        }
        output.insert(source.to_string(), target);
    }
    Ok(output)
}

pub fn remap_control_name(name: &str, patch_size: i64) -> Result<Option<String>, WeightError> {
    if patch_size <= 0 {
        return Err(WeightError::NotBuilt(format!(
            "Z-Image patch size must be positive, got {patch_size}"
        )));
    }
    if !name.starts_with("control_") {
        return Ok(None);
    }
    let prefix = format!("{patch_size}-1");
    let target = name
        .replace(
            &format!("control_all_x_embedder.{prefix}"),
            "control_x_embedder",
        )
        .replace("attention_norm1.", "attn_norm1.")
        .replace("attention_norm2.", "attn_norm2.")
        .replace(".to_out.0.", ".to_out.")
        .replace(".adaLN_modulation.1.", ".adaLN_modulation.0.");
    Ok(Some(target))
}

pub fn remap_lora_module_prefix(module: &str, patch_size: i64) -> Result<String, WeightError> {
    if patch_size <= 0 {
        return Err(WeightError::NotBuilt(format!(
            "Z-Image patch size must be positive, got {patch_size}"
        )));
    }
    let mut module = module.trim().trim_matches('.').to_string();
    while module.contains("..") {
        module = module.replace("..", ".");
    }
    for prefix in [
        "transformer.",
        "z_image_transformer.",
        "diffusion_model.",
        "pipe.transformer.",
        "model.",
    ] {
        if let Some(rest) = module.strip_prefix(prefix) {
            module = rest.to_string();
            break;
        }
    }
    let prefix = format!("{patch_size}-1");
    module = module
        .replace(".base_model.model.", ".")
        .replace(".default.", ".")
        .replace(&format!("all_x_embedder.{prefix}"), "x_embedder")
        .replace(&format!("all_final_layer.{prefix}"), "final_layer")
        .replace("attention_norm1.", "attn_norm1.")
        .replace("attention_norm2.", "attn_norm2.")
        .replace(".to_out.0.", ".to_out.")
        .replace("t_embedder.mlp.0.", "t_embedder.linear1.")
        .replace("t_embedder.mlp.2.", "t_embedder.linear2.")
        .replace("cap_embedder.0.", "cap_norm.")
        .replace("cap_embedder.1.", "cap_embedder.")
        .replace(".adaLN_modulation.1.", ".adaLN_modulation.0.");
    if let Some(rest) = module.strip_suffix(".default") {
        module = rest.trim_end_matches('.').to_string();
    }
    if let Some(rest) = module.strip_suffix(".weight") {
        module = rest.trim_end_matches('.').to_string();
    } else if let Some(rest) = module.strip_suffix(".bias") {
        module = rest.trim_end_matches('.').to_string();
    }
    Ok(module.trim_matches('.').to_string())
}

fn remap_name(name: &str, prefix: &str) -> String {
    name.replace(&format!("all_x_embedder.{prefix}"), "x_embedder")
        .replace(&format!("all_final_layer.{prefix}"), "final_layer")
        .replace("t_embedder.mlp.0.", "t_embedder.linear1.")
        .replace("t_embedder.mlp.2.", "t_embedder.linear2.")
        .replace("attention_norm1.", "attn_norm1.")
        .replace("attention_norm2.", "attn_norm2.")
        .replace(".to_out.0.", ".to_out.")
        .replace(".adaLN_modulation.1.", ".adaLN_modulation.0.")
        .replace("cap_embedder.0.", "cap_norm.")
        .replace("cap_embedder.1.", "cap_embedder.")
}

#[cfg(test)]
mod tests {
    use super::{remap_control_name, remap_lora_module_prefix, remap_weight_names};

    #[test]
    fn diffusers_names_match_main_remap() {
        let names = [
            "all_x_embedder.2-1.weight",
            "all_final_layer.2-1.adaLN_modulation.1.bias",
            "t_embedder.mlp.0.weight",
            "t_embedder.mlp.2.weight",
            "layers.0.attention_norm1.weight",
            "layers.0.attention_norm2.weight",
            "layers.0.attention.to_out.0.weight",
            "cap_embedder.0.weight",
            "cap_embedder.1.bias",
        ];
        let mapped = remap_weight_names(names, 2).unwrap();
        assert_eq!(mapped[names[0]], "x_embedder.weight");
        assert_eq!(mapped[names[1]], "final_layer.adaLN_modulation.0.bias");
        assert_eq!(mapped[names[2]], "t_embedder.linear1.weight");
        assert_eq!(mapped[names[3]], "t_embedder.linear2.weight");
        assert_eq!(mapped[names[4]], "layers.0.attn_norm1.weight");
        assert_eq!(mapped[names[5]], "layers.0.attn_norm2.weight");
        assert_eq!(mapped[names[6]], "layers.0.attention.to_out.weight");
        assert_eq!(mapped[names[7]], "cap_norm.weight");
        assert_eq!(mapped[names[8]], "cap_embedder.bias");
    }

    #[test]
    fn collisions_and_invalid_patch_sizes_fail_loud() {
        let error = remap_weight_names(["cap_embedder.0.weight", "cap_norm.weight"], 2)
            .unwrap_err()
            .to_string();
        assert!(error.contains("remap collision"), "{error}");
        assert!(remap_weight_names(["x"], 0).is_err());
    }

    #[test]
    #[ignore]
    fn real_dit_shards_cover_the_official_index_without_remap_collisions() {
        let root = std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_DIT_DIR").unwrap());
        let stream = danmo_mlx::Stream::cpu().unwrap();
        let mut sources = std::collections::BTreeSet::new();
        let mut targets = std::collections::BTreeSet::new();
        for shard in 1..=3 {
            let path = root.join(format!(
                "diffusion_pytorch_model-{shard:05}-of-00003.safetensors"
            ));
            let tensors = danmo_mlx::SafeTensors::load(&path, &stream).unwrap();
            let names = tensors.names().unwrap();
            let mapped = remap_weight_names(names, 2).unwrap();
            for (source, target) in mapped {
                assert!(sources.insert(source), "duplicate source tensor");
                assert!(targets.insert(target), "remap collision across shards");
            }
        }
        assert_eq!(sources.len(), 521);
        let first = danmo_mlx::SafeTensors::load(
            &root.join("diffusion_pytorch_model-00001-of-00003.safetensors"),
            &stream,
        )
        .unwrap();
        for (name, shape) in [
            ("all_x_embedder.2-1.weight", vec![3840, 64]),
            ("cap_embedder.1.weight", vec![3840, 2560]),
            ("layers.0.attention.to_q.weight", vec![3840, 3840]),
            ("all_final_layer.2-1.linear.weight", vec![64, 3840]),
        ] {
            assert_eq!(first.get(name).unwrap().unwrap().shape(), shape, "{name}");
        }
    }

    /// Published prequantized bundle (mflux mlx-q4/q8): every shard loads,
    /// the remap has no collisions, affine linears stay packed, and the
    /// registry bits/group size validate against the shard metadata.
    #[test]
    #[ignore]
    fn real_quantized_bundle_loads_packed_without_remap_collisions() {
        let root =
            std::path::PathBuf::from(std::env::var("DANQING_TEST_ZIMAGE_QUANT_DIT_DIR").unwrap());
        let bits: i32 = std::env::var("DANQING_TEST_ZIMAGE_QUANT_BITS")
            .unwrap_or_else(|_| "4".into())
            .parse()
            .unwrap();
        let config = crate::families::z_image::mlx::MlxZImageConfig::published();
        let quantization = danmo_mlx::AffineQuantization::new(bits, 64).unwrap();
        let weights = crate::families::z_image::mlx::model::load_directory(
            &root,
            config,
            crate::weights::WeightDtype::Native,
            Some(quantization),
        )
        .unwrap();
        assert_eq!(weights.quantization(), Some(quantization));
        let packed: Vec<&str> = weights.packed_names().collect();
        assert!(
            packed.len() > 100,
            "expected the DiT linears to stay packed: {}",
            packed.len()
        );
        // Every packed linear must answer through LinearWeight and keep the
        // checkpoint group size; spot-check the first block projections.
        for name in [
            "layers.0.attention.to_q.weight",
            "layers.0.attention.to_out.weight",
            "layers.0.feed_forward.w1.weight",
            "layers.29.feed_forward.w2.weight",
        ] {
            let linear = weights.linear(name).unwrap();
            assert_eq!(linear.compute_dtype(), danmo_mlx::Dtype::BFloat16);
        }
        // Norms / pad tokens stay dense tensors.
        for name in ["layers.0.attn_norm1.weight", "x_pad_token", "cap_pad_token"] {
            assert!(weights.tensor(name).is_ok(), "{name} must stay dense");
        }
        println!(
            "quantized bundle: {} packed linears, {} resident bytes",
            packed.len(),
            weights.nbytes()
        );
    }

    #[test]
    fn control_and_lora_prefixes_match_main() {
        assert_eq!(
            remap_control_name("control_all_x_embedder.2-1.weight", 2).unwrap(),
            Some("control_x_embedder.weight".into())
        );
        assert_eq!(remap_control_name("layers.0.weight", 2).unwrap(), None);
        assert_eq!(
            remap_lora_module_prefix("diffusion_model.layers.0.attention.to_out.0.weight", 2)
                .unwrap(),
            "layers.0.attention.to_out"
        );
        assert_eq!(
            remap_lora_module_prefix("transformer.cap_embedder.0.weight", 2).unwrap(),
            "cap_norm"
        );
    }
}
