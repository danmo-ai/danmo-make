mod common;
mod families;
mod pipeline;
pub mod runtime;

pub use families::{
    all_specs, load_safetensors_f32, remap_flux2_lora_target, remap_flux2_weights, spec,
    FamilySpec, Flux2Config, Flux2Denoise, Media, Shape,
};
pub use pipeline::{audio_create, image_create, upscale, video_create, EngineError, RunOutput};
pub use runtime::{identity_probe, BackendKind, GgmlContext};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::{remap_flux2_lora_target, remap_flux2_weights};
    use crate::runtime::identity_probe;
    use std::collections::BTreeMap;

    #[test]
    fn graph_is_reserved_and_replayed() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut graph = identity_probe(&ctx).unwrap();
        graph.set_f32("w", &[1.0, 0.0, 0.0, 1.0]).unwrap();
        graph.set_f32("x", &[3.0, 4.0]).unwrap();
        let y1 = graph.compute().unwrap();
        graph.set_f32("x", &[1.0, 2.0]).unwrap();
        let y2 = graph.compute().unwrap();
        assert_eq!(graph.compute_count(), 2);
        assert_ne!(y1, y2);
        for (got, exp) in y2.iter().zip([1.0f32, 2.0]) {
            assert!((got - exp).abs() < 1e-4, "got {y2:?}");
        }
    }

    #[test]
    fn flux2_remap_matches_python_rules() {
        let mut weights = BTreeMap::new();
        weights.insert("blocks.0.attn.to_out.0.weight".into(), vec![1.0f32]);
        weights.insert(
            "time_guidance_embed.timestep_embedder.linear_1.weight".into(),
            vec![2.0f32],
        );
        let mapped = remap_flux2_weights(weights);
        assert!(mapped.contains_key("blocks.0.attn.to_out.weight"));
        assert!(mapped.contains_key("time_guidance_embed.linear_1.weight"));
        assert_eq!(
            remap_flux2_lora_target("diffusion_model.double_blocks.3.img_mlp.0.lora_A"),
            Some("transformer_blocks.3.ff.linear_in".into())
        );
        assert_eq!(
            remap_flux2_lora_target("double_blocks.1.img_attn.norm.lora_down"),
            None
        );
    }

    #[test]
    fn flux2_denoise_reuses_one_graph() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let config = Flux2Config::smoke();
        let mut model = families::Flux2Denoise::build(&ctx, config).unwrap();
        model
            .load_weight_map(&BTreeMap::new(), config.inner_dim as usize)
            .unwrap();
        let mut ticks = 0u32;
        let out = model
            .denoise(
                vec![0.2; config.inner_dim as usize],
                3,
                64,
                64,
                true,
                3.5,
                |_, _| {
                    ticks += 1;
                },
            )
            .unwrap();
        assert_eq!(ticks, 3);
        assert_eq!(model.graph_computes(), 3);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn unknown_family_and_unbuilt_graph_fail_loud() {
        let req = danmo_contracts::ImageGenerationRequest {
            model: "nope".into(),
            title: String::new(),
            prompt: "a cat".into(),
            negative_prompt: String::new(),
            size: "64x64".into(),
            n: 1,
            steps: Some(1),
            guidance: None,
            seed: Some(1),
            scheduler: None,
            priority: "normal".into(),
        };
        assert!(image_create(&req, "not-a-family", BackendKind::Cpu, |_, _| ()).is_err());
        assert!(image_create(&req, "z_image", BackendKind::Cpu, |_, _| ()).is_err());
        assert!(video_create("wan").is_err());
        assert!(audio_create("ace_step").is_err());
        assert!(upscale("esrgan").is_err());
        assert!(spec("minimax_h3").unwrap().shape == Shape::GenerationFacade);
        assert_eq!(all_specs().len(), 23);
    }

    #[test]
    fn missing_backend_fails_loud() {
        assert!(GgmlContext::new(BackendKind::Cuda).is_err());
    }

    #[test]
    fn safetensors_file_is_remapped_before_the_graph_sees_it() {
        let dir = std::env::temp_dir().join(format!("flux2-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.safetensors");
        let values = [1.0f32, 0.0, 0.0, 1.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], &bytes)
                .unwrap();
        let file =
            safetensors::serialize(vec![("blocks.0.attn.to_out.0.weight", view)], &None).unwrap();
        std::fs::write(&path, file).unwrap();
        let mapped = load_safetensors_f32(&path).unwrap();
        let weight = mapped.get("blocks.0.attn.to_out.weight").unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut graph = identity_probe(&ctx).unwrap();
        graph.set_f32("w", weight).unwrap();
        graph.set_f32("x", &[5.0, 6.0]).unwrap();
        let y = graph.compute().unwrap();
        assert!((y[0] - 5.0).abs() < 1e-4 && (y[1] - 6.0).abs() < 1e-4);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
