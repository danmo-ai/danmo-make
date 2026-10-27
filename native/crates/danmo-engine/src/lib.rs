mod common;
mod families;
mod pipeline;
pub mod runtime;
mod weights;

pub use families::{
    all_specs, spec, weight_layout, write_chain_bundle, write_config_bundle, write_q8_bundle,
    write_qwen3_bundle, write_smoke_bundle, FamilySpec, Flux2Config, Flux2Denoise, Media, Shape,
};
pub use pipeline::{audio_create, image_create, upscale, video_create, EngineError, RunOutput};
pub use runtime::{identity_probe, BackendKind, GgmlContext};
pub use weights::{assert_gguf_bundle, lora_base_name, write_f32_gguf, GgufWeights, WeightError};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::identity_probe;
    use danmo_contracts::ImageGenerationRequest;
    use std::path::PathBuf;

    fn request() -> ImageGenerationRequest {
        ImageGenerationRequest {
            model: "flux2-klein-4b".into(),
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
        }
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
        dir
    }

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
    fn gguf_tensor_name_is_not_rewritten() {
        let dir = scratch("gguf-name");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.gguf");
        let stored = "blocks.0.attn.to_out.0.weight";
        write_f32_gguf(&path, stored, 2, 2, &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let loaded = GgufWeights::open(&path).unwrap();
        assert!(loaded.contains(stored));
        assert!(!loaded.contains("blocks.0.attn.to_out.weight"));
        let weight = loaded.f32_exact(stored, &[2, 2]).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut graph = identity_probe(&ctx).unwrap();
        graph.set_f32("w", &weight).unwrap();
        graph.set_f32("x", &[5.0, 6.0]).unwrap();
        let y = graph.compute().unwrap();
        assert!((y[0] - 5.0).abs() < 1e-4 && (y[1] - 6.0).abs() < 1e-4);
        assert_eq!(
            lora_base_name("velocity.weight.lora_a"),
            Some("velocity.weight")
        );
        assert_eq!(
            lora_base_name("diffusion_model.double_blocks.3.img_mlp.0.lora_A"),
            None
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn safetensors_and_legacy_dirs_are_refused() {
        let dir = scratch("legacy");
        std::fs::create_dir_all(&dir).unwrap();
        let stale = dir.join("w.safetensors");
        std::fs::write(&stale, b"not-gguf").unwrap();
        let err = match GgufWeights::open(&stale) {
            Err(err) => err,
            Ok(_) => panic!("safetensors path was accepted"),
        };
        assert!(err.to_string().contains("Re-download"), "{err}");

        let legacy = dir.join("flux2-klein-4b-mlx-q4");
        write_smoke_bundle(&legacy, 4).unwrap();
        let err = match assert_gguf_bundle(&legacy) {
            Err(err) => err,
            Ok(_) => panic!("legacy directory was accepted"),
        };
        assert!(err.to_string().contains("Re-download"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn flux2_denoise_loads_gguf_by_name_and_reuses_one_graph() {
        let dir = scratch("flux2-bundle");
        write_smoke_bundle(&dir, 4).unwrap();
        let req = request();
        let out = image_create(&req, "flux2", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        assert_eq!(out.graph_computes, 1);
        assert!(out.latents.iter().all(|v| v.is_finite()));
        assert_eq!(&out.preview_png[..8], b"\x89PNG\r\n\x1a\n");

        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let config = Flux2Config::smoke();
        let grid = (64 / config.vae_scale) as i64;
        let mut model = Flux2Denoise::build(&ctx, config, grid, grid, 1).unwrap();
        let weights = GgufWeights::open(&dir.join("transformer.gguf")).unwrap();
        assert!(weights.contains("transformer_blocks.0.attn.to_q.weight"));
        assert!(!weights.contains("velocity.weight"));
        for (name, ne0, ne1) in weight_layout(config) {
            let data = weights.f32_exact(&name, &[ne0, ne1]).unwrap();
            model.load_f32(&name, &data).unwrap();
        }
        let txt = vec![0.1f32; config.joint_attention_dim as usize];
        let mut ticks = 0u32;
        let latents = model
            .denoise(
                vec![0.2; model.latent_len()],
                &txt,
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
        assert_eq!(model.graph_computes(), 6);
        assert_eq!(model.host_readbacks(), 0);
        assert!(latents.iter().all(|v| v.is_finite()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn flux2_multi_head_denoise_is_finite() {
        let dir = scratch("flux2-heads");
        let config = Flux2Config::with_heads(16, 2);
        write_config_bundle(&dir, config).unwrap();
        let out = image_create(&request(), "flux2", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        assert_eq!(out.graph_computes, 1);
        assert!(out.latents.iter().all(|v| v.is_finite()));
        assert_eq!(&out.preview_png[..8], b"\x89PNG\r\n\x1a\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn flux2_q8_0_weights_stay_quantized() {
        let dir = scratch("flux2-q8");
        let config = Flux2Config::with_heads(32, 1);
        write_q8_bundle(&dir, config).unwrap();
        let weights = GgufWeights::open(&dir.join("transformer.gguf")).unwrap();
        assert_eq!(
            weights.ggml_type_of("x_embedder.weight").unwrap(),
            danmo_ggml_sys::ggml_type_GGML_TYPE_Q8_0
        );
        assert_eq!(
            weights
                .ggml_type_of("transformer_blocks.0.attn.norm_q.weight")
                .unwrap(),
            danmo_ggml_sys::ggml_type_GGML_TYPE_F32
        );
        let out = image_create(&request(), "flux2", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        assert_eq!(out.graph_computes, 1);
        assert!(out.latents.iter().all(|v| v.is_finite()));
        assert_eq!(&out.preview_png[..8], b"\x89PNG\r\n\x1a\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn flux2_qwen3_prompt_embeds_drive_denoise() {
        let dir = scratch("flux2-qwen3");
        let mut config = Flux2Config::from_dim(8);
        config.joint_attention_dim = 16;
        write_qwen3_bundle(&dir, config).unwrap();
        let out = image_create(&request(), "flux2", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        assert_eq!(out.graph_computes, 1);
        assert!(out.latents.iter().all(|v| v.is_finite()));
        assert_eq!(&out.preview_png[..8], b"\x89PNG\r\n\x1a\n");
        let mut missing = request();
        missing.prompt = "a missing".into();
        let err = match image_create(&missing, "flux2", &dir, BackendKind::Cpu, |_, _| ()) {
            Err(err) => err,
            Ok(_) => panic!("unknown prompt token was accepted"),
        };
        assert!(err.to_string().contains("no id for missing"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn flux2_chain_pack_decodes_any_prompt_at_the_requested_size() {
        let dir = scratch("flux2-chain");
        write_chain_bundle(&dir).unwrap();
        let mut cat = request();
        cat.prompt = "一只猫".into();
        let cat_out = image_create(&cat, "flux2", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        assert_eq!(png_edge(&cat_out.preview_png), (64, 64));
        assert!(cat_out.graph_computes >= 1);
        let mut dog = request();
        dog.prompt = "a dog".into();
        let dog_out = image_create(&dog, "flux2", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        assert_eq!(png_edge(&dog_out.preview_png), (64, 64));
        assert!(
            cat_out.latents.iter().all(|v| v.is_finite()),
            "cat latent {:?}",
            &cat_out.latents[..8]
        );
        assert_ne!(
            cat_out.preview_png, dog_out.preview_png,
            "prompt did not change the image"
        );
        let mut huge = request();
        huge.size = "512x512".into();
        let err = match image_create(&huge, "flux2", &dir, BackendKind::Cpu, |_, _| ()) {
            Err(err) => err,
            Ok(_) => panic!("oversized chain pack request was accepted"),
        };
        assert!(err.to_string().contains("edges up to 128"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn png_edge(png: &[u8]) -> (u32, u32) {
        let width = u32::from_be_bytes(png[16..20].try_into().unwrap());
        let height = u32::from_be_bytes(png[20..24].try_into().unwrap());
        (width, height)
    }

    #[test]
    fn unknown_family_and_unbuilt_graph_fail_loud() {
        let dir = scratch("unbuilt");
        write_smoke_bundle(&dir, 4).unwrap();
        let req = request();
        assert!(image_create(&req, "not-a-family", &dir, BackendKind::Cpu, |_, _| ()).is_err());
        let err = match image_create(&req, "z_image", &dir, BackendKind::Cpu, |_, _| ()) {
            Err(err) => err,
            Ok(_) => panic!("z_image graph should not be built"),
        };
        assert!(err.to_string().contains("no ggml graph"), "{err}");
        assert!(video_create("wan", &dir)
            .unwrap_err()
            .to_string()
            .contains("no ggml graph"));
        assert!(audio_create("ace_step", &dir).is_err());
        assert!(upscale("esrgan", &dir).is_err());
        let missing = dir.join("missing");
        let err = match image_create(&req, "flux2", &missing, BackendKind::Cpu, |_, _| ()) {
            Err(err) => err,
            Ok(_) => panic!("missing bundle should fail"),
        };
        assert!(err.to_string().contains("Re-download"), "{err}");
        assert!(spec("minimax_h3").unwrap().shape == Shape::GenerationFacade);
        assert_eq!(all_specs().len(), 23);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_backend_fails_loud() {
        assert!(GgmlContext::new(BackendKind::Cuda).is_err());
    }

    #[test]
    fn metal_diffusion_uses_the_ggml_device_or_fails_loud() {
        match crate::runtime::measure_diffusion_steps(BackendKind::Metal, 2) {
            Ok(timing) => {
                assert!(
                    timing.device.to_lowercase().contains("metal"),
                    "device {}",
                    timing.device
                );
                assert_eq!(timing.forward_readbacks, 0);
                assert_eq!(timing.steps, 2);
            }
            Err(err) => {
                let msg = err.to_string();
                assert!(msg.contains("metal") || msg.contains("init"), "{msg}");
            }
        }
        let cpu = crate::runtime::measure_diffusion_steps(BackendKind::Cpu, 2).unwrap();
        assert_eq!(cpu.forward_readbacks, 0);
        assert!(!cpu.device.is_empty(), "cpu device name was empty");
    }
}
