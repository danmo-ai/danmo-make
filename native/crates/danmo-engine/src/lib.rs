mod common;
mod families;
mod pipeline;
pub mod runtime;
mod weights;

pub use families::{
    all_specs, spec, weight_layout, write_smoke_bundle, FamilySpec, Flux2Config, Flux2Denoise,
    Media, Shape,
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
        let seq_img = (64 / config.vae_scale) * (64 / config.vae_scale);
        let mut model = Flux2Denoise::build(&ctx, config, seq_img as i64).unwrap();
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
        assert!(latents.iter().all(|v| v.is_finite()));
        let _ = std::fs::remove_dir_all(&dir);
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
}
