#[cfg(test)]
mod architecture;
mod adapter;
mod common;
mod contract;
mod engine;
mod error;
mod families;
mod pipeline;
mod ports;
pub mod runtime;
mod session;
#[cfg(feature = "mlx")]
pub mod train;
mod weights;

#[cfg(feature = "mlx")]
pub use families::run_family_lora_training;

pub use adapter::engine_request;
pub use common::image;
pub use common::png;
pub use contract::{
    Action, Artifact, AudioCreateRequest as EngineAudioCreateRequest,
    AudioEditRequest as EngineAudioEditRequest, AudioLmRequest as EngineAudioLmRequest,
    CancellationToken, EngineOutput, EngineRequest, EngineResult, ExecutionContext, NoopObserver,
    ResolvedRun, RunObserver, Sidecar,
};
pub use engine::Engine;
pub use error::EngineError;
pub use families::{
    all_specs, run_install_hook, spec, write_seedvr2_smoke_bundle,
    write_wan_smoke_bundle, write_z_image_control_smoke_bundle, write_z_image_edit_smoke_bundle,
    FamilySpec, Media, Shape,
};
#[cfg(test)]
pub use pipeline::legacy::{
    audio_create, audio_edit, audio_lm, image_create, read_audio_wav_pcm16, upscale, video_create,
    AudioCreateRequest, AudioEditRequest, AudioLmOutput, AudioLmRequest, AudioRunOutput, RunOutput,
};
pub use ports::{ArtifactSink, PersistedAsset, ResolvePort};
pub use runtime::{identity_probe, select_from_env, BackendKind, BackendSelection, GgmlContext};
pub use session::ensure_available;
pub use weights::{
    assert_gguf_bundle, is_ace_step_safetensors_bundle, is_controlnet_safetensors_dir,
    is_safetensors_bundle, is_safetensors_image_bundle, is_seedvr2_safetensors_bundle,
    lora_base_name, write_f32_gguf, GgufWeights, WeightError,
};

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
            scheduler_shift: None,
            use_empirical_mu: false,
            adapters: Vec::new(),
            structural_guide: None,
            metadata: serde_json::json!({}),
            preview: true,
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

    fn write_marker_bundle(dir: &std::path::Path) {
        std::fs::create_dir_all(dir).unwrap();
        for name in ["transformer.gguf", "text_encoder.gguf", "vae.gguf"] {
            write_f32_gguf(&dir.join(name), "marker", 1, 1, &[0.0]).unwrap();
        }
        std::fs::write(dir.join("tokenizer.json"), "{}\n").unwrap();
        std::fs::write(
            dir.join("bundle.manifest.json"),
            r#"{
  "format": "gguf",
  "quant": "bf16",
  "files": {
    "transformer": "transformer.gguf",
    "text_encoder": "text_encoder.gguf",
    "vae": "vae.gguf"
  }
}
"#,
        )
        .unwrap();
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
    fn gguf_assert_rejects_safetensors_product_dirs_without_legacy_wording() {
        let dir = scratch("legacy");
        std::fs::create_dir_all(&dir).unwrap();
        let stale = dir.join("w.safetensors");
        std::fs::write(&stale, b"not-gguf").unwrap();
        let err = match GgufWeights::open(&stale) {
            Err(err) => err,
            Ok(_) => panic!("garbage safetensors path was accepted"),
        };
        assert!(err.to_string().contains("safetensors"), "{err}");

        // Product MLX path names are valid; GGUF assert fails for layout, not
        // because the directory looks "legacy".
        let mlx = dir.join("flux2-klein-4b-mlx-q4");
        std::fs::create_dir_all(mlx.join("transformer")).unwrap();
        std::fs::create_dir_all(mlx.join("text_encoder")).unwrap();
        std::fs::create_dir_all(mlx.join("vae")).unwrap();
        std::fs::write(mlx.join("transformer/model.safetensors"), b"st").unwrap();
        std::fs::write(mlx.join("text_encoder/model.safetensors"), b"st").unwrap();
        std::fs::write(mlx.join("vae/model.safetensors"), b"st").unwrap();
        assert!(is_safetensors_image_bundle(&mlx));
        let err = match assert_gguf_bundle(&mlx) {
            Err(err) => err,
            Ok(_) => panic!("safetensors image bundle was accepted as GGUF"),
        };
        let message = err.to_string();
        assert!(message.contains("safetensors/MLX"), "{err}");
        assert!(!message.contains("Re-download"), "{err}");
        assert!(!message.contains("old MLX"), "{err}");

        let ready = dir.join("with-companions");
        write_marker_bundle(&ready);
        std::fs::create_dir_all(ready.join("vae")).unwrap();
        std::fs::write(ready.join("vae/diffusion_pytorch_model.safetensors"), b"st").unwrap();
        assert!(assert_gguf_bundle(&ready).is_ok());

        let seedvr = dir.join("seedvr2-7b-fp16");
        std::fs::create_dir_all(&seedvr).unwrap();
        std::fs::write(seedvr.join("seedvr2_ema_7b_fp16.safetensors"), b"st").unwrap();
        std::fs::write(seedvr.join("ema_vae_fp16.safetensors"), b"st").unwrap();
        assert!(is_seedvr2_safetensors_bundle(&seedvr));
        let err = match assert_gguf_bundle(&seedvr) {
            Err(err) => err,
            Ok(_) => panic!("seedvr2 safetensors bundle was accepted as GGUF"),
        };
        assert!(err.to_string().contains("safetensors/MLX"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_family_and_unbuilt_graph_fail_loud() {
        let dir = scratch("unbuilt");
        write_marker_bundle(&dir);
        let req = request();
        assert!(image_create(&req, "not-a-family", &dir, BackendKind::Cpu, |_, _| ()).is_err());
        let err = match image_create(&req, "fibo", &dir, BackendKind::Cpu, |_, _| ()) {
            Err(err) => err,
            Ok(_) => panic!("fibo graph should not be built"),
        };
        assert!(err.to_string().contains("no ggml graph"), "{err}");
        assert!(video_create("wan", &dir)
            .unwrap_err()
            .to_string()
            .contains("no ggml graph"));
        let ace = audio_create(
            "ace_step",
            &dir,
            &AudioCreateRequest {
                prompt: "test",
                lyrics: "[inst]",
                duration_sec: 5.0,
                seed: Some(0),
                guidance: 1.0,
                steps: None,
                vocal_language: "en",
                bpm: None,
                keyscale: "",
                timesignature: "",
                lm_expansion: "off",
                lm_bundle: None,
                audio_code_indices: None,
                timbre_latents: None,
            },
            BackendKind::Cpu,
            |_, _, _| (),
        )
        .unwrap_err();
        assert!(ace.to_string().contains("ACE-Step"), "{ace}");
        assert!(upscale("esrgan", &dir).is_err());
        ensure_available("esrgan", crate::contract::Action::ImageUpscale).unwrap();
        let err = ensure_available("diffrhythm", crate::contract::Action::AudioCreate).unwrap_err();
        assert!(matches!(err, EngineError::GraphNotBuilt { .. }), "{err}");
        let missing = dir.join("missing");
        let err = match image_create(&req, "flux2", &missing, BackendKind::Cpu, |_, _| ()) {
            Err(err) => err,
            Ok(_) => panic!("missing bundle should fail"),
        };
        assert!(err.to_string().contains("directory is missing"), "{err}");
        assert!(spec("minimax_h3").unwrap().shape == Shape::GenerationFacade);
        assert!(spec("wan").unwrap().shape == Shape::GenerationFacade);
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
                assert_eq!(timing.backend, "metal");
                assert!(!timing.device.is_empty());
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
