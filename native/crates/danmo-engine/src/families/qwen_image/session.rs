//! Qwen-Image create. The ggml graph for this action is gone.
//!
//! Safetensors bundles run on the MLX session when the engine is built with
//! `mlx`. A GGUF bundle fails here instead of falling back to ggml.

use crate::contract::ResolvedRun;
use crate::error::EngineError;
use crate::families::DitImageFamily;
use crate::ports::ResolvePort;
use crate::weights::{LoraFile, WeightError};
use danmo_contracts::ImageGenerationRequest;

pub struct QwenImageFamily;

impl DitImageFamily for QwenImageFamily {
    fn load(
        &self,
        run: &ResolvedRun,
        request: &ImageGenerationRequest,
        loras: Vec<LoraFile>,
        _resolver: &dyn ResolvePort,
    ) -> Result<Box<dyn crate::families::DitImageSession>, EngineError> {
        if request.structural_guide.is_some() {
            return Err(WeightError::NotBuilt(
                "structural guide graph is not built for qwen_image".into(),
            )
            .into());
        }
        #[cfg(feature = "mlx")]
        if crate::weights::is_safetensors_image_bundle(&run.bundle) {
            return Ok(Box::new(super::mlx::session::MlxSession::prepare(
                &run.bundle,
                request,
                loras,
            )?));
        }
        let _ = (run, request, loras);
        Err(WeightError::NotBuilt(
            "qwen-image create no longer has a ggml graph; install the safetensors bundle and build with the mlx feature"
                .into(),
        )
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::QwenImageFamily;
    use crate::contract::{Action, ResolvedRun};
    use crate::families::DitImageFamily;
    use crate::ports::ResolvePort;
    use crate::runtime::BackendKind;
    use danmo_contracts::ImageGenerationRequest;

    struct NoResolver;

    impl ResolvePort for NoResolver {
        fn resolve(
            &self,
            model: &str,
            _action: Action,
        ) -> Result<ResolvedRun, crate::error::EngineError> {
            Err(crate::error::EngineError::Resolve(format!(
                "{model} is not resolvable in this test"
            )))
        }
        fn resolve_component(
            &self,
            _run: &ResolvedRun,
            _role: &str,
            _model: &str,
        ) -> Result<std::path::PathBuf, crate::error::EngineError> {
            Err(crate::error::EngineError::Resolve(
                "no companion in test".into(),
            ))
        }
        fn read_source_audio(
            &self,
            _run: &ResolvedRun,
            _source: &str,
        ) -> Result<Vec<u8>, crate::error::EngineError> {
            Err(crate::error::EngineError::Resolve(
                "no audio in test".into(),
            ))
        }
    }

    #[test]
    fn gguf_create_fails_loud_without_the_ggml_graph() {
        let dir = std::env::temp_dir().join(format!("qwen-image-gguf-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let request = ImageGenerationRequest {
            model: "qwen-image".into(),
            title: String::new(),
            prompt: "a red apple".into(),
            negative_prompt: String::new(),
            size: "512x512".into(),
            n: 1,
            steps: Some(4),
            guidance: Some(1.0),
            seed: Some(1),
            scheduler: None,
            scheduler_shift: None,
            use_empirical_mu: false,
            use_pe: None,
            adapters: Vec::new(),
            structural_guide: None,
            metadata: serde_json::json!({}),
            preview: true,
            priority: "normal".into(),
        };
        let run = ResolvedRun {
            model: request.model.clone(),
            version: Some("q4_k".into()),
            family: "qwen_image".into(),
            action: Action::ImageCreate,
            bundle: dir.clone(),
            backend: BackendKind::Cpu,
        };
        let err = match QwenImageFamily.load(&run, &request, Vec::new(), &NoResolver) {
            Ok(_) => panic!("gguf create still ran"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("no longer has a ggml graph"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
