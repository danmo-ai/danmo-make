use crate::contract::ResolvedRun;
use crate::error::EngineError;
use crate::families::DitImageFamily;
use crate::ports::ResolvePort;
use crate::weights::{LoraFile, WeightError};
use danmo_contracts::ImageGenerationRequest;

pub struct Flux2Family;

impl DitImageFamily for Flux2Family {
    fn load(
        &self,
        run: &ResolvedRun,
        request: &ImageGenerationRequest,
        loras: Vec<LoraFile>,
        _resolver: &dyn ResolvePort,
    ) -> Result<Box<dyn crate::families::DitImageSession>, EngineError> {
        if request.structural_guide.is_some() {
            return Err(WeightError::NotBuilt(
                "structural guide graph is not built for flux2".into(),
            )
            .into());
        }
        if !request.negative_prompt.trim().is_empty() {
            return Err(WeightError::NotBuilt(
                "negative prompt is not built for this family".into(),
            )
            .into());
        }
        match request
            .scheduler
            .as_deref()
            .unwrap_or("flow_match_euler_discrete")
        {
            "flow_match_euler" | "flow_match_euler_discrete" => {}
            other => {
                return Err(WeightError::NotBuilt(format!("scheduler {other} is not built")).into())
            }
        }
        #[cfg(feature = "mlx")]
        if crate::weights::is_safetensors_image_bundle(&run.bundle) {
            return Ok(Box::new(super::mlx::session::MlxSession::prepare(
                &run.bundle,
                request,
                loras,
            )?));
        }
        let _ = (run, loras);
        Err(WeightError::NotBuilt(
            "flux2 image create no longer has a ggml graph; install the safetensors bundle and build with the mlx feature"
                .into(),
        )
        .into())
    }
}
