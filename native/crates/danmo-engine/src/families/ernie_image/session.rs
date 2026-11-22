use crate::contract::ResolvedRun;
use crate::error::EngineError;
use crate::families::{DitImageFamily, DitImageSession};
use crate::ports::ResolvePort;
use crate::weights::LoraFile;
#[cfg(not(feature = "mlx"))]
use crate::weights::WeightError;
use danmo_contracts::ImageGenerationRequest;

pub struct ErnieImageFamily;

impl DitImageFamily for ErnieImageFamily {
    fn load(
        &self,
        run: &ResolvedRun,
        request: &ImageGenerationRequest,
        loras: Vec<LoraFile>,
        _resolver: &dyn ResolvePort,
    ) -> Result<Box<dyn DitImageSession>, EngineError> {
        if !loras.is_empty() {
            return Err(EngineError::bad_request(
                "ERNIE-Image LoRA inference is not built",
            ));
        }
        if request.structural_guide.is_some() {
            return Err(EngineError::bad_request(
                "ERNIE-Image structural guides are not built",
            ));
        }
        #[cfg(feature = "mlx")]
        return Ok(Box::new(super::mlx::session::MlxSession::prepare(
            run, request,
        )?));
        #[cfg(not(feature = "mlx"))]
        {
            let _ = (run, request);
            Err(WeightError::NotBuilt(
                "ERNIE-Image requires a safetensors bundle and a binary built with MLX support"
                    .into(),
            )
            .into())
        }
    }
}
