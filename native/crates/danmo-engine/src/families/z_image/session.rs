//! Z-Image image create. The ggml graph for this action is gone.
//!
//! Safetensors bundles run on the MLX session when the engine is built with
//! `mlx`. A GGUF bundle fails here instead of falling back to ggml. Edit
//! still uses the ggml DiT until that action is migrated.

use crate::contract::ResolvedRun;
use crate::error::EngineError;
use crate::families::{DitImageFamily, DitImageSession};
use crate::ports::ResolvePort;
use crate::weights::{LoraFile, WeightError};
use danmo_contracts::ImageGenerationRequest;

pub struct ZImageFamily;

impl DitImageFamily for ZImageFamily {
    fn load(
        &self,
        run: &ResolvedRun,
        request: &ImageGenerationRequest,
        loras: Vec<LoraFile>,
        _resolver: &dyn ResolvePort,
    ) -> Result<Box<dyn DitImageSession>, EngineError> {
        let _ = &run.bundle;
        if !loras.is_empty() {
            return Err(EngineError::bad_request(
                "Z-Image create does not apply LoRA; the ggml create graph has been removed",
            ));
        }
        #[cfg(feature = "mlx")]
        if super::mlx::session::is_mlx_bundle(&run.bundle) {
            return Ok(Box::new(super::mlx::session::MlxSession::prepare(
                &run.bundle,
                request,
            )?));
        }
        let _ = request;
        Err(WeightError::NotBuilt(
            "z_image image create no longer has a ggml graph; install the safetensors bundle and build with the mlx feature"
                .into(),
        )
        .into())
    }
}
