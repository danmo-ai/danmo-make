//! Shared execution pipelines. Family dispatch happens only in `families/mod.rs`.

pub(crate) mod denoise;
pub(crate) mod dit;
pub(crate) mod facade;
#[cfg(test)]
pub(crate) mod legacy;

use crate::error::EngineError;
use crate::weights::LoraFile;
use danmo_contracts::{ImageAdapter, ImageGenerationRequest};

fn request_loras(request: &ImageGenerationRequest) -> Result<Vec<LoraFile>, EngineError> {
    lora_files(&request.adapters)
}

pub(crate) fn lora_files(adapters: &[ImageAdapter]) -> Result<Vec<LoraFile>, EngineError> {
    let mut files = Vec::new();
    for adapter in adapters {
        if adapter.id.trim().is_empty() {
            return Err(EngineError::InvalidRequest("lora id is required".into()));
        }
        let path = std::path::PathBuf::from(&adapter.path);
        if adapter.path.is_empty() || !path.is_file() {
            return Err(EngineError::InvalidRequest(format!(
                "lora {} is not installed",
                adapter.id
            )));
        }
        if !adapter.weight.is_finite() {
            return Err(EngineError::InvalidRequest(format!(
                "lora {} weight is not a finite number",
                adapter.id
            )));
        }
        files.push(LoraFile {
            id: adapter.id.clone(),
            path,
            weight: adapter.weight,
        });
    }
    Ok(files)
}
