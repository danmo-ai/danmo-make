use crate::contract::{Action, Artifact, ResolvedRun};
use crate::error::EngineError;
use std::path::PathBuf;

pub trait ResolvePort {
    fn resolve(&self, model: &str, action: Action) -> Result<ResolvedRun, EngineError>;

    fn resolve_component(
        &self,
        run: &ResolvedRun,
        role: &str,
        model: &str,
    ) -> Result<PathBuf, EngineError>;

    fn read_source_audio(&self, run: &ResolvedRun, source: &str) -> Result<Vec<u8>, EngineError>;

    /// RGB24 pixels of an image asset. The default refuses; hosts override it.
    fn read_source_image(
        &self,
        _run: &ResolvedRun,
        _source_asset_id: &str,
    ) -> Result<(Vec<u8>, u32, u32), EngineError> {
        Err(EngineError::Resolve(
            "source image reading is not available".into(),
        ))
    }
}

#[derive(Debug)]
pub struct PersistedAsset {
    pub id: String,
    pub path: PathBuf,
}

pub trait ArtifactSink {
    fn persist(
        &self,
        run: &ResolvedRun,
        task_id: &str,
        artifacts: Vec<Artifact>,
    ) -> Result<Vec<PersistedAsset>, EngineError>;
}
