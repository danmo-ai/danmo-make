use crate::contract::{EngineRequest, EngineResult, ExecutionContext};
use crate::error::EngineError;
use crate::ports::{ArtifactSink, ResolvePort};
use crate::runtime::{BackendKind, GgmlContext};
use crate::session;

pub struct Engine<'a> {
    resolver: &'a dyn ResolvePort,
    artifacts: &'a dyn ArtifactSink,
    backend: BackendKind,
}

impl<'a> Engine<'a> {
    pub fn new(
        resolver: &'a dyn ResolvePort,
        artifacts: &'a dyn ArtifactSink,
        backend: BackendKind,
    ) -> Self {
        Self {
            resolver,
            artifacts,
            backend,
        }
    }

    pub fn execute(
        &self,
        request: EngineRequest,
        exec: ExecutionContext,
    ) -> Result<EngineResult, EngineError> {
        exec.phase("resolve")?;
        let mut run = self.resolver.resolve(request.model(), request.action())?;
        run.backend = self.backend;
        let ctx = GgmlContext::new(self.backend)?;
        probe_backend(&ctx, &exec)?;
        let mut output = session::execute(&ctx, &run, &request, self.resolver, &exec)?;
        let capacity = crate::runtime::capacity_snapshot(&ctx);
        exec.log("info", &capacity.summary())?;
        if !output.metadata.is_object() {
            output.metadata = serde_json::json!({});
        }
        output.metadata["capacity"] = serde_json::json!({
            "backend": capacity.backend.name(),
            "weight_bytes": capacity.weight_bytes,
            "conditioning_bytes": capacity.conditioning_bytes,
            "graph_bytes": capacity.graph_bytes,
            "managed_bytes": capacity.managed_bytes(),
            "available_bytes": capacity.available_bytes,
            "device_bytes": capacity.device_bytes,
        });
        exec.phase("persist")?;
        let artifacts = std::mem::take(&mut output.artifacts);
        let persisted = self.artifacts.persist(&run, &exec.task_id, artifacts)?;
        let asset_ids = persisted.iter().map(|asset| asset.id.clone()).collect();
        let primary_asset_id = persisted.first().map(|asset| asset.id.clone());
        let output_paths = persisted.into_iter().map(|asset| asset.path).collect();
        Ok(EngineResult {
            output,
            asset_ids,
            primary_asset_id,
            output_paths,
        })
    }
}

fn probe_backend(ctx: &GgmlContext, exec: &ExecutionContext) -> Result<(), EngineError> {
    if ctx.kind() != BackendKind::Metal {
        return Ok(());
    }
    static PROBED: std::sync::OnceLock<std::sync::Mutex<Vec<BackendKind>>> =
        std::sync::OnceLock::new();
    let probed = PROBED.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    if probed.lock().expect("backend probe").contains(&ctx.kind()) {
        return Ok(());
    }
    let report = crate::runtime::probe_attention(ctx)?;
    exec.log("info", &report.summary())?;
    probed.lock().expect("backend probe").push(ctx.kind());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{Action, Artifact, ResolvedRun, RunObserver};
    use crate::ports::PersistedAsset;
    use crate::{image_create, write_smoke_bundle};
    use danmo_contracts::ImageGenerationRequest;
    use std::path::PathBuf;
    use std::sync::Mutex;

    struct Resolver {
        run: ResolvedRun,
    }

    impl ResolvePort for Resolver {
        fn resolve(&self, _model: &str, _action: Action) -> Result<ResolvedRun, EngineError> {
            Ok(self.run.clone())
        }

        fn resolve_component(
            &self,
            _run: &ResolvedRun,
            _role: &str,
            _model: &str,
        ) -> Result<PathBuf, EngineError> {
            Err(EngineError::Resolve("no companion in test".into()))
        }

        fn read_source_audio(
            &self,
            _run: &ResolvedRun,
            _source: &str,
        ) -> Result<Vec<u8>, EngineError> {
            Err(EngineError::Resolve("no audio in test".into()))
        }
    }

    #[derive(Default)]
    struct Sink {
        artifacts: Mutex<Vec<Artifact>>,
    }

    impl ArtifactSink for Sink {
        fn persist(
            &self,
            _run: &ResolvedRun,
            _task_id: &str,
            artifacts: Vec<Artifact>,
        ) -> Result<Vec<PersistedAsset>, EngineError> {
            let count = artifacts.len();
            self.artifacts.lock().unwrap().extend(artifacts);
            Ok((0..count)
                .map(|index| PersistedAsset {
                    id: format!("asset-{index}"),
                    path: PathBuf::from(format!("asset-{index}.png")),
                })
                .collect())
        }
    }

    #[derive(Default)]
    struct Observer {
        phases: Mutex<Vec<String>>,
    }

    impl RunObserver for Observer {
        fn progress(&self, _phase: &str, _step: u32, _total: u32) -> Result<(), EngineError> {
            Ok(())
        }

        fn log(&self, _level: &str, _message: &str) -> Result<(), EngineError> {
            Ok(())
        }

        fn phase(&self, phase: &str) -> Result<(), EngineError> {
            self.phases.lock().unwrap().push(phase.into());
            Ok(())
        }
    }

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

    #[test]
    fn phased_engine_matches_legacy_flux2_and_persists_artifacts() {
        let dir = std::env::temp_dir().join(format!("danmo-engine-phased-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_smoke_bundle(&dir, 4).unwrap();
        let expected =
            image_create(&request(), "flux2", &dir, BackendKind::Cpu, |_, _| ()).unwrap();
        let run = ResolvedRun {
            model: request().model,
            version: Some("bf16".into()),
            family: "flux2".into(),
            action: Action::ImageCreate,
            bundle: dir.clone(),
            backend: BackendKind::Cpu,
        };
        let resolver = Resolver { run };
        let sink = Sink::default();
        let observer = std::sync::Arc::new(Observer::default());
        let result = Engine::new(&resolver, &sink, BackendKind::Cpu)
            .execute(
                EngineRequest::ImageCreate(request()),
                ExecutionContext::new("task", observer.clone()),
            )
            .unwrap();
        assert_eq!(result.asset_ids, ["asset-0"]);
        assert_eq!(result.output.latent_len, expected.latents.len());
        let artifacts = sink.artifacts.lock().unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].bytes, expected.preview_png);
        assert_eq!(
            *observer.phases.lock().unwrap(),
            ["resolve", "load", "encode", "schedule", "infer", "decode", "persist"]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
