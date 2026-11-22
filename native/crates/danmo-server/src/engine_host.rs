use crate::app::{gguf_bundle, AppState};
use crate::product;
use danmo_engine::{
    Action, Artifact, ArtifactSink, CancellationToken, Engine, EngineError, ExecutionContext,
    PersistedAsset, ResolvePort, ResolvedRun, RunObserver,
};
use danmo_store::AssetRow;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct EngineHost<'a> {
    state: &'a AppState,
}

impl<'a> EngineHost<'a> {
    pub fn new(state: &'a AppState) -> Self {
        Self { state }
    }
}

impl ResolvePort for EngineHost<'_> {
    fn resolve(&self, model: &str, action: Action) -> Result<ResolvedRun, EngineError> {
        let catalog = danmo_catalog::Catalog::open(&self.state.registry)
            .map_err(|error| EngineError::Resolve(error.to_string()))?;
        let entry = catalog
            .model(model)
            .map_err(|error| EngineError::Resolve(error.to_string()))?;
        let bundle = gguf_bundle(self.state, model).map_err(EngineError::Resolve)?;
        Ok(ResolvedRun {
            model: model.into(),
            version: model
                .split_once(':')
                .map(|(_, version)| version.to_string()),
            family: entry.family,
            action,
            bundle,
            backend: self.state.backend,
        })
    }

    fn resolve_component(
        &self,
        _run: &ResolvedRun,
        role: &str,
        model: &str,
    ) -> Result<PathBuf, EngineError> {
        if role.trim().is_empty() {
            return Err(EngineError::Resolve("component role is required".into()));
        }
        gguf_bundle(self.state, model).map_err(EngineError::Resolve)
    }

    fn read_source_audio(&self, _run: &ResolvedRun, source: &str) -> Result<Vec<u8>, EngineError> {
        if let Some(encoded) = source.strip_prefix("base64:") {
            return product::base64_decode(encoded).map_err(EngineError::Resolve);
        }
        if let Some(asset_id) = source.strip_prefix("asset:") {
            let row = self
                .state
                .scheduler
                .get_asset(asset_id)
                .map_err(|error| EngineError::Resolve(error.to_string()))?
                .ok_or_else(|| {
                    EngineError::Resolve(format!("source asset {asset_id} is missing"))
                })?;
            let path = product::resolve_asset_fs_path(
                &self.state.outputs,
                &self.state.workspace,
                &row.file_path,
            );
            return std::fs::read(&path).map_err(|error| {
                EngineError::Resolve(format!("source audio {}: {error}", path.display()))
            });
        }
        let path = Path::new(source);
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.state.workspace.join(path)
        };
        std::fs::read(&resolved).map_err(|error| {
            EngineError::Resolve(format!("source audio {}: {error}", resolved.display()))
        })
    }

    fn read_source_image(
        &self,
        _run: &ResolvedRun,
        source_asset_id: &str,
    ) -> Result<(Vec<u8>, u32, u32), EngineError> {
        let row = self
            .state
            .scheduler
            .get_asset(source_asset_id)
            .map_err(|error| EngineError::Resolve(error.to_string()))?
            .ok_or_else(|| {
                EngineError::Resolve(format!("source asset {source_asset_id} is missing"))
            })?;
        let path = product::resolve_asset_fs_path(
            &self.state.outputs,
            &self.state.workspace,
            &row.file_path,
        );
        danmo_engine::image::decode_file_rgb(&path).map_err(EngineError::Resolve)
    }

    fn read_source_video_path(
        &self,
        _run: &ResolvedRun,
        source_asset_id: &str,
    ) -> Result<PathBuf, EngineError> {
        let row = self
            .state
            .scheduler
            .get_asset(source_asset_id)
            .map_err(|error| EngineError::Resolve(error.to_string()))?
            .ok_or_else(|| {
                EngineError::Resolve(format!("source asset {source_asset_id} is missing"))
            })?;
        Ok(product::resolve_asset_fs_path(
            &self.state.outputs,
            &self.state.workspace,
            &row.file_path,
        ))
    }
}

impl ArtifactSink for EngineHost<'_> {
    fn persist(
        &self,
        run: &ResolvedRun,
        task_id: &str,
        artifacts: Vec<Artifact>,
    ) -> Result<Vec<PersistedAsset>, EngineError> {
        let mut persisted = Vec::new();
        for artifact in artifacts {
            let id = product::new_id("ast_");
            let asset_dir = self.state.outputs.join("assets");
            std::fs::create_dir_all(&asset_dir)
                .map_err(|error| EngineError::Persist(error.to_string()))?;
            let path = asset_dir.join(format!("{id}.{}", artifact.extension));
            atomic_write(&path, &artifact.bytes)?;
            if artifact.kind == "image" {
                let preview_dir = self.state.outputs.join("previews");
                std::fs::create_dir_all(&preview_dir)
                    .map_err(|error| EngineError::Persist(error.to_string()))?;
                atomic_write(&preview_dir.join(format!("{task_id}.png")), &artifact.bytes)?;
            }
            for sidecar in &artifact.sidecars {
                let sidecar_path = asset_dir.join(format!("{id}{}", sidecar.suffix));
                atomic_write(&sidecar_path, &sidecar.bytes)?;
            }
            let mut metadata = artifact.metadata.clone();
            metadata["model"] = json!(run.model);
            metadata["family"] = json!(run.family);
            self.state
                .scheduler
                .insert_asset(&AssetRow {
                    id: id.clone(),
                    kind: artifact.kind,
                    mime_type: artifact.mime,
                    file_path: path.display().to_string(),
                    thumbnail_path: None,
                    width: artifact.width,
                    height: artifact.height,
                    duration_seconds: artifact.duration_seconds,
                    source_task_id: Some(task_id.to_string()),
                    source_action: Some(artifact.source_action),
                    metadata: metadata.to_string(),
                    created_at: String::new(),
                    parent_asset_id: artifact.parent_asset_id,
                    relation_type: artifact.relation_type,
                    group_id: None,
                })
                .map_err(|error| EngineError::Persist(error.to_string()))?;
            persisted.push(PersistedAsset { id, path });
        }
        Ok(persisted)
    }
}

struct TaskObserver {
    scheduler: danmo_schedule::Scheduler,
    task_id: String,
    preview_dir: PathBuf,
    preview: bool,
    trace: Option<Arc<danmo_otel::TaskTrace>>,
    telemetry: Option<danmo_otel::Telemetry>,
}

impl TaskObserver {
    /// Open `phase.<name>` on the task root. Closed phases are flushed so the
    /// graph / trace poll sees them without waiting for the batch timer.
    fn record_phase(&self, phase: &str) {
        if phase.is_empty() {
            return;
        }
        let Some(trace) = self.trace.as_ref() else {
            return;
        };
        if trace.current_phase().as_deref() == Some(phase) {
            return;
        }
        trace.phase(phase);
        if let Some(telemetry) = self.telemetry.as_ref() {
            if let Err(error) = telemetry.flush() {
                eprintln!("[otel] {error}");
            }
        }
    }
}

impl RunObserver for TaskObserver {
    fn progress(&self, phase: &str, step: u32, total: u32) -> Result<(), EngineError> {
        let progress = if total == 0 {
            0.0
        } else {
            step as f64 / total as f64
        };
        if let Some(trace) = self.trace.as_ref() {
            trace.progress(phase, step, total);
        }
        // Queue bar + SSE tip only — no task_logs row. Step history lives on
        // the open phase span and becomes queryable when that phase ends.
        self.scheduler
            .set_progress_tip(&self.task_id, progress, phase, step, total)
            .map_err(|error| EngineError::Observer(error.to_string()))
    }

    fn log(&self, level: &str, message: &str) -> Result<(), EngineError> {
        if let Some(trace) = self.trace.as_ref() {
            trace.log_event(level, message);
            return Ok(());
        }
        // Tracing disabled: keep a single fail-loud path for freeform engine
        // lines so they are not silently dropped.
        self.scheduler
            .log(&self.task_id, level, message)
            .map_err(|error| EngineError::Observer(error.to_string()))
    }

    fn phase(&self, phase: &str) -> Result<(), EngineError> {
        self.record_phase(phase);
        Ok(())
    }

    fn annotate(&self, attributes: &[(&str, String)]) -> Result<(), EngineError> {
        let Some(trace) = self.trace.as_ref() else {
            return Ok(());
        };
        let attrs = attributes
            .iter()
            .map(|(key, value)| danmo_otel::attribute((*key).to_string(), value.clone()))
            .collect::<Vec<_>>();
        trace.set_attributes(attrs);
        Ok(())
    }

    fn preview_interval_steps(&self) -> Option<u32> {
        if self.preview && !self.task_id.is_empty() {
            Some(2)
        } else {
            None
        }
    }

    fn preview_max_edge(&self) -> u32 {
        512
    }

    fn write_step_preview(&self, png: &[u8]) -> Result<(), EngineError> {
        if png.len() < 8 || &png[..8] != b"\x89PNG\r\n\x1a\n" {
            return Err(EngineError::Observer("step preview is not a PNG".into()));
        }
        std::fs::create_dir_all(&self.preview_dir)
            .map_err(|error| EngineError::Observer(error.to_string()))?;
        atomic_write(&self.preview_dir.join(format!("{}.png", self.task_id)), png)
    }
}

pub fn execute(
    state: &AppState,
    kind: &str,
    model: &str,
    params: &Value,
    cancellation: CancellationToken,
) -> Result<Value, EngineError> {
    let task_id = params.get("task_id").and_then(Value::as_str).unwrap_or("");
    let host = EngineHost::new(state);
    let request = danmo_engine::engine_request(kind, model, params)?;
    let engine = Engine::new(&host, &host, state.backend);
    let trace = state.telemetry.as_ref().and_then(|telemetry| {
        if task_id.is_empty() {
            return None;
        }
        Some(Arc::new(telemetry.start_task(
            task_id,
            vec![
                danmo_otel::attribute("task.id", task_id),
                danmo_otel::attribute("task.kind", kind),
                danmo_otel::attribute("gen_ai.request.model", model),
                danmo_otel::attribute("danmo.backend", state.backend.name()),
            ],
        )))
    });
    let observer = Arc::new(TaskObserver {
        scheduler: state.scheduler.clone(),
        task_id: task_id.into(),
        preview_dir: state.outputs.join("previews"),
        preview: request_preview(params),
        trace: trace.clone(),
        telemetry: state.telemetry.clone(),
    });
    // Capability line lands on the root span (or task_logs when tracing is off).
    if !task_id.is_empty() {
        if let Some(capability) = crate::app::mlx_capability_summary() {
            observer.log("info", capability)?;
        }
    }
    let exec = ExecutionContext::new(task_id, observer).with_cancellation(cancellation);
    let outcome = engine.execute(request, exec);
    if let Some(trace) = trace.as_ref() {
        match &outcome {
            Ok(_) => {}
            Err(EngineError::Cancelled) => trace.cancelled(),
            Err(error) => trace.fail(&error.to_string()),
        }
        trace.finish();
        if let Some(telemetry) = state.telemetry.as_ref() {
            // Flush so the trace page has the finished spans as soon as the task
            // reports completion; failures stay visible in stderr rather than
            // turning a successful generation into an error.
            if let Err(error) = telemetry.flush() {
                eprintln!("[otel] {error}");
            }
        }
    }
    outcome.map(|result| result.task_json())
}

fn request_preview(params: &Value) -> bool {
    match params.pointer("/request/preview") {
        Some(value) => value.as_bool().unwrap_or(true),
        None => true,
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), EngineError> {
    let temporary = path.with_extension(format!(
        "{}.tmp-{}",
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or("asset"),
        std::process::id()
    ));
    std::fs::write(&temporary, bytes).map_err(|error| EngineError::Persist(error.to_string()))?;
    std::fs::rename(&temporary, path).map_err(|error| EngineError::Persist(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use danmo_engine::{engine_request, Action, EngineRequest};

    #[cfg(feature = "mlx")]
    #[test]
    fn mlx_capability_line_is_present_and_well_formed() {
        let capability = crate::app::mlx_capability_summary().expect("mlx build");
        assert!(capability.contains("deployment_target="), "{capability}");
        assert!(capability.contains("nax_built="), "{capability}");
    }

    #[cfg(not(feature = "mlx"))]
    #[test]
    fn mlx_capability_line_is_absent_without_mlx() {
        assert!(crate::app::mlx_capability_summary().is_none());
    }

    #[test]
    fn video_generation_routes_wan_and_ltx25() {
        let params = json!({
            "family": "wan",
            "request": {"model": "wan-2.2-ti2v-5b", "prompt": "a cat", "num_frames": 81},
        });
        let request = engine_request("video.generation", "wan-2.2-ti2v-5b", &params).unwrap();
        match request {
            EngineRequest::VideoCreate(request) => {
                assert_eq!(request.prompt, "a cat");
                assert_eq!(request.num_frames, Some(81));
                assert_eq!(request.size, None);
                assert_eq!(request.fps, None);
            }
            other => panic!("wan video generation routed to {other:?}"),
        }
        let params = json!({
            "family": "ltx25",
            "request": {"model": "ltx-2.5-distilled:mlx-q4", "prompt": "a cat", "num_frames": 25},
        });
        let request =
            engine_request("video.generation", "ltx-2.5-distilled:mlx-q4", &params).unwrap();
        match request {
            EngineRequest::VideoCreate(request) => {
                assert_eq!(request.prompt, "a cat");
                assert_eq!(request.num_frames, Some(25));
            }
            other => panic!("ltx25 video generation routed to {other:?}"),
        }
        let params = json!({
            "family": "minimax_h3",
            "request": {"model": "minimax-h3-fl2va", "prompt": "x"},
        });
        let request = engine_request("video.generation", "minimax-h3-fl2va", &params).unwrap();
        assert!(matches!(
            request,
            EngineRequest::Unsupported {
                action: Action::VideoCreate,
                ..
            }
        ));
    }

    #[test]
    fn video_edit_routes_ltx25() {
        let params = json!({
            "family": "ltx25",
            "request": {
                "model": "ltx-2.5-distilled:mlx-q4",
                "source_asset_id": "ast_src",
                "prompt": "the cube rotates",
                "num_frames": 25,
                "size": "640x640",
            },
        });
        let request = engine_request("video.edit", "ltx-2.5-distilled:mlx-q4", &params).unwrap();
        match request {
            EngineRequest::VideoEdit(request) => {
                assert_eq!(request.source_asset_id, "ast_src");
                assert_eq!(request.num_frames, Some(25));
                assert_eq!(request.size.as_deref(), Some("640x640"));
            }
            other => panic!("ltx25 video edit routed to {other:?}"),
        }
        let params = json!({
            "family": "wan",
            "request": {"model": "wan-2.2-ti2v-5b", "source_asset_id": "ast_src", "prompt": "x"},
        });
        let request = engine_request("video.edit", "wan-2.2-ti2v-5b", &params).unwrap();
        assert!(matches!(
            request,
            EngineRequest::Unsupported {
                action: Action::VideoEdit,
                ..
            }
        ));
    }
}
