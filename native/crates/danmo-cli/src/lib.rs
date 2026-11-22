//! Shared host for the `danmo-*` command line tools.
//!
//! The CLI is a thin host over the same engine the HTTP server drives:
//! registry + workspace resolution goes through `danmo-catalog` /
//! `danmo-resolve`, task mapping through `danmo_engine::engine_request`, and
//! generation through `Engine::execute`. No family branches live here.

use danmo_engine::{
    Action, ArtifactSink, BackendKind, Engine, EngineError, ExecutionContext, PersistedAsset,
    ResolvePort, ResolvedRun, RunObserver,
};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Minimal `--key value` / `--flag` parser. Repeatable keys collect in order.
#[derive(Debug, Default)]
pub struct Args {
    values: std::collections::HashMap<String, Vec<String>>,
}

impl Args {
    pub fn parse_env() -> Result<Self, String> {
        let argv: Vec<String> = std::env::args().skip(1).collect();
        Self::parse(&argv)
    }

    pub fn parse(argv: &[String]) -> Result<Self, String> {
        let mut values: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        let mut index = 0;
        while index < argv.len() {
            let arg = &argv[index];
            let Some(key) = arg.strip_prefix("--") else {
                return Err(format!("unexpected argument {arg:?} (want --key value)"));
            };
            let (key, inline) = match key.split_once('=') {
                Some((key, value)) => (key.to_string(), Some(value.to_string())),
                None => (key.to_string(), None),
            };
            let value = match inline {
                Some(value) => value,
                None => {
                    let next = argv.get(index + 1);
                    match next {
                        Some(value) if !value.starts_with("--") => {
                            index += 1;
                            value.clone()
                        }
                        _ => "true".to_string(),
                    }
                }
            };
            values.entry(key).or_default().push(value);
            index += 1;
        }
        Ok(Self { values })
    }

    pub fn has(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key)?.last().map(String::as_str)
    }

    pub fn require(&self, key: &str) -> Result<String, String> {
        self.get(key)
            .map(str::to_string)
            .ok_or_else(|| format!("--{key} is required"))
    }

    pub fn all(&self, key: &str) -> Vec<&str> {
        self.values
            .get(key)
            .map(|rows| rows.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }

    pub fn u32(&self, key: &str) -> Result<Option<u32>, String> {
        self.get(key)
            .map(|raw| {
                raw.parse::<u32>()
                    .map_err(|_| format!("--{key} {raw:?} is not a positive integer"))
            })
            .transpose()
    }

    pub fn i64(&self, key: &str) -> Result<Option<i64>, String> {
        self.get(key)
            .map(|raw| {
                raw.parse::<i64>()
                    .map_err(|_| format!("--{key} {raw:?} is not an integer"))
            })
            .transpose()
    }

    pub fn f32(&self, key: &str) -> Result<Option<f32>, String> {
        self.get(key)
            .map(|raw| {
                raw.parse::<f32>()
                    .map_err(|_| format!("--{key} {raw:?} is not a number"))
            })
            .transpose()
    }
}

/// CLI host: workspace + registry + backend, and the engine ports.
pub struct Host {
    pub workspace: PathBuf,
    pub registry: PathBuf,
    pub backend: BackendKind,
    output: PathBuf,
}

impl Host {
    pub fn new(args: &Args) -> Result<Self, String> {
        let workspace = args
            .get("workspace")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("DANQING_WORKSPACE").map(PathBuf::from))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let registry = args
            .get("registry")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("DANQING_REGISTRY").map(PathBuf::from))
            .unwrap_or_else(|| {
                let local = workspace.join("config").join("models_registry.json");
                if local.is_file() {
                    local
                } else {
                    PathBuf::from("default_config/models_registry.json")
                }
            });
        if !registry.is_file() {
            return Err(format!(
                "registry {} is missing; pass --registry",
                registry.display()
            ));
        }
        let selection = danmo_engine::select_from_env().map_err(|error| error.to_string())?;
        eprintln!(
            "[cli] backend {} ({})",
            selection.backend.name(),
            selection.reason
        );
        let output = PathBuf::from(args.require("output")?);
        Ok(Self {
            workspace,
            registry,
            backend: selection.backend,
            output,
        })
    }

    /// A CLI source reference is a file path (absolute or workspace-relative).
    /// `asset:` ids belong to the HTTP asset store and fail loud here.
    fn source_path(&self, source: &str) -> Result<PathBuf, String> {
        if let Some(id) = source.strip_prefix("asset:") {
            return Err(format!("CLI has no asset store for {id}; pass a file path"));
        }
        let path = Path::new(source);
        Ok(if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.workspace.join(path)
        })
    }
}

impl ResolvePort for Host {
    fn resolve(&self, model: &str, action: Action) -> Result<ResolvedRun, EngineError> {
        let catalog = danmo_catalog::Catalog::open(&self.registry)
            .map_err(|e| EngineError::Resolve(e.to_string()))?;
        let bare = model.split(':').next().unwrap_or(model);
        let entry = catalog
            .model(bare)
            .map_err(|e| EngineError::Resolve(e.to_string()))?;
        let bundle = danmo_resolve::resolve_bundle(&self.workspace, &self.registry, model)
            .map_err(EngineError::Resolve)?;
        Ok(ResolvedRun {
            model: model.into(),
            version: model
                .split_once(':')
                .map(|(_, version)| version.to_string()),
            family: entry.family,
            action,
            bundle,
            backend: self.backend,
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
        danmo_resolve::resolve_bundle(&self.workspace, &self.registry, model)
            .map_err(EngineError::Resolve)
    }

    fn read_source_audio(&self, _run: &ResolvedRun, source: &str) -> Result<Vec<u8>, EngineError> {
        if source.strip_prefix("base64:").is_some() {
            return Err(EngineError::Resolve(
                "CLI does not accept base64 audio; pass a file path".into(),
            ));
        }
        let path = self.source_path(source).map_err(EngineError::Resolve)?;
        std::fs::read(&path).map_err(|error| {
            EngineError::Resolve(format!("source audio {}: {error}", path.display()))
        })
    }

    fn read_source_image(
        &self,
        _run: &ResolvedRun,
        source_asset_id: &str,
    ) -> Result<(Vec<u8>, u32, u32), EngineError> {
        let path = self
            .source_path(source_asset_id)
            .map_err(EngineError::Resolve)?;
        danmo_engine::image::decode_file_rgb(&path).map_err(EngineError::Resolve)
    }

    fn read_source_video_path(
        &self,
        _run: &ResolvedRun,
        source_asset_id: &str,
    ) -> Result<PathBuf, EngineError> {
        let path = self
            .source_path(source_asset_id)
            .map_err(EngineError::Resolve)?;
        if !path.is_file() {
            return Err(EngineError::Resolve(format!(
                "source video {} is missing",
                path.display()
            )));
        }
        Ok(path)
    }
}

impl ArtifactSink for Host {
    fn persist(
        &self,
        _run: &ResolvedRun,
        task_id: &str,
        artifacts: Vec<danmo_engine::Artifact>,
    ) -> Result<Vec<PersistedAsset>, EngineError> {
        let mut persisted = Vec::new();
        for (index, artifact) in artifacts.into_iter().enumerate() {
            let path = if index == 0 {
                self.output.clone()
            } else {
                sibling_path(&self.output, &format!("-{index}"), &artifact.extension)
            };
            write_bytes(&path, &artifact.bytes)?;
            for sidecar in &artifact.sidecars {
                let sidecar_path = sibling_path(&path, "", &sidecar.suffix);
                write_bytes(&sidecar_path, &sidecar.bytes)?;
            }
            persisted.push(PersistedAsset {
                id: format!("{task_id}-{index}"),
                path,
            });
        }
        Ok(persisted)
    }
}

fn sibling_path(output: &Path, tag: &str, extension: &str) -> PathBuf {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let stem = output
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("output");
    parent.join(format!("{stem}{tag}.{extension}"))
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<(), EngineError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| EngineError::Persist(error.to_string()))?;
    }
    std::fs::write(path, bytes).map_err(|error| EngineError::Persist(error.to_string()))
}

struct CliObserver {
    trace: Option<Arc<danmo_otel::TaskTrace>>,
}

impl CliObserver {
    fn record_phase(&self, phase: &str) {
        let Some(trace) = self.trace.as_ref() else {
            return;
        };
        if phase.is_empty() || trace.current_phase().as_deref() == Some(phase) {
            return;
        }
        trace.phase(phase);
    }
}

impl RunObserver for CliObserver {
    fn progress(&self, phase: &str, step: u32, total: u32) -> Result<(), EngineError> {
        if let Some(trace) = self.trace.as_ref() {
            trace.progress(phase, step, total);
        }
        let stride = (total / 20).max(1);
        if step == total || step % stride == 0 {
            eprintln!("[cli] {phase} {step}/{total}");
        }
        Ok(())
    }

    fn log(&self, level: &str, message: &str) -> Result<(), EngineError> {
        if let Some(trace) = self.trace.as_ref() {
            trace.log_event(level, message);
        }
        eprintln!("[cli] {level}: {message}");
        Ok(())
    }

    fn phase(&self, phase: &str) -> Result<(), EngineError> {
        self.record_phase(phase);
        eprintln!("[cli] == {phase}");
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
}

/// Submit one task to the engine and print the persisted output paths.
pub fn run_task(host: &Host, kind: &str, model: &str, mut request: Value) -> Result<(), String> {
    let catalog = danmo_catalog::Catalog::open(&host.registry).map_err(|e| e.to_string())?;
    let bare = model.split(':').next().unwrap_or(model);
    let entry = catalog.model(bare).map_err(|e| e.to_string())?;
    fill_registry_defaults(&catalog, &entry.id, &mut request);
    ensure_seed(&mut request);
    let obj = request.as_object_mut().ok_or("request must be an object")?;
    obj.entry("preview".to_string())
        .or_insert_with(|| json!(false));
    obj.entry("model".to_string())
        .or_insert_with(|| json!(model));

    let task_id = format!(
        "cli-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|span| span.as_millis())
            .unwrap_or(0)
    );
    let params = json!({
        "task_id": task_id,
        "family": entry.family,
        "request": request,
    });
    let engine_request =
        danmo_engine::engine_request(kind, model, &params).map_err(|e| e.to_string())?;
    let engine = Engine::new(host, host, host.backend);
    // CLI runs land in the same trace store as the API so a local generation can
    // be inspected in the task page. A store that cannot be opened fails loud:
    // silently skipping would make "no trace" indistinguishable from "not traced".
    let telemetry =
        danmo_otel::Telemetry::init(&danmo_otel::cli_db_path(&host.workspace), "danmo-cli")
            .map_err(|error| error.to_string())?;
    let trace = Arc::new(telemetry.start_task(
        &task_id,
        vec![
            danmo_otel::attribute("task.id", task_id.clone()),
            danmo_otel::attribute("task.kind", kind),
            danmo_otel::attribute("gen_ai.request.model", model),
            danmo_otel::attribute("danmo.backend", host.backend.name()),
            danmo_otel::attribute("danmo.source", "cli"),
        ],
    ));
    let observer = Arc::new(CliObserver {
        trace: Some(trace.clone()),
    });
    let exec = ExecutionContext::new(task_id, observer);
    let outcome = engine.execute(engine_request, exec);
    match &outcome {
        Ok(_) => {}
        Err(EngineError::Cancelled) => trace.cancelled(),
        Err(error) => trace.fail(&error.to_string()),
    }
    trace.finish();
    telemetry.shutdown().map_err(|error| error.to_string())?;
    let result = outcome.map_err(|e| e.to_string())?;
    for path in &result.output_paths {
        println!("{}", path.display());
    }
    Ok(())
}

/// Registry `ui.parameters` defaults for the knobs the CLI leaves unset.
fn fill_registry_defaults(catalog: &danmo_catalog::Catalog, id: &str, body: &mut Value) {
    let Some(params) = catalog
        .root()
        .pointer(&format!("/models/{id}/ui/parameters"))
        .and_then(|value| value.as_object())
    else {
        return;
    };
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    for key in ["steps", "guidance", "scheduler", "scheduler_shift"] {
        let present = obj.get(key).is_some_and(|value| !value.is_null());
        if present {
            continue;
        }
        let Some(spec) = params.get(key) else {
            continue;
        };
        let default = spec.get("default").unwrap_or(spec);
        if default.is_null() || default.is_object() || default.is_array() {
            continue;
        }
        obj.insert((*key).to_string(), default.clone());
    }
    stamp_edit_card(params, obj);
}

fn stamp_edit_card(
    params: &serde_json::Map<String, Value>,
    body: &mut serde_json::Map<String, Value>,
) {
    const KEYS: &[&str] = &[
        "edit_use_vl_vision",
        "edit_conditioning_latent_concat",
        "edit_use_picture_prefix",
        "edit_plus_multi_image",
        "edit_max_reference_images",
    ];
    if !KEYS.iter().any(|key| params.contains_key(*key)) {
        return;
    }
    let metadata = body
        .entry("metadata".to_string())
        .or_insert_with(|| json!({}));
    let Some(metadata) = metadata.as_object_mut() else {
        return;
    };
    for key in KEYS {
        if metadata.contains_key(*key) {
            continue;
        }
        if let Some(value) = params.get(*key) {
            if value.is_boolean() || value.is_number() {
                metadata.insert((*key).to_string(), value.clone());
            }
        }
    }
}

fn ensure_seed(request: &mut Value) {
    let missing = request.get("seed").is_none_or(|seed| seed.is_null());
    if missing {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|span| span.subsec_nanos() as i64 ^ (span.as_secs() as i64))
            .unwrap_or(42)
            .abs();
        request["seed"] = json!(seed);
    }
}

/// Parse one `--lora` value: `id[:weight]` or `path[:weight]`. Resolution
/// order: file on disk, workspace `config/user_loras.json`, registry card.
pub fn parse_lora(host: &Host, spec: &str) -> Result<Value, String> {
    let (id, weight) = match spec.rsplit_once(':') {
        Some((head, tail)) if !head.is_empty() && tail.parse::<f32>().is_ok() => {
            (head, tail.parse::<f32>().unwrap_or(1.0))
        }
        _ => (spec, 1.0),
    };
    if !weight.is_finite() || !(0.0..=2.0).contains(&weight) {
        return Err(format!("lora {id} weight is not between 0 and 2"));
    }
    let candidate = host.source_path(id)?;
    let path = if candidate.is_file() {
        candidate
    } else if let Some(path) = user_lora_path(host, id)? {
        path
    } else {
        registry_lora_path(host, id)?
    };
    Ok(json!({
        "id": id,
        "weight": weight,
        "path": path,
    }))
}

pub fn collect_loras(host: &Host, args: &Args) -> Result<Vec<Value>, String> {
    args.all("lora")
        .into_iter()
        .map(|spec| parse_lora(host, spec))
        .collect()
}

fn user_lora_path(host: &Host, id: &str) -> Result<Option<PathBuf>, String> {
    let path = host.workspace.join("config").join("user_loras.json");
    if !path.is_file() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let root: Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    let Some(items) = root.get("items").and_then(Value::as_array) else {
        return Ok(None);
    };
    for item in items {
        if item.get("id").and_then(Value::as_str) != Some(id) {
            continue;
        }
        let local = item
            .get("local_path")
            .and_then(Value::as_str)
            .filter(|local| !local.is_empty())
            .ok_or_else(|| format!("user lora {id} has no local_path"))?;
        let dir = danmo_resolve::resolve_model_path(&host.workspace, local);
        return danmo_resolve::lora_weight_file(&dir).map(Some);
    }
    Ok(None)
}

fn registry_lora_path(host: &Host, id: &str) -> Result<PathBuf, String> {
    let catalog = danmo_catalog::Catalog::open(&host.registry).map_err(|e| e.to_string())?;
    let (bare, version) = match id.split_once(':') {
        Some((bare, version)) => (bare, Some(version)),
        None => (id, None),
    };
    let entry = catalog
        .root()
        .pointer(&format!("/models/{bare}"))
        .ok_or_else(|| format!("lora {id} is not a file, user LoRA, or registry card"))?;
    danmo_resolve::registry_lora_weight(&host.workspace, entry, version)?
        .ok_or_else(|| format!("lora {id} is not installed"))
}

/// Insert optional scalar flags into the request object when present.
pub fn apply_options(
    request: &mut Map<String, Value>,
    args: &Args,
    keys: &[&str],
) -> Result<(), String> {
    for &key in keys {
        if let Some(raw) = args.get(key) {
            let field = key.replace('-', "_");
            let value = if let Ok(int) = raw.parse::<i64>() {
                json!(int)
            } else if let Ok(float) = raw.parse::<f64>() {
                json!(float)
            } else {
                json!(raw)
            };
            request.insert(field, value);
        }
    }
    Ok(())
}

/// Workspace root for tools that take no `--output` (e.g. `danmo-quantize`).
pub fn workspace_root(args: &Args) -> Result<PathBuf, String> {
    Ok(args
        .get("workspace")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("DANQING_WORKSPACE").map(PathBuf::from))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))))
}

/// Factory checkout root: the parent of `default_config/quant_recipes`.
pub fn repo_root(args: &Args) -> Result<PathBuf, String> {
    if let Some(explicit) = args
        .get("repo-root")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("DANQING_REPO_ROOT").map(PathBuf::from))
    {
        if !explicit.join("default_config/quant_recipes").is_dir() {
            return Err(format!(
                "{} has no default_config/quant_recipes",
                explicit.display()
            ));
        }
        return Ok(explicit);
    }
    let mut probe = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    for _ in 0..6 {
        if probe.join("default_config/quant_recipes").is_dir() {
            return Ok(probe);
        }
        if !probe.pop() {
            break;
        }
    }
    Err("cannot find default_config/quant_recipes; pass --repo-root".into())
}

/// Resolve a workspace-relative or absolute path.
pub fn resolve_path(workspace: &Path, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        workspace.join(path)
    }
}

/// Every recipe id shipped in `default_config/quant_recipes`.
pub fn list_recipes(workspace: &Path, repo_root: &Path) -> Result<Vec<String>, String> {
    let mut ids = Vec::new();
    for dir in [
        workspace.join("config/quant_recipes"),
        repo_root.join("default_config/quant_recipes"),
    ] {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            if !ids.iter().any(|id| id == stem) {
                ids.push(stem.to_string());
            }
        }
    }
    ids.sort();
    Ok(ids)
}

/// Dense bundle layout decides which family recipe applies. Unknown layouts
/// fail loud instead of guessing a recipe.
pub fn default_recipe_for(base: &Path) -> Result<String, String> {
    let has = |name: &str| base.join(name).is_dir();
    let any_file = |ext: &str| {
        std::fs::read_dir(base)
            .ok()
            .map(|entries| {
                entries.flatten().any(|entry| {
                    entry.path().extension().and_then(|value| value.to_str()) == Some(ext)
                })
            })
            .unwrap_or(false)
    };
    if has("FL2VA") {
        return Ok("minimax-h3-q4-mixed-v1".into());
    }
    if has("dit") && !has("transformer") {
        return Ok("longcat-avatar-q4-mixed-v1".into());
    }
    if base.join("diffusion_models").is_dir() {
        return Ok("ltx25-q4-mixed-v1".into());
    }
    if has("transformer") && has("text_encoder") && has("vae") {
        return Ok("z-image-q4-mixed-v1".into());
    }
    if has("transformer") {
        return Ok("flux2-q4-mixed-v1".into());
    }
    if any_file("safetensors") {
        // Flat bundle: an LLM tree or an upscaler. A tokenizer beside the
        // shards means an LLM.
        if base.join("tokenizer.json").is_file() {
            return Ok("qwen3-q4-mixed-v1".into());
        }
        return Ok("seedvr2-q4-mixed-v1".into());
    }
    Err(format!(
        "cannot infer a quantization recipe for {}; pass --recipe",
        base.display()
    ))
}

/// Default output directory beside the dense base. Same layout rule as the
/// installer (`danmo_engine::derived_layout`), so CLI and API cannot drift.
pub fn default_output(base: &Path) -> String {
    danmo_engine::derived_layout(Path::new(""), &base.display().to_string(), "q4-mixed")
        .display()
        .to_string()
}

/// Convert a dense bundle. Thin wrapper over the engine entry point so the CLI
/// and the installer cannot drift.
pub fn run_quantize(
    workspace: &Path,
    repo_root: &Path,
    base: &Path,
    output: &str,
    recipe_id: &str,
) -> Result<danmo_engine::QuantizeReport, String> {
    danmo_engine::run_derive(
        workspace,
        repo_root,
        &base.display().to_string(),
        output,
        recipe_id,
        Box::new(|| false),
        Box::new(|message| eprintln!("[danmo-quantize] {message}")),
    )
}

/// Find a recipe file for `--print-recipe`.
pub fn find_recipe(workspace: &Path, repo_root: &Path, id: &str) -> Result<String, String> {
    danmo_engine::find_recipe(workspace, repo_root, id).map(|path| path.display().to_string())
}
