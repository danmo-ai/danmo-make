//! Pure HTTP download client for the models page.
//! File URLs come from the registry or from a hub file listing. Names are not invented.
//! The DiT weight must be GGUF. A version may also list `components`: the text encoder,
//! VAE, and tokenizer published beside that GGUF. Those files land in the version
//! directory. When the last file lands, that install writes `bundle.manifest.json`
//! there. The engine still loads GGUF only.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use ureq::Agent;

const USER_AGENT: &str = "danmo-make/gguf";

#[derive(Clone)]
pub struct Downloads {
    inner: Arc<Mutex<Inner>>,
    agent: Agent,
    workspace: PathBuf,
    persist: PathBuf,
    loras_path: PathBuf,
    settings_path: PathBuf,
}

struct Inner {
    tasks: Vec<Task>,
    cancels: HashMap<String, Arc<AtomicBool>>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct Task {
    id: String,
    kind: String,
    model_name: String,
    version: String,
    filename: String,
    url: String,
    status: String,
    progress: f64,
    downloaded_size: u64,
    total_size: u64,
    speed: String,
    error_message: String,
    files: Vec<PlannedFile>,
    job: Job,
    created_at: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PlannedFile {
    url: String,
    dest: String,
    size: Option<u64>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Job {
    Model {
        spec: Value,
    },
    Direct {
        files: Vec<PlannedFile>,
    },
    Lora {
        source: String,
        repo_id: String,
        filename: String,
        url: String,
        base_model: String,
        display_name: String,
    },
}

#[derive(Debug)]
enum Plan {
    Files(Vec<PlannedFile>),
}

enum Done {
    Finished,
    Cancelled,
}

#[derive(Debug)]
struct RemoteEntry {
    path: String,
    size: Option<u64>,
}

struct RepoQuery {
    source: String,
    repo_id: String,
    revision: String,
}

#[derive(Clone)]
struct HubAuth {
    huggingface: String,
    civitai: String,
    nsfw: bool,
}

pub fn resolve_model_path(workspace: &Path, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        workspace.join(path)
    }
}

pub fn version_ready(path: &Path) -> bool {
    if path.join("bundle.manifest.json").is_file() {
        return true;
    }
    let mut gguf = false;
    let mut partial = false;
    walk_marks(path, &mut gguf, &mut partial);
    gguf && !partial
}

fn walk_marks(path: &Path, gguf: &mut bool, partial: &mut bool) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            walk_marks(&path, gguf, partial);
        } else if name.ends_with(".partial") {
            *partial = true;
        } else if name.to_ascii_lowercase().ends_with(".gguf") {
            *gguf = true;
        }
    }
}

impl Downloads {
    pub fn open(workspace: &Path) -> Self {
        let workspace = if workspace.as_os_str().is_empty() {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        } else {
            workspace.to_path_buf()
        };
        let config = workspace.join("config");
        let _ = fs::create_dir_all(&config);
        let persist = config.join("download_tasks.json");
        let loras_path = config.join("downloaded_loras.json");
        let settings_path = config.join(".app_config.json");
        let tasks = load_tasks(&persist);
        let agent: Agent = Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(20)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            .user_agent(USER_AGENT)
            .build()
            .into();
        Self {
            inner: Arc::new(Mutex::new(Inner {
                tasks,
                cancels: HashMap::new(),
            })),
            agent,
            workspace,
            persist,
            loras_path,
            settings_path,
        }
    }

    fn resolve(&self, path: &str) -> PathBuf {
        resolve_model_path(&self.workspace, path)
    }

    pub fn list(&self) -> Vec<Value> {
        self.lock().tasks.iter().rev().map(task_json).collect()
    }

    pub fn snapshot(&self, id: &str) -> Option<Value> {
        self.lock()
            .tasks
            .iter()
            .find(|task| task.id == id)
            .map(task_json)
    }

    pub fn install_model(
        &self,
        model_id: &str,
        version: &str,
        spec: &Value,
    ) -> Result<String, String> {
        {
            let inner = self.lock();
            if let Some(task) = inner.tasks.iter().find(|task| {
                task.model_name == model_id
                    && task.version == version
                    && matches!(task.status.as_str(), "pending" | "running" | "paused")
            }) {
                return Ok(task.id.clone());
            }
        }
        let task = Task {
            id: new_id(),
            kind: "model".into(),
            model_name: model_id.into(),
            version: version.into(),
            filename: format!("{model_id}:{version}"),
            url: String::new(),
            status: "pending".into(),
            progress: 0.0,
            downloaded_size: 0,
            total_size: 0,
            speed: String::new(),
            error_message: String::new(),
            files: Vec::new(),
            job: Job::Model { spec: spec.clone() },
            created_at: now_rfc3339(),
        };
        self.push_and_start(task, true)
    }

    pub fn start_url(&self, url: &str, filename: &str, kind: &str) -> Result<String, String> {
        let url = url.trim();
        if !(url.starts_with("https://") || url.starts_with("http://")) {
            return Err("download URL must be http or https".into());
        }
        let raw_name = if filename.trim().is_empty() {
            file_name_from_url(url)
        } else {
            filename.trim().to_string()
        };
        let filename = sanitize_name(&raw_name);
        if filename.is_empty() {
            return Err("download file name is empty".into());
        }
        if kind != "lora"
            && !filename.to_ascii_lowercase().ends_with(".gguf")
            && !url_path_is_gguf(url)
        {
            return Err(format!("download URL is not a GGUF file: {url}"));
        }
        let dest = self.resolve("models/imports").join(&filename);
        let file = PlannedFile {
            url: url.into(),
            dest: dest.display().to_string(),
            size: None,
        };
        let task = Task {
            id: new_id(),
            kind: kind.into(),
            model_name: String::new(),
            version: String::new(),
            filename: filename.clone(),
            url: url.into(),
            status: "pending".into(),
            progress: 0.0,
            downloaded_size: 0,
            total_size: 0,
            speed: String::new(),
            error_message: String::new(),
            files: vec![file.clone()],
            job: Job::Direct { files: vec![file] },
            created_at: now_rfc3339(),
        };
        self.push_and_start(task, true)
    }

    pub fn start_lora(&self, body: &Value) -> Result<String, String> {
        let source = body
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let repo_id = body
            .get("repo_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let filename = body
            .get("filename")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let url = body
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let base_model = body
            .get("base_model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let display_name = body
            .get("display_name")
            .and_then(|v| v.as_str())
            .filter(|name| !name.is_empty())
            .unwrap_or(if !filename.is_empty() {
                filename.as_str()
            } else {
                repo_id.as_str()
            })
            .to_string();
        if url.is_empty() && repo_id.is_empty() {
            return Err("no downloadable file was published for this LoRA".into());
        }
        if !url.is_empty() && !(url.starts_with("https://") || url.starts_with("http://")) {
            return Err("download URL must be http or https".into());
        }
        let task = Task {
            id: new_id(),
            kind: "lora".into(),
            model_name: String::new(),
            version: String::new(),
            filename: if filename.is_empty() {
                display_name.clone()
            } else {
                filename.clone()
            },
            url: url.clone(),
            status: "pending".into(),
            progress: 0.0,
            downloaded_size: 0,
            total_size: 0,
            speed: String::new(),
            error_message: String::new(),
            files: Vec::new(),
            job: Job::Lora {
                source,
                repo_id,
                filename,
                url,
                base_model,
                display_name,
            },
            created_at: now_rfc3339(),
        };
        self.push_and_start(task, true)
    }

    pub fn cancel(&self, id: &str) -> Result<(), String> {
        let inner = self.lock();
        let task = inner
            .tasks
            .iter()
            .find(|task| task.id == id)
            .ok_or_else(|| "download task not found".to_string())?;
        if matches!(task.status.as_str(), "completed" | "cancelled") {
            return Ok(());
        }
        if let Some(flag) = inner.cancels.get(id) {
            flag.store(true, Ordering::Relaxed);
        }
        drop(inner);
        self.set_status(id, "cancelled", "");
        Ok(())
    }

    pub fn resume(&self, id: &str) -> Result<(), String> {
        {
            let inner = self.lock();
            let task = inner
                .tasks
                .iter()
                .find(|task| task.id == id)
                .ok_or_else(|| "download task not found".to_string())?;
            if matches!(task.status.as_str(), "running" | "pending") {
                return Ok(());
            }
            if task.status == "completed" {
                return Err("download already completed".into());
            }
        }
        self.set_status(id, "pending", "");
        self.spawn(id.to_string(), false);
        Ok(())
    }

    pub fn delete_task(&self, id: &str) -> Result<(), String> {
        if let Some(flag) = self.lock().cancels.get(id).cloned() {
            flag.store(true, Ordering::Relaxed);
        }
        let mut inner = self.lock();
        let Some(pos) = inner.tasks.iter().position(|task| task.id == id) else {
            return Err("download task not found".into());
        };
        let task = inner.tasks.remove(pos);
        inner.cancels.remove(id);
        drop(inner);
        for file in &task.files {
            let partial = partial_path(Path::new(&file.dest));
            let _ = fs::remove_file(partial);
        }
        self.persist();
        Ok(())
    }

    pub fn search(&self, source: &str, query: &str, limit: usize) -> Result<Vec<Value>, String> {
        let auth = self.auth();
        let wanted: Vec<&str> = match source {
            "huggingface" | "modelscope" | "civitai" => vec![source],
            _ => vec!["huggingface", "modelscope", "civitai"],
        };
        let mut items = Vec::new();
        let mut errors = Vec::new();
        for hub in wanted {
            match self.search_one(hub, query, limit, &auth) {
                Ok(mut rows) => items.append(&mut rows),
                Err(err) => errors.push(format!("{hub}: {err}")),
            }
        }
        if items.is_empty() && !errors.is_empty() {
            return Err(errors.join("; "));
        }
        items.truncate(limit);
        Ok(items)
    }

    pub fn list_loras(&self) -> Value {
        let mut items = load_lora_items(&self.loras_path);
        items.retain(|item| {
            item.get("local_path")
                .and_then(|v| v.as_str())
                .map(|path| Path::new(path).is_file())
                .unwrap_or(false)
        });
        json!({ "items": items })
    }

    pub fn delete_lora(&self, id: &str, remove_files: bool) -> Result<Value, String> {
        let mut items = load_lora_items(&self.loras_path);
        let Some(pos) = items
            .iter()
            .position(|item| item.get("id").and_then(|v| v.as_str()) == Some(id))
        else {
            return Err("lora not found".into());
        };
        let item = items.remove(pos);
        if remove_files {
            if let Some(path) = item.get("local_path").and_then(|v| v.as_str()) {
                let path = PathBuf::from(path);
                if path.is_file() {
                    let _ = fs::remove_file(&path);
                }
            }
        }
        save_lora_items(&self.loras_path, &items)?;
        Ok(json!({ "ok": true }))
    }

    fn search_one(
        &self,
        hub: &str,
        query: &str,
        limit: usize,
        auth: &HubAuth,
    ) -> Result<Vec<Value>, String> {
        match hub {
            "huggingface" => {
                let url = format!(
                    "https://huggingface.co/api/models?search={}&limit={limit}&filter=lora",
                    encode_component(query)
                );
                let value = self.get_json(&url, token_header("huggingface", auth))?;
                parse_hf_search(&value, auth.nsfw)
            }
            "modelscope" => {
                let url = "https://www.modelscope.cn/api/v1/dolphin/models";
                let body = json!({
                    "PageSize": limit,
                    "PageNumber": 1,
                    "SortBy": "Default",
                    "Name": query,
                    "SingleCriterion": [],
                    "Criterion": [],
                });
                let value = self.put_json(url, &body.to_string())?;
                parse_ms_search(&value)
            }
            "civitai" => {
                let nsfw = if auth.nsfw { "true" } else { "false" };
                let url = format!(
                    "https://civitai.com/api/v1/models?types=LORA&limit={limit}&nsfw={nsfw}&query={}",
                    encode_component(query)
                );
                let value = self.get_json(&url, token_header("civitai", auth))?;
                parse_civitai_search(&value, auth.nsfw)
            }
            other => Err(format!("unknown LoRA source {other}")),
        }
    }

    fn push_and_start(&self, task: Task, force: bool) -> Result<String, String> {
        let id = task.id.clone();
        self.lock().tasks.push(task);
        self.persist();
        self.spawn(id.clone(), force);
        Ok(id)
    }

    fn spawn(&self, id: String, force: bool) {
        let cancel = Arc::new(AtomicBool::new(false));
        self.lock().cancels.insert(id.clone(), cancel.clone());
        let hub = self.clone();
        let _ = std::thread::Builder::new()
            .name(format!("download-{id}"))
            .spawn(move || hub.run(&id, force, &cancel));
    }

    fn run(&self, id: &str, force: bool, cancel: &AtomicBool) {
        if cancel.load(Ordering::Relaxed) {
            self.set_status(id, "cancelled", "");
            return;
        }
        self.set_status(id, "running", "");
        let job = {
            let inner = self.lock();
            let Some(task) = inner.tasks.iter().find(|task| task.id == id) else {
                return;
            };
            task.job.clone()
        };
        let outcome = match job {
            Job::Model { spec } => self.run_model(id, &spec, force, cancel),
            Job::Direct { files } => self.run_files(id, &files, force, cancel),
            Job::Lora {
                source,
                repo_id,
                filename,
                url,
                base_model,
                display_name,
            } => self.run_lora(
                id,
                &source,
                &repo_id,
                &filename,
                &url,
                &base_model,
                &display_name,
                force,
                cancel,
            ),
        };
        match outcome {
            Ok(Done::Finished) => self.finish(id),
            Ok(Done::Cancelled) => self.set_status(id, "cancelled", ""),
            Err(err) => self.set_status(id, "failed", &err),
        }
    }

    fn run_model(
        &self,
        id: &str,
        spec: &Value,
        force: bool,
        cancel: &AtomicBool,
    ) -> Result<Done, String> {
        let auth = self.auth();
        let workspace = self.workspace.clone();
        let plan = plan_version(spec, &mut |query| self.list_repo(query, &auth), &|path| {
            resolve_model_path(&workspace, path)
        })?;
        let Plan::Files(files) = plan;
        self.remember_files(id, &files);
        let done = self.run_files(id, &files, force, cancel)?;
        if matches!(done, Done::Finished) && version_lists_components(spec) {
            let quant = self.task_version(id)?;
            let local = spec
                .get("local_path")
                .and_then(|v| v.as_str())
                .filter(|path| !path.is_empty())
                .ok_or_else(|| "version has no local_path".to_string())?;
            let root = resolve_model_path(&workspace, local);
            write_bundle_manifest(&root, &quant, &files)?;
        }
        Ok(done)
    }

    fn task_version(&self, id: &str) -> Result<String, String> {
        let inner = self.lock();
        let version = inner
            .tasks
            .iter()
            .find(|task| task.id == id)
            .map(|task| task.version.clone())
            .unwrap_or_default();
        if !matches!(version.as_str(), "bf16" | "q8_0" | "q4_k") {
            return Err(format!("version {version} is not bf16, q8_0, or q4_k"));
        }
        Ok(version)
    }

    fn run_lora(
        &self,
        id: &str,
        source: &str,
        repo_id: &str,
        filename: &str,
        url: &str,
        base_model: &str,
        display_name: &str,
        force: bool,
        cancel: &AtomicBool,
    ) -> Result<Done, String> {
        let auth = self.auth();
        let (file_url, name) = if !url.is_empty() {
            let name = if filename.is_empty() {
                file_name_from_url(url)
            } else {
                filename.to_string()
            };
            (url.to_string(), sanitize_name(&name))
        } else {
            let revision = default_revision(source).to_string();
            let entries = self.list_repo(
                &RepoQuery {
                    source: source.into(),
                    repo_id: repo_id.into(),
                    revision: revision.clone(),
                },
                &auth,
            )?;
            let chosen = if filename.is_empty() {
                pick_lora_file(&entries)?
            } else if entries.iter().any(|entry| entry.path == filename) {
                filename.to_string()
            } else {
                return Err(format!("{repo_id} does not publish {filename}"));
            };
            let file_url = resolve_file_url(source, repo_id, &revision, &chosen)?;
            (file_url, sanitize_name(&chosen))
        };
        if name.is_empty() {
            return Err("download file name is empty".into());
        }
        let folder = sanitize_name(if display_name.is_empty() {
            &name
        } else {
            display_name
        });
        let dest = self.resolve("models/loras").join(folder).join(&name);
        let file = PlannedFile {
            url: file_url,
            dest: dest.display().to_string(),
            size: None,
        };
        self.remember_files(id, &[file.clone()]);
        {
            let mut inner = self.lock();
            if let Some(task) = inner.tasks.iter_mut().find(|task| task.id == id) {
                task.filename = name;
                if let Job::Lora {
                    base_model: slot,
                    display_name: label,
                    ..
                } = &mut task.job
                {
                    *slot = base_model.to_string();
                    if !display_name.is_empty() {
                        *label = display_name.to_string();
                    }
                }
            }
        }
        self.run_files(id, &[file], force, cancel)
    }

    fn run_files(
        &self,
        id: &str,
        files: &[PlannedFile],
        force: bool,
        cancel: &AtomicBool,
    ) -> Result<Done, String> {
        if files.is_empty() {
            return Err("no GGUF file is published for this download".into());
        }
        let known: u64 = files.iter().filter_map(|file| file.size).sum();
        let all_known = files.iter().all(|file| file.size.is_some());
        self.set_total(id, if all_known { known } else { 0 });
        let auth = self.auth();
        let mut downloaded = 0u64;
        let mut stamp = Instant::now();
        let mut stamp_bytes = 0u64;
        for file in files {
            if cancel.load(Ordering::Relaxed) {
                return Ok(Done::Cancelled);
            }
            let dest = PathBuf::from(&file.dest);
            let token = token_for(&file.url, &auth);
            let mut last = downloaded;
            let result = download_file(
                &self.agent,
                &file.url,
                &dest,
                token.as_deref(),
                file.size,
                force,
                cancel,
                &mut |got| {
                    let now = downloaded + got;
                    let elapsed = stamp.elapsed().as_secs_f64();
                    if elapsed >= 0.4 && now >= stamp_bytes {
                        let speed = ((now - stamp_bytes) as f64 / elapsed) as u64;
                        stamp = Instant::now();
                        stamp_bytes = now;
                        self.set_progress(id, now, speed);
                    }
                    last = now;
                },
            );
            downloaded = last;
            self.set_progress(id, downloaded, 0);
            match result {
                Ok(_) => {}
                Err(Stop::Cancel) => return Ok(Done::Cancelled),
                Err(Stop::Fail(err)) => return Err(err),
            }
        }
        write_receipts(files)?;
        Ok(Done::Finished)
    }

    fn finish(&self, id: &str) {
        let lora = {
            let mut inner = self.lock();
            let Some(task) = inner.tasks.iter_mut().find(|task| task.id == id) else {
                return;
            };
            task.status = "completed".into();
            task.progress = 1.0;
            task.speed = String::new();
            task.error_message = String::new();
            if task.total_size > 0 {
                task.downloaded_size = task.total_size;
            }
            lora_record(task)
        };
        if let Some(record) = lora {
            let _ = self.remember_lora(&record);
        }
        self.persist();
    }

    fn remember_lora(&self, record: &Value) -> Result<(), String> {
        let mut items = load_lora_items(&self.loras_path);
        if let Some(path) = record.get("local_path").and_then(|v| v.as_str()) {
            items.retain(|item| item.get("local_path").and_then(|v| v.as_str()) != Some(path));
        }
        items.push(record.clone());
        save_lora_items(&self.loras_path, &items)
    }

    fn remember_files(&self, id: &str, files: &[PlannedFile]) {
        let total: u64 = files.iter().filter_map(|file| file.size).sum();
        let all_known = !files.is_empty() && files.iter().all(|file| file.size.is_some());
        {
            let mut inner = self.lock();
            if let Some(task) = inner.tasks.iter_mut().find(|task| task.id == id) {
                task.files = files.to_vec();
                task.url = files
                    .first()
                    .map(|file| file.url.clone())
                    .unwrap_or_default();
                if all_known {
                    task.total_size = total;
                }
                if task.filename.is_empty() {
                    if let Some(file) = files.first() {
                        task.filename = Path::new(&file.dest)
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("")
                            .to_string();
                    }
                }
            }
        }
        self.persist();
    }

    fn list_repo(&self, query: &RepoQuery, auth: &HubAuth) -> Result<Vec<RemoteEntry>, String> {
        let url = list_url(&query.source, &query.repo_id, &query.revision)?;
        let token = token_for(&url, auth);
        let value = self.get_json(&url, token.as_deref())?;
        match query.source.as_str() {
            "huggingface" => parse_hf_tree(&value),
            "modelscope" => parse_ms_files(&value),
            other => Err(format!("{other} has no HTTP file listing")),
        }
    }

    fn get_json(&self, url: &str, token: Option<&str>) -> Result<Value, String> {
        let mut request = self.agent.get(url).header("accept", "application/json");
        if let Some(token) = token {
            request = request.header("authorization", &format!("Bearer {token}"));
        }
        let mut response = request
            .call()
            .map_err(|err| format!("GET {url} failed: {err}"))?;
        let status = response.status().as_u16();
        let mut body = String::new();
        response
            .body_mut()
            .as_reader()
            .take(20 * 1024 * 1024)
            .read_to_string(&mut body)
            .map_err(|err| format!("reading {url} failed: {err}"))?;
        if !(200..300).contains(&status) {
            return Err(format!("GET {url} returned {status}: {}", snippet(&body)));
        }
        serde_json::from_str(&body).map_err(|err| format!("{url} was not JSON: {err}"))
    }

    fn put_json(&self, url: &str, payload: &str) -> Result<Value, String> {
        let mut response = self
            .agent
            .put(url)
            .header("accept", "application/json")
            .header("content-type", "application/json")
            .send(payload)
            .map_err(|err| format!("PUT {url} failed: {err}"))?;
        let status = response.status().as_u16();
        let mut body = String::new();
        response
            .body_mut()
            .as_reader()
            .take(20 * 1024 * 1024)
            .read_to_string(&mut body)
            .map_err(|err| format!("reading {url} failed: {err}"))?;
        if !(200..300).contains(&status) {
            return Err(format!("PUT {url} returned {status}: {}", snippet(&body)));
        }
        serde_json::from_str(&body).map_err(|err| format!("{url} was not JSON: {err}"))
    }

    fn auth(&self) -> HubAuth {
        read_hub_auth(&self.settings_path)
    }

    fn set_status(&self, id: &str, status: &str, error: &str) {
        {
            let mut inner = self.lock();
            if let Some(task) = inner.tasks.iter_mut().find(|task| task.id == id) {
                task.status = status.into();
                task.error_message = error.into();
                if status != "running" {
                    task.speed.clear();
                }
            }
        }
        self.persist();
    }

    fn set_total(&self, id: &str, total: u64) {
        if total == 0 {
            return;
        }
        self.lock()
            .tasks
            .iter_mut()
            .filter(|task| task.id == id)
            .for_each(|task| task.total_size = total);
    }

    fn set_progress(&self, id: &str, downloaded: u64, speed: u64) {
        let mut inner = self.lock();
        if let Some(task) = inner.tasks.iter_mut().find(|task| task.id == id) {
            task.downloaded_size = downloaded;
            if task.total_size > 0 {
                task.progress = (downloaded as f64 / task.total_size as f64).clamp(0.0, 1.0);
            }
            if speed > 0 {
                task.speed = format!("{}/s", human_bytes(speed));
            }
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn persist(&self) {
        let tasks = self.lock().tasks.clone();
        let _ = save_tasks(&self.persist, &tasks);
    }
}

fn lora_record(task: &Task) -> Option<Value> {
    let Job::Lora {
        source,
        repo_id,
        base_model,
        display_name,
        ..
    } = &task.job
    else {
        return None;
    };
    let path = task.files.first()?.dest.clone();
    Some(json!({
        "id": task.id,
        "name": display_name,
        "base_model": base_model,
        "repo_id": repo_id,
        "remote_hub_source": source,
        "local_path": path,
        "filename": task.filename,
        "created_at": task.created_at,
    }))
}

fn task_json(task: &Task) -> Value {
    json!({
        "id": task.id,
        "kind": task.kind,
        "status": task.status,
        "model_name": task.model_name,
        "version": if task.version.is_empty() { Value::Null } else { json!(task.version) },
        "filename": task.filename,
        "url": task.url,
        "progress": task.progress,
        "downloaded_size": task.downloaded_size,
        "total_size": task.total_size,
        "speed": task.speed,
        "error_message": task.error_message,
        "created_at": task.created_at,
    })
}

#[derive(Debug)]
enum Stop {
    Cancel,
    Fail(String),
}

fn download_file(
    agent: &Agent,
    url: &str,
    dest: &Path,
    token: Option<&str>,
    size_hint: Option<u64>,
    force: bool,
    cancel: &AtomicBool,
    on_progress: &mut dyn FnMut(u64),
) -> Result<u64, Stop> {
    if cancel.load(Ordering::Relaxed) {
        return Err(Stop::Cancel);
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|err| Stop::Fail(err.to_string()))?;
    }
    let partial = partial_path(dest);
    if force {
        let _ = fs::remove_file(dest);
        let _ = fs::remove_file(&partial);
    } else if dest.is_file() {
        let len = dest.metadata().map(|meta| meta.len()).unwrap_or(0);
        if size_hint.map(|size| size == len).unwrap_or(true) {
            on_progress(len);
            return Ok(len);
        }
    }
    let mut start = if partial.is_file() {
        partial.metadata().map(|meta| meta.len()).unwrap_or(0)
    } else {
        0
    };
    if size_hint.is_some_and(|size| start > size) {
        let _ = fs::remove_file(&partial);
        start = 0;
    }
    let (status, length, reader) = open_download(agent, url, token, start)?;
    if status == 416 {
        if start > 0 && size_hint.map(|size| start >= size).unwrap_or(true) {
            fs::rename(&partial, dest).map_err(|err| Stop::Fail(err.to_string()))?;
            on_progress(start);
            return Ok(start);
        }
        let _ = fs::remove_file(&partial);
        let (status, length, reader) = open_download(agent, url, token, 0)?;
        return write_body(
            url,
            dest,
            &partial,
            status,
            length,
            reader,
            0,
            false,
            cancel,
            on_progress,
        );
    }
    let append = status == 206 && start > 0;
    if !append {
        start = 0;
    }
    write_body(
        url,
        dest,
        &partial,
        status,
        length,
        reader,
        start,
        append,
        cancel,
        on_progress,
    )
}

fn open_download(
    agent: &Agent,
    url: &str,
    token: Option<&str>,
    start: u64,
) -> Result<(u16, Option<u64>, Box<dyn Read + Send>), Stop> {
    let mut request = agent.get(url).header("accept", "*/*");
    if let Some(token) = token {
        request = request.header("authorization", &format!("Bearer {token}"));
    }
    if start > 0 {
        request = request.header("range", &format!("bytes={start}-"));
    }
    let response = request
        .call()
        .map_err(|err| Stop::Fail(format!("GET {url} failed: {err}")))?;
    let status = response.status().as_u16();
    if status == 416 {
        return Ok((status, None, Box::new(std::io::empty())));
    }
    if !(200..300).contains(&status) {
        let mut text = String::new();
        let mut response = response;
        let _ = response
            .body_mut()
            .as_reader()
            .take(400)
            .read_to_string(&mut text);
        let hint = if status == 401 || status == 403 {
            " Add a HuggingFace or Civitai token in Settings if this repository is gated."
        } else {
            ""
        };
        return Err(Stop::Fail(format!(
            "GET {url} returned {status}: {}{hint}",
            snippet(&text)
        )));
    }
    let headers = response.headers().clone();
    let length = if status == 206 {
        content_range_total(headers.get("content-range").and_then(|v| v.to_str().ok()))
    } else {
        headers
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|text| text.parse().ok())
    };
    let reader = response.into_body().into_reader();
    Ok((status, length, Box::new(reader)))
}

fn write_body(
    url: &str,
    dest: &Path,
    partial: &Path,
    status: u16,
    length: Option<u64>,
    mut reader: Box<dyn Read + Send>,
    start: u64,
    append: bool,
    cancel: &AtomicBool,
    on_progress: &mut dyn FnMut(u64),
) -> Result<u64, Stop> {
    if !(status == 200 || status == 206) {
        return Err(Stop::Fail(format!("GET {url} returned {status}")));
    }
    let mut file = if append {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(partial)
            .map_err(|err| Stop::Fail(err.to_string()))?
    } else {
        File::create(partial).map_err(|err| Stop::Fail(err.to_string()))?
    };
    let mut got = start;
    let mut buf = [0u8; 64 * 1024];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(Stop::Cancel);
        }
        let n = reader
            .read(&mut buf)
            .map_err(|err| Stop::Fail(format!("reading {url}: {err}")))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|err| Stop::Fail(err.to_string()))?;
        got += n as u64;
        on_progress(got);
    }
    file.flush().ok();
    drop(file);
    if let Some(total) = length {
        if got < total {
            return Err(Stop::Fail(format!(
                "GET {url} ended at {got} bytes before the published {total}"
            )));
        }
    }
    fs::rename(partial, dest).map_err(|err| Stop::Fail(err.to_string()))?;
    Ok(got)
}

fn content_range_total(header: Option<&str>) -> Option<u64> {
    let header = header?;
    let total = header.split('/').nth(1)?.trim();
    if total == "*" {
        None
    } else {
        total.parse().ok()
    }
}

fn partial_path(dest: &Path) -> PathBuf {
    PathBuf::from(format!("{}.partial", dest.display()))
}

fn plan_version(
    spec: &Value,
    list: &mut dyn FnMut(&RepoQuery) -> Result<Vec<RemoteEntry>, String>,
    resolve: &dyn Fn(&str) -> PathBuf,
) -> Result<Plan, String> {
    let mut repos = Vec::new();
    if let Some(bundles) = spec
        .get("bundle_repos")
        .and_then(|v| v.as_array())
        .filter(|rows| !rows.is_empty())
    {
        repos.extend(bundles.iter().cloned());
    } else if spec.get("repo_id").and_then(|v| v.as_str()).is_some()
        || spec.get("url").and_then(|v| v.as_str()).is_some()
        || spec.get("files").and_then(|v| v.as_array()).is_some()
    {
        repos.push(spec.clone());
    }
    if repos.is_empty() {
        return Err(
            "no GGUF download URL is published for this model. Re-download when a bf16, q8_0, or q4_k package is listed"
                .into(),
        );
    }
    let mut files = Vec::new();
    for repo in &repos {
        append_files(&mut files, plan_repo(repo, spec, list, resolve)?)?;
    }
    append_files(&mut files, plan_components(spec, list, resolve)?)?;
    if !files.iter().any(|file| is_gguf(&file.dest)) {
        return Err("no GGUF file is published for this model".into());
    }
    Ok(Plan::Files(files))
}

fn append_files(files: &mut Vec<PlannedFile>, extra: Vec<PlannedFile>) -> Result<(), String> {
    for file in extra {
        if let Some(existing) = files.iter().find(|row| row.dest == file.dest) {
            if existing.url != file.url {
                return Err(format!(
                    "two published files would download to {}",
                    file.dest
                ));
            }
            continue;
        }
        files.push(file);
    }
    Ok(())
}

fn plan_components(
    spec: &Value,
    list: &mut dyn FnMut(&RepoQuery) -> Result<Vec<RemoteEntry>, String>,
    resolve: &dyn Fn(&str) -> PathBuf,
) -> Result<Vec<PlannedFile>, String> {
    let Some(components) = spec.get("components") else {
        return Ok(Vec::new());
    };
    let rows = components
        .as_array()
        .ok_or_else(|| "components is not a list".to_string())?;
    if rows.is_empty() {
        return Err("components is empty".into());
    }
    let mut files = Vec::new();
    for component in rows {
        if !component.is_object() {
            return Err("component entry is not an object".into());
        }
        append_files(&mut files, plan_component(component, spec, list, resolve)?)?;
    }
    Ok(files)
}

fn plan_component(
    component: &Value,
    parent: &Value,
    list: &mut dyn FnMut(&RepoQuery) -> Result<Vec<RemoteEntry>, String>,
    resolve: &dyn Fn(&str) -> PathBuf,
) -> Result<Vec<PlannedFile>, String> {
    let local = parent
        .get("local_path")
        .and_then(|v| v.as_str())
        .filter(|path| !path.is_empty())
        .ok_or_else(|| "version has no local_path".to_string())?;
    let root = resolve(local);
    let repo_id = component
        .get("repo_id")
        .and_then(|v| v.as_str())
        .filter(|id| !id.is_empty())
        .ok_or_else(|| "component has no repo_id".to_string())?;
    let source = component
        .get("source")
        .and_then(|v| v.as_str())
        .or_else(|| parent.get("source").and_then(|v| v.as_str()))
        .unwrap_or("huggingface");
    let revision = revision_of(component, parent, source);
    let patterns = patterns_of(component);
    if patterns.is_empty() {
        return Err("component allow_patterns is empty".into());
    }
    let entries = list(&RepoQuery {
        source: source.into(),
        repo_id: repo_id.into(),
        revision: revision.clone(),
    })?;
    let selected = select_component_files(&entries, &patterns)?;
    let mut files = Vec::new();
    for entry in selected {
        let url = resolve_file_url(source, repo_id, &revision, &entry.path)?;
        files.push(PlannedFile {
            url,
            dest: root.join(&entry.path).display().to_string(),
            size: entry.size,
        });
    }
    Ok(files)
}

fn plan_repo(
    repo: &Value,
    parent: &Value,
    list: &mut dyn FnMut(&RepoQuery) -> Result<Vec<RemoteEntry>, String>,
    resolve: &dyn Fn(&str) -> PathBuf,
) -> Result<Vec<PlannedFile>, String> {
    let local = repo
        .get("local_path")
        .and_then(|v| v.as_str())
        .or_else(|| parent.get("local_path").and_then(|v| v.as_str()))
        .filter(|path| !path.is_empty())
        .ok_or_else(|| "version has no local_path".to_string())?;
    let root = resolve(local);
    if let Some(url) = repo
        .get("url")
        .and_then(|v| v.as_str())
        .filter(|url| !url.is_empty())
    {
        if !url_path_is_gguf(url) {
            return Err(format!("download URL is not a GGUF file: {url}"));
        }
        let name = file_name_from_url(url);
        return Ok(vec![PlannedFile {
            url: url.into(),
            dest: root.join(name).display().to_string(),
            size: repo.get("size_bytes").and_then(|v| v.as_u64()),
        }]);
    }
    if let Some(listed) = repo.get("files").and_then(|v| v.as_array()) {
        let mut files = Vec::new();
        for item in listed {
            let path = item
                .get("path")
                .and_then(|v| v.as_str())
                .or_else(|| item.get("filename").and_then(|v| v.as_str()))
                .unwrap_or("");
            if path.is_empty() || path.contains("..") {
                return Err("registry file path is empty".into());
            }
            let url = if let Some(url) = item
                .get("url")
                .and_then(|v| v.as_str())
                .filter(|url| !url.is_empty())
            {
                url.to_string()
            } else {
                let source = repo
                    .get("source")
                    .and_then(|v| v.as_str())
                    .or_else(|| parent.get("source").and_then(|v| v.as_str()))
                    .unwrap_or("huggingface");
                let repo_id = repo
                    .get("repo_id")
                    .and_then(|v| v.as_str())
                    .or_else(|| parent.get("repo_id").and_then(|v| v.as_str()))
                    .ok_or_else(|| "registry file has no url and no repo_id".to_string())?;
                let revision = revision_of(repo, parent, source);
                resolve_file_url(source, repo_id, &revision, path)?
            };
            if !path.to_ascii_lowercase().ends_with(".gguf") && !url_path_is_gguf(&url) {
                return Err(format!("registry file is not GGUF: {path}"));
            }
            files.push(PlannedFile {
                url,
                dest: root.join(path).display().to_string(),
                size: item.get("size").and_then(|v| v.as_u64()),
            });
        }
        return Ok(files);
    }
    let repo_id = repo
        .get("repo_id")
        .and_then(|v| v.as_str())
        .filter(|id| !id.is_empty())
        .ok_or_else(|| "no GGUF download URL is published for this model".to_string())?;
    let source = repo
        .get("source")
        .and_then(|v| v.as_str())
        .or_else(|| parent.get("source").and_then(|v| v.as_str()))
        .unwrap_or("huggingface");
    let revision = revision_of(repo, parent, source);
    let patterns = patterns_of(repo);
    let entries = list(&RepoQuery {
        source: source.into(),
        repo_id: repo_id.into(),
        revision: revision.clone(),
    })?;
    let selected = select_weight_files(&entries, &patterns)?;
    let mut files = Vec::new();
    for entry in selected {
        let url = resolve_file_url(source, repo_id, &revision, &entry.path)?;
        files.push(PlannedFile {
            url,
            dest: root.join(&entry.path).display().to_string(),
            size: entry.size,
        });
    }
    Ok(files)
}

fn patterns_of(repo: &Value) -> Vec<String> {
    repo.get("allow_patterns")
        .and_then(|v| v.as_array())
        .map(|rows| {
            rows.iter()
                .filter_map(|row| row.as_str().map(|text| text.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

fn revision_of(repo: &Value, parent: &Value, source: &str) -> String {
    repo.get("revision")
        .and_then(|v| v.as_str())
        .or_else(|| parent.get("revision").and_then(|v| v.as_str()))
        .filter(|rev| !rev.is_empty())
        .unwrap_or_else(|| default_revision(source))
        .to_string()
}

fn default_revision(source: &str) -> &str {
    if source == "modelscope" {
        "master"
    } else {
        "main"
    }
}

fn select_weight_files(
    entries: &[RemoteEntry],
    patterns: &[String],
) -> Result<Vec<RemoteEntry>, String> {
    let matched: Vec<&RemoteEntry> = entries
        .iter()
        .filter(|entry| {
            patterns.is_empty()
                || patterns
                    .iter()
                    .any(|pattern| glob_match(pattern, &entry.path))
        })
        .collect();
    if !patterns.is_empty() && matched.is_empty() {
        return Err("allow_patterns matched no published files".into());
    }
    let gguf: Vec<RemoteEntry> = matched
        .iter()
        .filter(|entry| is_gguf(&entry.path))
        .map(|entry| RemoteEntry {
            path: entry.path.clone(),
            size: entry.size,
        })
        .collect();
    if gguf.is_empty() {
        return Err(
            "no GGUF file is published. This runtime downloads GGUF only and does not fetch safetensors"
                .into(),
        );
    }
    let mut out = gguf;
    for entry in &matched {
        if is_sidecar(&entry.path) {
            out.push(RemoteEntry {
                path: entry.path.clone(),
                size: entry.size,
            });
        }
    }
    Ok(out)
}

fn pick_lora_file(entries: &[RemoteEntry]) -> Result<String, String> {
    if let Some(file) = entries.iter().find(|entry| is_gguf(&entry.path)) {
        return Ok(file.path.clone());
    }
    if let Some(file) = entries
        .iter()
        .find(|entry| entry.path.to_ascii_lowercase().ends_with(".safetensors"))
    {
        return Ok(file.path.clone());
    }
    Err("no downloadable LoRA file is published".into())
}

fn select_component_files(
    entries: &[RemoteEntry],
    patterns: &[String],
) -> Result<Vec<RemoteEntry>, String> {
    if patterns.is_empty() {
        return Err("component allow_patterns is empty".into());
    }
    let mut out: Vec<RemoteEntry> = Vec::new();
    for pattern in patterns {
        let mut matched = false;
        for entry in entries {
            if !glob_match(pattern, &entry.path) {
                continue;
            }
            if entry.path.contains("..") || entry.path.starts_with('/') {
                return Err(format!("component path escapes the bundle: {}", entry.path));
            }
            if !is_component_file(&entry.path) {
                continue;
            }
            matched = true;
            if out.iter().any(|row| row.path == entry.path) {
                continue;
            }
            out.push(RemoteEntry {
                path: entry.path.clone(),
                size: entry.size,
            });
        }
        if !matched {
            return Err(format!(
                "component pattern {pattern} matched no published file"
            ));
        }
    }
    Ok(out)
}

fn is_component_file(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.ends_with(".gguf")
        || lower.ends_with(".safetensors")
        || lower.ends_with(".json")
        || lower.ends_with(".txt")
        || lower.ends_with(".jinja")
}

fn is_gguf(path: &str) -> bool {
    path.to_ascii_lowercase().ends_with(".gguf")
}

fn is_sidecar(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.ends_with(".json")
        || lower.ends_with(".txt")
        || lower.ends_with(".md")
        || lower.ends_with(".jinja")
}

fn url_path_is_gguf(url: &str) -> bool {
    let path = url.split('?').next().unwrap_or(url);
    is_gguf(path)
}

fn list_url(source: &str, repo_id: &str, revision: &str) -> Result<String, String> {
    match source {
        "huggingface" => Ok(format!(
            "https://huggingface.co/api/models/{}/tree/{}?recursive=1",
            encode_repo_path(repo_id),
            encode_component(revision)
        )),
        "modelscope" => Ok(format!(
            "https://www.modelscope.cn/api/v1/models/{}/repo/files?Revision={}&Recursive=True",
            encode_repo_path(repo_id),
            encode_component(revision)
        )),
        other => Err(format!("{other} has no HTTP file listing")),
    }
}

fn resolve_file_url(
    source: &str,
    repo_id: &str,
    revision: &str,
    path: &str,
) -> Result<String, String> {
    let repo = encode_repo_path(repo_id);
    let rev = encode_component(revision);
    let file = encode_repo_path(path);
    match source {
        "huggingface" => Ok(format!(
            "https://huggingface.co/{repo}/resolve/{rev}/{file}"
        )),
        "modelscope" => Ok(format!(
            "https://www.modelscope.cn/models/{repo}/resolve/{rev}/{file}"
        )),
        other => Err(format!("{other} has no HTTP resolve URL")),
    }
}

fn file_name_from_url(url: &str) -> String {
    let path = url.split('?').next().unwrap_or(url);
    path.rsplit('/')
        .find(|part| !part.is_empty())
        .unwrap_or("download.gguf")
        .to_string()
}

fn sanitize_name(name: &str) -> String {
    let name = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | ' ') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    out.trim_matches('.').trim().to_string()
}

fn encode_repo_path(path: &str) -> String {
    path.split('/')
        .map(encode_component)
        .collect::<Vec<_>>()
        .join("/")
}

fn encode_component(text: &str) -> String {
    let mut out = String::new();
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn glob_match(pattern: &str, path: &str) -> bool {
    fn rec(pat: &[&str], path: &[&str]) -> bool {
        if pat.is_empty() {
            return path.is_empty();
        }
        if pat[0] == "**" {
            if rec(&pat[1..], path) {
                return true;
            }
            return !path.is_empty() && rec(pat, &path[1..]);
        }
        if path.is_empty() {
            return false;
        }
        segment_match(pat[0], path[0]) && rec(&pat[1..], &path[1..])
    }
    let pat: Vec<&str> = pattern.split('/').filter(|part| !part.is_empty()).collect();
    let path: Vec<&str> = path.split('/').filter(|part| !part.is_empty()).collect();
    rec(&pat, &path)
}

fn segment_match(pattern: &str, text: &str) -> bool {
    fn rec(pattern: &[u8], text: &[u8]) -> bool {
        if pattern.is_empty() {
            return text.is_empty();
        }
        if pattern[0] == b'*' {
            if rec(&pattern[1..], text) {
                return true;
            }
            return !text.is_empty() && rec(pattern, &text[1..]);
        }
        if text.is_empty() {
            return false;
        }
        (pattern[0] == b'?' || pattern[0] == text[0]) && rec(&pattern[1..], &text[1..])
    }
    rec(pattern.as_bytes(), text.as_bytes())
}

fn parse_hf_tree(value: &Value) -> Result<Vec<RemoteEntry>, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| "huggingface tree was not a file list".to_string())?;
    let mut out = Vec::new();
    for item in rows {
        let path = item.get("path").and_then(|v| v.as_str()).unwrap_or("");
        if path.is_empty() {
            continue;
        }
        let kind = item.get("type").and_then(|v| v.as_str()).unwrap_or("file");
        if kind != "file" {
            continue;
        }
        let size = item
            .get("size")
            .and_then(|v| v.as_u64())
            .or_else(|| item.pointer("/lfs/size").and_then(|v| v.as_u64()));
        out.push(RemoteEntry {
            path: path.into(),
            size,
        });
    }
    Ok(out)
}

fn parse_ms_files(value: &Value) -> Result<Vec<RemoteEntry>, String> {
    let rows = value
        .pointer("/Data/Files")
        .or_else(|| value.pointer("/data/Files"))
        .or_else(|| value.pointer("/Data/files"))
        .and_then(|v| v.as_array())
        .ok_or_else(|| "modelscope file list was not understood".to_string())?;
    let mut out = Vec::new();
    for item in rows {
        let kind = item
            .get("Type")
            .or_else(|| item.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("blob");
        if matches!(kind, "tree" | "dir" | "directory") {
            continue;
        }
        let path = item
            .get("Path")
            .or_else(|| item.get("path"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if path.is_empty() {
            continue;
        }
        let size = item
            .get("Size")
            .or_else(|| item.get("size"))
            .and_then(|v| v.as_u64());
        out.push(RemoteEntry {
            path: path.into(),
            size,
        });
    }
    Ok(out)
}

fn parse_hf_search(value: &Value, nsfw: bool) -> Result<Vec<Value>, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| "huggingface search was not a model list".to_string())?;
    let mut items = Vec::new();
    for row in rows {
        let repo = row.get("id").and_then(|v| v.as_str()).unwrap_or("");
        if repo.is_empty() {
            continue;
        }
        let tags = row
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|tags| {
                tags.iter()
                    .filter_map(|tag| tag.as_str())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if !nsfw && tags.iter().any(|tag| tag.eq_ignore_ascii_case("nsfw")) {
            continue;
        }
        let filename = row
            .get("siblings")
            .and_then(|v| v.as_array())
            .and_then(|files| {
                files.iter().find_map(|file| {
                    let name = file.get("rfilename").and_then(|v| v.as_str())?;
                    is_gguf(name).then_some(name)
                })
            })
            .or_else(|| {
                row.get("siblings")
                    .and_then(|v| v.as_array())
                    .and_then(|files| {
                        files.iter().find_map(|file| {
                            let name = file.get("rfilename").and_then(|v| v.as_str())?;
                            name.to_ascii_lowercase()
                                .ends_with(".safetensors")
                                .then_some(name)
                        })
                    })
            })
            .unwrap_or("");
        items.push(json!({
            "id": format!("hf:{repo}"),
            "source": "huggingface",
            "name": repo.rsplit('/').next().unwrap_or(repo),
            "repo_id": repo,
            "filename": filename,
            "downloads": row.get("downloads").cloned().unwrap_or(json!(0)),
            "likes": row.get("likes").cloned().unwrap_or(json!(0)),
            "tags": tags,
        }));
    }
    Ok(items)
}

fn parse_ms_search(value: &Value) -> Result<Vec<Value>, String> {
    if value.get("Success").and_then(|v| v.as_bool()) == Some(false) {
        let message = value
            .get("Message")
            .and_then(|v| v.as_str())
            .unwrap_or("modelscope search failed");
        return Err(message.into());
    }
    let rows = value
        .pointer("/Data/Model/Models")
        .or_else(|| value.pointer("/Data/Models"))
        .and_then(|v| v.as_array())
        .ok_or_else(|| "modelscope search was not understood".to_string())?;
    let mut items = Vec::new();
    for row in rows {
        let path = row
            .get("Path")
            .or_else(|| row.get("path"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let model_name = row.get("Name").and_then(|v| v.as_str()).unwrap_or("");
        let repo = if path.contains('/') || model_name.is_empty() {
            path.to_string()
        } else {
            format!("{path}/{model_name}")
        };
        if repo.is_empty() {
            continue;
        }
        let name = row
            .get("ChineseName")
            .and_then(|v| v.as_str())
            .filter(|name| !name.is_empty())
            .unwrap_or(if model_name.is_empty() {
                repo.as_str()
            } else {
                model_name
            });
        items.push(json!({
            "id": format!("ms:{repo}"),
            "source": "modelscope",
            "name": name,
            "description": row.get("Description").cloned().unwrap_or(json!("")),
            "repo_id": repo,
            "downloads": row.get("Downloads").cloned().unwrap_or(json!(0)),
            "likes": row.get("Stars").cloned().unwrap_or(json!(0)),
        }));
    }
    Ok(items)
}

fn parse_civitai_search(value: &Value, nsfw: bool) -> Result<Vec<Value>, String> {
    let rows = value
        .get("items")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "civitai search was not a model list".to_string())?;
    let mut items = Vec::new();
    for row in rows {
        if !nsfw && row.get("nsfw").and_then(|v| v.as_bool()).unwrap_or(false) {
            continue;
        }
        let id = row.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
        if id == 0 {
            continue;
        }
        let versions = row
            .get("modelVersions")
            .and_then(|v| v.as_array())
            .map(|versions| {
                versions
                    .iter()
                    .map(|version| {
                        json!({
                            "id": version.get("id").cloned().unwrap_or(json!(0)),
                            "name": version.get("name").cloned().unwrap_or(json!("")),
                            "base_model": version.get("baseModel").cloned().unwrap_or(json!("")),
                            "files": version.get("files").and_then(|v| v.as_array()).map(|files| {
                                files.iter().map(|file| json!({
                                    "name": file.get("name").cloned().unwrap_or(json!("")),
                                    "download_url": file.get("downloadUrl").cloned().unwrap_or(json!("")),
                                    "primary": file.get("primary").cloned().unwrap_or(json!(false)),
                                })).collect::<Vec<_>>()
                            }).unwrap_or_default(),
                            "images": version.get("images").and_then(|v| v.as_array()).map(|images| {
                                images.iter().filter_map(|image| image.get("url").map(|url| json!({ "url": url }))).collect::<Vec<_>>()
                            }).unwrap_or_default(),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let first = versions.first();
        let preview = first
            .and_then(|version| version.pointer("/images/0/url"))
            .cloned()
            .unwrap_or(Value::Null);
        items.push(json!({
            "id": format!("civitai:{id}"),
            "source": "civitai",
            "name": row.get("name").cloned().unwrap_or(json!("")),
            "description": row.get("description").cloned().unwrap_or(json!("")),
            "preview_url": preview,
            "downloads": row.get("stats").and_then(|v| v.get("downloadCount")).cloned().unwrap_or(json!(0)),
            "likes": row.get("stats").and_then(|v| v.get("thumbsUpCount")).cloned().unwrap_or(json!(0)),
            "nsfw": row.get("nsfw").cloned().unwrap_or(json!(false)),
            "creator": row.pointer("/creator/username").cloned().unwrap_or(json!("")),
            "civitai_model_id": id,
            "civitai_version_id": first.and_then(|version| version.get("id")).cloned().unwrap_or(json!(0)),
            "versions": versions,
        }));
    }
    Ok(items)
}

fn read_hub_auth(path: &Path) -> HubAuth {
    let value = fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .unwrap_or_else(|| json!({}));
    let setting = |key: &str| {
        value
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string()
    };
    let env = |key: &str| std::env::var(key).unwrap_or_default();
    let huggingface = first_nonempty(&[
        setting("huggingface_token"),
        env("DANQING_HF_TOKEN"),
        env("HF_TOKEN"),
        env("HUGGING_FACE_HUB_TOKEN"),
    ]);
    let civitai = first_nonempty(&[
        setting("civitai_token"),
        env("CIVITAI_TOKEN"),
        env("DANQING_CIVITAI_TOKEN"),
    ]);
    HubAuth {
        huggingface,
        civitai,
        nsfw: value
            .get("nsfw_enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    }
}

fn first_nonempty(values: &[String]) -> String {
    values
        .iter()
        .find(|value| !value.trim().is_empty())
        .cloned()
        .unwrap_or_default()
}

fn token_for(url: &str, auth: &HubAuth) -> Option<String> {
    if url.contains("huggingface.co") && !auth.huggingface.is_empty() {
        Some(auth.huggingface.clone())
    } else if url.contains("civitai.com") && !auth.civitai.is_empty() {
        Some(auth.civitai.clone())
    } else {
        None
    }
}

fn token_header<'a>(hub: &str, auth: &'a HubAuth) -> Option<&'a str> {
    match hub {
        "huggingface" if !auth.huggingface.is_empty() => Some(auth.huggingface.as_str()),
        "civitai" if !auth.civitai.is_empty() => Some(auth.civitai.as_str()),
        _ => None,
    }
}

fn version_lists_components(spec: &Value) -> bool {
    spec.get("components")
        .and_then(|v| v.as_array())
        .is_some_and(|rows| !rows.is_empty())
}

fn write_bundle_manifest(root: &Path, quant: &str, files: &[PlannedFile]) -> Result<(), String> {
    let body = bundle_manifest_value(quant, root, files)?;
    fs::create_dir_all(root).map_err(|err| err.to_string())?;
    let mut bytes = serde_json::to_vec_pretty(&body).map_err(|err| err.to_string())?;
    bytes.push(b'\n');
    fs::write(root.join("bundle.manifest.json"), bytes).map_err(|err| err.to_string())
}

fn bundle_manifest_value(quant: &str, root: &Path, files: &[PlannedFile]) -> Result<Value, String> {
    if !matches!(quant, "bf16" | "q8_0" | "q4_k") {
        return Err(format!("version {quant} is not bf16, q8_0, or q4_k"));
    }
    let mut transformer = Vec::new();
    let mut text_encoder = Vec::new();
    let mut vae = Vec::new();
    for file in files {
        let rel = rel_under(root, &file.dest)?;
        let Some(role) = manifest_role(&rel) else {
            continue;
        };
        let slot = match role {
            "transformer" => &mut transformer,
            "text_encoder" => &mut text_encoder,
            "vae" => &mut vae,
            _ => continue,
        };
        if !slot.iter().any(|path| path == &rel) {
            slot.push(rel);
        }
    }
    transformer.sort();
    text_encoder.sort();
    vae.sort();
    if transformer.is_empty() {
        return Err("bundle manifest has no transformer file".into());
    }
    if text_encoder.is_empty() {
        return Err("bundle manifest has no text_encoder file".into());
    }
    if vae.is_empty() {
        return Err("bundle manifest has no vae file".into());
    }
    Ok(json!({
        "format": "gguf",
        "quant": quant,
        "files": {
            "transformer": role_value(transformer),
            "text_encoder": role_value(text_encoder),
            "vae": role_value(vae),
        }
    }))
}

fn role_value(paths: Vec<String>) -> Value {
    if paths.len() == 1 {
        json!(paths[0])
    } else {
        json!(paths)
    }
}

fn rel_under(root: &Path, dest: &str) -> Result<String, String> {
    let rel = Path::new(dest)
        .strip_prefix(root)
        .map_err(|_| format!("{} is outside {}", dest, root.display()))?;
    let text = rel
        .components()
        .map(|part| part.as_os_str().to_string_lossy().to_string())
        .collect::<Vec<_>>()
        .join("/");
    if text.is_empty() || text.split('/').any(|part| part == "..") {
        return Err(format!("manifest path escapes the bundle: {text}"));
    }
    Ok(text)
}

fn manifest_role(rel: &str) -> Option<&'static str> {
    let lower = rel.to_ascii_lowercase();
    let weight = lower.ends_with(".gguf") || lower.ends_with(".safetensors");
    if !weight {
        return None;
    }
    if rel.starts_with("text_encoder/") {
        Some("text_encoder")
    } else if rel.starts_with("vae/") {
        Some("vae")
    } else if lower.ends_with(".gguf") {
        Some("transformer")
    } else {
        None
    }
}

fn write_receipts(files: &[PlannedFile]) -> Result<(), String> {
    let mut roots: Vec<PathBuf> = Vec::new();
    for file in files {
        if let Some(parent) = Path::new(&file.dest).parent() {
            if !roots.iter().any(|root| root == parent) {
                roots.push(parent.to_path_buf());
            }
        }
    }
    for root in roots {
        let listed: Vec<Value> = files
            .iter()
            .filter(|file| Path::new(&file.dest).parent() == Some(root.as_path()))
            .map(|file| {
                json!({
                    "path": Path::new(&file.dest).file_name().and_then(|name| name.to_str()).unwrap_or(""),
                    "url": file.url,
                    "bytes": file.size,
                })
            })
            .collect();
        let body = json!({
            "format": "gguf-http",
            "files": listed,
            "completed_at": now_rfc3339(),
        });
        fs::create_dir_all(&root).map_err(|err| err.to_string())?;
        fs::write(
            root.join("download-receipt.json"),
            serde_json::to_vec_pretty(&body).map_err(|err| err.to_string())?,
        )
        .map_err(|err| err.to_string())?;
    }
    Ok(())
}

fn load_tasks(path: &Path) -> Vec<Task> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(mut tasks) = serde_json::from_str::<Vec<Task>>(&text) else {
        return Vec::new();
    };
    for task in &mut tasks {
        if matches!(task.status.as_str(), "running" | "pending") {
            task.status = "paused".into();
            task.error_message = "download stopped when the API restarted".into();
            task.speed.clear();
        }
    }
    tasks
}

fn save_tasks(path: &Path, tasks: &[Task]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let body = serde_json::to_vec_pretty(tasks).map_err(|err| err.to_string())?;
    fs::write(path, body).map_err(|err| err.to_string())
}

fn load_lora_items(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|value| value.get("items").and_then(|v| v.as_array()).cloned())
        .unwrap_or_default()
}

fn save_lora_items(path: &Path, items: &[Value]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let body = json!({ "items": items });
    fs::write(
        path,
        serde_json::to_vec_pretty(&body).map_err(|err| err.to_string())?,
    )
    .map_err(|err| err.to_string())
}

fn snippet(text: &str) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    compact.chars().take(180).collect()
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn new_id() -> String {
    let raw = uuid::Uuid::new_v4().simple().to_string();
    format!("dl{}", &raw[..20])
}

fn now_rfc3339() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = (secs / 86400) as i64;
    let tod = secs % 86400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        tod / 3600,
        (tod % 3600) / 60,
        tod % 60
    )
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let year = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    (year, month as u32, day as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::Mutex as StdMutex;

    fn entry(path: &str, size: u64) -> RemoteEntry {
        RemoteEntry {
            path: path.into(),
            size: Some(size),
        }
    }

    #[test]
    fn keeps_gguf_and_refuses_safetensors_only() {
        let mixed = vec![
            entry("model.safetensors", 10),
            entry("model.gguf", 4),
            entry("tokenizer.json", 2),
            entry("preview.png", 1),
        ];
        let files = select_weight_files(&mixed, &[]).unwrap();
        let names: Vec<_> = files.iter().map(|file| file.path.as_str()).collect();
        assert_eq!(names, ["model.gguf", "tokenizer.json"]);

        let weights = vec![entry("model.safetensors", 10)];
        let err = select_weight_files(&weights, &[]).unwrap_err();
        assert!(err.contains("GGUF"), "{err}");
    }

    #[test]
    fn allow_patterns_keep_only_the_published_subtree() {
        let entries = vec![
            entry("vae/config.json", 1),
            entry("vae/diffusion.gguf", 2),
            entry("text/model.gguf", 3),
        ];
        let files = select_weight_files(&entries, &["vae/**".into()]).unwrap();
        let names: Vec<_> = files.iter().map(|file| file.path.as_str()).collect();
        assert_eq!(names, ["vae/diffusion.gguf", "vae/config.json"]);
    }

    #[test]
    fn explicit_url_is_used_and_safetensors_urls_fail() {
        let spec = json!({
            "local_path": "/tmp/danmo-demo",
            "url": "https://example.com/weights/demo.gguf"
        });
        let plan = plan_version(&spec, &mut |_| panic!("listed"), &|path| {
            PathBuf::from(path.to_string())
        })
        .unwrap();
        let Plan::Files(files) = plan;
        assert_eq!(files.len(), 1);
        assert!(files[0].dest.ends_with("demo.gguf"), "{}", files[0].dest);
        assert!(files[0].url.ends_with("demo.gguf"));
        let blocked = json!({
            "local_path": "/tmp/danmo-demo",
            "url": "https://example.com/model.safetensors"
        });
        let err = plan_version(&blocked, &mut |_| panic!("listed"), &|path| {
            PathBuf::from(path.to_string())
        })
        .unwrap_err();
        assert!(err.contains("GGUF"), "{err}");
    }

    fn published_version(model: &str, weight: &str) -> Value {
        let registry = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../default_config/models_registry.json");
        let catalog = danmo_catalog::Catalog::open(&registry).expect("factory registry");
        catalog
            .root()
            .pointer(&format!("/models/{model}/distribution/versions/{weight}"))
            .unwrap_or_else(|| panic!("{model}:{weight} is not in the registry"))
            .clone()
    }

    fn published_listing(query: &RepoQuery) -> Result<Vec<RemoteEntry>, String> {
        assert_eq!(query.source, "modelscope", "{}", query.repo_id);
        assert_eq!(query.revision, "master", "{}", query.repo_id);
        let files: &[&str] = match query.repo_id.as_str() {
            "unsloth/FLUX.2-klein-4B-GGUF" => &[
                "not-selected.gguf",
                "flux-2-klein-4b-BF16.gguf",
                "flux-2-klein-4b-Q8_0.gguf",
                "flux-2-klein-4b-Q4_K_M.gguf",
            ],
            "black-forest-labs/FLUX.2-klein-4B" => &[
                "text_encoder",
                "text_encoder/config.json",
                "text_encoder/generation_config.json",
                "text_encoder/model-00001-of-00002.safetensors",
                "text_encoder/model-00002-of-00002.safetensors",
                "text_encoder/model.safetensors.index.json",
                "tokenizer/added_tokens.json",
                "tokenizer/chat_template.jinja",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
                "transformer/diffusion_pytorch_model.safetensors",
                "flux-2-klein-4b.safetensors",
                "scheduler/scheduler_config.json",
            ],
            "unsloth/Z-Image-GGUF" => &[
                "z-image-BF16.gguf",
                "z-image-Q8_0.gguf",
                "z-image-Q4_K_M.gguf",
                "z-image-Q2_K.gguf",
                "text_encoder",
                "text_encoder/config.json",
                "text_encoder/generation_config.json",
                "text_encoder/model-00001-of-00003.safetensors",
                "text_encoder/model-00002-of-00003.safetensors",
                "text_encoder/model-00003-of-00003.safetensors",
                "text_encoder/model.safetensors.index.json",
                "tokenizer/merges.txt",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
                "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
                "transformer/config.json",
                "scheduler/scheduler_config.json",
            ],
            other => return Err(format!("unexpected repo {other}")),
        };
        Ok(files
            .iter()
            .enumerate()
            .map(|(index, path)| entry(path, (index as u64) + 1))
            .collect())
    }

    fn planned_names(spec: &Value, workspace: &Path) -> Vec<String> {
        let plan = plan_version(spec, &mut published_listing, &|path| {
            resolve_model_path(workspace, path)
        })
        .unwrap();
        let Plan::Files(files) = plan;
        files
            .into_iter()
            .map(|file| {
                let dest = PathBuf::from(&file.dest);
                let rel = dest
                    .strip_prefix(workspace)
                    .unwrap_or_else(|_| panic!("dest left the workspace: {}", file.dest));
                let rel = rel.display().to_string();
                assert!(
                    file.url.contains("https://www.modelscope.cn/models/"),
                    "{}",
                    file.url
                );
                assert!(file.url.contains("/resolve/master/"), "{}", file.url);
                rel
            })
            .collect()
    }

    #[test]
    fn smallest_quant_lands_in_the_download_directory() {
        let spec = published_version("flux2-klein-4b", "q4_k");
        let local = spec["local_path"].as_str().expect("local_path");
        let workspace = PathBuf::from("/tmp/danmo-weight-root");
        let names = planned_names(&spec, &workspace);
        assert!(names
            .iter()
            .any(|name| name == &format!("{local}/flux-2-klein-4b-Q4_K_M.gguf")));
        for suffix in [
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
            "tokenizer/tokenizer.json",
            "tokenizer/chat_template.jinja",
            "tokenizer/merges.txt",
            "vae/diffusion_pytorch_model.safetensors",
        ] {
            assert!(
                names
                    .iter()
                    .any(|name| name == &format!("{local}/{suffix}")),
                "missing {suffix} in {names:?}"
            );
        }
        assert!(
            names
                .iter()
                .all(|name| name.starts_with(&format!("{local}/"))),
            "{names:?}"
        );
        assert!(!names.iter().any(|name| name.contains("transformer/")));
        assert!(!names
            .iter()
            .any(|name| name.ends_with("flux-2-klein-4b.safetensors")));
        assert!(!names.iter().any(|name| name.ends_with("not-selected.gguf")));
        assert_eq!(names.len(), 15, "{names:?}");

        let bf16 = published_version("flux2-klein-4b", "bf16");
        let bf16_names = planned_names(&bf16, &workspace);
        assert!(bf16_names
            .iter()
            .any(|name| name.ends_with("flux-2-klein-4b-BF16.gguf")));
        assert!(bf16_names
            .iter()
            .all(|name| name.starts_with("models/flux2-klein-4b/bf16/")));
        assert!(names
            .iter()
            .all(|name| name.starts_with("models/flux2-klein-4b/q4_k/")));

        let err = plan_version(
            &json!({ "graph_pack": "flux2-chain" }),
            &mut |_| panic!("listed"),
            &|path| PathBuf::from(path.to_string()),
        )
        .unwrap_err();
        assert!(err.contains("no GGUF download"), "{err}");
    }

    #[test]
    fn z_image_quant_copies_companions_into_its_own_directory() {
        let spec = published_version("z-image", "q4_k");
        let workspace = PathBuf::from("/tmp/danmo-weight-root");
        let names = planned_names(&spec, &workspace);
        assert!(names
            .iter()
            .any(|name| name == "models/z-image/q4_k/z-image-Q4_K_M.gguf"));
        assert_eq!(
            names.iter().filter(|name| name.ends_with(".gguf")).count(),
            1,
            "{names:?}"
        );
        for suffix in [
            "text_encoder/model-00001-of-00003.safetensors",
            "text_encoder/model-00003-of-00003.safetensors",
            "tokenizer/vocab.json",
            "vae/diffusion_pytorch_model.safetensors",
        ] {
            assert!(
                names
                    .iter()
                    .any(|name| name == &format!("models/z-image/q4_k/{suffix}")),
                "missing {suffix} in {names:?}"
            );
        }
        assert!(!names.iter().any(|name| name.contains("transformer/")));
        assert!(!names.iter().any(|name| name.contains("Q2_K")));
        assert!(!names
            .iter()
            .any(|name| name.starts_with("models/flux2-klein-4b/")));
        assert_eq!(names.len(), 13, "{names:?}");
    }

    #[test]
    fn component_pattern_must_match_a_published_file() {
        let spec = json!({
            "local_path": "models/demo/q4_k",
            "source": "modelscope",
            "repo_id": "acme/dit",
            "allow_patterns": ["demo-Q4_K_M.gguf"],
            "components": [{
                "source": "modelscope",
                "repo_id": "acme/aux",
                "allow_patterns": ["vae/**"]
            }]
        });
        let err = plan_version(
            &spec,
            &mut |query| {
                if query.repo_id == "acme/dit" {
                    Ok(vec![entry("demo-Q4_K_M.gguf", 1)])
                } else {
                    Ok(vec![entry("text_encoder/model.safetensors", 2)])
                }
            },
            &|path| PathBuf::from(path.to_string()),
        )
        .unwrap_err();
        assert!(err.contains("vae/**"), "{err}");

        let empty = json!({
            "local_path": "models/demo/q4_k",
            "source": "modelscope",
            "repo_id": "acme/dit",
            "allow_patterns": ["demo-Q4_K_M.gguf"],
            "components": [{
                "source": "modelscope",
                "repo_id": "acme/aux",
                "allow_patterns": []
            }]
        });
        let err = plan_version(
            &empty,
            &mut |_| Ok(vec![entry("demo-Q4_K_M.gguf", 1)]),
            &|path| PathBuf::from(path.to_string()),
        )
        .unwrap_err();
        assert!(err.contains("component allow_patterns is empty"), "{err}");
    }

    #[test]
    fn component_install_writes_manifest_from_the_planned_files() {
        let workspace = PathBuf::from("/tmp/danmo-weight-root");
        let spec = published_version("flux2-klein-4b", "q4_k");
        assert!(version_lists_components(&spec));
        let root = resolve_model_path(&workspace, spec["local_path"].as_str().unwrap());
        let Plan::Files(files) = plan_version(&spec, &mut published_listing, &|path| {
            resolve_model_path(&workspace, path)
        })
        .unwrap();
        let body = bundle_manifest_value("q4_k", &root, &files).unwrap();
        assert_eq!(body["format"], "gguf");
        assert_eq!(body["quant"], "q4_k");
        assert_eq!(body["files"]["transformer"], "flux-2-klein-4b-Q4_K_M.gguf");
        assert_eq!(
            body["files"]["vae"],
            "vae/diffusion_pytorch_model.safetensors"
        );
        let text: Vec<_> = body["files"]["text_encoder"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        assert_eq!(
            text,
            [
                "text_encoder/model-00001-of-00002.safetensors",
                "text_encoder/model-00002-of-00002.safetensors"
            ]
        );
        assert!(!body.to_string().contains("tokenizer.json"));
        assert!(!body.to_string().contains("transformer/"));

        let z = published_version("z-image", "bf16");
        let zroot = resolve_model_path(&workspace, z["local_path"].as_str().unwrap());
        let Plan::Files(zfiles) = plan_version(&z, &mut published_listing, &|path| {
            resolve_model_path(&workspace, path)
        })
        .unwrap();
        let zbody = bundle_manifest_value("bf16", &zroot, &zfiles).unwrap();
        assert_eq!(zbody["files"]["transformer"], "z-image-BF16.gguf");
        assert_eq!(
            zbody["files"]["vae"],
            "vae/diffusion_pytorch_model.safetensors"
        );
        assert!(zbody["files"]["text_encoder"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "text_encoder/model-00001-of-00003.safetensors"));
        assert!(!zbody.to_string().contains("models/flux2-klein-4b"));

        let bare = json!({
            "local_path": "models/demo/q4_k",
            "repo_id": "acme/dit",
            "allow_patterns": ["a.gguf"]
        });
        assert!(!version_lists_components(&bare));
        let err = bundle_manifest_value("fp16", &root, &files).unwrap_err();
        assert!(err.contains("fp16"), "{err}");
        let err = bundle_manifest_value("q4_k", &root, &files[..1]).unwrap_err();
        assert!(err.contains("text_encoder"), "{err}");

        let scratch = std::env::temp_dir().join(format!(
            "danmo-manifest-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let files = vec![
            PlannedFile {
                url: "https://example.com/dit.gguf".into(),
                dest: scratch.join("dit.gguf").display().to_string(),
                size: Some(4),
            },
            PlannedFile {
                url: "https://example.com/te".into(),
                dest: scratch
                    .join("text_encoder/model.safetensors")
                    .display()
                    .to_string(),
                size: Some(8),
            },
            PlannedFile {
                url: "https://example.com/tok".into(),
                dest: scratch
                    .join("tokenizer/tokenizer.json")
                    .display()
                    .to_string(),
                size: Some(2),
            },
            PlannedFile {
                url: "https://example.com/vae".into(),
                dest: scratch
                    .join("vae/diffusion_pytorch_model.safetensors")
                    .display()
                    .to_string(),
                size: Some(3),
            },
        ];
        write_bundle_manifest(&scratch, "q8_0", &files).unwrap();
        let text = fs::read_to_string(scratch.join("bundle.manifest.json")).unwrap();
        let written: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(written["quant"], "q8_0");
        assert_eq!(written["files"]["transformer"], "dit.gguf");
        assert_eq!(
            written["files"]["text_encoder"],
            "text_encoder/model.safetensors"
        );
        assert_eq!(
            written["files"]["vae"],
            "vae/diffusion_pytorch_model.safetensors"
        );
        let _ = fs::remove_dir_all(&scratch);
    }

    #[test]
    fn repo_listing_builds_the_resolve_url_from_published_names() {
        let spec = json!({
            "local_path": "/tmp/danmo-demo",
            "repo_id": "acme/demo",
            "source": "huggingface",
            "revision": "main"
        });
        let plan = plan_version(
            &spec,
            &mut |query| {
                assert_eq!(query.repo_id, "acme/demo");
                Ok(vec![
                    entry("demo-q4.gguf", 8),
                    entry("weights.safetensors", 9),
                ])
            },
            &|path| PathBuf::from(path.to_string()),
        )
        .unwrap();
        let Plan::Files(files) = plan;
        assert_eq!(
            files[0].url,
            "https://huggingface.co/acme/demo/resolve/main/demo-q4.gguf"
        );
    }

    #[test]
    fn parses_hub_payloads() {
        let tree = json!([
            {"type": "file", "path": "a.gguf", "size": 3},
            {"type": "directory", "path": "vae"}
        ]);
        assert_eq!(parse_hf_tree(&tree).unwrap().len(), 1);
        let ms = json!({"Data": {"Files": [
            {"Type": "blob", "Path": "a.gguf", "Size": 3},
            {"Type": "tree", "Path": "vae"}
        ]}});
        assert_eq!(parse_ms_files(&ms).unwrap().len(), 1);
        let hf = json!([{
            "id": "acme/lora",
            "downloads": 4,
            "likes": 1,
            "siblings": [{"rfilename": "adapter.gguf"}]
        }]);
        let items = parse_hf_search(&hf, false).unwrap();
        assert_eq!(items[0]["filename"], "adapter.gguf");
        let ms = json!({"Success": true, "Data": {"Model": {"Models": [{
            "Path": "black-forest-labs",
            "Name": "FLUX.2-klein-9B",
            "Downloads": 10,
            "Stars": 2
        }]}}});
        let items = parse_ms_search(&ms).unwrap();
        assert_eq!(items[0]["repo_id"], "black-forest-labs/FLUX.2-klein-9B");
        let civ = json!({"items": [{
            "id": 9,
            "name": "soft",
            "nsfw": false,
            "modelVersions": [{
                "id": 3,
                "name": "v1",
                "baseModel": "Flux",
                "files": [{"name": "a.safetensors", "downloadUrl": "https://civitai.com/api/download/models/3", "primary": true}],
                "images": [{"url": "https://image.example/a.jpg"}]
            }]
        }]});
        let items = parse_civitai_search(&civ, false).unwrap();
        assert_eq!(
            items[0]["versions"][0]["files"][0]["download_url"],
            "https://civitai.com/api/download/models/3"
        );
    }

    struct HitServer {
        port: u16,
        ranges: Arc<StdMutex<Vec<String>>>,
    }

    fn serve_bytes(body: &'static [u8]) -> HitServer {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let ranges = Arc::new(StdMutex::new(Vec::new()));
        let recorded = ranges.clone();
        std::thread::spawn(move || {
            for _ in 0..8 {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let (method_path, headers) = read_request(&mut stream);
                let range = headers
                    .iter()
                    .find(|(key, _)| key.eq_ignore_ascii_case("range"))
                    .map(|(_, value)| value.clone())
                    .unwrap_or_default();
                recorded.lock().unwrap().push(range.clone());
                let (status, payload, extra) = if method_path.contains("/missing") {
                    (404, b"missing".to_vec(), String::new())
                } else if let Some(start) = range
                    .strip_prefix("bytes=")
                    .and_then(|text| text.trim_end_matches('-').parse::<usize>().ok())
                {
                    if start >= body.len() {
                        (416, Vec::new(), String::new())
                    } else {
                        (
                            206,
                            body[start..].to_vec(),
                            format!(
                                "content-range: bytes {start}-{end}/{total}\r\n",
                                end = body.len().saturating_sub(1),
                                total = body.len()
                            ),
                        )
                    }
                } else {
                    (200, body.to_vec(), String::new())
                };
                let header = format!(
                    "HTTP/1.1 {status} OK\r\ncontent-length: {}\r\nconnection: close\r\n{extra}\r\n",
                    payload.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(&payload);
            }
        });
        HitServer { port, ranges }
    }

    fn read_request(stream: &mut TcpStream) -> (String, Vec<(String, String)>) {
        let mut buf = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            if stream.read(&mut byte).ok().filter(|n| *n == 1).is_none() {
                break;
            }
            buf.push(byte[0]);
            if buf.ends_with(b"\r\n\r\n") {
                break;
            }
            if buf.len() > 8192 {
                break;
            }
        }
        let text = String::from_utf8_lossy(&buf);
        let mut lines = text.split("\r\n");
        let start = lines.next().unwrap_or("").to_string();
        let headers = lines
            .filter(|line| !line.is_empty())
            .filter_map(|line| {
                let (key, value) = line.split_once(':')?;
                Some((key.trim().to_string(), value.trim().to_string()))
            })
            .collect();
        (start, headers)
    }

    #[test]
    fn http_client_resumes_with_range_and_installs_the_gguf() {
        let server = serve_bytes(b"abcdefghijklmnopqrstuvwxyz");
        let dir = std::env::temp_dir().join(format!("danmo-dl-{}", uuid::Uuid::new_v4().simple()));
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("demo.gguf");
        fs::write(format!("{}.partial", dest.display()), b"abcdefghij").unwrap();
        let url = format!("http://127.0.0.1:{}/demo.gguf", server.port);
        let agent: Agent = Agent::config_builder().build().into();
        let cancel = AtomicBool::new(false);
        let mut seen = 0u64;
        let got = download_file(
            &agent,
            &url,
            &dest,
            None,
            Some(26),
            false,
            &cancel,
            &mut |n| seen = n,
        )
        .unwrap();
        assert_eq!(got, 26);
        assert_eq!(fs::read(&dest).unwrap(), b"abcdefghijklmnopqrstuvwxyz");
        let ranges = server.ranges.lock().unwrap().clone();
        assert!(
            ranges.iter().any(|range| range.starts_with("bytes=10-")),
            "{ranges:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
