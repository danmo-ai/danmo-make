//! Provider bootstrap and the per-task span tree.
//!
//! The engine only exposes `RunObserver` callbacks (`phase`, `progress`, `log`),
//! so the trace shape is derived here: one `engine.execute` root span per task
//! and one `phase.<name>` child per pipeline phase. Progress steps become span
//! events instead of spans so a 30-step denoise loop does not flood the store.
//!
//! Trace IDs are not random: each task pins the OTel `trace_id` to a 128-bit
//! value derived from `task.id` (`tsk_<uuid>` → UUID octets). Product UI can
//! show a single identifier.

use opentelemetry::trace::{
    Span as _, SpanBuilder, SpanContext, SpanId, SpanKind, Status, TraceContextExt, TraceId,
    Tracer as _, TracerProvider as _,
};
use opentelemetry::{Context, KeyValue};
use opentelemetry_sdk::trace::{IdGenerator, RandomIdGenerator, SdkTracer, SdkTracerProvider};
use opentelemetry_sdk::Resource;
use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use thiserror::Error;

use crate::exporter::SqliteSpanExporter;

/// Root span name for one `Engine::execute` call.
pub const ROOT_SPAN: &str = "engine.execute";
/// Prefix for pipeline phase spans.
pub const PHASE_SPAN_PREFIX: &str = "phase.";

thread_local! {
    /// Next root span's TraceId. Set by [`Telemetry::start_task`] so the SDK's
    /// IdGenerator (which no longer exposes `with_trace_id` on SpanBuilder)
    /// still produces a deterministic id for this task.
    static FORCED_TRACE_ID: Cell<Option<TraceId>> = const { Cell::new(None) };
}

/// Id generator that honours a one-shot forced TraceId for the task root.
#[derive(Debug, Default)]
struct TaskBoundIdGenerator {
    inner: RandomIdGenerator,
}

impl IdGenerator for TaskBoundIdGenerator {
    fn new_trace_id(&self) -> TraceId {
        if let Some(forced) = FORCED_TRACE_ID.with(Cell::take) {
            return forced;
        }
        self.inner.new_trace_id()
    }

    fn new_span_id(&self) -> SpanId {
        self.inner.new_span_id()
    }
}

/// Map a product task id to the OTel TraceId used for its span tree.
///
/// - `tsk_<uuid>` / bare uuid → TraceId from the 16 UUID bytes
/// - anything else → stable FNV-style 128-bit digest (never all-zero)
pub fn trace_id_for_task(task_id: &str) -> TraceId {
    let trimmed = task_id.trim();
    let body = trimmed
        .strip_prefix("tsk_")
        .or_else(|| trimmed.strip_prefix("tsk-"))
        .unwrap_or(trimmed);
    let hex: String = body
        .chars()
        .filter(|ch| ch.is_ascii_hexdigit())
        .flat_map(|ch| ch.to_lowercase())
        .collect();
    if hex.len() == 32 {
        if let Ok(id) = TraceId::from_hex(&hex) {
            if id != TraceId::INVALID {
                return id;
            }
        }
    }
    hash_trace_id(trimmed)
}

fn hash_trace_id(task_id: &str) -> TraceId {
    let mut h1: u64 = 0xcbf29ce484222325;
    let mut h2: u64 = 0x100000001b3;
    for byte in task_id.as_bytes() {
        h1 ^= u64::from(*byte);
        h1 = h1.wrapping_mul(0x100000001b3);
        h2 ^= u64::from(*byte);
        h2 = h2.wrapping_mul(0xcbf29ce484222325);
    }
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&h1.to_be_bytes());
    bytes[8..].copy_from_slice(&h2.to_be_bytes());
    if bytes.iter().all(|b| *b == 0) {
        bytes[15] = 1;
    }
    TraceId::from_bytes(bytes)
}

/// Lowercase 32-char hex form of [`trace_id_for_task`] (matches exported spans).
pub fn trace_id_hex_for_task(task_id: &str) -> String {
    trace_id_for_task(task_id).to_string()
}

#[derive(Debug, Error)]
pub enum TelemetryError {
    #[error("{0}")]
    Message(String),
}

#[derive(Clone)]
pub struct Telemetry {
    inner: Arc<TelemetryInner>,
}

struct TelemetryInner {
    provider: SdkTracerProvider,
    tracer: SdkTracer,
    db: PathBuf,
}

impl std::fmt::Debug for Telemetry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Telemetry")
            .field("db", &self.inner.db)
            .finish()
    }
}

impl Telemetry {
    /// Build a provider that batches completed spans into `db`.
    pub fn init(db: &Path, service_name: &str) -> Result<Self, TelemetryError> {
        if let Some(parent) = db.parent() {
            std::fs::create_dir_all(parent).map_err(|error| {
                TelemetryError::Message(format!(
                    "cannot create trace directory {}: {error}",
                    parent.display()
                ))
            })?;
        }
        let exporter = SqliteSpanExporter::new(db.to_path_buf());
        let resource = Resource::builder()
            .with_service_name(service_name.to_string())
            .with_attribute(KeyValue::new(
                "service.version",
                env!("CARGO_PKG_VERSION").to_string(),
            ))
            .build();
        let provider = SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .with_id_generator(TaskBoundIdGenerator::default())
            .with_resource(resource)
            .build();
        let tracer = provider.tracer("danmo-engine");
        Ok(Self {
            inner: Arc::new(TelemetryInner {
                provider,
                tracer,
                db: db.to_path_buf(),
            }),
        })
    }

    pub fn database_path(&self) -> &Path {
        &self.inner.db
    }

    /// Flush pending spans. Called explicitly on CLI exit paths that bypass
    /// destructors (`std::process::exit`).
    pub fn shutdown(&self) -> Result<(), TelemetryError> {
        self.inner
            .provider
            .shutdown()
            .map_err(|error| TelemetryError::Message(format!("trace shutdown failed: {error}")))
    }

    /// Block until every queued span has been handed to the exporter.
    pub fn flush(&self) -> Result<(), TelemetryError> {
        self.inner
            .provider
            .force_flush()
            .map_err(|error| TelemetryError::Message(format!("trace flush failed: {error}")))
    }

    /// Start the trace for one engine execution.
    ///
    /// `task_id` becomes both the `task.id` attribute and the OTel TraceId
    /// (via [`trace_id_for_task`]), so product and span stores share one id.
    pub fn start_task(&self, task_id: &str, mut attributes: Vec<KeyValue>) -> TaskTrace {
        let has_task_id = attributes.iter().any(|kv| kv.key.as_str() == "task.id");
        if !has_task_id {
            attributes.push(KeyValue::new("task.id", task_id.to_string()));
        }
        FORCED_TRACE_ID.with(|cell| cell.set(Some(trace_id_for_task(task_id))));
        let root = self
            .inner
            .tracer
            .span_builder(ROOT_SPAN)
            .with_kind(SpanKind::Internal)
            .with_attributes(attributes.clone())
            .start(&self.inner.tracer);
        // Clear any unused force in case the sampler dropped the span without
        // consuming a new_trace_id call.
        FORCED_TRACE_ID.with(|cell| {
            cell.set(None);
        });
        let root_context = root.span_context().clone();
        debug_assert_eq!(
            root_context.trace_id(),
            trace_id_for_task(task_id),
            "root TraceId must equal task-derived id"
        );
        TaskTrace {
            tracer: self.inner.tracer.clone(),
            root_context,
            // Children are separate spans; OTel does not inherit attributes, so
            // the denormalized `task.id` has to be re-applied for the task-page
            // query to find a whole trace. Mutex so resolve-time annotate can
            // merge dims after the root has already started.
            span_attributes: Mutex::new(attributes),
            root: Mutex::new(Some(root)),
            phase: Mutex::new(None),
            phase_name: Mutex::new(None),
        }
    }
}

/// One task's span tree. All methods are cheap and never fail; the observer
/// contract has no error channel for telemetry (`fail loud` stays in the log
/// path; a broken exporter surfaces through the SDK's own error reporting).
pub struct TaskTrace {
    tracer: SdkTracer,
    root_context: SpanContext,
    span_attributes: Mutex<Vec<KeyValue>>,
    root: Mutex<Option<opentelemetry_sdk::trace::Span>>,
    phase: Mutex<Option<opentelemetry_sdk::trace::Span>>,
    phase_name: Mutex<Option<String>>,
}

impl std::fmt::Debug for TaskTrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskTrace")
            .field("trace_id", &self.root_context.trace_id().to_string())
            .finish()
    }
}

impl TaskTrace {
    pub fn trace_id(&self) -> String {
        self.root_context.trace_id().to_string()
    }

    /// Merge attributes onto the root span and into the clone set used by
    /// later phase children. Safe to call after resolve when family/dims are known.
    pub fn set_attributes(&self, attributes: Vec<KeyValue>) {
        if attributes.is_empty() {
            return;
        }
        {
            let mut shared = lock(&self.span_attributes);
            for attr in &attributes {
                if let Some(existing) = shared.iter_mut().find(|row| row.key == attr.key) {
                    *existing = attr.clone();
                } else {
                    shared.push(attr.clone());
                }
            }
        }
        if let Some(span) = lock(&self.root).as_mut() {
            for attr in &attributes {
                span.set_attribute(attr.clone());
            }
        }
        if let Some(span) = lock(&self.phase).as_mut() {
            for attr in &attributes {
                span.set_attribute(attr.clone());
            }
        }
    }

    /// Close the previous phase and open `phase.<name>` as a child of the root.
    pub fn phase(&self, name: &str) {
        self.end_phase();
        let span = self.start_child(&format!("{PHASE_SPAN_PREFIX}{name}"));
        *lock(&self.phase) = Some(span);
        *lock(&self.phase_name) = Some(name.to_string());
    }

    /// Name of the open phase, if any. Used by callers that want to avoid
    /// re-opening a phase the engine reports twice.
    pub fn current_phase(&self) -> Option<String> {
        lock(&self.phase_name).clone()
    }

    /// Attach a progress step to the open phase span (or the root when the
    /// engine reports progress before a phase start).
    pub fn progress(&self, phase: &str, step: u32, total: u32) {
        self.progress_with(phase, step, total, Vec::new());
    }

    /// Progress event with extra attributes (loss, sigma, stage, …).
    pub fn progress_with(&self, phase: &str, step: u32, total: u32, mut extra: Vec<KeyValue>) {
        let mut attributes = vec![
            KeyValue::new("phase", phase.to_string()),
            KeyValue::new("progress.step", step as i64),
            KeyValue::new("progress.total", total as i64),
        ];
        attributes.append(&mut extra);
        if let Some(span) = lock(&self.phase).as_mut() {
            span.add_event("progress", attributes);
            return;
        }
        if let Some(span) = lock(&self.root).as_mut() {
            span.add_event("progress", attributes);
        }
    }

    /// Named span event (structured context that is not a freeform log line).
    pub fn add_event(&self, name: &str, attributes: Vec<KeyValue>) {
        if let Some(span) = lock(&self.phase).as_mut() {
            span.add_event(name.to_string(), attributes);
            return;
        }
        if let Some(span) = lock(&self.root).as_mut() {
            span.add_event(name.to_string(), attributes);
        }
    }

    /// Record a log line on the open phase span.
    pub fn log_event(&self, level: &str, message: &str) {
        let attributes = vec![
            KeyValue::new("log.level", level.to_string()),
            KeyValue::new("log.message", message.to_string()),
        ];
        if let Some(span) = lock(&self.phase).as_mut() {
            span.add_event("log", attributes);
            return;
        }
        if let Some(span) = lock(&self.root).as_mut() {
            span.add_event("log", attributes);
        }
    }

    /// Mark the trace and the open phase as failed.
    pub fn fail(&self, message: &str) {
        let status = Status::error(message.to_string());
        if let Some(span) = lock(&self.phase).as_mut() {
            span.set_status(status.clone());
        }
        if let Some(span) = lock(&self.root).as_mut() {
            span.set_status(status);
        }
    }

    /// Mark the trace as cancelled.
    pub fn cancelled(&self) {
        let status = Status::error("cancelled");
        if let Some(span) = lock(&self.phase).as_mut() {
            span.set_status(status.clone());
        }
        if let Some(span) = lock(&self.root).as_mut() {
            span.set_status(status);
        }
    }

    /// Close the phase and root spans. Idempotent.
    pub fn finish(&self) {
        self.end_phase();
        if let Some(mut span) = lock(&self.root).take() {
            span.end();
        }
    }

    fn end_phase(&self) {
        if let Some(mut span) = lock(&self.phase).take() {
            span.end();
        }
        *lock(&self.phase_name) = None;
    }

    fn start_child(&self, name: &str) -> opentelemetry_sdk::trace::Span {
        let parent = parent_context(&self.root_context);
        let attributes = lock(&self.span_attributes).clone();
        let builder = SpanBuilder::from_name(name.to_string())
            .with_kind(SpanKind::Internal)
            .with_attributes(attributes);
        self.tracer.build_with_context(builder, &parent)
    }
}

/// A context carrying the parent span context as a *local* (non-remote) parent,
/// so children share the trace id and point at the root span id.
fn parent_context(parent: &SpanContext) -> Context {
    let local = SpanContext::new(
        parent.trace_id(),
        parent.span_id(),
        parent.trace_flags(),
        false,
        parent.trace_state().clone(),
    );
    Context::new().with_remote_span_context(local)
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Convenience so callers need no direct `opentelemetry` dependency.
pub fn attribute(key: impl Into<String>, value: impl Into<String>) -> KeyValue {
    KeyValue::new(key.into(), value.into())
}

/// Trace store shared with the API: `{workspace}/db/otel.db`.
pub fn cli_db_path(workspace: &Path) -> PathBuf {
    workspace.join("db").join("otel.db")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::spans_for_task;

    fn temp_db(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("danmo-otel-trace-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir.join("otel.db")
    }

    #[test]
    fn phases_nest_under_one_root_span() {
        let db = temp_db("phases");
        let telemetry = Telemetry::init(&db, "danmo-test").expect("telemetry");
        let task_id = "tsk_a";
        let trace = telemetry.start_task(task_id, vec![]);
        trace.phase("load");
        trace.progress("load", 1, 4);
        trace.phase("infer");
        trace.progress("infer", 2, 8);
        trace.finish();
        telemetry.shutdown().expect("shutdown");

        let spans = spans_for_task(&db, task_id).expect("query");
        let root = spans
            .iter()
            .find(|span| span.name == ROOT_SPAN)
            .unwrap_or_else(|| panic!("root span missing: {spans:?}"));
        assert_eq!(root.trace_id, trace_id_hex_for_task(task_id));
        assert!(
            root.parent_span_id.is_none()
                || root.parent_span_id.as_deref() == Some("0000000000000000")
        );
        let phases: Vec<_> = spans
            .iter()
            .filter(|span| span.name.starts_with(PHASE_SPAN_PREFIX))
            .collect();
        assert_eq!(phases.len(), 2, "{spans:?}");
        for phase in phases {
            assert_eq!(phase.trace_id, root.trace_id, "same trace");
            assert_eq!(
                phase.parent_span_id.as_deref(),
                Some(root.span_id.as_str()),
                "phase parented to root"
            );
        }
        assert!(spans_for_task(&db, "tsk_absent").unwrap().is_empty());
    }

    #[test]
    fn uuid_task_id_becomes_trace_id_octets() {
        let task_id = "tsk_91d3d5fd-38ba-423c-a596-3c219cfb766e";
        assert_eq!(
            trace_id_hex_for_task(task_id),
            "91d3d5fd38ba423ca5963c219cfb766e"
        );
        assert_eq!(
            trace_id_hex_for_task("91d3d5fd38ba423ca5963c219cfb766e"),
            "91d3d5fd38ba423ca5963c219cfb766e"
        );
    }

    #[test]
    fn failure_and_cancel_set_error_status() {
        let db = temp_db("status");
        let telemetry = Telemetry::init(&db, "danmo-test").expect("telemetry");
        let failed = telemetry.start_task("tsk_fail", vec![]);
        failed.phase("decode");
        failed.fail("vae decode exploded");
        failed.finish();

        let cancelled = telemetry.start_task("tsk_cancel", vec![]);
        cancelled.phase("infer");
        cancelled.cancelled();
        cancelled.finish();
        telemetry.shutdown().expect("shutdown");

        let failed = spans_for_task(&db, "tsk_fail").expect("query");
        assert!(
            failed.iter().all(|span| span.status == "error"),
            "{failed:?}"
        );
        let cancelled = spans_for_task(&db, "tsk_cancel").expect("query");
        assert!(
            cancelled.iter().all(|span| span.status == "error"),
            "{cancelled:?}"
        );
    }

    #[test]
    fn progress_before_any_phase_lands_on_root() {
        let db = temp_db("early");
        let telemetry = Telemetry::init(&db, "danmo-test").expect("telemetry");
        let trace = telemetry.start_task("tsk_early", vec![]);
        trace.progress("", 1, 4);
        trace.log_event("info", "warmup");
        trace.finish();
        telemetry.shutdown().expect("shutdown");

        let spans = spans_for_task(&db, "tsk_early").expect("query");
        assert_eq!(spans.len(), 1, "{spans:?}");
        let events = spans[0].events.as_array().expect("events array");
        assert_eq!(events.len(), 2, "{events:?}");
    }
}
