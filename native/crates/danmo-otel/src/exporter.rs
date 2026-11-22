//! `SpanExporter` that batches spans into the local SQLite file.
//!
//! Writes are synchronous on the batch-processor thread with WAL enabled, so a
//! completed task's spans land without blocking the engine. A failed insert
//! returns an SDK error — the SDK logs it — instead of dropping the batch.

use crate::store::SPAN_SCHEMA;
use opentelemetry::trace::Status;
use opentelemetry::Value;
use opentelemetry_sdk::error::{OTelSdkError, OTelSdkResult};
use opentelemetry_sdk::trace::{SpanData, SpanExporter};
use rusqlite::{params, Connection};
use serde_json::{Map, Value as Json};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;
use std::time::UNIX_EPOCH;

/// The task attribute the store denormalizes into `task_id`.
pub const TASK_ID_ATTRIBUTE: &str = "task.id";

#[derive(Debug)]
pub struct SqliteSpanExporter {
    path: PathBuf,
    connection: Mutex<Option<Connection>>,
}

impl SqliteSpanExporter {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            connection: Mutex::new(None),
        }
    }

    /// Open (or reuse) the writer connection. Called lazily on first export so
    /// building a provider never touches the filesystem.
    fn with_connection(
        &self,
        f: impl FnOnce(&Connection) -> rusqlite::Result<()>,
    ) -> OTelSdkResult {
        let mut guard = self
            .connection
            .lock()
            .map_err(|_| OTelSdkError::InternalFailure("span store mutex poisoned".into()))?;
        if guard.is_none() {
            if let Some(parent) = self.path.parent() {
                std::fs::create_dir_all(parent).map_err(|error| {
                    OTelSdkError::InternalFailure(format!(
                        "cannot create span store directory {}: {error}",
                        parent.display()
                    ))
                })?;
            }
            let conn = Connection::open(&self.path).map_err(|error| {
                OTelSdkError::InternalFailure(format!(
                    "cannot open span store {}: {error}",
                    self.path.display()
                ))
            })?;
            conn.busy_timeout(Duration::from_millis(3000))
                .map_err(internal)?;
            conn.execute_batch(&format!("PRAGMA journal_mode=WAL;{SPAN_SCHEMA}"))
                .map_err(internal)?;
            *guard = Some(conn);
        }
        let conn = guard
            .as_ref()
            .ok_or_else(|| OTelSdkError::InternalFailure("span store unavailable".into()))?;
        f(conn).map_err(internal)
    }
}

impl SpanExporter for SqliteSpanExporter {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        if batch.is_empty() {
            return Ok(());
        }
        self.with_connection(|conn| {
            let tx = conn.unchecked_transaction()?;
            {
                let mut stmt = tx.prepare_cached(
                    "INSERT OR REPLACE INTO spans (
                        span_id, trace_id, parent_span_id, task_id, name,
                        start_unix_nano, end_unix_nano, status_code, status_message,
                        attributes, events
                     ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
                )?;
                for span in &batch {
                    let scope = span.instrumentation_scope.name();
                    let mut attributes = attributes_json(&span.attributes);
                    if !scope.is_empty() {
                        if let Json::Object(map) = &mut attributes {
                            map.entry("otel.scope.name")
                                .or_insert_with(|| Json::String(scope.to_string()));
                        }
                    }
                    let task_id = attributes
                        .get(TASK_ID_ATTRIBUTE)
                        .and_then(Json::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let (status_code, status_message) = status_fields(&span.status);
                    stmt.execute(params![
                        span.span_context.span_id().to_string(),
                        span.span_context.trace_id().to_string(),
                        span.parent_span_id.to_string(),
                        task_id,
                        span.name.as_ref(),
                        unix_nanos(span.start_time),
                        unix_nanos(span.end_time),
                        status_code,
                        status_message,
                        attributes.to_string(),
                        events_json(&span.events.events).to_string(),
                    ])?;
                }
            }
            tx.commit()
        })
    }

    fn shutdown_with_timeout(&self, _timeout: Duration) -> OTelSdkResult {
        let mut guard = self
            .connection
            .lock()
            .map_err(|_| OTelSdkError::InternalFailure("span store mutex poisoned".into()))?;
        if let Some(conn) = guard.take() {
            // Flush the WAL into the main database so external readers (the API
            // process, sqlite3 CLI) see completed spans immediately.
            conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
                .map_err(internal)?;
        }
        Ok(())
    }
}

fn internal(error: rusqlite::Error) -> OTelSdkError {
    OTelSdkError::InternalFailure(format!("span store write failed: {error}"))
}

fn unix_nanos(time: std::time::SystemTime) -> i64 {
    time.duration_since(UNIX_EPOCH)
        .map(|since| since.as_nanos().min(i64::MAX as u128) as i64)
        .unwrap_or(0)
}

fn status_fields(status: &Status) -> (&'static str, Option<String>) {
    match status {
        Status::Unset => ("unset", None),
        Status::Ok => ("ok", None),
        Status::Error { description } => ("error", Some(description.to_string())),
    }
}

fn attributes_json(attributes: &[opentelemetry::KeyValue]) -> Json {
    let mut map = Map::new();
    for pair in attributes {
        map.insert(pair.key.as_str().to_string(), value_json(&pair.value));
    }
    Json::Object(map)
}

fn events_json(events: &[opentelemetry::trace::Event]) -> Json {
    let rows: Vec<Json> = events
        .iter()
        .map(|event| {
            let mut row = Map::new();
            row.insert("name".into(), Json::String(event.name.to_string()));
            row.insert(
                "time_unix_nanos".into(),
                Json::from(unix_nanos(event.timestamp)),
            );
            row.insert("attributes".into(), attributes_json(&event.attributes));
            Json::Object(row)
        })
        .collect();
    Json::Array(rows)
}

fn value_json(value: &Value) -> Json {
    match value {
        Value::Bool(flag) => Json::Bool(*flag),
        Value::I64(number) => Json::from(*number),
        Value::F64(number) => Json::from(*number),
        Value::String(text) => Json::String(text.to_string()),
        Value::Array(array) => Json::String(array.to_string()),
        other => Json::String(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{spans_for_task, SPAN_SCHEMA};
    use opentelemetry::trace::{SpanContext, SpanId, SpanKind, TraceFlags, TraceId};
    use opentelemetry::{InstrumentationScope, KeyValue};
    use opentelemetry_sdk::trace::SpanEvents;
    use std::time::SystemTime;

    fn span_data(task_id: &str, name: &str, span_id: u64, parent: u64) -> SpanData {
        let trace_id = TraceId::from_bytes([7; 16]);
        SpanData {
            span_context: SpanContext::new(
                trace_id,
                SpanId::from_bytes(span_id.to_be_bytes()),
                TraceFlags::SAMPLED,
                false,
                Default::default(),
            ),
            parent_span_id: SpanId::from_bytes(parent.to_be_bytes()),
            parent_span_is_remote: false,
            span_kind: SpanKind::Internal,
            name: name.to_string().into(),
            start_time: SystemTime::now(),
            end_time: SystemTime::now() + Duration::from_millis(12),
            attributes: vec![KeyValue::new(TASK_ID_ATTRIBUTE, task_id.to_string())],
            dropped_attributes_count: 0,
            events: SpanEvents::default(),
            links: Default::default(),
            status: Status::Ok,
            instrumentation_scope: InstrumentationScope::builder("danmo-test").build(),
        }
    }

    #[test]
    fn exported_spans_are_queryable_by_task() {
        let dir = std::env::temp_dir().join(format!("danmo-otel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("otel.db");

        let exporter = SqliteSpanExporter::new(db.clone());
        futures::executor::block_on(exporter.export(vec![
            span_data("tsk_root", "engine.execute", 1, 0),
            span_data("tsk_root", "phase.load", 2, 1),
        ]))
        .expect("export");
        exporter.shutdown().expect("shutdown");

        let spans = spans_for_task(&db, "tsk_root").expect("query");
        assert_eq!(spans.len(), 2, "both spans persisted: {spans:?}");
        let names: Vec<&str> = spans.iter().map(|span| span.name.as_str()).collect();
        assert!(names.contains(&"engine.execute"), "{names:?}");
        assert!(names.contains(&"phase.load"), "{names:?}");
        assert!(spans[0].duration_ms >= 0.0);

        // Unknown task is empty, not an error.
        assert!(spans_for_task(&db, "tsk_absent").expect("query").is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_store_is_reported_not_silently_empty() {
        let missing = std::env::temp_dir().join("danmo-otel-does-not-exist/otel.db");
        let error = spans_for_task(&missing, "tsk_x").expect_err("missing store must fail");
        assert!(
            matches!(error, crate::store::TraceSpanQueryError::Missing(_)),
            "{error}"
        );
    }

    #[test]
    fn schema_is_idempotent() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(SPAN_SCHEMA).unwrap();
        conn.execute_batch(SPAN_SCHEMA).unwrap();
    }
}
