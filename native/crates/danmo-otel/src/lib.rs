//! OpenTelemetry trace collection for engine inference, exported into a local
//! SQLite file (`{workspace}/db/otel.db`).
//!
//! The crate is deliberately free of engine types: callers (server worker, CLI)
//! drive [`TaskTrace`] from their observer callbacks. Failures to persist spans
//! are loud — [`SqliteSpanExporter`] returns an SDK error instead of silently
//! dropping a batch.

mod exporter;
mod store;
mod trace;

pub use exporter::SqliteSpanExporter;
pub use store::{
    purge_older_than, purge_task_spans, spans_for_task, TraceSpan, TraceSpanQueryError,
};
pub use trace::{
    attribute, cli_db_path, trace_id_for_task, trace_id_hex_for_task, TaskTrace, Telemetry,
    TelemetryError, PHASE_SPAN_PREFIX, ROOT_SPAN,
};
