//! SQLite schema and read/queries for persisted spans.
//!
//! One table, `spans`, with denormalized `task_id` so the task page can query a
//! whole trace without joining. `CREATE TABLE IF NOT EXISTS` only — no ALTER.

use rusqlite::{params, Connection, OpenFlags};
use serde::Serialize;
use std::path::Path;
use thiserror::Error;

pub const SPAN_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_span_id TEXT,
    task_id TEXT NOT NULL,
    name TEXT NOT NULL,
    start_unix_nano INTEGER NOT NULL,
    end_unix_nano INTEGER NOT NULL,
    status_code TEXT NOT NULL,
    status_message TEXT,
    attributes TEXT NOT NULL,
    events TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spans_task ON spans(task_id, start_unix_nano);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id, start_unix_nano);
CREATE INDEX IF NOT EXISTS idx_spans_end ON spans(end_unix_nano);
";

#[derive(Debug, Error)]
pub enum TraceSpanQueryError {
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    #[error("trace store {0} is missing")]
    Missing(String),
}

/// One exported span as the UI consumes it. Times are absolute wall clock;
/// `duration_ms` is derived so the waterfall does not need to rescale.
#[derive(Debug, Clone, Serialize)]
pub struct TraceSpan {
    pub span_id: String,
    pub trace_id: String,
    pub parent_span_id: Option<String>,
    pub task_id: String,
    pub name: String,
    pub start_unix_nanos: i64,
    pub end_unix_nanos: i64,
    pub start_ms: f64,
    pub duration_ms: f64,
    pub status: String,
    pub status_message: Option<String>,
    pub attributes: serde_json::Value,
    pub events: serde_json::Value,
}

/// Open a reader connection. Read/write is not needed here; a missing file is a
/// loud error rather than an empty result, so the API can say why it is empty.
pub fn open_reader(path: &Path) -> Result<Connection, TraceSpanQueryError> {
    if !path.exists() {
        return Err(TraceSpanQueryError::Missing(path.display().to_string()));
    }
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    conn.busy_timeout(std::time::Duration::from_millis(2000))?;
    Ok(conn)
}

/// All spans for one task, ordered by start time so span[0] is the root.
pub fn spans_for_task(path: &Path, task_id: &str) -> Result<Vec<TraceSpan>, TraceSpanQueryError> {
    let conn = open_reader(path)?;
    let mut stmt = conn.prepare(
        "SELECT span_id, trace_id, parent_span_id, task_id, name, start_unix_nano,
                end_unix_nano, status_code, status_message, attributes, events
         FROM spans WHERE task_id=?1 ORDER BY start_unix_nano ASC, span_id ASC",
    )?;
    let rows = stmt.query_map(params![task_id], read_span)?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(TraceSpanQueryError::from)
}

/// Delete every span belonging to a purged task.
pub fn purge_task_spans(path: &Path, task_id: &str) -> Result<usize, TraceSpanQueryError> {
    if !path.exists() {
        return Ok(0);
    }
    let conn = Connection::open(path)?;
    conn.busy_timeout(std::time::Duration::from_millis(2000))?;
    let deleted = conn.execute("DELETE FROM spans WHERE task_id=?1", params![task_id])?;
    Ok(deleted)
}

/// Drop spans older than `cutoff_unix_nano`.
pub fn purge_older_than(path: &Path, cutoff_unix_nano: i64) -> Result<usize, TraceSpanQueryError> {
    if !path.exists() {
        return Ok(0);
    }
    let conn = Connection::open(path)?;
    conn.busy_timeout(std::time::Duration::from_millis(2000))?;
    let deleted = conn.execute(
        "DELETE FROM spans WHERE end_unix_nano < ?1",
        params![cutoff_unix_nano],
    )?;
    Ok(deleted)
}

fn read_span(row: &rusqlite::Row<'_>) -> rusqlite::Result<TraceSpan> {
    let start: i64 = row.get(5)?;
    let end: i64 = row.get(6)?;
    let attributes: String = row.get(9)?;
    let events: String = row.get(10)?;
    Ok(TraceSpan {
        span_id: row.get(0)?,
        trace_id: row.get(1)?,
        parent_span_id: row.get(2)?,
        task_id: row.get(3)?,
        name: row.get(4)?,
        start_unix_nanos: start,
        end_unix_nanos: end,
        start_ms: start as f64 / 1_000_000.0,
        duration_ms: (end.saturating_sub(start)) as f64 / 1_000_000.0,
        status: row.get(7)?,
        status_message: row.get(8)?,
        attributes: serde_json::from_str(&attributes).unwrap_or(serde_json::Value::Null),
        events: serde_json::from_str(&events).unwrap_or(serde_json::Value::Null),
    })
}
