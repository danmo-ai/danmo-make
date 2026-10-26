//! SQLite task store. Same columns as `V3TaskStore`. `CREATE TABLE IF NOT EXISTS` only — no ALTER.

use rusqlite::{params, Connection};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
}

#[derive(Debug, Clone)]
pub struct TaskRow {
    pub id: String,
    pub kind: String,
    pub model_id: String,
    pub status: String,
    pub priority: i64,
    pub progress: f64,
    pub params: String,
    pub result: Option<String>,
    pub error_message: Option<String>,
}

pub struct TaskStore {
    conn: Connection,
}

impl TaskStore {
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                model_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                priority INTEGER NOT NULL DEFAULT 100,
                progress REAL NOT NULL DEFAULT 0.0,
                params TEXT NOT NULL,
                result TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS task_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL
            );
            ",
        )?;
        Ok(Self { conn })
    }

    pub fn insert_task(
        &self,
        id: &str,
        kind: &str,
        model_id: &str,
        params: &str,
        priority: i64,
    ) -> Result<(), StoreError> {
        self.conn.execute(
            "INSERT INTO tasks (id, kind, model_id, status, priority, progress, params, created_at)
             VALUES (?1, ?2, ?3, 'queued', ?4, 0, ?5, datetime('now'))",
            params![id, kind, model_id, priority, params],
        )?;
        Ok(())
    }

    pub fn update_status(
        &self,
        id: &str,
        status: &str,
        progress: f64,
        result: Option<&str>,
        error_message: Option<&str>,
    ) -> Result<(), StoreError> {
        self.conn.execute(
            "UPDATE tasks SET status=?2, progress=?3, result=COALESCE(?4, result),
             error_message=COALESCE(?5, error_message),
             started_at=CASE WHEN ?2='running' THEN datetime('now') ELSE started_at END,
             completed_at=CASE WHEN ?2 IN ('succeeded','failed','cancelled') THEN datetime('now') ELSE completed_at END
             WHERE id=?1",
            params![id, status, progress, result, error_message],
        )?;
        Ok(())
    }

    pub fn log(&self, task_id: &str, level: &str, message: &str) -> Result<(), StoreError> {
        self.conn.execute(
            "INSERT INTO task_logs (task_id, ts, level, message) VALUES (?1, datetime('now'), ?2, ?3)",
            params![task_id, level, message],
        )?;
        Ok(())
    }

    pub fn get(&self, id: &str) -> Result<Option<TaskRow>, StoreError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, model_id, status, priority, progress, params, result, error_message
             FROM tasks WHERE id=?1",
        )?;
        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(TaskRow {
                id: row.get(0)?,
                kind: row.get(1)?,
                model_id: row.get(2)?,
                status: row.get(3)?,
                priority: row.get(4)?,
                progress: row.get(5)?,
                params: row.get(6)?,
                result: row.get(7)?,
                error_message: row.get(8)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn list_queued(&self, image_first: bool) -> Result<Vec<TaskRow>, StoreError> {
        let order = if image_first {
            "CASE WHEN kind LIKE 'image.%' THEN 0 ELSE 1 END, priority ASC, created_at ASC"
        } else {
            "priority ASC, created_at ASC"
        };
        let sql = format!(
            "SELECT id, kind, model_id, status, priority, progress, params, result, error_message
             FROM tasks WHERE status='queued' ORDER BY {order}"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| {
            Ok(TaskRow {
                id: row.get(0)?,
                kind: row.get(1)?,
                model_id: row.get(2)?,
                status: row.get(3)?,
                priority: row.get(4)?,
                progress: row.get(5)?,
                params: row.get(6)?,
                result: row.get(7)?,
                error_message: row.get(8)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(StoreError::from)
    }
}
