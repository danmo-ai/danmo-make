//! SQLite studio store. Same task, asset, and canvas tables as the previous product.
//! `CREATE TABLE IF NOT EXISTS` only — no ALTER.

use rusqlite::{params, Connection};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    #[error("{0}")]
    Message(String),
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
    pub created_at: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LogRow {
    pub id: i64,
    pub message: String,
    pub level: String,
    pub time: String,
}

#[derive(Debug, Clone)]
pub struct AssetRow {
    pub id: String,
    pub kind: String,
    pub mime_type: String,
    pub file_path: String,
    pub thumbnail_path: Option<String>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub duration_seconds: Option<f64>,
    pub source_task_id: Option<String>,
    pub source_action: Option<String>,
    pub metadata: String,
    pub created_at: String,
    pub parent_asset_id: Option<String>,
    pub relation_type: Option<String>,
    pub group_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CanvasRow {
    pub id: String,
    pub media: String,
    pub title: String,
    pub state_json: String,
    pub created_at: String,
    pub updated_at: String,
}

pub struct TaskStore {
    conn: Connection,
}

const TASK_COLS: &str = "id, kind, model_id, status, priority, progress, params, result, error_message, created_at, started_at, completed_at";

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
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                thumbnail_path TEXT,
                width INTEGER,
                height INTEGER,
                duration_seconds REAL,
                source_task_id TEXT,
                source_action TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                parent_asset_id TEXT,
                relation_type TEXT,
                group_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_assets_kind ON assets(kind);
            CREATE INDEX IF NOT EXISTS idx_assets_created ON assets(created_at);
            CREATE TABLE IF NOT EXISTS asset_groups (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                kind TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS canvas_sessions (
                id TEXT PRIMARY KEY,
                media TEXT NOT NULL DEFAULT 'image',
                title TEXT NOT NULL,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_canvas_sessions_media ON canvas_sessions(media);
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

    pub fn mark_running(&self, id: &str) -> Result<usize, StoreError> {
        let n = self.conn.execute(
            "UPDATE tasks SET status='running', progress=0, started_at=datetime('now')
             WHERE id=?1 AND status='queued'",
            params![id],
        )?;
        Ok(n)
    }

    pub fn set_progress(&self, id: &str, progress: f64) -> Result<(), StoreError> {
        self.conn.execute(
            "UPDATE tasks SET progress=?2 WHERE id=?1 AND status='running'",
            params![id, progress],
        )?;
        Ok(())
    }

    pub fn finish(
        &self,
        id: &str,
        status: &str,
        progress: f64,
        result: Option<&str>,
        error_message: Option<&str>,
    ) -> Result<usize, StoreError> {
        let n = self.conn.execute(
            "UPDATE tasks SET status=?2, progress=?3, result=COALESCE(?4, result),
             error_message=COALESCE(?5, error_message), completed_at=datetime('now')
             WHERE id=?1 AND status IN ('queued','running')",
            params![id, status, progress, result, error_message],
        )?;
        Ok(n)
    }

    pub fn set_priority(&self, id: &str, priority: i64) -> Result<usize, StoreError> {
        let n = self.conn.execute(
            "UPDATE tasks SET priority=?2 WHERE id=?1 AND status='queued'",
            params![id, priority],
        )?;
        Ok(n)
    }

    pub fn log(&self, task_id: &str, level: &str, message: &str) -> Result<(), StoreError> {
        self.conn.execute(
            "INSERT INTO task_logs (task_id, ts, level, message) VALUES (?1, datetime('now'), ?2, ?3)",
            params![task_id, level, message],
        )?;
        Ok(())
    }

    pub fn logs(&self, task_id: &str, offset: i64, limit: i64) -> Result<Vec<LogRow>, StoreError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, message, level, ts FROM task_logs
             WHERE task_id=?1 ORDER BY id ASC LIMIT ?2 OFFSET ?3",
        )?;
        let rows = stmt.query_map(params![task_id, limit, offset], |row| {
            Ok(LogRow {
                id: row.get(0)?,
                message: row.get(1)?,
                level: row.get(2)?,
                time: row.get(3)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(StoreError::from)
    }

    pub fn get(&self, id: &str) -> Result<Option<TaskRow>, StoreError> {
        let mut stmt = self
            .conn
            .prepare(&format!("SELECT {TASK_COLS} FROM tasks WHERE id=?1"))?;
        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(read_task(row)?))
        } else {
            Ok(None)
        }
    }

    pub fn list_tasks(
        &self,
        limit: i64,
        offset: i64,
        kind: Option<&str>,
        status: Option<&str>,
        since: Option<&str>,
    ) -> Result<Vec<TaskRow>, StoreError> {
        let mut sql = format!("SELECT {TASK_COLS} FROM tasks WHERE 1=1");
        let mut args: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        if let Some(kind) = kind.filter(|k| !k.is_empty()) {
            if kind.contains('.') {
                sql.push_str(" AND kind=?");
                args.push(Box::new(kind.to_string()));
            } else {
                sql.push_str(" AND kind LIKE ?");
                args.push(Box::new(format!("{kind}.%")));
            }
        }
        if let Some(status) = status.filter(|s| !s.is_empty()) {
            sql.push_str(" AND status=?");
            args.push(Box::new(status.to_string()));
        }
        if let Some(since) = since.filter(|s| !s.is_empty()) {
            sql.push_str(" AND created_at>=?");
            args.push(Box::new(since.to_string()));
        }
        sql.push_str(" ORDER BY created_at DESC LIMIT ? OFFSET ?");
        args.push(Box::new(limit));
        args.push(Box::new(offset));
        let refs: Vec<&dyn rusqlite::types::ToSql> = args.iter().map(|a| a.as_ref()).collect();
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(refs.as_slice(), read_task)?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(StoreError::from)
    }

    pub fn list_queued(&self, image_first: bool) -> Result<Vec<TaskRow>, StoreError> {
        let order = if image_first {
            "CASE WHEN kind LIKE 'image.%' THEN 0 ELSE 1 END, priority ASC, created_at ASC"
        } else {
            "priority ASC, created_at ASC"
        };
        let sql = format!("SELECT {TASK_COLS} FROM tasks WHERE status='queued' ORDER BY {order}");
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], read_task)?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(StoreError::from)
    }

    pub fn insert_asset(&self, row: &AssetRow) -> Result<(), StoreError> {
        self.conn.execute(
            "INSERT INTO assets (
                id, kind, mime_type, file_path, thumbnail_path, width, height, duration_seconds,
                source_task_id, source_action, metadata, created_at, parent_asset_id, relation_type, group_id
             ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,datetime('now'),?12,?13,?14)",
            params![
                row.id,
                row.kind,
                row.mime_type,
                row.file_path,
                row.thumbnail_path,
                row.width,
                row.height,
                row.duration_seconds,
                row.source_task_id,
                row.source_action,
                row.metadata,
                row.parent_asset_id,
                row.relation_type,
                row.group_id,
            ],
        )?;
        Ok(())
    }

    pub fn get_asset(&self, id: &str) -> Result<Option<AssetRow>, StoreError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, mime_type, file_path, thumbnail_path, width, height, duration_seconds,
                    source_task_id, source_action, metadata, created_at, parent_asset_id, relation_type, group_id
             FROM assets WHERE id=?1",
        )?;
        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(read_asset(row)?))
        } else {
            Ok(None)
        }
    }

    pub fn list_assets(
        &self,
        kind: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<(Vec<AssetRow>, i64), StoreError> {
        let (where_sql, arg) = match kind.filter(|k| !k.is_empty()) {
            Some(kind) => ("WHERE kind=?", Some(kind.to_string())),
            None => ("", None),
        };
        let total: i64 = match &arg {
            Some(kind) => self.conn.query_row(
                &format!("SELECT COUNT(*) FROM assets {where_sql}"),
                params![kind],
                |row| row.get(0),
            )?,
            None => self
                .conn
                .query_row("SELECT COUNT(*) FROM assets", [], |row| row.get(0))?,
        };
        let sql = format!(
            "SELECT id, kind, mime_type, file_path, thumbnail_path, width, height, duration_seconds,
                    source_task_id, source_action, metadata, created_at, parent_asset_id, relation_type, group_id
             FROM assets {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = if let Some(kind) = arg {
            stmt.query_map(params![kind, limit, offset], read_asset)?
                .collect::<Result<Vec<_>, _>>()?
        } else {
            stmt.query_map(params![limit, offset], read_asset)?
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok((rows, total))
    }

    pub fn delete_asset(&self, id: &str) -> Result<Option<AssetRow>, StoreError> {
        let row = self.get_asset(id)?;
        if row.is_some() {
            self.conn
                .execute("DELETE FROM assets WHERE id=?1", params![id])?;
        }
        Ok(row)
    }

    pub fn child_assets(&self, parent_id: &str) -> Result<Vec<AssetRow>, StoreError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, kind, mime_type, file_path, thumbnail_path, width, height, duration_seconds,
                    source_task_id, source_action, metadata, created_at, parent_asset_id, relation_type, group_id
             FROM assets WHERE parent_asset_id=?1 ORDER BY created_at ASC",
        )?;
        let rows = stmt.query_map(params![parent_id], read_asset)?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(StoreError::from)
    }

    pub fn insert_canvas(&self, row: &CanvasRow) -> Result<(), StoreError> {
        self.conn.execute(
            "INSERT INTO canvas_sessions (id, media, title, state_json, created_at, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6)",
            params![
                row.id,
                row.media,
                row.title,
                row.state_json,
                row.created_at,
                row.updated_at
            ],
        )?;
        Ok(())
    }

    pub fn update_canvas(
        &self,
        id: &str,
        title: Option<&str>,
        state_json: Option<&str>,
        updated_at: &str,
    ) -> Result<bool, StoreError> {
        let n = if let (Some(title), Some(state)) = (title, state_json) {
            self.conn.execute(
                "UPDATE canvas_sessions SET title=?2, state_json=?3, updated_at=?4 WHERE id=?1",
                params![id, title, state, updated_at],
            )?
        } else if let Some(title) = title {
            self.conn.execute(
                "UPDATE canvas_sessions SET title=?2, updated_at=?3 WHERE id=?1",
                params![id, title, updated_at],
            )?
        } else if let Some(state) = state_json {
            self.conn.execute(
                "UPDATE canvas_sessions SET state_json=?2, updated_at=?3 WHERE id=?1",
                params![id, state, updated_at],
            )?
        } else {
            0
        };
        Ok(n > 0)
    }

    pub fn get_canvas(&self, id: &str) -> Result<Option<CanvasRow>, StoreError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, media, title, state_json, created_at, updated_at FROM canvas_sessions WHERE id=?1",
        )?;
        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(read_canvas(row)?))
        } else {
            Ok(None)
        }
    }

    pub fn list_canvas(&self, media: &str, limit: i64) -> Result<Vec<CanvasRow>, StoreError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, media, title, state_json, created_at, updated_at
             FROM canvas_sessions WHERE media=?1 ORDER BY updated_at DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![media, limit], read_canvas)?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(StoreError::from)
    }

    pub fn delete_canvas(&self, id: &str) -> Result<bool, StoreError> {
        Ok(self
            .conn
            .execute("DELETE FROM canvas_sessions WHERE id=?1", params![id])?
            > 0)
    }
}

fn read_task(row: &rusqlite::Row<'_>) -> rusqlite::Result<TaskRow> {
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
        created_at: row.get(9)?,
        started_at: row.get(10)?,
        completed_at: row.get(11)?,
    })
}

fn read_asset(row: &rusqlite::Row<'_>) -> rusqlite::Result<AssetRow> {
    Ok(AssetRow {
        id: row.get(0)?,
        kind: row.get(1)?,
        mime_type: row.get(2)?,
        file_path: row.get(3)?,
        thumbnail_path: row.get(4)?,
        width: row.get(5)?,
        height: row.get(6)?,
        duration_seconds: row.get(7)?,
        source_task_id: row.get(8)?,
        source_action: row.get(9)?,
        metadata: row
            .get::<_, Option<String>>(10)?
            .unwrap_or_else(|| "{}".into()),
        created_at: row.get(11)?,
        parent_asset_id: row.get(12)?,
        relation_type: row.get(13)?,
        group_id: row.get(14)?,
    })
}

fn read_canvas(row: &rusqlite::Row<'_>) -> rusqlite::Result<CanvasRow> {
    Ok(CanvasRow {
        id: row.get(0)?,
        media: row.get(1)?,
        title: row.get(2)?,
        state_json: row.get(3)?,
        created_at: row.get(4)?,
        updated_at: row.get(5)?,
    })
}
