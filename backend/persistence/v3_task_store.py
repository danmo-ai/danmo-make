"""v3 unified task table: any media, any params JSON."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from backend.core.interfaces import IV3TaskStore, TaskStatus


class V3TaskStore(IV3TaskStore):
    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self._db_path), check_same_thread=False, timeout=30.0)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA busy_timeout=30000")
        return c

    def _init_db(self) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(
                    """
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
                    CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                    CREATE INDEX IF NOT EXISTS idx_tasks_kind ON tasks(kind);
                    CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_task_logs_task ON task_logs(task_id);
                    """
                )
                conn.commit()

    def insert_task(
        self,
        task_id: str,
        kind: str,
        model_id: str,
        params: dict[str, Any],
        *,
        priority: int = 100,
        status: TaskStatus = TaskStatus.QUEUED,
    ) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO tasks
                    (id, kind, model_id, status, priority, progress, params, created_at)
                    VALUES (?, ?, ?, ?, ?, 0.0, ?, ?)
                    """,
                    (
                        task_id,
                        kind,
                        model_id,
                        status.value,
                        priority,
                        json.dumps(params, ensure_ascii=False),
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()

    def update_status(self, task_id: str, status: TaskStatus) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE id = ?",
                    (status.value, task_id),
                )
                conn.commit()

    def update_progress(self, task_id: str, progress: float) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE tasks SET progress = ? WHERE id = ?",
                    (progress, task_id),
                )
                conn.commit()

    def update_task_priority(self, task_id: str, priority: int) -> bool:
        """Only ``queued`` rows can change ``priority`` (50=high / 100=normal). Returns True on success."""
        with self._lock:
            with self._conn() as conn:
                cur = conn.execute(
                    "UPDATE tasks SET priority = ? WHERE id = ? AND status = ?",
                    (priority, task_id, TaskStatus.QUEUED.value),
                )
                conn.commit()
                return cur.rowcount > 0

    def mark_running(self, task_id: str) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    UPDATE tasks SET status = ?, started_at = ?, progress = 0.0
                    WHERE id = ?
                    """,
                    (TaskStatus.RUNNING.value, datetime.now().isoformat(), task_id),
                )
                conn.commit()

    def mark_completed(self, task_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    UPDATE tasks SET status = ?, progress = 1.0, result = ?, completed_at = ?
                    WHERE id = ?
                    """,
                    (
                        TaskStatus.COMPLETED.value,
                        json.dumps(result, ensure_ascii=False),
                        datetime.now().isoformat(),
                        task_id,
                    ),
                )
                conn.commit()

    def mark_failed(self, task_id: str, message: str) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    UPDATE tasks SET status = ?, error_message = ?, completed_at = ?
                    WHERE id = ?
                    """,
                    (TaskStatus.FAILED.value, message, datetime.now().isoformat(), task_id),
                )
                conn.commit()

    def mark_cancelled(self, task_id: str) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    UPDATE tasks SET status = ?, completed_at = ?
                    WHERE id = ?
                    """,
                    (TaskStatus.CANCELLED.value, datetime.now().isoformat(), task_id),
                )
                conn.commit()

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            return None
        return self._row_dict(row)

    def delete_task(self, task_id: str) -> bool:
        with self._lock:
            with self._conn() as conn:
                conn.execute("DELETE FROM task_logs WHERE task_id = ?", (task_id,))
                cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                conn.commit()
                return cur.rowcount > 0

    def list_tasks(
        self,
        limit: int = 200,
        offset: int = 0,
        *,
        kind: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[dict[str, Any]]:
        where = ["1=1"]
        args: list[Any] = []
        if kind:
            if "." in kind:
                where.append("kind = ?")
                args.append(kind)
            else:
                where.append("kind LIKE ?")
                args.append(f"{kind}.%")
        if status:
            where.append("status = ?")
            args.append(status)
        if since:
            where.append("created_at >= ?")
            args.append(since)
        sql = (
            "SELECT * FROM tasks WHERE "
            + " AND ".join(where)
            + " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        args.extend([limit, offset])
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._row_dict(r) for r in rows]

    def append_log(self, task_id: str, message: str, level: str = "info") -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO task_logs (task_id, message, level, ts)
                    VALUES (?, ?, ?, ?)
                    """,
                    (task_id, message, level, datetime.now().isoformat()),
                )
                conn.commit()

    def get_logs(self, task_id: str, offset: int = 0, limit: int = 500) -> List[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT message, level, ts FROM task_logs
                WHERE task_id = ? ORDER BY id ASC LIMIT ? OFFSET ?
                """,
                (task_id, limit, offset),
            ).fetchall()
        return [{"message": r["message"], "level": r["level"], "time": r["ts"]} for r in rows]

    def _row_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "kind": row["kind"],
            "model_id": row["model_id"],
            "status": row["status"],
            "priority": row["priority"],
            "progress": row["progress"],
            "params": json.loads(row["params"]),
            "result": json.loads(row["result"]) if row["result"] else None,
            "error_message": row["error_message"] or "",
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
        }
