"""SQLite persistence for Studio canvas sessions."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def _default_state() -> dict[str, Any]:
    return {
        "items": {},
        "viewport": {"zoom": 1, "panX": 0, "panY": 0},
        "staging": {"x": 240, "y": 180, "width": 512, "height": 512, "visible": True},
        "active_asset_path": "",
        "overlays": {"reference": None, "control": None},
        "edges": [],
    }


class CanvasSessionStore:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            c = sqlite3.connect(str(self._db_path), check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            self._local.conn = c
        return self._local.conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._conn()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS canvas_sessions (
                    id TEXT PRIMARY KEY,
                    media TEXT NOT NULL DEFAULT 'image',
                    title TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_canvas_sessions_media
                    ON canvas_sessions(media);
                CREATE INDEX IF NOT EXISTS idx_canvas_sessions_updated
                    ON canvas_sessions(updated_at);
                """
            )
            conn.commit()

    def list_sessions(self, *, media: str = "image", limit: int = 50) -> list[dict[str, Any]]:
        lim = max(1, min(int(limit), 200))
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, media, title, state_json, created_at, updated_at
                FROM canvas_sessions
                WHERE media = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (media, lim),
            ).fetchall()
        return [self._row_to_summary(r) for r in rows]

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT id, media, title, state_json, created_at, updated_at
                FROM canvas_sessions WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_detail(row)

    def create_session(
        self,
        *,
        media: str = "image",
        title: str = "",
        state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        sid = "cvs_" + uuid.uuid4().hex[:20]
        now = datetime.now().isoformat()
        st = dict(state or _default_state())
        title_clean = (title or "").strip() or self._default_title(media)
        with self._lock:
            conn = self._conn()
            conn.execute(
                """
                INSERT INTO canvas_sessions (id, media, title, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (sid, media, title_clean, json.dumps(st, ensure_ascii=False), now, now),
            )
            conn.commit()
        return {
            "id": sid,
            "media": media,
            "title": title_clean,
            "state": st,
            "created_at": now,
            "updated_at": now,
        }

    def update_session(
        self,
        session_id: str,
        *,
        title: Optional[str] = None,
        state: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                """
                SELECT id, media, title, state_json, created_at, updated_at
                FROM canvas_sessions WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            if not row:
                return None
            existing = self._row_to_detail(row)
            new_title = (title.strip() if title is not None else existing["title"])
            new_state = state if state is not None else existing["state"]
            conn.execute(
                """
                UPDATE canvas_sessions
                SET title = ?, state_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    new_title,
                    json.dumps(new_state, ensure_ascii=False),
                    now,
                    session_id,
                ),
            )
            conn.commit()
        return {
            "id": session_id,
            "media": existing["media"],
            "title": new_title,
            "state": new_state,
            "created_at": existing["created_at"],
            "updated_at": now,
        }

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            conn = self._conn()
            cur = conn.execute("DELETE FROM canvas_sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cur.rowcount > 0

    @staticmethod
    def _default_title(media: str) -> str:
        if media == "image":
            return "Canvas"
        return f"{media} canvas"

    @staticmethod
    def _parse_state(raw: str) -> dict[str, Any]:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                base = _default_state()
                base.update(data)
                if not isinstance(base.get("items"), dict):
                    base["items"] = {}
                if not isinstance(base.get("viewport"), dict):
                    base["viewport"] = _default_state()["viewport"]
                if not isinstance(base.get("staging"), dict):
                    base["staging"] = _default_state()["staging"]
                if not isinstance(base.get("overlays"), dict):
                    base["overlays"] = _default_state()["overlays"]
                if not isinstance(base.get("edges"), list):
                    base["edges"] = []
                return base
        except (json.JSONDecodeError, TypeError):
            pass
        return _default_state()

    def _row_to_detail(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "media": row["media"],
            "title": row["title"],
            "state": self._parse_state(row["state_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _row_to_summary(self, row: sqlite3.Row) -> dict[str, Any]:
        st = self._parse_state(row["state_json"])
        return {
            "id": row["id"],
            "media": row["media"],
            "title": row["title"],
            "item_count": len(st.get("items") or {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
