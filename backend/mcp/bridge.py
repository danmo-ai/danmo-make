"""HTTP bridge from MCP tools to Danmo Make REST API."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from backend.mcp.base_url import resolve_api_base_url


class MakeAPIBridge:
    """Thin async client over ``/api/*`` (same contracts as the SPA/CLI)."""

    def __init__(self, base_url: str | None = None, timeout: float = 60.0) -> None:
        self.base_url = (base_url or resolve_api_base_url()).rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        to = timeout if timeout is not None else self.timeout
        async with httpx.AsyncClient(timeout=to) as client:
            resp = await client.request(
                method,
                self._url(path),
                json=json_body,
                params=params,
            )
        if resp.status_code >= 400:
            detail: Any
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(
                f"Make API {method} {path} → HTTP {resp.status_code}: {detail}"
            )
        if resp.status_code == 204 or not resp.content:
            return None
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            return resp.json()
        return {"content_type": ctype, "bytes": len(resp.content)}

    async def get(self, path: str, **kwargs: Any) -> Any:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, json_body: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return await self.request("POST", path, json_body=json_body, **kwargs)

    async def delete(self, path: str, **kwargs: Any) -> Any:
        return await self.request("DELETE", path, **kwargs)

    async def upload_file(self, file_path: str) -> dict[str, Any]:
        p = Path(file_path).expanduser().resolve()
        if not p.is_file():
            raise RuntimeError(f"upload_asset: file not found: {p}")
        mime, _ = mimetypes.guess_type(str(p))
        mime = mime or "application/octet-stream"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with p.open("rb") as fh:
                resp = await client.post(
                    self._url("/api/assets"),
                    files={"file": (p.name, fh, mime)},
                )
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"upload_asset failed HTTP {resp.status_code}: {detail}")
        return resp.json()


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
