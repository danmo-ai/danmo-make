"""Resolve Danmo Make REST base URL for MCP bridges (stdio / HTTP client)."""

from __future__ import annotations

import os

from backend.utils.user_home import resolve_control_plane_dir


def resolve_api_base_url() -> str:
    """Order: ``DANQING_MCP_BASE_URL`` → ``DANQING_HTTP_*`` → ``api.port`` → ``7800``."""
    explicit = (os.environ.get("DANQING_MCP_BASE_URL") or "").strip().rstrip("/")
    if explicit:
        return explicit

    host = (os.environ.get("DANQING_HTTP_HOST") or "").strip() or "127.0.0.1"
    if host in ("0.0.0.0", "::", "[::]"):
        host = "127.0.0.1"
    port_env = (os.environ.get("DANQING_HTTP_PORT") or "").strip()
    if port_env:
        return f"http://{host}:{port_env}"

    port_file = resolve_control_plane_dir() / "api.port"
    if port_file.is_file():
        raw = port_file.read_text(encoding="utf-8").strip()
        if raw.isdigit():
            return f"http://{host}:{raw}"

    return f"http://{host}:7800"


def resolve_api_base_url_from_app_port(port: int | None = None) -> str:
    """Prefer live listen port when the MCP is mounted in-process."""
    if port and port > 0:
        host = (os.environ.get("DANQING_HTTP_HOST") or "").strip() or "127.0.0.1"
        if host in ("0.0.0.0", "::", "[::]"):
            host = "127.0.0.1"
        return f"http://{host}:{port}"
    return resolve_api_base_url()
