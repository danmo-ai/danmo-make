"""Loopback-open / remote API-key gate for REST and MCP.

Storage (settings file): ``v1:<salt_b64url>:<hmac_sha256_hex>`` —
per-key random salt + HMAC-SHA256(salt, plaintext). Never store plaintext.

Client always sends plaintext; server re-derives the digest and compares.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import os
import secrets
from typing import Literal

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.core.container import get_container
from backend.core.interfaces import ISettingsService

ScopeKind = Literal["http_api", "mcp", "none"]
KeyKind = Literal["http", "mcp"]

_V1_PREFIX = "v1:"
_SALT_BYTES = 16


def generate_api_key(kind: KeyKind) -> str:
    """Opaque high-entropy key with scope prefix (shown once to the user)."""
    prefix = "dmh_" if kind == "http" else "dmm_"
    return prefix + secrets.token_urlsafe(32)


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def hash_api_key(plaintext: str, *, salt: bytes | None = None) -> str:
    """Return ``v1:<salt>:<hmac>`` for durable storage (never store plaintext)."""
    if salt is None:
        salt = secrets.token_bytes(_SALT_BYTES)
    digest = hmac.new(salt, plaintext.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{_V1_PREFIX}{_b64url_encode(salt)}:{digest}"


def key_hint(plaintext: str) -> str:
    """Stable display hint: first 8 + … + last 4."""
    p = (plaintext or "").strip()
    if len(p) <= 14:
        return p[:4] + "…" if len(p) > 4 else p
    return f"{p[:8]}…{p[-4:]}"


def verify_stored_key(presented: str, stored: str) -> bool:
    """Verify client plaintext against ``v1:<salt>:<hmac>`` only."""
    presented = (presented or "").strip()
    stored = (stored or "").strip()
    if not presented or not stored.startswith(_V1_PREFIX):
        return False

    body = stored[len(_V1_PREFIX) :]
    salt_b64, sep, digest = body.partition(":")
    if not sep or not salt_b64 or not digest:
        return False
    try:
        salt = _b64url_decode(salt_b64)
    except Exception:
        return False
    expected = hmac.new(salt, presented.encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, digest)


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    h = host.strip().lower()
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    if "%" in h:
        h = h.split("%", 1)[0]
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return h in ("localhost",)


def client_host(request: Request) -> str | None:
    """Use direct peer only — do not trust X-Forwarded-For for auth bypass."""
    if request.client is None:
        return None
    return request.client.host


def auth_scope_for_path(path: str) -> ScopeKind:
    if path == "/mcp" or path.startswith("/mcp/"):
        return "mcp"
    if path.startswith("/api/") or path == "/api":
        return "http_api"
    return "none"


def extract_presented_key(request: Request) -> str:
    auth = (request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    header_key = (request.headers.get("x-api-key") or "").strip()
    if header_key:
        return header_key
    return (
        (request.query_params.get("api_key") or request.query_params.get("access_token") or "")
        .strip()
    )


def _stored_http_secret() -> str:
    try:
        return (get_container().resolve(ISettingsService).get_settings().http_api_key or "").strip()
    except Exception:
        return ""


def _stored_mcp_secret() -> str:
    try:
        return (get_container().resolve(ISettingsService).get_settings().mcp_api_key or "").strip()
    except Exception:
        return ""


def http_api_key_configured() -> bool:
    return bool(_stored_http_secret()) or bool((os.environ.get("DANQING_HTTP_API_KEY") or "").strip())


def mcp_api_key_configured() -> bool:
    return bool(_stored_mcp_secret()) or bool((os.environ.get("DANQING_MCP_API_KEY") or "").strip())


def verify_http_api_key(presented: str) -> bool:
    env = (os.environ.get("DANQING_HTTP_API_KEY") or "").strip()
    if env:
        return hmac.compare_digest(presented.encode("utf-8"), env.encode("utf-8"))
    return verify_stored_key(presented, _stored_http_secret())


def verify_mcp_api_key(presented: str) -> bool:
    env = (os.environ.get("DANQING_MCP_API_KEY") or "").strip()
    if env:
        return hmac.compare_digest(presented.encode("utf-8"), env.encode("utf-8"))
    return verify_stored_key(presented, _stored_mcp_secret())


def access_key_public_view(settings) -> dict[str, object]:
    """Safe fields for GET /api/settings — never include secrets."""
    http_stored = (getattr(settings, "http_api_key", None) or "").strip()
    mcp_stored = (getattr(settings, "mcp_api_key", None) or "").strip()
    return {
        "http_api_key_configured": bool(http_stored)
        or bool((os.environ.get("DANQING_HTTP_API_KEY") or "").strip()),
        "http_api_key_hint": (getattr(settings, "http_api_key_hint", None) or "").strip(),
        "http_api_key_from_env": bool((os.environ.get("DANQING_HTTP_API_KEY") or "").strip()),
        "mcp_api_key_configured": bool(mcp_stored)
        or bool((os.environ.get("DANQING_MCP_API_KEY") or "").strip()),
        "mcp_api_key_hint": (getattr(settings, "mcp_api_key_hint", None) or "").strip(),
        "mcp_api_key_from_env": bool((os.environ.get("DANQING_MCP_API_KEY") or "").strip()),
    }


class AccessAuthMiddleware:
    """ASGI middleware: remote clients must present scope-specific API keys."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        if request.method == "OPTIONS":
            await self.app(scope, receive, send)
            return

        path = request.url.path or "/"
        kind = auth_scope_for_path(path)
        if kind == "none":
            await self.app(scope, receive, send)
            return

        if is_loopback_host(client_host(request)):
            await self.app(scope, receive, send)
            return

        configured = http_api_key_configured() if kind == "http_api" else mcp_api_key_configured()
        label = "HTTP API" if kind == "http_api" else "MCP"
        env_name = "DANQING_HTTP_API_KEY" if kind == "http_api" else "DANQING_MCP_API_KEY"

        if not configured:
            resp = JSONResponse(
                status_code=401,
                content={
                    "code": "auth_required",
                    "message": (
                        f"Non-loopback access to {label} requires an API key, but none is configured. "
                        f"Create one in Settings → Integrations (loopback), or set env {env_name}."
                    ),
                },
            )
            await resp(scope, receive, send)
            return

        presented = extract_presented_key(request)
        ok = verify_http_api_key(presented) if kind == "http_api" else verify_mcp_api_key(presented)
        if not ok:
            resp = JSONResponse(
                status_code=401,
                content={
                    "code": "unauthorized",
                    "message": (
                        f"Invalid or missing {label} key. "
                        "Send Authorization: Bearer <key> or X-API-Key: <key>."
                    ),
                },
            )
            await resp(scope, receive, send)
            return

        await self.app(scope, receive, send)
