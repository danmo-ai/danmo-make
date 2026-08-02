"""``python -m backend.mcp`` — stdio MCP (proxies to local Make REST)."""

from __future__ import annotations

from backend.mcp.bridge import MakeAPIBridge
from backend.mcp.base_url import resolve_api_base_url
from backend.mcp.server import create_mcp, set_bridge


def main() -> None:
    base = resolve_api_base_url()
    set_bridge(MakeAPIBridge(base_url=base))
    create_mcp().run(transport="stdio")


if __name__ == "__main__":
    main()
