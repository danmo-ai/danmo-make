"""Danmo Make MCP server (Agent / connector surface over REST)."""

from backend.mcp.server import create_mcp, mcp, set_bridge

__all__ = ["create_mcp", "mcp", "set_bridge"]
