from __future__ import annotations

import importlib.metadata as _meta
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..config import get_settings
from ..services.docker_executor import ensure_docker_available


def bind(mcp: FastMCP) -> None:
    @mcp.tool()
    def health() -> dict[str, Any]:
        """Checks Docker availability and basic server settings."""
        try:
            ensure_docker_available()
            ok = True
            message = "ok"
        except Exception as e:
            ok = False
            message = str(e)
        return {"ok": ok, "message": message}

    @mcp.tool()
    def version() -> dict[str, Any]:
        """Returns server version and default execution parameters."""
        s = get_settings()
        try:
            ver = _meta.version("shell-mcp")
        except Exception:
            ver = "0.0.0"
        return {
            "name": "shell-mcp",
            "version": ver,
            "defaults": {
                "image": s.DEFAULT_IMAGE,
                "workdir": s.CONTAINER_WORKDIR,
                "network": s.DEFAULT_NETWORK,
            },
        }
