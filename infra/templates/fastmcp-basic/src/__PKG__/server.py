from __future__ import annotations

import argparse
import importlib.metadata as _meta
import json
import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from .config import get_settings
from .tools import example as example_tools


logger = logging.getLogger(__name__)


def _make_server() -> FastMCP:
    settings = get_settings()
    name = "__APP_NAME__"
    instructions = "MCP server: __APP_NAME__. See README for tools and usage."
    mcp = FastMCP(name=name, instructions=instructions)

    # Register built-in tools
    @mcp.tool()
    def health() -> dict[str, Any]:
        """Quick health check. Extend to verify downstreams."""
        return {"status": "ok", "mode": settings.MODE}

    @mcp.tool()
    def version() -> dict[str, Any]:
        """Return package version and selected mode."""
        try:
            ver = _meta.version("__CLI__")
        except Exception:
            ver = "0.0.0"
        return {"version": ver, "mode": settings.MODE}

    # Register example domain tools
    example_tools.bind(mcp)

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="__APP_NAME__ MCP server")
    parser.add_argument("--print-config", action="store_true", help="Print effective config and exit")
    parser.add_argument("--host", default=None, help="Bind host for HTTP transport")
    parser.add_argument("--port", type=int, default=None, help="Bind port for HTTP transport")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.print_config:
        print(json.dumps(get_settings().model_dump(), indent=2))
        return

    mcp = _make_server()
    if args.host and args.port:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
