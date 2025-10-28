from __future__ import annotations

import argparse
import json
import logging
import os

from mcp.server.fastmcp import FastMCP

from .config import get_settings
from .tools import meta as meta_tools
from .tools import run as run_tools


def _make_server() -> FastMCP:
    mcp = FastMCP(
        name="Docker Executor",
        instructions=(
            "Execute code snippets inside disposable Docker containers. "
            "Tool run_code_simple executes provided source inside a selected image."
        ),
    )
    run_tools.bind(mcp)
    meta_tools.bind(mcp)
    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="shell-mcp: Docker-backed MCP server")
    parser.add_argument(
        "--image", default=None, help="Default Docker image (overrides config)"
    )
    parser.add_argument(
        "--network",
        dest="network_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable networking inside containers by default",
    )
    parser.add_argument("--host", default=None, help="HTTP host (optional)")
    parser.add_argument("--port", type=int, default=None, help="HTTP port (optional)")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print current configuration and exit",
    )
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    # Apply CLI overrides
    if args.image:
        os.environ["SHELL_MCP_DEFAULT_IMAGE"] = args.image
    if args.network_enabled is not None:
        os.environ["SHELL_MCP_DEFAULT_NETWORK"] = (
            "true" if args.network_enabled else "false"
        )

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
