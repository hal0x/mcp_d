"""MCP server entry point with support for stdio and HTTP transports."""

import argparse
import sys
import logging

import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the shell MCP server."""
    parser = argparse.ArgumentParser(description="Shell MCP Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the HTTP server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8070,
        help="Port to bind the HTTP server to"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default="http",
        help="Transport type (default: http)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development mode)"
    )
    parser.add_argument("--image", default=None, help="Default Docker image")
    parser.add_argument("--network", dest="network_enabled", action=argparse.BooleanOptionalAction, default=None)
    
    args = parser.parse_args()
    
    if args.transport == "http":
        run_http_server(host=args.host, port=args.port, log_level=args.log_level, reload=args.reload)
    else:
        run_stdio_server()


def run_http_server(host: str = "0.0.0.0", port: int = 8070, log_level: str = "INFO", reload: bool = False) -> None:
    """Run the MCP server as HTTP server."""
    from src.shell_mcp.api import create_app
    
    app = create_app()
    
    print(f"Starting Shell MCP server on http://{host}:{port}")
    print(f"Transport: HTTP")
    print(f"Log level: {log_level}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level.lower(),
        reload=reload
    )


def run_stdio_server() -> None:
    """Run the MCP server in stdio mode."""
    from mcp.server.stdio import stdio_server
    from src.shell_mcp.server import _make_server
    
    import asyncio
    
    server = _make_server()
    
    print("Starting Shell MCP server in stdio mode")
    
    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    
    asyncio.run(run())


if __name__ == "__main__":
    main()

