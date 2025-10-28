#!/usr/bin/env python3
"""Entry point for TradingView MCP server module."""

import sys
import os
import argparse

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tradingview_mcp.server import mcp

def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="TradingView MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--transport", default=None, help="Transport mode")
    
    args = parser.parse_args()
    
    # Determine transport mode
    transport = args.transport or os.getenv("DEFAULT_TRANSPORT", "stdio")
    
    if transport in ["streamable-http", "http"]:
        # HTTP mode
        host = args.host or os.getenv("HOST", "0.0.0.0")
        port = args.port or int(os.getenv("PORT", "8000"))
        print(f"Starting TradingView MCP server in HTTP mode on {host}:{port}")
        # FastMCP использует run() без параметров, транспорт определяется через переменные окружения
        mcp.run()
    else:
        # Stdio mode (default)
        print("Starting TradingView MCP server in stdio mode")
        mcp.run()

if __name__ == "__main__":
    main()
