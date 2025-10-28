#!/usr/bin/env python3
"""Entry point for TradingView MCP server."""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import and run from the new mcp_server
from mcp_server import main


if __name__ == "__main__":
    main()