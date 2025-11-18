#!/usr/bin/env python3
"""Simple entry point to run the Coqui TTS MCP server."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from coqui_tts_mcp.server import main

if __name__ == "__main__":
    main()




