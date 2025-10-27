"""Policy MCP Server - Decision profiles and policy management."""

import asyncio
from fastmcp import FastMCP
from .config import get_settings
from .services.profile_service import ProfileService
from .tools import register_tools


def create_server() -> FastMCP:
    """Create and configure the policy MCP server."""
    mcp = FastMCP("policy-mcp")
    
    # Initialize services
    profile_service = ProfileService()
    
    # Register tools
    register_tools(mcp, profile_service)
    
    return mcp


def main() -> None:
    """Main entry point."""
    import sys
    
    server = create_server()
    
    if "--stdio" in sys.argv:
        server.run_stdio()
    else:
        # HTTP mode
        host = "0.0.0.0"
        port = 8000
        
        # Parse host and port from args
        for i, arg in enumerate(sys.argv):
            if arg == "--host" and i + 1 < len(sys.argv):
                host = sys.argv[i + 1]
            elif arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        
        asyncio.run(server.run_http_async(host=host, port=port))


if __name__ == "__main__":
    main()

