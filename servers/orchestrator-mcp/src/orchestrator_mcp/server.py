"""Orchestrator MCP entry point."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Optional

from fastmcp import FastMCP

from .clients import LearningClient, PolicyClient, SupervisorClient
from .config import Settings, get_settings
from .services import OrchestratorService
from .tools import register_tools


def create_server(settings: Optional[Settings] = None) -> FastMCP:
    settings = settings or get_settings()
    mcp = FastMCP("orchestrator-mcp")

    supervisor_client = SupervisorClient()
    policy_client = PolicyClient()
    learning_client = LearningClient()
    orchestrator_service = OrchestratorService(
        supervisor_client=supervisor_client,
        policy_client=policy_client,
        learning_client=learning_client,
    )

    register_tools(mcp, orchestrator_service)

    @mcp.http_app.on_event("shutdown")  # pragma: no cover - lifecycle hook
    async def _shutdown_clients() -> None:
        await supervisor_client.close()
        await policy_client.close()
        await learning_client.close()

    return mcp


def main() -> None:
    import sys

    settings = get_settings()
    server = create_server(settings=settings)

    if "--stdio" in sys.argv:
        server.run_stdio()
        return

    # HTTP mode
    host = "0.0.0.0"
    port = 8010

    for i, arg in enumerate(sys.argv):
        if arg == "--host" and i + 1 < len(sys.argv):
            host = sys.argv[i + 1]
        elif arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])

    asyncio.run(server.run_http_async(host=host, port=port))


if __name__ == "__main__":
    main()
