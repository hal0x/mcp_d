"""Learning MCP Server - Central learning system for HAL ecosystem."""

import asyncio
import contextlib

from fastmcp import FastMCP

from .config import get_settings
from .services.pattern_analyzer import PatternAnalyzer
from .services.policy_client import PolicyClient
from .services.supervisor_client import SupervisorClient
from .services.trainer import TrainerService
from .services.online_learning import OnlineLearningService
from .tools import register_tools

settings = get_settings()


def create_server() -> FastMCP:
    """Create and configure the learning MCP server."""
    mcp = FastMCP("learning-mcp")

    # Initialize services
    supervisor_client = SupervisorClient()
    trainer_service = TrainerService(supervisor_client=supervisor_client)
    pattern_analyzer = PatternAnalyzer()
    policy_client = PolicyClient()
    online_learning_service = OnlineLearningService(
        trainer_service=trainer_service,
        supervisor_client=supervisor_client,
        policy_client=policy_client,
    )

    # Register tools
    register_tools(
        mcp,
        trainer_service=trainer_service,
        pattern_analyzer=pattern_analyzer,
        policy_client=policy_client,
        supervisor_client=supervisor_client,
        online_learning_service=online_learning_service,
    )

    @mcp.http_app.on_event("shutdown")  # pragma: no cover - FastAPI lifecycle
    async def _shutdown_supervisor_client() -> None:
        await supervisor_client.close()
        await policy_client.close()

    @mcp.http_app.post("/orchestrator/trigger")
    async def orchestrator_trigger() -> dict:
        return await online_learning_service.run_cycle()

    @mcp.http_app.get("/orchestrator/policy-profiles")
    async def orchestrator_policy_profiles(active_only: bool = False) -> dict:
        response = await policy_client.list_profiles(active_only=active_only)
        return response

    @mcp.http_app.on_event("startup")  # pragma: no cover - FastAPI lifecycle
    async def _start_online_learning() -> None:
        app = mcp.http_app
        app.state.online_learning_task = asyncio.create_task(_online_learning_loop())

    @mcp.http_app.on_event("shutdown")  # pragma: no cover
    async def _stop_online_learning() -> None:
        task = getattr(mcp.http_app.state, "online_learning_task", None)
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _online_learning_loop() -> None:
        interval = online_learning_service.interval_seconds
        while True:
            try:
                await online_learning_service.run_cycle()
            except Exception:
                pass
            await asyncio.sleep(interval)

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
        port = 8003

        # Parse host and port from args
        for i, arg in enumerate(sys.argv):
            if arg == "--host" and i + 1 < len(sys.argv):
                host = sys.argv[i + 1]
            elif arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])

        server.run_http(host=host, port=port)


if __name__ == "__main__":
    main()
