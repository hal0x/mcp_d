"""Supervisor MCP Server - Central supervisor for MCP ecosystem."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime
from functools import partial
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastmcp import FastMCP

from .cache import get_redis
from .config import Settings, get_settings
from .db import get_session, init_db
from .pydantic_models import Fact, Metric
from .services.alerts import AlertsService
from .services.health import HealthService
from .services.metrics import MetricsService
from .services.registry import RegistryService
from .services.scraper import ScraperService
from .tools import register_tools


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime strings for query parameters."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


def _serialize_fact(fact: Fact) -> Dict[str, Any]:
    """Serialize Fact model for JSON responses."""
    data = fact.model_dump()
    data["ts"] = fact.ts.isoformat()
    return data


def _register_http_routes(
    mcp: FastMCP,
    metrics_service: MetricsService,
    settings: Settings,
    app: Optional[FastAPI] = None,
) -> None:
    """Attach REST endpoints for aggregates and facts."""
    if app is None:
        # Try to get from mcp, but create new if needed
        try:
            app = mcp.http_app
            if not isinstance(app, FastAPI):
                raise AttributeError("http_app is not a FastAPI instance")
        except (AttributeError, TypeError):
            app = FastAPI()

    async def _aggregation_scheduler() -> None:
        kinds = ("business", "technical")
        while True:
            try:
                for kind in kinds:
                    for window in settings.aggregation_windows:
                        await metrics_service.get_aggregation(kind=kind, window=window)
                await metrics_service.purge_expired_data()
            except Exception:  # pragma: no cover - guard against background failures
                pass
            await asyncio.sleep(settings.aggregation_refresh_seconds)

    @app.on_event("startup")
    async def start_background_tasks() -> None:  # pragma: no cover - FastAPI event
        app.state.aggregation_task = asyncio.create_task(_aggregation_scheduler())

    @app.on_event("shutdown")
    async def stop_background_tasks() -> None:  # pragma: no cover - FastAPI event
        task = getattr(app.state, "aggregation_task", None)
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    @app.get("/query/agg")
    async def get_aggregation(kind: str = "business", window: str = "7d") -> Dict[str, Any]:
        """Return aggregated metrics window."""
        aggregation = await metrics_service.get_aggregation(kind=kind, window=window)
        payload = aggregation.model_dump()
        payload["period_start"] = aggregation.period_start.isoformat()
        payload["period_end"] = aggregation.period_end.isoformat()
        return payload

    @app.get("/query/facts")
    async def get_facts(
        kind: Optional[str] = None,
        actor: Optional[str] = None,
        start_time: Optional[str] = Query(None, description="ISO8601 timestamp"),
        end_time: Optional[str] = Query(None, description="ISO8601 timestamp"),
    ) -> Dict[str, Any]:
        """Return facts filtered by optional parameters."""
        facts = await metrics_service.query_facts(
            kind=kind,
            actor=actor,
            start_time=_parse_datetime(start_time),
            end_time=_parse_datetime(end_time),
        )
        serialized = [_serialize_fact(fact) for fact in facts]
        return {"facts": serialized, "count": len(serialized)}


def create_server(settings: Optional[Settings] = None) -> FastMCP:
    """Create and configure the supervisor MCP server."""
    settings = settings or get_settings()
    mcp = FastMCP("supervisor-mcp")

    session_provider = partial(get_session, settings=settings)
    redis_client = get_redis(settings=settings)

    # Initialize services
    registry_service = RegistryService(session_provider=session_provider)
    health_service = HealthService(session_provider=session_provider)
    metrics_service = MetricsService(
        session_provider=session_provider,
        redis_client=redis_client,
        cache_ttl_seconds=settings.redis_cache_ttl_seconds,
    )
    alerts_service = AlertsService(session_provider=session_provider)
    scraper_service = ScraperService()

    # Register tools
    register_tools(
        mcp,
        registry_service=registry_service,
        health_service=health_service,
        metrics_service=metrics_service,
        alerts_service=alerts_service,
        scraper_service=scraper_service,
    )

    # Attach REST API for aggregates / facts (will use mcp.http_app if available)
    _register_http_routes(mcp, metrics_service=metrics_service, settings=settings)

    return mcp


def main() -> None:
    """Main entry point."""
    import sys

    settings = get_settings()
    
    # Initialize database
    try:
        asyncio.run(init_db(settings=settings))
    except Exception as e:
        import logging
        logging.error(f"Failed to initialize database: {e}")
        # Continue anyway - database might not be critical for startup
    
    server = create_server(settings=settings)

    if "--stdio" in sys.argv:
        server.run_stdio()
        return

    # HTTP mode
    host = "0.0.0.0"
    port = 8000

    # Parse host and port from args
    for i, arg in enumerate(sys.argv):
        if arg == "--host" and i + 1 < len(sys.argv):
            host = sys.argv[i + 1]
        elif arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])

    # Use FastMCP's built-in HTTP transport
    asyncio.run(server.run_http_async(host=host, port=port))


if __name__ == "__main__":
    main()
