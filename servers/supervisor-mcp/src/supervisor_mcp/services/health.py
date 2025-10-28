"""Health service for monitoring MCP server health."""

from __future__ import annotations

import time
from contextlib import suppress
from datetime import datetime
from typing import Any, AsyncContextManager, Callable, Dict, List, Optional

import httpx
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import HealthStatus, MCPInfo
from ..models.orm import HealthStatusORM

SessionProvider = Callable[[], AsyncContextManager[AsyncSession]]


class HealthService:
    """Service for monitoring MCP server health."""

    def __init__(self, session_provider: SessionProvider):
        self._session_provider = session_provider
        self._health_cache: Dict[str, HealthStatus] = {}
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=5.0)
        return self._http_client

    async def check_health(self, mcp_info: MCPInfo) -> HealthStatus:
        """Check health for specific MCP server and persist snapshot."""
        start_time = time.perf_counter()
        status = "healthy"
        error = None
        response_time_ms: Optional[float] = None

        try:
            if mcp_info.protocol == "http" and mcp_info.endpoint:
                client = await self._get_http_client()
                health_url = f"{mcp_info.endpoint}/health"
                response = await client.get(health_url)
                response_time_ms = (time.perf_counter() - start_time) * 1000
                if response.status_code != 200:
                    status = "unhealthy"
                    error = f"HTTP {response.status_code}"
            else:
                response_time_ms = (time.perf_counter() - start_time) * 1000
        except Exception as exc:  # pragma: no cover - defensive path
            status = "unhealthy"
            error = str(exc)
            response_time_ms = (time.perf_counter() - start_time) * 1000

        snapshot = HealthStatus(
            name=mcp_info.name,
            status=status,
            response_time_ms=response_time_ms,
            error=error,
            last_check=datetime.utcnow(),
        )

        await self._persist_health_status(snapshot)
        self._health_cache[mcp_info.name] = snapshot
        return snapshot

    async def get_health_status(self, name: Optional[str] = None) -> List[HealthStatus]:
        """Get latest health statuses."""
        async with self._session_provider() as session:
            if name:
                stmt = (
                    select(HealthStatusORM)
                    .where(HealthStatusORM.name == name)
                    .order_by(HealthStatusORM.last_check.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                record = result.scalar_one_or_none()
                return [self._to_model(record)] if record else []

            subquery = (
                select(
                    HealthStatusORM.name,
                    func.max(HealthStatusORM.last_check).label("last_check"),
                )
                .group_by(HealthStatusORM.name)
                .subquery()
            )
            stmt = (
                select(HealthStatusORM)
                .join(
                    subquery,
                    (HealthStatusORM.name == subquery.c.name)
                    & (HealthStatusORM.last_check == subquery.c.last_check),
                )
                .order_by(HealthStatusORM.name.asc())
            )
            result = await session.execute(stmt)
            records = result.scalars().all()
            return [self._to_model(record) for record in records]

    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall health summary."""
        statuses = await self.get_health_status()
        if not statuses:
            return {"status": "unknown", "total": 0, "healthy": 0, "unhealthy": 0}

        healthy_count = sum(1 for status in statuses if status.status == "healthy")
        unhealthy_count = len(statuses) - healthy_count
        overall_status = (
            "healthy"
            if unhealthy_count == 0
            else "degraded"
            if unhealthy_count < len(statuses)
            else "unhealthy"
        )

        return {
            "status": overall_status,
            "total": len(statuses),
            "healthy": healthy_count,
            "unhealthy": unhealthy_count,
            "last_check": max(status.last_check for status in statuses),
        }

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def _persist_health_status(self, status: HealthStatus) -> None:
        async with self._session_provider() as session:
            session.add(
                HealthStatusORM(
                    name=status.name,
                    status=status.status,
                    response_time_ms=status.response_time_ms,
                    error=status.error,
                    last_check=status.last_check,
                    uptime_seconds=status.uptime_seconds,
                )
            )
            await session.commit()

    def _to_model(self, record: HealthStatusORM) -> HealthStatus:
        return HealthStatus(
            name=record.name,
            status=record.status,
            response_time_ms=record.response_time_ms,
            error=record.error,
            last_check=record.last_check,
            uptime_seconds=record.uptime_seconds,
        )
