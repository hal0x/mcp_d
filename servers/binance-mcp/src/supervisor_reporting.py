"""Supervisor MCP integration for metrics and facts."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
import structlog

from .config import get_config

logger = structlog.get_logger(__name__)


class SupervisorClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def send_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        payload = {
            "name": name,
            "value": value,
            "ts": datetime.utcnow().isoformat(),
            "tags": tags or {},
        }
        await self._client.post(f"{self.base_url}/ingest/metric", json=payload)

    async def send_fact(
        self,
        kind: str,
        payload: Dict[str, Any],
        actor: str = "binance-mcp",
        correlation_id: Optional[str] = None,
    ) -> None:
        body = {
            "kind": kind,
            "payload": payload,
            "actor": actor,
            "correlation_id": correlation_id or "",
            "ts": datetime.utcnow().isoformat(),
        }
        await self._client.post(f"{self.base_url}/ingest/fact", json=body)


class SupervisorReporter:
    _client: Optional[SupervisorClient] = None
    _lock = asyncio.Lock()

    @classmethod
    async def _get_client(cls) -> Optional[SupervisorClient]:
        config = get_config()
        if not config.supervisor_metrics_enabled or not config.supervisor_url:
            return None
        if cls._client is None:
            async with cls._lock:
                if cls._client is None:
                    cls._client = SupervisorClient(
                        base_url=config.supervisor_url,
                        timeout=config.api_backoff_base * 5,
                    )
        return cls._client

    @classmethod
    async def emit_metric(cls, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        client = await cls._get_client()
        if client is None:
            return
        try:
            await client.send_metric(name, value, tags)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("supervisor_metric_failed", metric=name, error=str(exc))

    @classmethod
    async def emit_fact(
        cls,
        kind: str,
        payload: Dict[str, Any],
        actor: str = "binance-mcp",
        correlation_id: Optional[str] = None,
    ) -> None:
        client = await cls._get_client()
        if client is None:
            return
        try:
            await client.send_fact(kind=kind, payload=payload, actor=actor, correlation_id=correlation_id)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("supervisor_fact_failed", kind=kind, error=str(exc))

    @classmethod
    async def order_created(cls, order: Dict[str, Any]) -> None:
        await cls.emit_metric("binance.order_created", 1.0, tags={"symbol": order.get("symbol", "unknown")})
        await cls.emit_fact("Fact:OrderCreated", order, correlation_id=order.get("client_order_id"))

    @classmethod
    async def order_cancelled(cls, order: Dict[str, Any]) -> None:
        await cls.emit_metric("binance.order_cancelled", 1.0, tags={"symbol": order.get("symbol", "unknown")})
        await cls.emit_fact("Fact:OrderCancelled", order, correlation_id=order.get("client_order_id"))

    @classmethod
    async def record_portfolio_metrics(cls, metrics: Dict[str, Any]) -> None:
        await cls.emit_metric("binance.portfolio_equity", metrics.get("total_equity", 0.0))
        await cls.emit_fact("Fact:PortfolioMetrics", metrics)

    @classmethod
    async def shutdown(cls) -> None:
        if cls._client is not None:
            await cls._client.close()
            cls._client = None
