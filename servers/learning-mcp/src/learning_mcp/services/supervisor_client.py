"""Async client for Supervisor MCP REST API."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

from ..config import get_settings


class SupervisorClient:
    """Thin wrapper over Supervisor MCP HTTP endpoints."""

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        settings = get_settings()
        self.base_url = (base_url or settings.supervisor_url).rstrip("/")
        self.timeout = timeout or settings.supervisor_timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Dispose underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_aggregates(self, window: str = "7d", kind: str = "business") -> Dict[str, Any]:
        """Fetch aggregated metrics for the specified window/kind."""
        client = await self._get_client()
        response = await client.get(
            f"{self.base_url}/query/agg",
            params={"window": window, "kind": kind},
        )
        response.raise_for_status()
        return response.json()

    async def fetch_facts(
        self,
        window: str = "7d",
        kind: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch facts within a time window."""
        client = await self._get_client()
        params: Dict[str, Any] = {}

        if window:
            start_time, end_time = self._window_to_range(window)
            params["start_time"] = start_time.isoformat()
            params["end_time"] = end_time.isoformat()

        if kind:
            params["kind"] = kind
        if actor:
            params["actor"] = actor

        response = await client.get(f"{self.base_url}/query/facts", params=params or None)
        response.raise_for_status()
        payload = response.json()
        return payload.get("facts", [])

    def _window_to_range(self, window: str) -> tuple[datetime, datetime]:
        """Convert window string into datetime range."""
        if window.endswith("d"):
            days = max(int(window[:-1]), 1)
        else:
            days = 7
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        return start_time, end_time

