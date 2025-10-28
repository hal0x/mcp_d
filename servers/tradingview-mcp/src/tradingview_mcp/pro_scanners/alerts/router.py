
"""Alert routing to HALv1 bot."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, List

import httpx

from ..config import HTTPServiceConfig
from ..models import AlertRouteResult, ScannerSignal
from .formatter import build_alert_payload


logger = logging.getLogger(__name__)


class AlertRouter:
    def __init__(self, config: HTTPServiceConfig, *, max_retries: int = 2, retry_delay: float = 1.0):
        self._config = config
        self._client: httpx.AsyncClient | None = None
        self._max_retries = max(0, max_retries)
        self._retry_delay = max(0.0, retry_delay)
        self._callbacks: List[Callable[[AlertRouteResult], Awaitable[None] | None]] = []

    async def startup(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._config.timeout)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def register_callback(self, callback: Callable[[AlertRouteResult], Awaitable[None] | None]) -> None:
        self._callbacks.append(callback)

    async def dispatch_signal(self, signal: ScannerSignal) -> AlertRouteResult:
        payload = build_alert_payload(signal)
        client = self._require_client()

        attempts = 0
        result: AlertRouteResult | None = None
        last_error: Exception | None = None

        while attempts <= self._max_retries:
            try:
                response = await client.post(self._config.url, json=payload.model_dump(mode="json"))
                response.raise_for_status()
                result = AlertRouteResult(payload=payload, status="sent", details=response.text)
                break
            except httpx.HTTPError as exc:  # pragma: no cover - network failure path
                attempts += 1
                last_error = exc
                logger.warning("Failed to dispatch alert attempt %s/%s: %s", attempts, self._max_retries + 1, exc)
                if attempts > self._max_retries:
                    result = AlertRouteResult(payload=payload, status="failed", details=str(exc))
                    break
                await asyncio.sleep(self._retry_delay * attempts)

        if result is None:
            details = str(last_error) if last_error else "unknown error"
            result = AlertRouteResult(payload=payload, status="failed", details=details)

        await self._notify_callbacks(result)
        return result

    async def _notify_callbacks(self, result: AlertRouteResult) -> None:
        for callback in self._callbacks:
            try:
                outcome = callback(result)
                if asyncio.iscoroutine(outcome):
                    await outcome
            except Exception:  # noqa: BLE001
                logger.exception("Alert callback failed")

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("AlertRouter not started")
        return self._client


__all__ = ["AlertRouter"]
