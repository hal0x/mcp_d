"""Bridge for recording trading signals in Memory MCP."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from ..alerts.router import AlertRouteResult
from ..config import HTTPServiceConfig
from ..models import ScannerSignal


logger = logging.getLogger(__name__)


class TradingMemoryRecorder:
    """Push trading signals into the shared Memory MCP service."""

    def __init__(self, config: HTTPServiceConfig | None):
        self._config = config
        self._url = config.url if config else None
        self._timeout = float(config.timeout) if config else 0.0
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    async def store_signal(
        self,
        signal: ScannerSignal,
        *,
        profile: str | None,
        strategy: str | None,
        alert_result: AlertRouteResult | None,
        signal_key: str,
    ) -> None:
        if not self._url:
            return

        payload = self._build_payload(
            signal=signal,
            profile=profile,
            strategy=strategy,
            alert_result=alert_result,
            signal_key=signal_key,
        )

        arguments = {"request": payload}

        try:
            async with self._lock:
                await self._invoke_tool("store_trading_signal", arguments)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to store trading signal in Memory MCP")

    async def _invoke_tool(self, tool: str, arguments: dict[str, Any]) -> None:
        assert self._url is not None  # for type checkers
        timeout = self._timeout or 30.0
        async with streamablehttp_client(self._url, timeout=timeout) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                await session.list_tools()
                await session.call_tool(tool, arguments)

    def _build_payload(
        self,
        *,
        signal: ScannerSignal,
        profile: str | None,
        strategy: str | None,
        alert_result: AlertRouteResult | None,
        signal_key: str,
    ) -> dict[str, Any]:
        indicators = signal.indicators.model_dump(mode="json")
        derivatives = (
            signal.derivatives.model_dump(mode="json") if signal.derivatives else None
        )
        risk = signal.risk.model_dump(mode="json")

        context: dict[str, Any] = {
            "timeframe": signal.timeframe,
            "profile": profile,
            "strategy": strategy,
            "generated_at": _isoformat(signal.generated_at),
            "reasons": list(signal.reasons),
            "indicators": indicators,
            "derivatives": derivatives,
            "risk": risk,
            "source": "tradingview_mcp",
            "signal_key": signal_key,
        }

        if alert_result is not None:
            context["alert"] = {
                "status": alert_result.status,
                "details": alert_result.details,
                "symbol": alert_result.payload.symbol,
                "timeframe": alert_result.payload.timeframe,
                "direction": alert_result.payload.direction.value,
                "entry": alert_result.payload.entry,
                "leverage": alert_result.payload.leverage,
                "confidence": alert_result.payload.confidence,
                "reasons": list(alert_result.payload.reasons),
                "metadata": dict(alert_result.payload.metadata),
            }

        return {
            "symbol": signal.symbol,
            "signal_type": strategy or "unknown",
            "direction": signal.direction.value,
            "entry": float(signal.entry),
            "confidence": float(signal.confidence) if signal.confidence is not None else None,
            "timestamp": _isoformat(signal.generated_at),
            "context": _strip_none(context),
        }


def _strip_none(data: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            nested = _strip_none(value)
            if nested:
                cleaned[key] = nested
        else:
            cleaned[key] = value
    return cleaned


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone().isoformat()


__all__ = ["TradingMemoryRecorder"]
