"""Adapter for binance-mcp service."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import httpx
from datetime import datetime

from ..config import HTTPServiceConfig
from ..models import DerivativeSnapshot
from ..storage.redis_cache import RedisCache
from ..utils import make_cache_key
from .base import BaseMarketDataAdapter



def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


logger = logging.getLogger(__name__)


class BinanceMCPAdapter(BaseMarketDataAdapter):
    """Lightweight client for the binance-mcp service."""

    def __init__(self, http_config: HTTPServiceConfig, cache: RedisCache | None = None):
        self._config = http_config
        self._cache = cache
        self._client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._config.url, timeout=self._config.timeout)

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_klines(self, symbol: str, interval: str, limit: int) -> Sequence[dict]:
        cache_key = make_cache_key("klines", [symbol, interval, limit])
        if self._cache:
            cached = await self._cache.get_json(cache_key)
            if cached:
                return cached

        try:
            client = self._require_client()
            response = await client.get(
                "/market/klines", params={"symbol": symbol, "interval": interval, "limit": limit}
            )
            response.raise_for_status()
            data = response.json()
            klines = self._normalize_klines(data)
            if self._cache:
                await self._cache.set_json(cache_key, klines, ttl=self._cache.indicator_ttl)
            return klines
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            logger.warning("Failed to fetch klines for %s: %s", symbol, exc)
            return []
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error fetching klines for %s: %s", symbol, exc)
            return []

    async def fetch_derivatives(self, symbol: str) -> DerivativeSnapshot:
        cache_key = make_cache_key("derivatives", [symbol])
        if self._cache:
            cached = await self._cache.get_derivatives_snapshot(cache_key)
            if cached:
                return cached

        try:
            ticker = await self.fetch_ticker_24h(symbol)
            snapshot = DerivativeSnapshot(
                open_interest=_coerce_float(ticker.get("open_interest") or ticker.get("openInterest")),
                funding_rate=_coerce_float(ticker.get("funding_rate") or ticker.get("fundingRate")),
                cvd=_coerce_float(ticker.get("quote_volume") or ticker.get("quoteVolume")),
                timestamp=datetime.utcnow(),
            )
            if self._cache:
                await self._cache.set_derivatives_snapshot(cache_key, snapshot)
            return snapshot
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch derivatives for %s: %s", symbol, exc)
            return DerivativeSnapshot(timestamp=datetime.utcnow())

    async def fetch_ticker_24h(self, symbol: str) -> dict[str, Any]:
        cache_key = make_cache_key("ticker24h", [symbol])
        if self._cache:
            cached = await self._cache.get_json(cache_key)
            if cached:
                return cached

        try:
            client = self._require_client()
            response = await client.get("/market/ticker/24h", params={"symbol": symbol})
            response.raise_for_status()
            payload = response.json() or {}
            if self._cache:
                await self._cache.set_json(cache_key, payload, ttl=self._cache.indicator_ttl)
            return payload
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            logger.warning("Failed to fetch ticker for %s: %s", symbol, exc)
            return {}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error fetching ticker for %s: %s", symbol, exc)
            return {}

    @staticmethod
    def _normalize_klines(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            raise TypeError("Unexpected klines payload type")
        items = payload.get("recent_klines") or []
        normalized: list[dict[str, Any]] = []
        for item in items:
            normalized.append(
                {
                    "open_time": item.get("open_time"),
                    "open": float(item.get("open", 0.0)),
                    "high": float(item.get("high", 0.0)),
                    "low": float(item.get("low", 0.0)),
                    "close": float(item.get("close", 0.0)),
                    "volume": float(item.get("volume", 0.0)),
                    "close_time": item.get("close_time"),
                }
            )
        return normalized

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Adapter not started")
        return self._client


__all__ = ["BinanceMCPAdapter"]
