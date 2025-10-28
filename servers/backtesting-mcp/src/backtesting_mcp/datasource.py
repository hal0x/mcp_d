"""Market data loaders for the backtesting MCP."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import httpx
import pandas as pd

from .config import get_settings


class MarketDataSource:
    """Facade for pulling OHLCV candles from upstream MCP servers."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def fetch_candles(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Optional[pd.DataFrame]:
        source = self.settings.preferred_source.lower()
        if source in {"binance", "both"}:
            candles = self._fetch_binance(symbol, timeframe, start, end)
            if candles is not None:
                return candles
        if source in {"tradingview", "both"}:
            candles = self._fetch_tradingview(symbol, timeframe, start, end)
            if candles is not None:
                return candles
        return None

    def _fetch_binance(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Optional[pd.DataFrame]:
        url = f"{self.settings.binance_url.rstrip('/')}/market/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": timeframe,
            "limit": 1000,
        }
        with httpx.Client(timeout=self.settings.request_timeout) as client:
            try:
                response = client.get(url, params=params)
                response.raise_for_status()
            except Exception:
                return None
        data = response.json() or {}
        raw = data.get("recent_klines") if isinstance(data, dict) else None
        if not raw:
            return None
        frame = pd.DataFrame(raw)
        if frame.empty:
            return None
        frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        frame = frame.set_index("timestamp").sort_index()
        # Convert start and end to UTC timezone-aware datetime
        start_utc = pd.to_datetime(start, utc=True)
        end_utc = pd.to_datetime(end, utc=True)
        frame = frame.loc[(frame.index >= start_utc) & (frame.index <= end_utc)]
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = frame[column].astype(float)
        return frame[["open", "high", "low", "close", "volume"]]

    def _fetch_tradingview(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Optional[pd.DataFrame]:
        url = f"{self.settings.tradingview_url.rstrip('/')}/tools/run"
        payload = {
            "name": "bollinger_scan",
            "arguments": {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "limit": 500,
            },
        }
        with httpx.Client(timeout=self.settings.request_timeout) as client:
            try:
                response = client.post(url, json=payload)
                response.raise_for_status()
            except Exception:
                return None
        data = response.json()
        if not isinstance(data, dict):
            return None
        items = data.get("results") or []
        if not items:
            return None
        records: List[Dict[str, float]] = []
        for item in items:
            candle = item.get("candles", [])[-1] if item.get("candles") else None
            if not candle:
                continue
            records.append(
                {
                    "timestamp": pd.to_datetime(candle.get("time"), utc=True, unit="ms"),
                    "open": float(candle.get("open", 0.0)),
                    "high": float(candle.get("high", 0.0)),
                    "low": float(candle.get("low", 0.0)),
                    "close": float(candle.get("close", 0.0)),
                    "volume": float(candle.get("volume", 0.0)),
                }
            )
        if not records:
            return None
        frame = pd.DataFrame(records).set_index("timestamp").sort_index()
        # Convert start and end to UTC timezone-aware datetime
        start_utc = pd.to_datetime(start, utc=True)
        end_utc = pd.to_datetime(end, utc=True)
        frame = frame.loc[(frame.index >= start_utc) & (frame.index <= end_utc)]
        return frame if not frame.empty else None


__all__ = ["MarketDataSource"]
