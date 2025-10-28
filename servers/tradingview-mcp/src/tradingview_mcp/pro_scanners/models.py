"""Pydantic models used across pro scanners."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Sequence

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


class DerivativeSnapshot(BaseModel):
    open_interest: float | None = Field(default=None)
    funding_rate: float | None = Field(default=None)
    cvd: float | None = Field(default=None, description="Cumulative volume delta")
    timestamp: datetime | None = Field(default=None)


class IndicatorSnapshot(BaseModel):
    ema_fast: float | None = None
    ema_slow: float | None = None
    adx: float | None = None
    volume_z: float | None = None


class RiskAssessment(BaseModel):
    position_size: PositiveFloat | None = None
    leverage: PositiveFloat | None = None
    stop_loss: float | None = None
    take_profit: Sequence[float] = Field(default_factory=tuple)
    notes: str | None = None


class ScannerSignal(BaseModel):
    symbol: str
    timeframe: str
    direction: SignalDirection
    entry: float
    confidence: PositiveInt | None = None
    indicators: IndicatorSnapshot = Field(default_factory=IndicatorSnapshot)
    derivatives: DerivativeSnapshot = Field(default_factory=DerivativeSnapshot)
    risk: RiskAssessment = Field(default_factory=RiskAssessment)
    reasons: Sequence[str] = Field(default_factory=tuple)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class BacktestRequest(BaseModel):
    symbols: Sequence[str]
    timeframe: str = Field(default="15m")
    lookback_limit: PositiveInt = Field(default=500)
    profile: str = Field(default="balanced")
    strategy: str = Field(default="momentum")


class BacktestResult(BaseModel):
    signals: Sequence[ScannerSignal]
    metrics: dict[str, float]
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    profile: str
    timeframe: str
    symbol_universe: Sequence[str] = Field(default_factory=tuple)
    strategy: str | None = None


class AlertPayload(BaseModel):
    symbol: str
    timeframe: str
    direction: SignalDirection
    entry: float
    leverage: float | None = None
    confidence: PositiveInt | None = None
    reasons: Sequence[str] = Field(default_factory=tuple)
    metadata: dict[str, str] = Field(default_factory=dict)


class CachedIndicator(BaseModel):
    key: str
    data: IndicatorSnapshot
    ttl: PositiveInt


class CacheEntry(BaseModel):
    key: str
    payload: dict
    ttl: PositiveInt


class AlertRouteResult(BaseModel):
    payload: AlertPayload
    status: Literal["sent", "skipped", "failed"]
    details: str | None = None
    dispatched_at: datetime = Field(default_factory=datetime.utcnow)


__all__ = [
    "AlertPayload",
    "AlertRouteResult",
    "BacktestRequest",
    "BacktestResult",
    "CacheEntry",
    "CachedIndicator",
    "DerivativeSnapshot",
    "IndicatorSnapshot",
    "RiskAssessment",
    "ScannerSignal",
    "SignalDirection",
]
