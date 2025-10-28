"""Breakout scanner implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..adapters.base import BaseMarketDataAdapter
from ..alerts.router import AlertRouter
from ..filters.context import ContextFilter
from ..filters.derivatives import DerivativesFilter
from ..models import IndicatorSnapshot, ScannerSignal, SignalDirection
from ..profiles import ScannerProfiles
from ..risk.calculator import RiskCalculator, RiskContext
from ..risk.manager import RiskManager
from ..storage.redis_cache import RedisCache
from ..utils import make_cache_key
from .base import BaseScanner
from ..backtesting.engine import BacktestEngine


@dataclass(slots=True)
class BreakoutEvaluation:
    direction: SignalDirection
    breakout_level: float
    atr: float
    reasons: list[str]
    confidence: int


class BreakoutScanner(BaseScanner):
    DEFAULT_INTERVAL = "15m"

    def __init__(
        self,
        adapter: BaseMarketDataAdapter,
        derivatives_filter: DerivativesFilter,
        context_filter: ContextFilter,
        risk_calculator: RiskCalculator,
        risk_manager: RiskManager,
        alert_router: AlertRouter,
        cache: RedisCache,
        profiles: ScannerProfiles | None = None,
    ):
        self._adapter = adapter
        self._derivatives_filter = derivatives_filter
        self._context_filter = context_filter
        self._risk_calculator = risk_calculator
        self._risk_manager = risk_manager
        self._alert_router = alert_router
        self._cache = cache
        self._profiles = profiles or ScannerProfiles()
        self._backtest_engine = BacktestEngine(adapter, self)
        self.strategy_name = "breakout"

    async def run(self, symbols: Sequence[str], profile: str) -> list[ScannerSignal]:
        profile_cfg = self._profiles.get(profile)
        results: list[ScannerSignal] = []
        for symbol in symbols:
            cache_key = make_cache_key("breakout:signal", [symbol, profile])
            cached = await self._cache.get_cached_signals(cache_key)
            if cached:
                results.extend(cached)
                continue

            klines = await self._adapter.fetch_klines(
                symbol=symbol,
                interval=self.DEFAULT_INTERVAL,
                limit=self._required_bars(),
            )
            if len(klines) < self._required_bars() // 2:
                continue

            evaluation = self._evaluate_klines(klines)
            if evaluation is None:
                continue

            derivatives = await self._adapter.fetch_derivatives(symbol)
            if not self._derivatives_filter.validate(derivatives):
                continue

            entry_price = float(klines[-1]["close"])
            signal = ScannerSignal(
                symbol=symbol,
                timeframe=self.DEFAULT_INTERVAL,
                direction=evaluation.direction,
                entry=entry_price,
                indicators=IndicatorSnapshot(
                    ema_fast=None,
                    ema_slow=None,
                    adx=None,
                    volume_z=None,
                ),
                derivatives=derivatives,
                confidence=evaluation.confidence,
                reasons=tuple(evaluation.reasons),
            )

            if not self._risk_manager.allow_signal(signal):
                continue

            signal.risk = self._risk_calculator.evaluate(
                signal,
                profile,
                context=RiskContext(atr=evaluation.atr, entry_price=entry_price),
            )
            if not self._context_filter.accept(signal):
                continue

            self._risk_manager.record_signal(signal)
            await self._cache.set_cached_signals(cache_key, [signal], ttl=profile_cfg.cache_ttl)
            results.append(signal)
        return results

    def get_cache_ttl(self, profile: str) -> int:
        return self._profiles.get(profile).cache_ttl

    def get_backtest_engine(self) -> BacktestEngine:
        return self._backtest_engine

    def generate_backtest_signals(self, symbol: str, klines: Sequence[dict], profile: str) -> list[ScannerSignal]:
        min_window = self._required_bars()
        results: list[ScannerSignal] = []
        if len(klines) < min_window:
            return results
        profile_cfg = self._profiles.get(profile)
        for end in range(min_window, len(klines) + 1):
            window = klines[:end]
            evaluation = self._evaluate_klines(window)
            if evaluation is None:
                continue
            entry_price = float(window[-1]["close"])
            signal = ScannerSignal(
                symbol=symbol,
                timeframe=self.DEFAULT_INTERVAL,
                direction=evaluation.direction,
                entry=entry_price,
                indicators=IndicatorSnapshot(),
                derivatives=None,
            )
            signal.risk = self._risk_calculator.evaluate(
                signal,
                profile,
                context=RiskContext(atr=evaluation.atr, entry_price=entry_price),
            )
            if self._context_filter.accept(signal):
                results.append(signal)
        return results

    @staticmethod
    def _required_bars() -> int:
        return 200

    def _evaluate_klines(self, klines: Sequence[dict]) -> BreakoutEvaluation | None:
        df = pd.DataFrame(klines)
        if df.empty:
            return None
        highs = df["high"].astype(float)
        lows = df["low"].astype(float)
        closes = df["close"].astype(float)
        volumes = df["volume"].astype(float)

        rolling_max = highs.rolling(window=50).max()
        rolling_min = lows.rolling(window=50).min()
        atr = self._calculate_atr(highs, lows, closes, period=14)
        volume_avg = volumes.rolling(window=20).mean()

        if rolling_max.isna().iloc[-1] or atr.isna().iloc[-1] or volume_avg.isna().iloc[-1]:
            return None

        last_close = closes.iloc[-1]
        last_high = rolling_max.iloc[-1]
        last_low = rolling_min.iloc[-1]
        last_atr = atr.iloc[-1]
        last_volume = volumes.iloc[-1]
        avg_volume = volume_avg.iloc[-1]

        reasons: list[str] = []
        direction: SignalDirection | None = None
        confidence = 55

        if last_close > last_high and last_volume > avg_volume * 1.5:
            direction = SignalDirection.LONG
            reasons.append(f"Breakout above {last_high:.2f}")
            reasons.append(f"Volume spike {last_volume:.0f}>{avg_volume:.0f}")
            confidence += 15
        elif last_close < last_low and last_volume > avg_volume * 1.5:
            direction = SignalDirection.SHORT
            reasons.append(f"Breakdown below {last_low:.2f}")
            reasons.append(f"Volume spike {last_volume:.0f}>{avg_volume:.0f}")
            confidence += 15
        else:
            return None

        if last_atr <= 0:
            return None

        confidence = min(confidence, 95)
        breakout_level = last_high if direction is SignalDirection.LONG else last_low
        reasons.append(f"ATR {last_atr:.2f}")

        return BreakoutEvaluation(
            direction=direction,
            breakout_level=float(breakout_level),
            atr=float(last_atr),
            reasons=reasons,
            confidence=confidence,
        )

    @staticmethod
    def _calculate_atr(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int) -> pd.Series:
        prev_close = closes.shift(1)
        tr = pd.concat(
            [
                highs - lows,
                (highs - prev_close).abs(),
                (lows - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()


__all__ = ["BreakoutScanner"]

