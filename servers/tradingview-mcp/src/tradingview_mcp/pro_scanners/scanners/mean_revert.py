"""Mean reversion scanner implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..adapters.base import BaseMarketDataAdapter
from ..alerts.router import AlertRouter
from ..filters.derivatives import DerivativesFilter
from ..filters.context import ContextFilter
from ..models import DerivativeSnapshot, IndicatorSnapshot, ScannerSignal, SignalDirection
from ..profiles import ScannerProfiles
from ..risk.calculator import RiskCalculator, RiskContext
from ..risk.manager import RiskManager
from ..storage.redis_cache import RedisCache
from ..utils import make_cache_key
from ..backtesting.engine import BacktestEngine
from .base import BaseScanner


@dataclass(slots=True)
class MeanReversionEvaluation:
    direction: SignalDirection
    basis: float
    upper_band: float
    lower_band: float
    rsi: float
    atr: float
    reasons: list[str]
    confidence: int
    entry: float


class MeanRevertScanner(BaseScanner):
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
        self._bollinger_window = 20
        self.strategy_name = "mean_revert"
        self._backtest_engine = BacktestEngine(adapter, self)

    async def run(self, symbols: Sequence[str], profile: str) -> list[ScannerSignal]:
        profile_cfg = self._profiles.get(profile)
        results: list[ScannerSignal] = []
        for symbol in symbols:
            cache_key = make_cache_key("mean_revert:signal", [symbol, profile])
            cached = await self._cache.get_cached_signals(cache_key)
            if cached:
                results.extend(cached)
                continue

            klines = await self._adapter.fetch_klines(
                symbol=symbol,
                interval=self.DEFAULT_INTERVAL,
                limit=self._required_bars(),
            )
            if len(klines) < self._bollinger_window:
                continue

            evaluation = self._evaluate_klines(klines)
            if evaluation is None:
                continue

            derivatives = await self._adapter.fetch_derivatives(symbol)
            if not self._derivatives_filter.validate(derivatives):
                continue

            signal = ScannerSignal(
                symbol=symbol,
                timeframe=self.DEFAULT_INTERVAL,
                direction=evaluation.direction,
                entry=evaluation.entry,
                indicators=IndicatorSnapshot(
                    ema_fast=evaluation.basis,
                    ema_slow=evaluation.upper_band,
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
                context=RiskContext(atr=evaluation.atr, entry_price=signal.entry),
            )
            if not self._context_filter.accept(signal):
                continue
            self._risk_manager.record_signal(signal)
            await self._cache.set_cached_signals(cache_key, [signal], ttl=profile_cfg.cache_ttl)
            results.append(signal)
        return results

    def get_cache_ttl(self, profile: str) -> int:
        return self._profiles.get(profile).cache_ttl

    def get_backtest_engine(self):
        return self._backtest_engine

    def generate_backtest_signals(self, symbol: str, klines: Sequence[dict], profile: str) -> list[ScannerSignal]:
        min_window = self._required_bars()
        results: list[ScannerSignal] = []
        if len(klines) < min_window:
            return results

        for end in range(min_window, len(klines) + 1):
            window = klines[:end]
            evaluation = self._evaluate_klines(window)
            if evaluation is None:
                continue

            signal = ScannerSignal(
                symbol=symbol,
                timeframe=self.DEFAULT_INTERVAL,
                direction=evaluation.direction,
                entry=evaluation.entry,
                indicators=IndicatorSnapshot(
                    ema_fast=evaluation.basis,
                    ema_slow=evaluation.upper_band,
                    adx=None,
                    volume_z=None,
                ),
                derivatives=DerivativeSnapshot(),
                confidence=evaluation.confidence,
                reasons=tuple(evaluation.reasons),
            )
            signal.risk = self._risk_calculator.evaluate(
                signal,
                profile,
                context=RiskContext(atr=evaluation.atr, entry_price=evaluation.entry),
            )
            if not self._context_filter.accept(signal):
                continue
            results.append(signal)

        return results

    def _evaluate_klines(self, klines: Sequence[dict]) -> MeanReversionEvaluation | None:
        df = pd.DataFrame(klines)
        if df.empty:
            return None
        closes = df["close"].astype(float)
        highs = df["high"].astype(float)
        lows = df["low"].astype(float)

        basis = closes.rolling(window=self._bollinger_window).mean()
        std = closes.rolling(window=self._bollinger_window).std(ddof=0)
        upper = basis + 2 * std
        lower = basis - 2 * std

        if basis.isna().iloc[-1] or upper.isna().iloc[-1] or lower.isna().iloc[-1]:
            return None

        rsi = self._calculate_rsi(closes, period=14)
        if rsi.isna().iloc[-1]:
            return None

        price = closes.iloc[-1]
        latest_basis = float(basis.iloc[-1])
        latest_upper = float(upper.iloc[-1])
        latest_lower = float(lower.iloc[-1])
        latest_rsi = float(rsi.iloc[-1])

        direction: SignalDirection | None = None
        reasons: list[str] = []

        if price < latest_lower and latest_rsi < 35:
            direction = SignalDirection.LONG
            reasons.append(f"Price below lower band {latest_lower:.4f}")
            reasons.append(f"RSI oversold {latest_rsi:.1f}")
        elif price > latest_upper and latest_rsi > 65:
            direction = SignalDirection.SHORT
            reasons.append(f"Price above upper band {latest_upper:.4f}")
            reasons.append(f"RSI overbought {latest_rsi:.1f}")
        else:
            return None

        atr = self._calculate_atr(highs, lows, closes, period=14)
        latest_atr = float(atr.iloc[-1]) if not atr.empty and not pd.isna(atr.iloc[-1]) else 0.0

        confidence = 55
        if direction is SignalDirection.LONG and latest_rsi < 30:
            confidence += 10
        if direction is SignalDirection.SHORT and latest_rsi > 70:
            confidence += 10
        confidence = min(90, confidence)

        if latest_atr <= 0:
            return None

        entry = float(price)
        reasons.append(f"Basis {latest_basis:.4f}")

        return MeanReversionEvaluation(
            direction=direction,
            basis=latest_basis,
            upper_band=latest_upper,
            lower_band=latest_lower,
            rsi=latest_rsi,
            atr=latest_atr,
            reasons=reasons,
            confidence=confidence,
            entry=entry,
        )

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

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

    def _required_bars(self) -> int:
        return max(200, self._bollinger_window * 4)


__all__ = ["MeanRevertScanner"]
