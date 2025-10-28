"""Momentum scanner implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..adapters.base import BaseMarketDataAdapter
from ..filters.derivatives import DerivativesFilter
from ..filters.context import ContextFilter
from ..models import DerivativeSnapshot, IndicatorSnapshot, ScannerSignal, SignalDirection
from ..profiles import ScannerProfiles
from ..risk.calculator import RiskCalculator, RiskContext
from ..risk.manager import RiskManager
from ..storage.redis_cache import RedisCache
from ..utils import make_cache_key
from ..backtesting.engine import BacktestEngine
from ..alerts.router import AlertRouter
from .base import BaseScanner


@dataclass(slots=True)
class MomentumEvaluation:
    direction: SignalDirection
    ema_fast: float
    ema_slow: float
    adx: float
    volume_z: float
    atr: float
    reasons: list[str]
    confidence: int


class MomentumScanner(BaseScanner):
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
        self.strategy_name = "momentum"

    async def run(self, symbols: Sequence[str], profile: str) -> list[ScannerSignal]:
        profile_cfg = self._profiles.get(profile)
        results: list[ScannerSignal] = []
        for symbol in symbols:
            cache_key = make_cache_key("momentum:signal", [symbol, profile])
            cached = await self._cache.get_cached_signals(cache_key)
            if cached:
                results.extend(cached)
                continue

            klines = await self._adapter.fetch_klines(
                symbol=symbol,
                interval=self.DEFAULT_INTERVAL,
                limit=self._required_bars(profile_cfg.params.ema_slow),
            )
            if len(klines) < self._required_bars(profile_cfg.params.ema_slow) // 2:
                continue

            evaluation = self._evaluate_klines(klines, profile_cfg.params)
            if evaluation is None or evaluation.adx < profile_cfg.params.adx_min:
                continue

            if evaluation.volume_z < profile_cfg.params.vol_z_min:
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
                    ema_fast=evaluation.ema_fast,
                    ema_slow=evaluation.ema_slow,
                    adx=evaluation.adx,
                    volume_z=evaluation.volume_z,
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
                context=RiskContext(
                    atr=evaluation.atr,
                    entry_price=signal.entry,
                ),
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
        profile_cfg = self._profiles.get(profile)
        min_window = self._required_bars(profile_cfg.params.ema_slow)
        results: list[ScannerSignal] = []
        if len(klines) < min_window:
            return results

        for end in range(min_window, len(klines) + 1):
            window = klines[:end]
            evaluation = self._evaluate_klines(window, profile_cfg.params)
            if evaluation is None or evaluation.adx < profile_cfg.params.adx_min:
                continue
            if evaluation.volume_z < profile_cfg.params.vol_z_min:
                continue

            entry_price = float(window[-1]["close"])
            signal = ScannerSignal(
                symbol=symbol,
                timeframe=self.DEFAULT_INTERVAL,
                direction=evaluation.direction,
                entry=entry_price,
                indicators=IndicatorSnapshot(
                    ema_fast=evaluation.ema_fast,
                    ema_slow=evaluation.ema_slow,
                    adx=evaluation.adx,
                    volume_z=evaluation.volume_z,
                ),
                derivatives=DerivativeSnapshot(),
                confidence=evaluation.confidence,
                reasons=tuple(evaluation.reasons),
            )
            signal.risk = self._risk_calculator.evaluate(
                signal,
                profile,
                context=RiskContext(atr=evaluation.atr, entry_price=entry_price),
            )
            if not self._context_filter.accept(signal):
                continue
            results.append(signal)

        return results

    @staticmethod
    def _required_bars(slow_period: int) -> int:
        return max(200, slow_period * 4)

    def _evaluate_klines(self, klines: Sequence[dict], params) -> MomentumEvaluation | None:
        df = pd.DataFrame(klines)
        for column in ("open", "high", "low", "close", "volume"):
            if column not in df:
                return None
        closes = df["close"].astype(float)
        highs = df["high"].astype(float)
        lows = df["low"].astype(float)
        volumes = df["volume"].astype(float)

        ema_fast = closes.ewm(span=params.ema_fast, adjust=False).mean()
        ema_slow = closes.ewm(span=params.ema_slow, adjust=False).mean()
        if ema_fast.empty or ema_slow.empty:
            return None

        adx = self._calculate_adx(highs, lows, closes, period=14)
        volume_z = self._zscore(volumes, window=30)

        direction = self._detect_cross(ema_fast, ema_slow)
        # Если нет пересечений, но есть сильный тренд - тоже считаем сигналом
        if direction is None:
            direction = self._detect_trend_strength(ema_fast, ema_slow, params)

        reasons: list[str] = []
        latest_ema_fast = float(ema_fast.iloc[-1])
        latest_ema_slow = float(ema_slow.iloc[-1])
        latest_adx = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
        latest_volume_z = float(volume_z.iloc[-1]) if not pd.isna(volume_z.iloc[-1]) else 0.0

        reasons.append(f"EMA{params.ema_fast} {latest_ema_fast:.2f}")
        reasons.append(f"EMA{params.ema_slow} {latest_ema_slow:.2f}")
        reasons.append(f"ADX {latest_adx:.1f}")
        reasons.append(f"Volume z-score {latest_volume_z:.2f}")

        atr_series = self._calculate_atr(highs, lows, closes, period=14)
        latest_atr = float(atr_series.iloc[-1]) if not atr_series.empty and not pd.isna(atr_series.iloc[-1]) else 0.0
        confidence = 60
        if latest_adx > params.adx_min + 5:
            confidence += 10
        if latest_volume_z > params.vol_z_min + 0.5:
            confidence += 10
        confidence = min(95, confidence)

        if latest_atr <= 0:
            return None

        return MomentumEvaluation(
            direction=direction,
            ema_fast=latest_ema_fast,
            ema_slow=latest_ema_slow,
            adx=latest_adx,
            volume_z=latest_volume_z,
            atr=latest_atr,
            reasons=reasons,
            confidence=confidence,
        )

    @staticmethod
    def _detect_trend_strength(ema_fast: pd.Series, ema_slow: pd.Series, params) -> SignalDirection | None:
        """Определяет направление тренда по силе расхождения EMA."""
        if len(ema_fast) < 5 or len(ema_slow) < 5:
            return None
        
        # Проверяем последние 5 баров
        recent_fast = ema_fast.iloc[-5:]
        recent_slow = ema_slow.iloc[-5:]
        
        # Быстрая EMA выше медленной и растет
        if recent_fast.iloc[-1] > recent_slow.iloc[-1] * 1.002:  # 0.2% разница
            fast_growing = recent_fast.iloc[-1] > recent_fast.iloc[-3]  # растет за 3 бара
            if fast_growing:
                return SignalDirection.LONG
        
        # Быстрая EMA ниже медленной и падает  
        if recent_fast.iloc[-1] < recent_slow.iloc[-1] * 0.998:  # 0.2% разница
            fast_falling = recent_fast.iloc[-1] < recent_fast.iloc[-3]  # падает за 3 бара
            if fast_falling:
                return SignalDirection.SHORT
        
        return None

    @staticmethod
    def _detect_cross(ema_fast: pd.Series, ema_slow: pd.Series) -> SignalDirection | None:
        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return None
        prev_fast, prev_slow = ema_fast.iloc[-2], ema_slow.iloc[-2]
        curr_fast, curr_slow = ema_fast.iloc[-1], ema_slow.iloc[-1]
        if curr_fast > curr_slow and prev_fast <= prev_slow:
            return SignalDirection.LONG
        if curr_fast < curr_slow and prev_fast >= prev_slow:
            return SignalDirection.SHORT
        return None

    @staticmethod
    def _calculate_adx(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int) -> pd.Series:
        high = highs.to_numpy()
        low = lows.to_numpy()
        close = closes.to_numpy()

        tr1 = high[1:] - high[:-1]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        true_range = np.maximum.reduce([np.abs(high[1:] - low[1:]), tr2, tr3])
        true_range = np.insert(true_range, 0, np.nan)

        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.insert(plus_dm, 0, np.nan)
        minus_dm = np.insert(minus_dm, 0, np.nan)

        tr_smooth = MomentumScanner._wilder_smooth(true_range, period)
        safe_tr = np.where(tr_smooth == 0, np.nan, tr_smooth)
        plus_di = 100 * MomentumScanner._wilder_smooth(plus_dm, period) / safe_tr
        minus_di = 100 * MomentumScanner._wilder_smooth(minus_dm, period) / safe_tr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
        adx = MomentumScanner._wilder_smooth(dx, period)
        adx_series = pd.Series(adx, index=closes.index)
        return adx_series.bfill()

    @staticmethod
    def _wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
        result = np.zeros_like(values, dtype=float)
        result[:period] = np.nan
        initial = np.nanmean(values[:period]) if np.any(~np.isnan(values[:period])) else 0.0
        if period - 1 < len(result):
            result[period - 1] = initial
        for i in range(period, len(values)):
            if np.isnan(result[i - 1]):
                result[i] = values[i]
            else:
                result[i] = (result[i - 1] * (period - 1) + values[i]) / period
        return result

    @staticmethod
    def _calculate_atr(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int) -> pd.Series:
        high = highs.astype(float)
        low = lows.astype(float)
        close = closes.astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        return atr

    @staticmethod
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std(ddof=0)
        z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
        return z.fillna(0.0)


__all__ = ["MomentumScanner"]
