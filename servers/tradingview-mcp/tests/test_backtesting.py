import asyncio
from datetime import datetime, timedelta
from typing import Sequence

import pytest

from tradingview_mcp.pro_scanners.backtesting.engine import BacktestEngine
from tradingview_mcp.pro_scanners.backtesting.metrics import MetricsTracker
from tradingview_mcp.pro_scanners.models import (
    BacktestRequest,
    DerivativeSnapshot,
    IndicatorSnapshot,
    RiskAssessment,
    ScannerSignal,
    SignalDirection,
)
from tradingview_mcp.pro_scanners.scanners.base import BaseScanner


class StubAdapter:
    def __init__(self):
        self._active = 0
        self.max_active = 0

    async def startup(self) -> None:  # pragma: no cover - not used in test
        return None

    async def shutdown(self) -> None:  # pragma: no cover - not used in test
        return None

    async def fetch_klines(self, symbol: str, interval: str, limit: int) -> Sequence[dict]:
        self._active += 1
        self.max_active = max(self.max_active, self._active)
        await asyncio.sleep(0)
        self._active -= 1
        candle = {
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1_000.0,
        }
        return [candle for _ in range(limit)]

    async def fetch_derivatives(self, symbol: str) -> DerivativeSnapshot:  # pragma: no cover - unused
        return DerivativeSnapshot()


class StubScanner(BaseScanner):
    def __init__(self):
        self.requests: list[str] = []
        self._ts = datetime(2024, 1, 1)

    async def run(self, symbols: Sequence[str], profile: str):
        return []

    def get_cache_ttl(self, profile: str) -> int:
        return 60

    def get_backtest_engine(self):  # pragma: no cover - not used
        return None

    def generate_backtest_signals(self, symbol: str, klines: Sequence[dict], profile: str):
        self.requests.append(symbol)
        entry = float(klines[-1]["close"])
        current_ts = self._ts
        self._ts += timedelta(minutes=1)
        signal = ScannerSignal(
            symbol=symbol,
            timeframe="15m",
            direction=SignalDirection.LONG if len(self.requests) % 2 else SignalDirection.SHORT,
            entry=entry,
            confidence=70,
            indicators=IndicatorSnapshot(),
            derivatives=DerivativeSnapshot(),
            risk=RiskAssessment(
                leverage=2.0,
                position_size=150.0,
                stop_loss=entry * 0.95,
                take_profit=(entry * 1.1,),
            ),
            generated_at=current_ts,
        )
        return [signal]


@pytest.mark.asyncio
async def test_backtest_engine_limits_concurrency() -> None:
    adapter = StubAdapter()
    scanner = StubScanner()
    engine = BacktestEngine(adapter, scanner, max_concurrency=2)
    request = BacktestRequest(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"], profile="balanced", timeframe="15m")

    result = await engine.execute(request)

    assert len(result.signals) == 3
    assert scanner.requests == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert adapter.max_active <= 2
    assert list(result.symbol_universe) == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert result.metrics["total_signals"] == 3.0
    assert result.signals == sorted(result.signals, key=lambda s: s.generated_at)


def _make_signal(direction: SignalDirection, entry: float, stop: float, tp_values: Sequence[float]) -> ScannerSignal:
    return ScannerSignal(
        symbol="TEST",
        timeframe="15m",
        direction=direction,
        entry=entry,
        confidence=80,
        indicators=IndicatorSnapshot(),
        derivatives=DerivativeSnapshot(),
        risk=RiskAssessment(
            leverage=3.0,
            position_size=200.0,
            stop_loss=stop,
            take_profit=tuple(tp_values),
        ),
    )


def test_metrics_tracker_produces_rich_stats() -> None:
    tracker = MetricsTracker()
    signals = [
        _make_signal(SignalDirection.LONG, entry=100.0, stop=95.0, tp_values=(110.0,)),
        _make_signal(SignalDirection.SHORT, entry=90.0, stop=95.0, tp_values=(85.0,)),
    ]

    metrics = tracker.calculate(signals)

    assert metrics["total_signals"] == 2.0
    assert metrics["long_signals"] == 1.0
    assert metrics["short_signals"] == 1.0
    assert metrics["avg_confidence"] == 80.0
    assert metrics["avg_leverage"] == 3.0
    assert metrics["avg_position_size"] == 200.0
    assert metrics["symbols_covered"] == 1.0
    assert metrics["signals_per_symbol_avg"] == 2.0
    assert metrics["avg_stop_loss_pct"] > 0.0
    assert metrics["avg_take_profit_pct"] > 0.0
