"""Backtesting engine."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Sequence

from ..models import BacktestRequest, BacktestResult, ScannerSignal
from ..scanners.base import BaseScanner
from .metrics import MetricsTracker


logger = logging.getLogger(__name__)


class BacktestEngine:
    """Coordinate historical signal generation for a scanner."""

    def __init__(self, adapter, scanner: BaseScanner, max_concurrency: int = 5):
        self._adapter = adapter
        self._scanner = scanner
        self._metrics = MetricsTracker()
        self._max_concurrency = max(1, max_concurrency)

    async def execute(self, request: BacktestRequest) -> BacktestResult:
        started_at = datetime.utcnow()
        semaphore = asyncio.Semaphore(self._max_concurrency)
        aggregated: list[ScannerSignal] = []

        async def _run_for_symbol(symbol: str) -> list[ScannerSignal]:
            async with semaphore:
                try:
                    klines: Sequence[dict] = await self._adapter.fetch_klines(
                        symbol=symbol,
                        interval=request.timeframe,
                        limit=request.lookback_limit,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Backtest fetch failed for %s: %s", symbol, exc)
                    return []
            try:
                return self._scanner.generate_backtest_signals(symbol, klines, request.profile)
            except Exception:  # noqa: BLE001
                logger.exception("Signal generation failed for %s", symbol)
                return []

        if request.symbols:
            results = await asyncio.gather(
                *[_run_for_symbol(symbol) for symbol in request.symbols],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    logger.exception("Backtest worker raised", exc_info=result)
                    continue
                aggregated.extend(result)

        aggregated.sort(key=lambda signal: signal.generated_at)
        metrics = self._metrics.calculate(aggregated)
        completed_at = datetime.utcnow()
        strategy = request.strategy or getattr(self._scanner, "strategy_name", None)
        return BacktestResult(
            signals=aggregated,
            metrics=metrics,
            started_at=started_at,
            completed_at=completed_at,
            profile=request.profile,
            timeframe=request.timeframe,
            symbol_universe=tuple(request.symbols),
            strategy=strategy,
        )


__all__ = ["BacktestEngine"]
