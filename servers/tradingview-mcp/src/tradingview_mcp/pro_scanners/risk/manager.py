"""Risk manager applying limits and cooldowns."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

from ..models import ScannerSignal


@dataclass(slots=True)
class RiskLimits:
    cooldown_seconds: float = 180.0
    max_signals_window: int = 12
    window_seconds: float = 900.0  # 15 minutes


class RiskManager:
    """Tracks cooldowns and rolling limits for generated signals."""

    def __init__(self, limits: RiskLimits | None = None):
        self._limits = limits or RiskLimits()
        self._last_trigger: dict[str, float] = {}
        self._recent: Deque[Tuple[float, str]] = deque()

    def allow_signal(self, signal: ScannerSignal) -> bool:
        now = time.time()
        limits = self._limits

        last = self._last_trigger.get(signal.symbol)
        if last and now - last < limits.cooldown_seconds:
            return False

        self._prune(now)

        if limits.max_signals_window > 0 and len(self._recent) >= limits.max_signals_window:
            return False

        return True

    def record_signal(self, signal: ScannerSignal) -> None:
        now = time.time()
        self._last_trigger[signal.symbol] = now
        self._recent.append((now, signal.symbol))
        self._prune(now)

    def reset_symbol(self, symbol: str) -> None:
        self._last_trigger.pop(symbol, None)
        self._recent = deque([(ts, sym) for ts, sym in self._recent if sym != symbol])

    def _prune(self, now: float) -> None:
        window = self._limits.window_seconds
        while self._recent and now - self._recent[0][0] > window:
            self._recent.popleft()


__all__ = ["RiskManager", "RiskLimits"]
