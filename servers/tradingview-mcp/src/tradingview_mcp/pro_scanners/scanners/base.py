"""Base scanner functionality."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..models import ScannerSignal


class BaseScanner(ABC):
    @abstractmethod
    async def run(self, symbols: Sequence[str], profile: str) -> list[ScannerSignal]:
        ...

    @abstractmethod
    def get_cache_ttl(self, profile: str) -> int:
        ...

    def get_backtest_engine(self):
        raise NotImplementedError

    def generate_backtest_signals(self, symbol: str, klines: Sequence[dict], profile: str) -> list[ScannerSignal]:
        raise NotImplementedError


__all__ = ["BaseScanner"]
