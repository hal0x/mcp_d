"""Base interface for market data adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..models import DerivativeSnapshot


class BaseMarketDataAdapter(ABC):
    """Abstract market data adapter."""

    @abstractmethod
    async def startup(self) -> None:
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        ...

    @abstractmethod
    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> Sequence[dict]:
        ...

    @abstractmethod
    async def fetch_derivatives(self, symbol: str) -> DerivativeSnapshot:
        ...

