"""Filters based on derivative market data."""

from __future__ import annotations

from dataclasses import dataclass
from ..models import DerivativeSnapshot


@dataclass(slots=True)
class DerivativesThresholds:
    min_open_interest: float | None = None
    max_funding_abs: float = 0.05
    min_cvd_abs: float | None = None


class DerivativesFilter:
    """Validates derivative context."""

    def __init__(self, thresholds: DerivativesThresholds | None = None):
        self._thresholds = thresholds or DerivativesThresholds()

    def validate(self, snapshot: DerivativeSnapshot) -> bool:
        t = self._thresholds
        if t.min_open_interest is not None and snapshot.open_interest is not None:
            if snapshot.open_interest < t.min_open_interest:
                return False
        if snapshot.funding_rate is not None and abs(snapshot.funding_rate) > t.max_funding_abs:
            return False
        if t.min_cvd_abs is not None:
            if snapshot.cvd is None or abs(snapshot.cvd) < t.min_cvd_abs:
                return False
        return True

    @property
    def thresholds(self) -> DerivativesThresholds:
        return self._thresholds


__all__ = ["DerivativesFilter", "DerivativesThresholds"]
