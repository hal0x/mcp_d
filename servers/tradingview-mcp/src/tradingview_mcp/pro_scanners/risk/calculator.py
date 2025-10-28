"""Risk calculation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..models import RiskAssessment, ScannerSignal, SignalDirection
from ..profiles import ScannerProfiles


@dataclass(slots=True)
class RiskContext:
    atr: float
    entry_price: float


class RiskCalculator:
    """Performs simplified position sizing."""

    def __init__(self, profiles: ScannerProfiles | None = None):
        self._profiles = profiles or ScannerProfiles()

    def evaluate(self, signal: ScannerSignal, profile: str, context: RiskContext) -> RiskAssessment:
        profile_cfg = self._profiles.get(profile)
        low, high = profile_cfg.risk.leverage_bounds
        leverage = max(low, min(high, (low + high) / 2))

        atr_stop = context.atr * profile_cfg.risk.atr_stop_mult if context.atr else 0.0
        if signal.direction is SignalDirection.LONG:
            stop_loss = max(signal.entry - atr_stop, 0.0) if atr_stop else None
            take_profit = tuple(signal.entry + context.atr * mult for mult in (1.5, 2.5)) if context.atr else ()
        else:
            stop_loss = signal.entry + atr_stop if atr_stop else None
            take_profit = tuple(signal.entry - context.atr * mult for mult in (1.5, 2.5)) if context.atr else ()

        position_size = None
        if context.atr and context.entry_price:
            risk_fraction = 0.01  # risk 1% of notional per trade
            risk_per_unit = context.atr * profile_cfg.risk.atr_stop_mult
            if risk_per_unit > 0:
                raw_size = (risk_fraction * context.entry_price) / risk_per_unit
                if isfinite(raw_size) and raw_size > 0:
                    position_size = raw_size * leverage

        risk = RiskAssessment(
            position_size=position_size,
            leverage=leverage,
            stop_loss=None,
            take_profit=(),
            notes=None,
        )
        risk.stop_loss = stop_loss
        risk.take_profit = take_profit
        return risk


__all__ = ["RiskCalculator", "RiskContext"]
