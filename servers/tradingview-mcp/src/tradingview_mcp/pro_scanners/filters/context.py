"""Context-based filters such as market regime checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..models import ScannerSignal


@dataclass(slots=True)
class ContextRules:
    min_confidence: int = 55
    max_leverage: float = 20.0
    require_stop_loss: bool = True


class ContextFilter:
    def __init__(self, rules: ContextRules | None = None):
        self._rules = rules or ContextRules()

    def accept(self, signal: ScannerSignal) -> bool:
        rules = self._rules
        if signal.confidence is not None and signal.confidence < rules.min_confidence:
            return False
        leverage = signal.risk.leverage
        if leverage is not None and leverage > rules.max_leverage:
            return False
        if rules.require_stop_loss and signal.risk.stop_loss is None:
            return False
        return True

    @property
    def rules(self) -> ContextRules:
        return self._rules


__all__ = ["ContextFilter", "ContextRules"]
