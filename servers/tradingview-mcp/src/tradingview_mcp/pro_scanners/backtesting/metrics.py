"""Metrics tracker producing aggregate statistics."""

from __future__ import annotations

from typing import Iterable

from ..models import ScannerSignal, SignalDirection


class MetricsTracker:
    def calculate(self, signals: Iterable[ScannerSignal]) -> dict[str, float]:
        signals_list = list(signals)
        total = len(signals_list)
        metrics: dict[str, float] = {
            "total_signals": float(total),
            "long_signals": 0.0,
            "short_signals": 0.0,
            "long_ratio": 0.0,
            "short_ratio": 0.0,
            "avg_confidence": 0.0,
            "avg_leverage": 0.0,
            "avg_position_size": 0.0,
            "avg_stop_loss_pct": 0.0,
            "avg_take_profit_pct": 0.0,
            "signals_per_symbol_avg": 0.0,
            "symbols_covered": 0.0,
            "take_rate": 0.0,
        }
        if total == 0:
            return metrics

        long_count = sum(1 for s in signals_list if s.direction is SignalDirection.LONG)
        short_count = total - long_count
        metrics["long_signals"] = float(long_count)
        metrics["short_signals"] = float(short_count)
        metrics["long_ratio"] = float(round(long_count / total, 4))
        metrics["short_ratio"] = float(round(short_count / total, 4))

        confidence_values = [float(s.confidence) for s in signals_list if s.confidence is not None]
        if confidence_values:
            metrics["avg_confidence"] = float(round(sum(confidence_values) / len(confidence_values), 2))

        leverage_values = [s.risk.leverage for s in signals_list if s.risk.leverage is not None]
        if leverage_values:
            metrics["avg_leverage"] = float(
                round(sum(float(value) for value in leverage_values) / len(leverage_values), 2)
            )

        position_sizes = [s.risk.position_size for s in signals_list if s.risk.position_size is not None]
        if position_sizes:
            metrics["avg_position_size"] = float(
                round(sum(float(value) for value in position_sizes) / len(position_sizes), 2)
            )

        stop_loss_pct: list[float] = []
        take_profit_pct: list[float] = []
        for signal in signals_list:
            entry = signal.entry
            denom = abs(entry)
            if denom <= 0:
                continue
            if signal.risk.stop_loss is not None:
                stop_loss_pct.append(abs(entry - signal.risk.stop_loss) / denom * 100)
            if signal.risk.take_profit:
                take_profit_pct.extend(abs(tp - entry) / denom * 100 for tp in signal.risk.take_profit)
        if stop_loss_pct:
            metrics["avg_stop_loss_pct"] = float(round(sum(stop_loss_pct) / len(stop_loss_pct), 2))
        if take_profit_pct:
            metrics["avg_take_profit_pct"] = float(round(sum(take_profit_pct) / len(take_profit_pct), 2))

        unique_symbols = {signal.symbol for signal in signals_list}
        metrics["symbols_covered"] = float(len(unique_symbols))
        if unique_symbols:
            metrics["signals_per_symbol_avg"] = float(round(total / len(unique_symbols), 2))

        return metrics


__all__ = ["MetricsTracker"]
