"""Formatting helpers for alerts."""

from __future__ import annotations

from ..models import AlertPayload, ScannerSignal


def build_alert_payload(signal: ScannerSignal) -> AlertPayload:
    signal_key = f"{signal.symbol.upper()}|{signal.timeframe}|{signal.generated_at.isoformat()}"
    return AlertPayload(
        symbol=signal.symbol,
        timeframe=signal.timeframe,
        direction=signal.direction,
        entry=signal.entry,
        leverage=signal.risk.leverage,
        confidence=signal.confidence,
        reasons=signal.reasons,
        metadata={
            "ema_fast": str(signal.indicators.ema_fast or ""),
            "ema_slow": str(signal.indicators.ema_slow or ""),
            "adx": str(signal.indicators.adx or ""),
            "volume_z": str(signal.indicators.volume_z or ""),
            "signal_generated_at": signal.generated_at.isoformat(),
            "signal_key": signal_key,
        },
    )


__all__ = ["build_alert_payload"]
