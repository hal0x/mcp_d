"""Risk management helpers ported from BNC project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

from .config import get_config

Side = Literal["long", "short", "LONG", "SHORT"]


@dataclass
class PositionSizeRequest:
    equity: float
    entry: float
    stop_loss: float
    side: Side


@dataclass
class PositionSizeResult:
    quantity: float
    effective_entry: float
    risk_amount: float


def _normalize_side(side: Side) -> Literal["long", "short"]:
    return "long" if str(side).lower() == "long" else "short"


def effective_entry_price(entry: float, side: Side) -> float:
    cfg = get_config()
    fee = cfg.taker_fee_pct
    slip = cfg.slippage_pct
    if _normalize_side(side) == "long":
        return entry * (1 + fee + slip)
    return entry * (1 - fee - slip)


def compute_position_size(request: PositionSizeRequest) -> PositionSizeResult:
    cfg = get_config()
    entry_eff = effective_entry_price(request.entry, request.side)
    risk_abs = request.equity * (cfg.risk_per_trade_pct / 100.0)
    sl_dist = abs(entry_eff - request.stop_loss)
    quantity = 0.0 if sl_dist <= 0 else risk_abs / sl_dist
    return PositionSizeResult(
        quantity=quantity, effective_entry=entry_eff, risk_amount=risk_abs
    )


def compute_rr(entry: float, tp: float, sl: float, side: Side) -> float:
    normalized = _normalize_side(side)
    if normalized == "long":
        risk = entry - sl
        reward = tp - entry
    else:
        risk = sl - entry
        reward = entry - tp
    if risk <= 0 or reward <= 0:
        return 0.0
    return reward / risk


def enforce_min_rr(
    entry: float, tp: float, sl: float, atr: float, side: Side
) -> Dict[str, float]:
    cfg = get_config()
    min_rr = cfg.tp_rr_min
    atr_mult = cfg.tp_atr_multiple
    current_rr = compute_rr(entry, tp, sl, side)
    normalized = _normalize_side(side)
    adjusted_tp = tp
    if current_rr < min_rr and atr > 0:
        if normalized == "long":
            cap_tp = entry + atr_mult * atr
            target_tp = entry + min_rr * abs(entry - sl)
            adjusted_tp = min(cap_tp, max(tp, target_tp))
        else:
            cap_tp = entry - atr_mult * atr
            target_tp = entry - min_rr * abs(entry - sl)
            adjusted_tp = max(cap_tp, min(tp, target_tp))
        current_rr = compute_rr(entry, adjusted_tp, sl, side)
    return {"tp": adjusted_tp, "rr": current_rr}


def should_halt(total_pnl_pct_today: float, consecutive_losses: int) -> bool:
    cfg = get_config()
    if total_pnl_pct_today <= -abs(cfg.daily_loss_cap_pct):
        return True
    if consecutive_losses >= cfg.max_consecutive_losses:
        return True
    return False


__all__ = [
    "PositionSizeRequest",
    "PositionSizeResult",
    "effective_entry_price",
    "compute_position_size",
    "compute_rr",
    "enforce_min_rr",
    "should_halt",
]
