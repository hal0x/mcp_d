"""Feature engineering utilities for Learning MCP."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Tuple

import pandas as pd


def build_feature_matrix(
    facts: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, List[int]]:
    """Convert supervisor facts into feature matrix and labels.

    Returns a tuple (X, y) where X is a DataFrame and y is the label list.
    Label is 1 when majority execution steps succeeded.
    """

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        corr_id = fact.get("correlation_id")
        if corr_id:
            grouped[corr_id].append(fact)

    if not grouped:
        raise ValueError("No correlation groups found in facts")

    feature_rows: List[Dict[str, Any]] = []
    labels: List[int] = []

    for corr_id, group in grouped.items():
        plan = [f for f in group if f.get("kind") == "Fact:Plan"]
        exec_steps = [f for f in group if f.get("kind") == "Fact:Execution"]
        errors = [f for f in group if f.get("kind") == "Fact:Error"]
        outcome = [f for f in group if f.get("kind") == "Fact:Outcome"]

        if not exec_steps:
            continue

        success_count = sum(1 for step in exec_steps if step.get("payload", {}).get("success"))
        label = 1 if success_count >= len(exec_steps) / 2 else 0
        labels.append(label)

        durations = [step.get("payload", {}).get("duration", 0.0) for step in exec_steps]
        timestamps = [step.get("ts") for step in exec_steps]
        timestamps = [datetime.fromisoformat(ts) if isinstance(ts, str) else ts for ts in timestamps if ts]

        row = {
            "plan_steps": plan[0].get("payload", {}).get("plan_steps", len(exec_steps)) if plan else len(exec_steps),
            "avg_duration": mean(durations) if durations else 0.0,
            "max_duration": max(durations) if durations else 0.0,
            "error_count": len(errors),
            "retry_count": max(0, len(exec_steps) - (plan[0].get("payload", {}).get("plan_steps", len(exec_steps)) if plan else len(exec_steps))),
            "outcome_success": 1 if any(f.get("payload", {}).get("success") for f in outcome) else 0,
            "outcome_confidence": outcome[0].get("payload", {}).get("confidence", 0.0) if outcome else 0.0,
            "execution_span_seconds": _compute_exec_span(timestamps),
        }

        feature_rows.append(row)

    if not feature_rows:
        raise ValueError("No feature rows generated from facts")

    X = pd.DataFrame(feature_rows)
    X.fillna(0, inplace=True)
    return X, labels


def _compute_exec_span(timestamps: List[datetime]) -> float:
    if len(timestamps) < 2:
        return 0.0
    timestamps = sorted(timestamps)
    return (timestamps[-1] - timestamps[0]).total_seconds()
