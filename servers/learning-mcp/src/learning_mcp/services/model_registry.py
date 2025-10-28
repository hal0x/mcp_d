"""Simple model registry for tracking decision profiles and metrics."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ..models import DecisionProfile, TrainingResult

REGISTRY_PATH = Path("servers/learning-mcp/data/model_registry.json")


def register_profile(profile: DecisionProfile, result: TrainingResult) -> None:
    """Persist profile metadata for traceability."""
    entry = {
        "profile_id": profile.profile_id,
        "version": profile.version,
        "created_at": profile.created_at.isoformat(),
        "trained_on_samples": profile.trained_on_samples,
        "confidence_score": profile.confidence_score,
        "performance_metrics": profile.performance_metrics,
        "training_duration": result.training_duration,
        "validation_score": result.validation_score,
        "registered_at": datetime.utcnow().isoformat(),
    }

    registry = _load_registry()
    registry.append(entry)
    _save_registry(registry)


def _load_registry() -> list[Dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        return []
    return json.loads(REGISTRY_PATH.read_text())


def _save_registry(entries: list[Dict[str, Any]]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(entries, indent=2))
