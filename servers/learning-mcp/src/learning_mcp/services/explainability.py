"""Explainability helpers for Learning MCP models."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.inspection import permutation_importance


def compute_permutation_importance(model, X, y, n_repeats: int = 5) -> Dict[str, float]:
    """Return permutation importance scores keyed by feature name."""
    if isinstance(X, list):
        X = np.array(X)
    if hasattr(X, "values"):
        feature_names = list(X.columns)
        data = X.values
    else:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        data = X

    result = permutation_importance(model, data, y, n_repeats=n_repeats, random_state=42)
    importances = {name: float(score) for name, score in zip(feature_names, result.importances_mean)}
    return importances
