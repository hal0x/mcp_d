"""Common vector mathematics utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine similarity between two vectors.

    If either vector has zero norm the result is ``0.0``.
    """
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    if va.shape != vb.shape:
        dim = max(len(va), len(vb))
        if len(va) < dim:
            va = np.pad(va, (0, dim - len(va)))
        if len(vb) < dim:
            vb = np.pad(vb, (0, dim - len(vb)))
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(va @ vb / (na * nb))


__all__ = ["cosine_similarity"]
