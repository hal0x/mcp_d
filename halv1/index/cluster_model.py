"""Data structures and helpers for clustering vector entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from utils.vector_math import cosine_similarity

from .vector_index import VectorEntry


@dataclass
class Cluster:
    """Simple cluster of vector entries."""

    id: str
    members: List[VectorEntry] = field(default_factory=list)
    centroid: List[float] = field(default_factory=list)
    medoid: VectorEntry | None = None
    summary: str | None = None
    pagerank: float = 0.0
    freshness_ts: float = 0.0
    freshness: float = 0.0
    source_quality: float = 0.0
    timeline: List[VectorEntry] = field(default_factory=list)

    def recompute_centroid(self) -> None:
        if not self.members:
            self.centroid = []
            return
        matrix = np.array([m.embedding for m in self.members])
        self.centroid = matrix.mean(axis=0).tolist()

    def recompute_medoid(self) -> None:
        if not self.members:
            self.medoid = None
            return
        old_medoid = self.medoid
        matrix = np.array([m.embedding for m in self.members])
        sims = matrix @ matrix.T
        norms = np.linalg.norm(matrix, axis=1)
        denom = norms[:, None] * norms[None, :]
        denom[denom == 0] = 1.0  # avoid division by zero
        cosine = sims / denom
        dists = 1 - cosine
        totals = dists.sum(axis=1)
        idx = int(totals.argmin())
        self.medoid = self.members[idx]
        if self.medoid and (
            old_medoid is None or self.medoid.chunk_id != old_medoid.chunk_id
        ):
            self.timeline.append(self.medoid)


__all__ = ["Cluster", "cosine_similarity"]
