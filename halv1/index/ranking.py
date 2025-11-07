"""Ranking helpers for combining vector and BM25 scores."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple


def combine_scores(
    vector_scores: Dict[str, float], bm25_hits: Iterable[Tuple[str, float]], weights: Counter[str]
) -> List[Tuple[str, float]]:
    """Combine cosine similarity scores with BM25 scores and per-id weights.

    Returns a list of (chunk_id, score) sorted descending by score.
    """
    bm25_scores: Dict[str, float] = {cid: score for cid, score in bm25_hits}
    combined_ids = set(vector_scores) | set(bm25_scores)
    scored: List[Tuple[str, float]] = []
    for cid in combined_ids:
        score = vector_scores.get(cid, 0.0) + bm25_scores.get(cid, 0.0)
        weight = weights.get(cid, 1)
        scored.append((cid, score * weight))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

