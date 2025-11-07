from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .cross_encoder import CrossEncoder


@dataclass
class Candidate:
    """Represent a document candidate returned from initial search."""

    node_id: str
    content: str
    embedding: Sequence[float]
    score: float


@dataclass
class RankedResult:
    """Final ranked result produced by the pipeline."""

    node_id: str
    fast_score: float
    score: float


class RetrievalError(Exception):
    """Raised when the retrieval pipeline encounters an unrecoverable error."""


class RetrievalPipeline:
    """Multi-stage retrieval pipeline.

    The pipeline executes the following stages:

    1. Candidate generation using both HNSW and FTS5 indices.
    2. Induction of a subgraph over the candidate set.
    3. Lightweight ranking via a simple PPR-style scoring (``fast_score``).
    4. Cross-encoder reranking of candidates.
    5. Diversified selection using a basic MMR procedure.
    """

    def __init__(
        self,
        hnsw_index,
        fts5_index,
        graph,
        cross_encoder: CrossEncoder | None = None,
    ) -> None:
        self.hnsw_index = hnsw_index
        self.fts5_index = fts5_index
        self.graph = graph
        self.cross_encoder = cross_encoder

    def _merge_candidates(
        self, cand_lists: Iterable[Sequence[Candidate]]
    ) -> List[Candidate]:
        """Merge candidate lists keeping the best score for duplicates."""

        merged: Dict[str, Candidate] = {}
        for cand_list in cand_lists:
            for cand in cand_list:
                existing = merged.get(cand.node_id)
                if existing is None or cand.score > existing.score:
                    merged[cand.node_id] = cand
        return list(merged.values())

    def generate_candidates(self, query: str, top_k: int = 50) -> List[Candidate]:
        """Return union of HNSW and FTS5 candidates for ``query``."""

        hnsw = self.hnsw_index.search(query, top_k)
        fts5 = self.fts5_index.search(query, top_k)
        return self._merge_candidates([hnsw, fts5])

    def induce_subgraph(self, candidates: Sequence[Candidate]):
        """Return an induced subgraph for ``candidates``."""

        node_ids = [c.node_id for c in candidates]
        return self.graph.induce(node_ids)

    def _ppr_rank(
        self, subgraph, candidates: Sequence[Candidate]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Return PPR, recency and centrality components for ``candidates``."""

        adjacency: Dict[str, Dict[str, float]] = {}
        timestamps: Dict[str, float] = {}
        for cand in candidates:
            data = subgraph.get(cand.node_id, {})
            if isinstance(data, dict):
                neighbors = list(data.get("neighbors", []))
                weights = data.get("weights", {})
                ts = data.get("timestamp")
                if ts is not None:
                    timestamps[cand.node_id] = float(ts)
            else:  # pragma: no cover - defensive
                neighbors = list(data)
                weights = {}
            adjacency[cand.node_id] = {
                nb: float(weights.get(nb, 1.0)) for nb in neighbors
            }

        node_ids = list(adjacency.keys())
        if not node_ids:
            return {}

        degrees = {nid: sum(wts.values()) for nid, wts in adjacency.items()}
        max_deg = max(degrees.values(), default=1.0)
        centrality = {
            nid: (degrees[nid] / max_deg if max_deg else 0.0) for nid in node_ids
        }

        if timestamps:
            max_ts = max(timestamps.values())
            min_ts = min(timestamps.values())
            span = max_ts - min_ts
            recency = {
                nid: ((ts - min_ts) / span if span > 0 else 0.0)
                for nid, ts in timestamps.items()
            }
        else:
            recency = {nid: 0.0 for nid in node_ids}

        alpha = 0.85
        epsilon = 1e-4
        n = len(node_ids)
        p = {nid: 0.0 for nid in node_ids}
        r = {nid: 1.0 / n for nid in node_ids}
        queue: List[str] = node_ids.copy()
        while queue:
            nid = queue.pop()
            residual = r[nid]
            if residual < epsilon:
                continue
            p[nid] += alpha * residual
            push = (1 - alpha) * residual
            r[nid] = 0.0
            deg = degrees.get(nid, 0.0)
            if deg == 0.0:
                continue
            for nb, w in adjacency[nid].items():
                share = push * (w / deg)
                r[nb] = r.get(nb, 0.0) + share
                if r[nb] >= epsilon and nb not in queue:
                    queue.append(nb)

        max_p = max(p.values(), default=1.0)
        ppr = {nid: (p[nid] / max_p if max_p else 0.0) for nid in node_ids}

        return ppr, recency, centrality

    def beam_search(
        self,
        subgraph,
        candidates: Sequence[Candidate],
        fast_scores: Dict[str, float],
        *,
        beam_width: int = 5,
        max_depth: int = 2,
    ) -> List[Candidate]:
        """Expand candidate paths using a simple beam search guided by ``fast_scores``.

        Only candidates present in ``candidates`` are considered during expansion.
        """

        candidate_map = {c.node_id: c for c in candidates}
        beams: List[Tuple[List[str], float]] = []
        for cand in candidates:
            heuristic = fast_scores.get(cand.node_id, 0.0)
            beams.append(([cand.node_id], heuristic))
        beams.sort(key=lambda x: x[1], reverse=True)
        beams = beams[:beam_width]

        visited = {nid for path, _ in beams for nid in path}
        for _ in range(max_depth):
            new_beams: List[Tuple[List[str], float]] = []
            for path, _ in beams:
                last = path[-1]
                data = subgraph.get(last, [])
                if isinstance(data, dict):
                    neighbors = list(data.get("neighbors", []))
                else:
                    neighbors = list(data)
                for nb in neighbors:
                    if nb in visited or nb not in candidate_map:
                        continue
                    score = fast_scores.get(nb, 0.0)
                    new_beams.append((path + [nb], score))
                    visited.add(nb)
            if not new_beams:
                break
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        ordered: List[str] = []
        for path, _ in beams:
            for nid in path:
                if nid not in ordered:
                    ordered.append(nid)
        return [candidate_map[nid] for nid in ordered]

    def _cross_encode(
        self, query: str, candidates: Sequence[Candidate]
    ) -> Dict[str, float]:
        """Return cross-encoder scores for ``candidates``."""

        if self.cross_encoder is None:
            self.cross_encoder = CrossEncoder()
        scores: Dict[str, float] = {}
        for cand in candidates:
            scores[cand.node_id] = self.cross_encoder.score(query, cand.content)
        return scores

    @staticmethod
    def _similarity(a: Sequence[float], b: Sequence[float]) -> float:
        """Return cosine similarity between two embeddings."""

        va = np.array(a)
        vb = np.array(b)
        if not va.any() or not vb.any():
            return 0.0
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

    def _mmr(
        self,
        ranked_candidates: Sequence[Candidate],
        fast_scores: Dict[str, float],
        final_scores: Dict[str, float],
        *,
        top_k: int = 20,
        lambda_: float = 0.7,
    ) -> List[RankedResult]:
        """Select ``top_k`` candidates using a simple MMR strategy."""

        remaining = list(ranked_candidates)
        selected: List[Candidate] = []
        results: List[RankedResult] = []
        while remaining and len(results) < top_k:
            best = None
            best_score = -float("inf")
            for cand in remaining:
                diversity = 0.0
                if selected:
                    diversity = max(
                        self._similarity(cand.embedding, s.embedding) for s in selected
                    )
                score = lambda_ * final_scores[cand.node_id] - (1 - lambda_) * diversity
                if score > best_score:
                    best = cand
                    best_score = score
            assert best is not None
            remaining.remove(best)
            selected.append(best)
            results.append(
                RankedResult(
                    node_id=best.node_id,
                    fast_score=fast_scores[best.node_id],
                    score=final_scores[best.node_id],
                )
            )
        return results

    def run(
        self, query: str, *, top_k: int = 20, candidate_k: int = 50
    ) -> List[RankedResult]:
        """Execute the full retrieval pipeline for ``query``."""

        candidates = self.generate_candidates(query, candidate_k)
        subgraph = self.induce_subgraph(candidates)
        ppr, recency, centrality = self._ppr_rank(subgraph, candidates)
        fast_scores = {
            cand.node_id: (
                0.55 * cand.score
                + 0.20 * ppr.get(cand.node_id, 0.0)
                + 0.15 * recency.get(cand.node_id, 0.0)
                + 0.10 * centrality.get(cand.node_id, 0.0)
            )
            for cand in candidates
        }
        candidates = self.beam_search(subgraph, candidates, fast_scores)
        fast_scores = {c.node_id: fast_scores[c.node_id] for c in candidates}
        ce_scores = self._cross_encode(query, candidates)
        final_scores = {
            cand.node_id: fast_scores[cand.node_id] + ce_scores[cand.node_id]
            for cand in candidates
        }
        return self._mmr(candidates, fast_scores, final_scores, top_k=top_k)
