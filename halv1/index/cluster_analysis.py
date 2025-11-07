from __future__ import annotations

# mypy: ignore-errors
import logging
import math
import random
import uuid
from typing import Dict, List, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - defensive
    faiss = None

from .cluster_model import Cluster, cosine_similarity

logger = logging.getLogger(__name__)


class ClusterAnalysisMixin:
    """Analysis helpers used by :class:`ClusterManager`."""

    # ------------------------------------------------------------------
    def _build_doc_graph(self) -> None:
        """Rebuild similarity graph using ``k_neighbors`` nearest neighbours."""
        self._doc_graph = {e.chunk_id: set() for e in self._entries}
        n = len(self._entries)
        if n < 2:
            return
        if faiss is not None and self._faiss_index is not None:
            xb_ptr = self._faiss_index.get_xb()
            matrix = np.array(
                faiss.rev_swig_ptr(
                    xb_ptr, self._faiss_index.ntotal * self._faiss_index.d
                ),
                dtype="float32",
            ).reshape(self._faiss_index.ntotal, self._faiss_index.d)
            hnsw = faiss.IndexHNSWFlat(matrix.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
            hnsw.hnsw.efSearch = max(32, self.k_neighbors * 2)
            hnsw.add(matrix)
            k = min(self.k_neighbors + 1, n)
            _, idx = hnsw.search(matrix, k)
            for i, entry in enumerate(self._entries):
                neigh = [int(j) for j in idx[i] if j != i][: self.k_neighbors]
                self._doc_graph[entry.chunk_id] = {
                    self._entries[j].chunk_id for j in neigh if 0 <= j < n
                }
            return
        matrix = np.array([e.embedding for e in self._entries])
        sims = matrix @ matrix.T
        norms = np.linalg.norm(matrix, axis=1)
        denom = norms[:, None] * norms[None, :]
        denom[denom == 0] = 1.0
        cosine = sims / denom
        for i, entry in enumerate(self._entries):
            order = np.argsort(cosine[i])[::-1]
            neighbors = [j for j in order if j != i][: self.k_neighbors]
            self._doc_graph[entry.chunk_id] = {
                self._entries[j].chunk_id for j in neighbors
            }

    # ------------------------------------------------------------------
    def rebuild_clusters(
        self,
        *,
        k: int | None = None,
        change_threshold: float = 0.5,
        max_iter: int = 100,
    ) -> None:
        """Recompute clusters from scratch using k-means++."""

        if not self._entries:
            self.clusters.clear()
            self._doc_graph.clear()
            self._centroid_history.clear()
            return

        matrix = np.array([e.embedding for e in self._entries])
        n = len(matrix)
        if k is None or k <= 0 or k > n:
            k = max(1, int(round(math.sqrt(n))))

        # k-means++ initialisation
        centers = [matrix[random.randrange(n)]]
        while len(centers) < k:
            dists = np.array(
                [min(np.linalg.norm(x - c) ** 2 for c in centers) for x in matrix]
            )
            if dists.sum() == 0:
                centers.append(matrix[random.randrange(n)])
            else:
                probs = dists / dists.sum()
                r = random.random()
                idx = int(np.searchsorted(np.cumsum(probs), r))
                centers.append(matrix[idx])
        centers = np.array(centers)

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dists = np.linalg.norm(matrix[:, None, :] - centers[None, :, :], axis=2)
            new_labels = dists.argmin(axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for j in range(k):
                members = matrix[labels == j]
                if len(members):
                    centers[j] = members.mean(axis=0)

        old_centroids = {cid: list(cl.centroid) for cid, cl in self.clusters.items()}
        self.clusters.clear()
        self._centroid_history.clear()

        for j in range(k):
            idxs = np.where(labels == j)[0]
            if not len(idxs):
                continue
            members = [self._entries[i] for i in idxs]
            cid = str(uuid.uuid4())
            cluster = Cluster(id=cid, members=members)
            cluster.recompute_centroid()
            cluster.recompute_medoid()
            self.clusters[cid] = cluster
            self._update_centroid_history(cluster)

        # rebuild similarity graph and drop clusters without members
        self.clusters = {cid: cl for cid, cl in self.clusters.items() if cl.members}
        self._build_doc_graph()

        # log change-points
        new_centroids = [cl.centroid for cl in self.clusters.values() if cl.centroid]
        for cid, prev in old_centroids.items():
            if not prev or not new_centroids:
                continue
            dists = [
                np.linalg.norm(np.array(prev) - np.array(c)) for c in new_centroids
            ]
            if dists and min(dists) > change_threshold:
                logger.info("change-point detected for cluster %s", cid)

    # ------------------------------------------------------------------
    def recompute_pagerank(
        self, *, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6
    ) -> None:
        """Recompute PageRank scores for clusters using the document graph."""
        if not self._doc_graph:
            for cl in self.clusters.values():
                cl.pagerank = 0.0
            return
        nodes = list(self._doc_graph.keys())
        n = len(nodes)
        pr = {node: 1.0 / n for node in nodes}
        for _ in range(max_iter):
            new_pr: Dict[str, float] = {}
            for node in nodes:
                rank_sum = 0.0
                for src in nodes:
                    outs = self._doc_graph.get(src, set())
                    if node in outs and outs:
                        rank_sum += pr[src] / len(outs)
                new_pr[node] = (1 - damping) / n + damping * rank_sum
            diff = sum(abs(new_pr[n] - pr[n]) for n in nodes)
            pr = new_pr
            if diff < tol:
                break
        for cluster in self.clusters.values():
            if cluster.members:
                total = sum(pr.get(m.chunk_id, 0.0) for m in cluster.members)
                cluster.pagerank = total / len(cluster.members)
            else:
                cluster.pagerank = 0.0

    # ------------------------------------------------------------------
    def summarise(self, summariser, *, max_tokens: int = 5) -> None:
        """Generate summaries for all clusters using ``summariser``."""
        for cluster in self.clusters.values():
            texts = [m.text for m in cluster.members[:max_tokens]]
            try:
                cluster.summary = summariser.summarize(texts)
            except Exception:  # pragma: no cover - defensive
                cluster.summary = ""

    # ------------------------------------------------------------------
    def get_insights(
        self,
        query_emb: Sequence[float],
        top_k: int = 5,
        *,
        cluster_weight: Dict[str, float] | None = None,
        bm25_score: Dict[str, float] | None = None,
        w1: float = 0.4,
        w2: float = 0.3,
        w3: float = 0.2,
        w4: float = 0.1,
        w5: float = 0.1,
        w6: float = 0.1,
    ) -> List[Dict[str, str]]:
        """Return top ``top_k`` cluster insights matching ``query_emb``."""
        cluster_weight = cluster_weight or {}
        bm25_score = bm25_score or {}
        scored: List[tuple[float, Cluster]] = []
        sims: List[float] = []
        prs: List[float] = []
        fresh_vals: List[float] = []
        qualities: List[float] = []
        weights: List[float] = []
        bm25_vals: List[float] = []
        tmp: List[tuple[Cluster, float, float, float, float, float, float]] = []
        for cluster in self.clusters.values():
            if not cluster.centroid:
                continue
            sim = cosine_similarity(cluster.centroid, query_emb)
            pr = cluster.pagerank
            fr = cluster.freshness
            sq = cluster.source_quality
            wt = cluster_weight.get(cluster.id, 0.0)
            bm = bm25_score.get(cluster.id, 0.0)
            sims.append(sim)
            prs.append(pr)
            fresh_vals.append(fr)
            qualities.append(sq)
            weights.append(wt)
            bm25_vals.append(bm)
            tmp.append((cluster, sim, pr, fr, sq, wt, bm))
        if not tmp:
            return []
        max_sim = max(sims) if sims else 0.0
        max_pr = max(prs) if prs else 0.0
        max_fr = max(fresh_vals) if fresh_vals else 0.0
        max_sq = max(qualities) if qualities else 0.0
        max_wt = max(weights) if weights else 0.0
        max_bm = max(bm25_vals) if bm25_vals else 0.0
        for cluster, sim, pr, fr, sq, wt, bm in tmp:
            score = (
                w1 * (sim / max_sim if max_sim else 0.0)
                + w2 * (pr / max_pr if max_pr else 0.0)
                + w3 * (fr / max_fr if max_fr else 0.0)
                + w4 * (sq / max_sq if max_sq else 0.0)
                + w5 * (wt / max_wt if max_wt else 0.0)
                + w6 * (bm / max_bm if max_bm else 0.0)
            )
            scored.append((score, cluster))
        scored.sort(key=lambda x: x[0], reverse=True)
        insights: List[Dict[str, str]] = []
        for _, cluster in scored[:top_k]:
            insights.append(
                {
                    "id": cluster.id,
                    "summary": cluster.summary or "",
                    "medoid": cluster.medoid.text if cluster.medoid else "",
                }
            )
        return insights

    # ------------------------------------------------------------------
    def mark_noise(self, doc_id: str, decay: float = 0.5) -> None:
        """Reduce weight of document identified as noise and speed up decay."""
        if not doc_id:
            return
        current = self.doc_weights.get(doc_id, 1.0)
        self.doc_weights[doc_id] = current * decay
        entry = next((e for e in self._entries if e.chunk_id == doc_id), None)
        if entry:
            entry.timestamp -= self.weight_half_life

    # ------------------------------------------------------------------
    def apply_decay(self, now: float) -> None:
        """Apply exponential decay to document weights using half-life."""
        half_life = self.weight_half_life
        if half_life <= 0:
            return
        for doc_id, weight in list(self.doc_weights.items()):
            entry = next((e for e in self._entries if e.chunk_id == doc_id), None)
            if entry is None:
                continue
            elapsed = now - getattr(entry, "timestamp", now)
            if elapsed <= 0:
                entry.timestamp = now
                continue
            factor = 0.5 ** (elapsed / half_life)
            self.doc_weights[doc_id] = weight * factor
            entry.timestamp = now
