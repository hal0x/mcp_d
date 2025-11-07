"""Manage clusters of vector entries and derive insights.

This module provides a small in-memory clustering helper that operates on
:class:`VectorEntry` instances from :mod:`index.vector_index`.  It performs
three main steps:

1. **Noise filtering** – entries with low embedding norm are skipped.
2. **Deduplication** – entries that are near duplicates (cosine similarity
   above ``dup_threshold``) are ignored.
3. **Clustering** – remaining entries are grouped by cosine similarity to
   existing cluster centroids.  For each cluster we keep the centroid and the
   medoid (the most central member).

Once clusters are built, a summariser can be used to produce short textual
insights for each cluster.  These insights together with medoid examples can
serve as a high‑level context – a lightweight “long‑term memory” of the
information bubble.

The implementation is intentionally simple and synchronous so it can be easily
unit tested without external dependencies.  Real indexing code is expected to
provide pre‑embedded :class:`VectorEntry` objects.
"""

# mypy: ignore-errors

from __future__ import annotations

import logging
import math
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - defensive
    faiss = None

from .cluster_analysis import ClusterAnalysisMixin
from .cluster_model import Cluster, cosine_similarity
from .cluster_persistence import ClusterPersistenceMixin
from .vector_index import VectorEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ClusterManager(ClusterAnalysisMixin, ClusterPersistenceMixin):
    """Build clusters from vector entries and provide insight summaries."""

    def __init__(
        self,
        *,
        dup_threshold: float = 0.92,
        attach_threshold: float = 0.75,
        min_norm: float = 1e-6,
        k_neighbors: int = 5,
        source_msg_limit: int = 100,
    ) -> None:
        self.dup_threshold = dup_threshold
        self.attach_threshold = attach_threshold
        self.min_norm = min_norm
        self.k_neighbors = k_neighbors
        self.clusters: Dict[str, Cluster] = {}
        self._entries: List[VectorEntry] = []
        # keep short history of centroids (timestamp, vector) to detect drift
        self._centroid_history: Dict[str, List[Tuple[float, List[float]]]] = {}
        # document weights for feedback adjustments
        self.doc_weights: Dict[str, float] = {}
        # half-life in seconds for document weight decay
        self.weight_half_life: float = 86400.0
        # path where missing facts are logged
        self.missing_facts_path = Path("db/missing_facts.log")
        # similarity graph of documents: doc_id -> set of neighbor ids
        self._doc_graph: Dict[str, set[str]] = {}
        # optional faiss index for duplicate detection
        self._faiss_index = None
        # per-source message counters
        self.source_msg_counts: Dict[str, int] = {}
        self.source_msg_limit = source_msg_limit
        # archived document ids
        self.archived_docs: set[str] = set()

    # ------------------------------------------------------------------
    def _is_duplicate(self, emb: Sequence[float]) -> str | None:
        """Return ``chunk_id`` of near duplicate or ``None``."""
        if (
            faiss is not None
            and self._faiss_index is not None
            and self._faiss_index.ntotal
        ):
            vec = np.array(emb, dtype="float32").reshape(1, -1)
            faiss.normalize_L2(vec)
            sims, idx = self._faiss_index.search(vec, 1)
            if sims.size and sims[0][0] >= self.dup_threshold:
                target = int(idx[0][0])
                if 0 <= target < len(self._entries):
                    return self._entries[target].chunk_id
                return ""
            return None
        for e in self._entries:
            if cosine_similarity(e.embedding, emb) >= self.dup_threshold:
                return e.chunk_id
        return None

    def _closest_cluster(self, emb: Sequence[float]) -> Cluster | None:
        best: tuple[float, Cluster] | None = None
        for cluster in self.clusters.values():
            if not cluster.centroid:
                continue
            sim = cosine_similarity(cluster.centroid, emb)
            if sim >= self.attach_threshold and (best is None or sim > best[0]):
                best = (sim, cluster)
        return best[1] if best else None

    def _update_centroid_history(self, cluster: Cluster, max_len: int = 10) -> None:
        """Store centroid history with timestamps for drift detection."""
        hist = self._centroid_history.setdefault(cluster.id, [])
        hist.append((time.time(), list(cluster.centroid)))
        if len(hist) > max_len:
            del hist[0]

    def _compute_source_quality(self, metadata: Dict[str, str]) -> float:
        """Estimate source quality from ``metadata``.

        A very small heuristic based on ``source`` and ``chat`` fields.  The
        result is normalised to ``[0, 1]`` so it can be mixed with other
        ranking signals.
        """
        source = (metadata.get("source") or "").lower()
        chat = str(metadata.get("chat", "")).lower()

        # default baseline
        quality = 0.5
        if source in {"official", "news", "trusted"}:
            quality = 1.0
        elif source in {"user", "unknown"}:
            quality = 0.3
        elif source in {"social", "telegram", "chat"}:
            quality = 0.6

        if chat:
            quality += 0.05

        if quality < 0.0:
            return 0.0
        if quality > 1.0:
            return 1.0
        return quality

    def detect_drift(self, cluster: Cluster, *, threshold: float = 0.5) -> bool:
        """Detect centroid drift using a simple CUSUM over centroid shifts."""
        hist = self._centroid_history.get(cluster.id, [])
        if len(hist) < 2:
            return False
        # compute distances between successive centroids
        diffs = [
            float(np.linalg.norm(np.array(hist[i][1]) - np.array(hist[i - 1][1])))
            for i in range(1, len(hist))
        ]
        cusum = 0.0
        for d in diffs:
            cusum = max(0.0, cusum + d)
            if cusum > threshold:
                return True
        return False

    def _split_cluster(self, cluster: Cluster, entry: VectorEntry) -> None:
        """Split ``cluster`` by moving ``entry`` into a new cluster."""
        if entry not in cluster.members or len(cluster.members) <= 1:
            return
        cluster.members.remove(entry)
        cluster.recompute_centroid()
        cluster.recompute_medoid()
        hist = self._centroid_history.get(cluster.id, [])
        if hist:
            ts, _ = hist[-1]
            hist[-1] = (ts, list(cluster.centroid))
        cid = str(uuid.uuid4())
        new_cluster = Cluster(id=cid, members=[entry])
        new_cluster.recompute_centroid()
        new_cluster.recompute_medoid()
        self.clusters[cid] = new_cluster
        self._update_centroid_history(new_cluster)

    def _archive_source_docs(self, source: str) -> None:
        """Reduce weights and mark documents from ``source`` as archived."""
        for entry in self._entries:
            if (
                entry.metadata.get("source") == source
                and entry.chunk_id not in self.archived_docs
            ):
                current = self.doc_weights.get(entry.chunk_id, 1.0)
                self.doc_weights[entry.chunk_id] = current * 0.5
                self.archived_docs.add(entry.chunk_id)
                entry.metadata["archived"] = True

    # ------------------------------------------------------------------
    def _add_to_index(self, emb: Sequence[float]) -> None:
        """Add ``emb`` to the faiss index if available."""
        if faiss is None:
            return
        vec = np.array(emb, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        if self._faiss_index is None:
            self._faiss_index = faiss.IndexFlatIP(vec.shape[1])
        self._faiss_index.add(vec)

    def _rebuild_index(self) -> None:
        """Rebuild faiss index from current entries."""
        if faiss is None:
            return
        if not self._entries:
            self._faiss_index = None
            return
        matrix = np.array([e.embedding for e in self._entries], dtype="float32")
        faiss.normalize_L2(matrix)
        self._faiss_index = faiss.IndexFlatIP(matrix.shape[1])
        self._faiss_index.add(matrix)

    def ingest(self, entries: Iterable[VectorEntry]) -> None:
        """Process ``entries`` updating internal clusters.

        Entries with low norm are skipped.  Near duplicates are ignored.
        Remaining entries are assigned to clusters based on cosine similarity
        of their embeddings to existing centroids.
        """
        for entry in entries:
            source = entry.metadata.get("source", "unknown")
            count = self.source_msg_counts.get(source, 0) + 1
            self.source_msg_counts[source] = count
            norm = math.sqrt(sum(x * x for x in entry.embedding))
            if norm < self.min_norm:
                if count > self.source_msg_limit:
                    self._archive_source_docs(source)
                continue
            dup_id = self._is_duplicate(entry.embedding)
            if dup_id:
                self.doc_weights[dup_id] = self.doc_weights.get(dup_id, 1.0) + 1.0
                cluster = next(
                    (
                        cl
                        for cl in self.clusters.values()
                        if any(m.chunk_id == dup_id for m in cl.members)
                    ),
                    None,
                )
                if cluster is not None:
                    cluster.freshness += self.doc_weights[dup_id]
                    cluster.freshness_ts = time.time()
                if count > self.source_msg_limit:
                    self._archive_source_docs(source)
                continue
            self._entries.append(entry)
            self._add_to_index(entry.embedding)
            cluster = self._closest_cluster(entry.embedding)
            if cluster is None:
                cid = str(uuid.uuid4())
                cluster = Cluster(id=cid, members=[entry])
                self.clusters[cid] = cluster
            else:
                cluster.members.append(entry)

            # update freshness and timestamp
            weight = self.doc_weights.get(entry.chunk_id, 1.0)
            cluster.freshness += weight
            cluster.freshness_ts = time.time()

            # update source quality as running average
            sq = self._compute_source_quality(entry.metadata)
            n = len(cluster.members)
            if n <= 1:
                cluster.source_quality = sq
            else:
                cluster.source_quality = ((cluster.source_quality * (n - 1)) + sq) / n
            cluster.recompute_centroid()
            cluster.recompute_medoid()
            self._update_centroid_history(cluster)
            if self.detect_drift(cluster):
                self._split_cluster(cluster, entry)
            if count > self.source_msg_limit:
                self._archive_source_docs(source)
        # rebuild similarity graph after ingesting new entries
        self._build_doc_graph()
