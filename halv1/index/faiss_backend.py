"""FAISS backend utilities for VectorIndex.

Encapsulates building and searching the FAISS index so that the main index
implementation can remain compact and focused on I/O and orchestration.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import numpy as np

try:
    import faiss  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover - import error path
    faiss = None
    _faiss_import_error: Exception | None = exc
else:  # pragma: no cover - simple attribute assignment
    _faiss_import_error = None


class FaissBackend:
    def __init__(self) -> None:
        if faiss is None:
            raise RuntimeError(
                "faiss-cpu package is required for FaissBackend; install it to enable vector search"
            ) from _faiss_import_error
        self.index: faiss.Index | None = None
        self._entries: List[object] = []  # holds objects aligned with vectors

    @property
    def entries(self) -> List[object]:
        return self._entries

    def clear(self) -> None:
        self.index = None
        self._entries = []

    def rebuild(
        self, entries: Iterable[object], get_vector: Callable[[object], List[float]]
    ) -> None:
        """Rebuild the FAISS index from ``entries`` using ``get_vector(obj)``.

        Entries with empty or zero-norm embeddings are ignored.
        """
        vecs: List[List[float]] = []
        kept: List[object] = []
        dim: int | None = None
        for e in entries:
            v = get_vector(e)
            # Accept only non-empty Python lists
            if not (isinstance(v, list) and v):
                continue
            # Establish target dimensionality from the first valid vector
            if dim is None:
                dim = len(v)
            # Skip vectors that don't match the established dimensionality
            if len(v) != dim:
                continue
            vecs.append(v)
            kept.append(e)
        if not vecs:
            self.clear()
            return
        vectors = np.array(vecs, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            self.clear()
            return
        norms = np.linalg.norm(vectors, axis=1)
        mask = norms > 0
        if not np.any(mask):
            self.clear()
            return
        vectors = vectors[mask]
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efSearch = 64
        index.add(vectors)
        self.index = index
        # keep only the masked entries
        self._entries = [e for e, m in zip(kept, mask) if m]

    def search(
        self, q: List[float], top_k: int, get_id: Callable[[object], str]
    ) -> Dict[str, float]:
        """Search nearest neighbors for ``q`` and return mapping id->score."""
        if self.index is None or not self._entries:
            return {}
        q_arr = np.array(q, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q_arr)
        sims, idxs = self.index.search(q_arr, top_k)
        scores: Dict[str, float] = {}
        for score, idx in zip(sims[0], idxs[0]):
            if idx == -1:
                continue
            entry = self._entries[idx]
            scores[get_id(entry)] = float(score)
        return scores
