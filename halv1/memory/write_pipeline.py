from __future__ import annotations

import hashlib
from array import array
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from llm.base_client import EmbeddingsClient
from utils.vector_math import cosine_similarity

from .episode_graph import EpisodeGraph
from .models import MemoryItem

EMBED_DIM = 384


@dataclass
class WriteResult:
    """Result of processing a memory write request."""

    node_id: int | None
    surprise: float
    is_surprising: bool
    is_useful: bool
    merged: bool
    related: List[int]


class WritePipeline:
    """Pipeline responsible for embedding, deduplication and graph insertion."""

    def __init__(
        self,
        graph: EpisodeGraph,
        embeddings_client: EmbeddingsClient | None = None,
        *,
        k: int = 32,
        tau_s: float = 0.5,
        tau_u: float = 0.5,
        tau_merge: float = 0.95,
    ) -> None:
        self.graph = graph
        self.embeddings_client = embeddings_client
        self.k = k
        self.tau_s = tau_s
        self.tau_u = tau_u
        self.tau_merge = tau_merge

    # ------------------------------------------------------------------
    def _embed(self, text: str) -> List[float]:
        """Return an embedding for ``text``.

        Uses ``embeddings_client`` when available and falls back to a deterministic
        hashing-based representation. The resulting vector always has ``EMBED_DIM``
        dimensions which keeps the SQLite schema happy even without the optional
        HNSW extension.
        """

        if self.embeddings_client:
            try:
                vec = self.embeddings_client.embed(text)
            except Exception:
                vec = []
        else:
            vec = []

        if not vec:
            # simple bag-of-words hashing similar to :mod:`memory.memory_store`
            tokens = text.lower().split()
            vec = [0.0] * EMBED_DIM
            for tok in tokens:
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                vec[h % EMBED_DIM] += 1.0
            norm = float(np.linalg.norm(vec))
            if norm:
                vec = [v / norm for v in vec]
        elif len(vec) < EMBED_DIM:
            vec = [float(v) for v in vec] + [0.0] * (EMBED_DIM - len(vec))
        else:
            vec = [float(v) for v in vec[:EMBED_DIM]]
        return vec

    # ------------------------------------------------------------------
    @staticmethod
    def _from_blob(blob: bytes) -> List[float]:
        arr = array("f")
        arr.frombytes(blob)
        return list(arr)

    # ------------------------------------------------------------------
    def _nearest(self, embedding: Sequence[float]) -> List[Tuple[int, float]]:
        """Return up to ``k`` nearest nodes and their similarity."""
        cur = self.graph.conn.cursor()
        blob = self.graph._to_blob(embedding)

        # try SQLite HNSW virtual table
        try:
            cur.execute(
                "SELECT rowid FROM items_hnsw ORDER BY distance(embedding, ?) LIMIT ?",
                (blob, self.k),
            )
            candidate_ids = [int(row[0]) for row in cur.fetchall()]
            if candidate_ids:
                placeholders = ",".join("?" for _ in candidate_ids)
                cur.execute(
                    f"SELECT id, embedding FROM items WHERE id IN ({placeholders})",
                    candidate_ids,
                )
                neighbours: List[Tuple[int, float]] = []
                for node_id, emb_blob in cur.fetchall():
                    if emb_blob is None:
                        continue
                    emb = self._from_blob(emb_blob)
                    sim = cosine_similarity(embedding, emb)
                    neighbours.append((int(node_id), sim))
                neighbours.sort(key=lambda x: x[1], reverse=True)
                return neighbours[: self.k]
        except Exception:
            pass

        # fallback to hnswlib when available
        try:
            import hnswlib  # type: ignore

            cur.execute("SELECT id, embedding FROM items")
            ids: List[int] = []
            vecs: List[List[float]] = []
            for node_id, emb_blob in cur.fetchall():
                if emb_blob is None:
                    continue
                ids.append(int(node_id))
                vecs.append(self._from_blob(emb_blob))
            if ids:
                dim = len(vecs[0])
                index = hnswlib.Index(space="cosine", dim=dim)
                index.init_index(max_elements=len(ids))
                index.add_items(np.array(vecs, dtype=np.float32), np.array(ids))
                index.set_ef(min(self.k * 2, len(ids)))
                labels, distances = index.knn_query(
                    np.array(embedding, dtype=np.float32),
                    k=min(self.k, len(ids)),
                )
                return [
                    (int(label), 1.0 - float(dist))
                    for label, dist in zip(labels[0], distances[0])
                ]
        except Exception:
            pass

        # brute force fallback
        cur.execute("SELECT id, embedding FROM items")
        neighbours: List[Tuple[int, float]] = []
        for node_id, emb_blob in cur.fetchall():
            if emb_blob is None:
                continue
            emb = self._from_blob(emb_blob)
            sim = cosine_similarity(embedding, emb)
            neighbours.append((int(node_id), sim))
        neighbours.sort(key=lambda x: x[1], reverse=True)
        return neighbours[: self.k]

    # ------------------------------------------------------------------
    def write_event(
        self,
        content: str,
        *,
        timestamp: float | int,
        entities: Iterable[str] | None = None,
        frozen: bool = False,
    ) -> WriteResult:
        """Insert ``content`` into the graph applying deduplication logic."""

        embedding = self._embed(content)
        neighbours = self._nearest(embedding)
        max_sim = neighbours[0][1] if neighbours else 0.0
        surprise = 1.0 - max_sim
        is_surprising = surprise >= self.tau_s
        is_useful = max_sim >= self.tau_u
        merged = bool(neighbours and max_sim >= self.tau_merge)

        if not frozen and not is_surprising and not is_useful:
            return WriteResult(
                node_id=None,
                surprise=surprise,
                is_surprising=is_surprising,
                is_useful=is_useful,
                merged=False,
                related=[],
            )

        if merged:
            node_id = neighbours[0][0]
            related_ids: List[int] = []
        else:
            related_ids = [nid for nid, sim in neighbours if sim >= self.tau_u]
            item = MemoryItem(
                content=content,
                embedding=embedding,
                timestamp=timestamp,
                entities=list(entities) if entities is not None else None,
                frozen=frozen,
            )
            node_id = self.graph.add_event(item, related=related_ids)

        return WriteResult(
            node_id=node_id,
            surprise=surprise,
            is_surprising=is_surprising,
            is_useful=is_useful,
            merged=merged,
            related=related_ids,
        )
