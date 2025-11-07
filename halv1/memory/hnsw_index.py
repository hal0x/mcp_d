from __future__ import annotations

import hashlib
from array import array
from typing import List

import numpy as np

from llm.base_client import EmbeddingsClient
from retriever.pipeline import Candidate
from utils.vector_math import cosine_similarity

from .episode_graph import EpisodeGraph

EMBED_DIM = 384


class HNSWIndex:
    """Wrapper around the SQLite HNSW index (``items_hnsw``)."""

    def __init__(
        self, graph: EpisodeGraph, embeddings_client: EmbeddingsClient | None = None
    ) -> None:
        self.graph = graph
        self.embeddings_client = embeddings_client

    # ------------------------------------------------------------------
    def _embed(self, text: str) -> List[float]:
        """Return an embedding for ``text`` using the configured client."""

        vec: List[float] | np.ndarray
        if self.embeddings_client:
            try:
                vec = self.embeddings_client.embed(text)
            except Exception:  # pragma: no cover - defensive
                vec = []
        else:
            vec = []

        if not vec:
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
        return vec  # type: ignore[return-value]

    @staticmethod
    def _from_blob(blob: bytes | None) -> List[float]:
        arr = array("f")
        if blob:
            arr.frombytes(blob)
        return list(arr)

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 25) -> List[Candidate]:
        """Return top ``top_k`` nearest items for ``query``."""

        embedding = self._embed(query)
        blob = self.graph._to_blob(embedding)
        cur = self.graph.conn.cursor()
        try:
            cur.execute(
                "SELECT rowid FROM items_hnsw ORDER BY distance(embedding, ?) LIMIT ?",
                (blob, top_k),
            )
        except Exception:  # pragma: no cover - missing extension
            return []
        ids = [int(row[0]) for row in cur.fetchall()]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        cur.execute(
            f"SELECT id, content, embedding FROM items WHERE id IN ({placeholders})",
            ids,
        )
        results: List[Candidate] = []
        for node_id, content, emb_blob in cur.fetchall():
            if emb_blob is None:
                continue
            emb = self._from_blob(emb_blob)
            score = cosine_similarity(embedding, emb)
            results.append(Candidate(str(node_id), str(content), emb, score))
        return results
