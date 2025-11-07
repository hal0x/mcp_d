from __future__ import annotations

from array import array
from typing import List
import sqlite3

from .episode_graph import EpisodeGraph
from retriever.pipeline import Candidate


class FTS5Index:
    """Simple wrapper around the SQLite FTS5 index."""

    def __init__(self, graph: EpisodeGraph) -> None:
        self.graph = graph

    @staticmethod
    def _from_blob(blob: bytes | None) -> List[float]:
        arr = array("f")
        if blob:
            arr.frombytes(blob)
        return list(arr)

    def search(self, query: str, top_k: int = 25) -> List[Candidate]:
        """Return top ``top_k`` documents matching ``query``."""

        cur = self.graph.conn.cursor()
        try:
            cur.execute(
                "SELECT rowid, content FROM items_fts WHERE items_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, top_k),
            )
        except sqlite3.DatabaseError:
            # Corrupted index; treat as no results rather than propagating
            return []
        rows = cur.fetchall()
        if not rows:
            return []

        ids = [int(row[0]) for row in rows]
        placeholders = ",".join("?" for _ in ids)

        cur.execute(
            f"SELECT rowid, bm25(items_fts) FROM items_fts WHERE rowid IN ({placeholders})",
            ids,
        )
        score_map = {int(rid): -float(score) for rid, score in cur.fetchall()}

        cur.execute(
            f"SELECT id, embedding FROM items WHERE id IN ({placeholders})",
            ids,
        )
        emb_map = {
            int(i): self._from_blob(blob)
            for i, blob in cur.fetchall()
            if blob is not None
        }

        results: List[Candidate] = []
        for rowid, content in rows:
            emb = emb_map.get(int(rowid), [])
            score = score_map.get(int(rowid), 0.0)
            results.append(Candidate(str(rowid), str(content), emb, score))
        return results
