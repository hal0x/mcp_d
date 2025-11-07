from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Protocol

import faiss  # type: ignore
import numpy as np

from .memory_store import MemoryEntry


class MemoryStoreLike(Protocol):
    long_term: List[MemoryEntry]
    short_term: List[MemoryEntry]

    def save(self) -> None:  # pragma: no cover - interface method
        ...


class MemoryArchive:
    """Persist removed memory entries and allow restoration.

    Entries are stored in a JSONL file for text recovery and in a FAISS index
    using an OPQ + PQ pipeline to compress embeddings. The FAISS index is saved
    to disk allowing later reconstruction of the original vectors.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        pq_path: str | Path,
        *,
        dim: int = 64,
        m: int = 4,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.pq_path = Path(pq_path)
        self.dim = dim
        self.m = m

    # ------------------------------------------------------------------
    def _create_index(self) -> faiss.Index:
        opq = faiss.OPQMatrix(self.dim, self.m)
        pq = faiss.IndexPQ(self.dim, self.m, 8)
        return faiss.IndexPreTransform(opq, pq)

    def _load_index(self) -> faiss.Index:
        if self.pq_path.exists():
            return faiss.read_index(str(self.pq_path))
        return self._create_index()

    # ------------------------------------------------------------------
    def archive(self, entries: Iterable[MemoryEntry]) -> None:
        """Append ``entries`` to the archive."""

        entries = list(entries)
        if not entries:
            return

        # Persist texts
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps({"text": e.text}, ensure_ascii=False) + "\n")

        # Persist embeddings using OPQ+PQ
        index = self._load_index()
        vectors = np.array([e.embedding for e in entries], dtype="float32")
        if not index.is_trained:
            ksub = 256
            train_x = vectors
            if train_x.shape[0] < ksub:
                rng = np.random.default_rng(0)
                extra = rng.standard_normal((ksub - train_x.shape[0], self.dim)).astype(
                    "float32"
                )
                train_x = np.vstack([train_x, extra])
            index.train(train_x)
        index.add(vectors)
        faiss.write_index(index, str(self.pq_path))

    # ------------------------------------------------------------------
    def restore(self) -> List[MemoryEntry]:
        """Return a list of archived entries without altering storage."""

        if not self.jsonl_path.exists() or not self.pq_path.exists():
            return []

        texts: List[str] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                texts.append(str(data.get("text", "")))

        index = faiss.read_index(str(self.pq_path))
        vecs = index.reconstruct_n(0, index.ntotal)
        return [MemoryEntry(text=t, embedding=v.tolist()) for t, v in zip(texts, vecs)]

    # ------------------------------------------------------------------
    def restore_to(self, store: MemoryStoreLike, *, long_term: bool = False) -> None:
        """Restore archived entries into ``store``."""

        entries = self.restore()
        if not entries:
            return
        target = store.long_term if long_term else store.short_term
        target.extend(entries)
        if long_term:
            store.save()
