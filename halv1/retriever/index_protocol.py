from __future__ import annotations

from typing import Any, List, Protocol, Sequence

from index.vector_index import VectorEntry


class IndexProtocol(Protocol):
    """Protocol for vector indexes used by :class:`Retriever`."""

    entries: Sequence[VectorEntry]
    bm25: Any
    _faiss: Any

    async def search(self, query: str, top_k: int = 25) -> List[VectorEntry]:
        """Return top ``top_k`` entries for ``query``."""

        ...

    async def embed(self, text: str) -> List[float]:
        """Public embedding method."""

        ...

    async def _embed(self, text: str) -> List[float]:
        """Internal embedding helper used for scoring."""

        ...
