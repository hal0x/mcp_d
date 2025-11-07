from __future__ import annotations

import logging
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

from llm.base_client import EmbeddingsClient
from retriever.cross_encoder import CrossEncoder
from retriever.pipeline import RetrievalError, RetrievalPipeline

from .episode_graph import EpisodeGraph
from .episodic_buffer import EpisodicBuffer
from .fts5_index import FTS5Index
from .hnsw_index import HNSWIndex
from .memory_store import MemoryEntry
from .write_pipeline import WritePipeline, WriteResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .archive import MemoryArchive


logger = logging.getLogger(__name__)


@dataclass
class EpisodicEvent:
    """Event stored in the episodic buffer before graph consolidation."""

    content: str
    timestamp: datetime
    entities: Optional[List[str]] = None
    frozen: bool = False


class MemoryService:
    """High-level wrapper around episodic memory components."""

    def __init__(
        self,
        path: str | Path,
        embeddings_client: EmbeddingsClient | None = None,
        *,
        buffer_ttl_days: int = 7,
        buffer_max_size: int = 1000,
    ) -> None:
        self.graph = EpisodeGraph(path)
        self.pipeline = WritePipeline(self.graph, embeddings_client)
        self.buffer = EpisodicBuffer(
            ttl_days=buffer_ttl_days,
            max_size=buffer_max_size,
            flush_callback=self._flush_events,
        )
        self.archive: MemoryArchive | None = None

    # ------------------------------------------------------------------
    def _flush_events(self, events: List[EpisodicEvent]) -> None:
        for ev in events:
            result = self.pipeline.write_event(
                ev.content,
                timestamp=ev.timestamp.timestamp(),
                entities=ev.entities,
                frozen=ev.frozen,
            )
            if result.node_id is None:
                continue

    # ------------------------------------------------------------------
    @staticmethod
    def _now() -> float:
        return datetime.now(timezone.utc).timestamp()

    # ------------------------------------------------------------------
    def remember(self, text: str, *, frozen: bool = False) -> WriteResult:
        """Store ``text`` in long-term memory."""

        return self.pipeline.write_event(text, timestamp=self._now(), frozen=frozen)

    # ------------------------------------------------------------------
    def write_event(
        self,
        content: str,
        *,
        timestamp: Optional[datetime] = None,
        entities: Iterable[str] | None = None,
        frozen: bool = False,
    ) -> None:
        """Record an episodic event into the L0 buffer."""

        ts = timestamp or datetime.now(timezone.utc)
        episode = EpisodicEvent(
            content=content,
            timestamp=ts,
            entities=list(entities) if entities is not None else None,
            frozen=frozen,
        )
        self.buffer.write(episode)

    # ------------------------------------------------------------------
    def read_events(self) -> List[str]:
        """Return current events from the episodic buffer."""

        return [ev.content for ev in self.buffer.read()]

    # ------------------------------------------------------------------
    def consolidate(self) -> None:
        """Flush events and cluster recent ones into schemas."""

        from . import schemas

        self.buffer.flush()
        schemas.consolidate_graph(self.graph)

    # ------------------------------------------------------------------
    def recall(self, long_term: bool = False) -> List[str]:
        """Return stored items from memory."""

        if not long_term:
            return [ev.content for ev in self.buffer.read()]
        cur = self.graph.conn.cursor()
        cur.execute("SELECT content FROM items")
        return [str(row[0]) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    def read_schemas(self) -> List[Tuple[str, List[str]]]:
        """Return consolidated schemas and their episodes."""

        cur = self.graph.conn.cursor()
        cur.execute("SELECT id, name FROM schemas")
        result: List[Tuple[str, List[str]]] = []
        for schema_id, name in cur.fetchall():
            cur.execute(
                "SELECT content FROM items WHERE schema_id = ?",
                (schema_id,),
            )
            episodes = [str(row[0]) for row in cur.fetchall()]
            result.append((str(name), episodes))
        return result

    # ------------------------------------------------------------------
    def semantic_search(
        self, query: str, long_term: bool | None = None, top_k: int = 5
    ) -> List[str]:
        """Return items most similar to ``query`` using cosine similarity."""

        from .unified_memory import UnifiedMemory  # Local import to avoid circular dependency

        store = UnifiedMemory(
            path=":memory:",
            embeddings_client=self.pipeline.embeddings_client,
        )
        
        # Отключаем автоматическую загрузку кэша
        store._cache_dirty = False

        # Загружаем данные из графа в long_term
        cur = self.graph.conn.cursor()
        cur.execute("SELECT content, embedding FROM items")
        long_term_entries = []
        for content, emb_blob in cur.fetchall():
            arr = array('f')
            arr.frombytes(emb_blob)
            long_term_entries.append(MemoryEntry(text=str(content), embedding=list(arr)))
        store._long_term_cache = long_term_entries

        # Загружаем данные из буфера в short_term
        short_term_entries = []
        for ev in self.buffer.read():
            short_term_entries.append(
                MemoryEntry(text=ev.content, embedding=store._embed(ev.content))
            )
        store._short_term_cache = short_term_entries

        return store.semantic_search(query, long_term=long_term, top_k=top_k)

    # ------------------------------------------------------------------
    def search(self, query: str, long_term: bool | None = None) -> List[str]:
        """Return items matching ``query`` from memory."""

        def _search_short() -> List[str]:
            items = [ev.content for ev in self.buffer.read()]
            if not query:
                return items
            q = query.lower()
            return [it for it in items if q in it.lower()]

        def _search_long() -> List[str]:
            cur = self.graph.conn.cursor()
            if not query:
                cur.execute("SELECT content FROM items")
                return [str(row[0]) for row in cur.fetchall()]

            hnsw = HNSWIndex(self.graph, self.pipeline.embeddings_client)
            fts5 = FTS5Index(self.graph)

            class _ZeroModel:
                def predict(self, pairs):  # pragma: no cover - trivial
                    return [0.0 for _ in pairs]

            ce = CrossEncoder(model=_ZeroModel())
            pipe = RetrievalPipeline(hnsw, fts5, self.graph, ce)
            try:
                results = pipe.run(query, top_k=20)
            except (IndexError, ValueError, RetrievalError) as exc:
                # If the retrieval pipeline encounters expected errors (e.g. malformed
                # index), log and fall back to returning no long-term matches instead
                # of raising.
                logger.warning(
                    "Retrieval pipeline error (%s): %s",
                    type(exc).__name__,
                    exc,
                )
                return []
            output: List[str] = []
            for res in results:
                node = self.graph.get_node(int(res.node_id))
                if node is not None:
                    output.append(node.content)
            return output

        if long_term is True:
            return _search_long()
        if long_term is False:
            return _search_short()
        return _search_short() + _search_long()

    # ------------------------------------------------------------------
    def forget(
        self,
        item: str,
        *,
        long_term: bool = False,
        archive: MemoryArchive | None = None,
    ) -> bool:
        """Remove ``item`` from memory."""

        cur = self.graph.conn.cursor()
        cur.execute("DELETE FROM items WHERE content = ?", (item,))
        removed = cur.rowcount > 0
        if removed:
            self.graph.conn.commit()
        return removed

    # ------------------------------------------------------------------
    def explain(self, item_id: int) -> List[Tuple[str, EpisodeGraph.Node]]:
        """Return a chain of edges starting from ``item_id``.

        The chain follows ``time``, ``entity`` and ``semantic`` relations in
        that order. Each element of the returned list is a ``(relation, node)``
        tuple with the first entry representing the starting node using the
        ``"self"`` relation. If a relation is missing at any step the traversal
        stops and the collected path is returned.
        """

        node = self.graph.get_node(item_id)
        if node is None:
            return []
        path: List[Tuple[str, EpisodeGraph.Node]] = [("self", node)]
        current_id = item_id
        for relation in ("time", "entity", "semantic"):
            targets = self.graph.get_edges(current_id, relation)
            if not targets:
                break
            next_node = self.graph.get_node(targets[0])
            if next_node is None:
                break
            path.append((relation, next_node))
            current_id = next_node.id
        return path
