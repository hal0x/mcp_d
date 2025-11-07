"""Simple in-memory vector index using LM Studio embeddings."""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np

from utils.performance import (
    measure_context_async,
    measure_time_async,
)

from .bm25_index import BM25Index
from .faiss_backend import FaissBackend
from .preprocess import normalize_text
from .ranking import combine_scores
from .storage import load_index as _load_index
from .storage import save_index as _save_index

logger = logging.getLogger(__name__)


@dataclass
class VectorEntry:
    chunk_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, str]
    timestamp: float = field(default_factory=lambda: time.time())


DEFAULT_MODEL = "dengcao/Qwen3-Embedding-4B:Q5_K_M"  # Специализированная модель для эмбеддингов
MIN_EMBED_NORM = 1e-6


class VectorIndex:
    """Store text chunks with LM Studio-provided embeddings."""

    def __init__(
        self,
        path: str = "db/index/index.json",
        model_name: str = DEFAULT_MODEL,
        host: str = "127.0.0.1",
        port: int = 11434,  # Ollama порт по умолчанию для embeddings
        api_key: str | None = None,
        bm25_path: str | None = None,
        num_ctx: int = 32000,  # Максимальный контекст для эмбеддингов
        num_batch: int = 1024,  # Размер батча для обработки
        keep_alive: str = "600s",  # Время жизни модели
    ) -> None:
        self.path = Path(path)
        self.model_name = model_name
        # Embeddings HTTP client
        from llm.embeddings_client import AsyncEmbeddingsClient

        # Определяем провайдера по порту
        provider = "ollama" if port == 11434 else "lmstudio"
        self.emb_client = AsyncEmbeddingsClient(
            model=model_name, 
            host=host, 
            port=port, 
            api_key=api_key, 
            provider=provider,
            num_ctx=num_ctx,
            num_batch=num_batch,
            keep_alive=keep_alive
        )
        self.entries: List[VectorEntry] = []
        self._id_map: Dict[str, VectorEntry] = {}
        self.weights: Counter[str] = Counter()
        self._faiss = FaissBackend()
        self.bm25 = BM25Index(bm25_path or "db/index/bm25.json")
        self._needs_rebuild = False
        if self.path.exists():
            self._load()
            self._rebuild_faiss_index()
        logger.info(
            "VectorIndex initialized: path=%s, model=%s",
            str(self.path),
            self.model_name,
        )

    # ------------------------------------------------------------------
    def _load(self) -> None:
        def _factory(item: dict[str, Any]) -> VectorEntry:
            return VectorEntry(
                chunk_id=item.get("chunk_id", ""),
                text=item.get("text", ""),
                embedding=item.get("embedding", []),
                metadata=item.get("metadata", {}),
                timestamp=item.get("timestamp", time.time()),
            )

        entries, weights = _load_index(self.path, entry_factory=_factory)
        self.entries = entries
        self._id_map = {e.chunk_id: e for e in entries}
        self.weights = weights

    def _save(self) -> None:
        _save_index(
            self.path,
            self.entries,
            self.weights,
            to_dict=lambda e: cast(Dict[str, Any], cast(Any, e).__dict__.copy()),
        )
        logger.debug(
            "VectorIndex saved: entries=%d, path=%s", len(self.entries), str(self.path)
        )

    def _rebuild_faiss_index(self) -> None:
        if not self.entries:
            self._faiss.clear()
            return
        self._faiss.rebuild(self.entries, get_vector=lambda e: e.embedding)

    # Factory for themed index paths
    @staticmethod
    def themed(
        base_dir: str,
        theme: str,
        model_name: str = DEFAULT_MODEL,
        host: str = "127.0.0.1",
        port: int = 1234,
        api_key: str | None = None,
        num_ctx: int = 32000,
        num_batch: int = 1024,
        keep_alive: str = "600s",
    ) -> "VectorIndex":
        safe = theme.replace("/", "_").strip() or "default"
        path = Path(base_dir) / f"index_{safe}.json"
        return VectorIndex(
            str(path), 
            model_name=model_name, 
            host=host, 
            port=port, 
            api_key=api_key,
            num_ctx=num_ctx,
            num_batch=num_batch,
            keep_alive=keep_alive
        )

    async def embed(self, text: str) -> List[float]:
        """Return an embedding vector for ``text``."""

        return cast(List[float], await self._embed(text))

    @measure_time_async("vector_index_embed")  # type: ignore[misc]
    async def _embed(self, text: str) -> List[float]:
        """Internal helper for retrieving an embedding.

        External callers should use :meth:`embed` instead.
        """
        snippet = (text or "")[:80].replace("\n", " ")
        logger.debug(
            "Embedding start: model=%s, len=%d, text='%s...'",
            self.model_name,
            len(text or ""),
            snippet,
        )
        emb = await self.emb_client.embed(text)
        if emb:
            logger.info("Embedding ok: dims=%d", len(emb))
        else:
            logger.warning("Embedding failed")
        return emb

    async def _embed_many(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for multiple ``texts`` via a single request."""
        logger.debug(
            "Embedding batch start: model=%s, count=%d",
            self.model_name,
            len(texts),
        )
        embeddings = await self.emb_client.embed_many(texts)
        if embeddings:
            logger.info(
                "Embedding batch ok: count=%d, dims=%d",
                len(embeddings),
                len(embeddings[0]) if embeddings else 0,
            )
        else:
            logger.warning(
                "Embedding batch failed: count=%d",
                len(texts),
            )
        return embeddings

    @measure_time_async("vector_index_add")  # type: ignore[misc]
    async def add(self, chunk_id: str, text: str, metadata: Dict[str, str]) -> None:
        text = normalize_text(text)
        async with measure_context_async("embedding_generation"):
            emb = await self._embed(text)
        # Allow BM25-only indexing when embeddings are unavailable
        if not emb:
            has_valid_emb = False
        else:
            if np.linalg.norm(emb) < MIN_EMBED_NORM:
                logger.warning(
                    "VectorIndex add skipped: low embedding norm, chunk_id=%s, chat=%s, date=%s, theme=%s, text_len=%d",
                    chunk_id,
                    metadata.get("chat"),
                    metadata.get("date"),
                    metadata.get("theme"),
                    len(text or ""),
                )
                return
            has_valid_emb = True
        if chunk_id in self._id_map:
            entry = self._id_map[chunk_id]
            entry.text = text
            entry.embedding = emb if has_valid_emb else []
            entry.metadata = metadata
            entry.timestamp = time.time()
            self.weights[chunk_id] += 1
            logger.info(
                "VectorIndex add update: chunk_id=%s, chat=%s, date=%s, theme=%s, total_entries=%d, path=%s",
                chunk_id,
                metadata.get("chat"),
                metadata.get("date"),
                metadata.get("theme"),
                len(self.entries),
                str(self.path),
            )
        else:
            entry = VectorEntry(chunk_id, text, emb if has_valid_emb else [], metadata)
            self.entries.append(entry)
            self._id_map[chunk_id] = entry
            self.weights[chunk_id] = 1
            logger.info(
                "VectorIndex add ok: chunk_id=%s, chat=%s, date=%s, theme=%s, total_entries=%d, path=%s",
                chunk_id,
                metadata.get("chat"),
                metadata.get("date"),
                metadata.get("theme"),
                len(self.entries),
                str(self.path),
            )
        async with measure_context_async("index_save"):
            self._save()
        async with measure_context_async("bm25_add"):
            self.bm25.add(chunk_id, text)
        # Отложенная пересборка FAISS индекса
        self._needs_rebuild = True

    async def add_many(
        self,
        items: List[tuple[str, str, Dict[str, str]]],
        embeddings: List[List[float]] | None = None,
    ) -> None:
        """Add multiple chunks to the index using a single embedding request."""
        if not items:
            return
        norm_items = [(cid, normalize_text(txt), meta) for cid, txt, meta in items]
        texts = [text for _, text, _ in norm_items]
        if embeddings is None:
            embeddings = await self._embed_many(texts)
        added = 0
        updated = 0
        bm25_items = []
        for (chunk_id, text, metadata), emb in zip(norm_items, embeddings):
            if not emb:
                has_valid_emb = False
            else:
                if np.linalg.norm(emb) < MIN_EMBED_NORM:
                    logger.warning(
                        "VectorIndex add_many skipped: low-norm embedding, chunk_id=%s, chat=%s, date=%s, theme=%s, text_len=%d",
                        chunk_id,
                        metadata.get("chat"),
                        metadata.get("date"),
                        metadata.get("theme"),
                        len(text or ""),
                    )
                    continue
                has_valid_emb = True
            if chunk_id in self._id_map:
                entry = self._id_map[chunk_id]
                entry.text = text
                entry.embedding = emb if has_valid_emb else []
                entry.metadata = metadata
                entry.timestamp = time.time()
                self.weights[chunk_id] += 1
                updated += 1
                bm25_items.append((chunk_id, text))
            else:
                entry = VectorEntry(
                    chunk_id, text, emb if has_valid_emb else [], metadata
                )
                self.entries.append(entry)
                self._id_map[chunk_id] = entry
                self.weights[chunk_id] = 1
                added += 1
                bm25_items.append((chunk_id, text))
        if added or updated:
            self._save()
            if bm25_items:
                self.bm25.add_many(bm25_items)
            # Отложенная пересборка FAISS индекса
            self._needs_rebuild = True
        logger.info(
            "VectorIndex add_many ok: added=%d, updated=%d, total_entries=%d, path=%s",
            added,
            updated,
            len(self.entries),
            str(self.path),
        )

    @measure_time_async("vector_index_search")  # type: ignore[misc]
    async def search(self, query: str, top_k: int = 25) -> List[VectorEntry]:
        """Return the *top_k* most relevant entries mixing vector and BM25 scores."""
        if not self.entries:
            return []
        if len(self._id_map) != len(self.entries):
            self._id_map = {e.chunk_id: e for e in self.entries}

        async with measure_context_async("query_embedding"):
            q_list = await self._embed(query)
        q_norm = float(np.linalg.norm(q_list)) if q_list else 0.0
        if q_norm <= MIN_EMBED_NORM:
            return []

        if self._needs_rebuild or self._faiss.index is None or not self._faiss.entries:
            async with measure_context_async("faiss_rebuild_search"):
                self._rebuild_faiss_index()
                self._needs_rebuild = False

        vector_scores: Dict[str, float] = {}
        if self._faiss.index is not None and self._faiss.entries:
            async with measure_context_async("faiss_search"):
                vector_scores = self._faiss.search(
                    q_list, top_k, get_id=lambda e: e.chunk_id
                )

        async with measure_context_async("bm25_search"):
            bm25_hits = self.bm25.search(query, top_k=top_k)
        async with measure_context_async("score_combination"):
            scored = combine_scores(vector_scores, bm25_hits, self.weights)
            return [
                self._id_map[cid] for cid, _ in scored[:top_k] if cid in self._id_map
            ]
