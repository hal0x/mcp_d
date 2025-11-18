"""Vector store integration backed by Qdrant."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse, urlunparse

from ..config import get_settings

try:  # pragma: no cover - optional dependency tested at runtime
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:  # pragma: no cover - fallback when qdrant is not installed
    QdrantClient = None  # type: ignore[assignment]
    qmodels = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _mask_url_in_log(url: str) -> str:
    """Маскирует учетные данные в URL для безопасного логирования."""
    try:
        parsed = urlparse(url)
        if parsed.username or parsed.password:
            # Маскируем username и password
            masked_netloc = f"{parsed.hostname or ''}"
            if parsed.port:
                masked_netloc += f":{parsed.port}"
            masked = urlunparse((
                parsed.scheme,
                masked_netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))
            return masked
        return url
    except Exception:
        # Если не удалось распарсить, возвращаем как есть (но это не должно происходить)
        return url


@dataclass
class VectorSearchResult:
    record_id: str
    score: float
    payload: Dict[str, object]


class VectorStore:
    """Thin wrapper around Qdrant for storing and searching embeddings."""

    def __init__(
        self,
        url: Optional[str],
        *,
        collection: str = "memory-records",
        distance: str = "cosine",
    ) -> None:
        self.url = url
        self.collection = collection
        self.distance = distance
        self.client: Optional[QdrantClient] = None
        self.vector_size: Optional[int] = None

        if url and QdrantClient is not None:
            try:
                self.client = QdrantClient(url=url, check_compatibility=False)
                logger.info("Vector store connected to %s", _mask_url_in_log(url))
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to connect to Qdrant at %s: %s", _mask_url_in_log(url), exc)
                self.client = None
        elif url:
            logger.warning("qdrant-client package not available; vector store disabled.")

    def available(self) -> bool:
        return self.client is not None

    def ensure_collection(self, vector_size: int) -> None:
        if not self.client or qmodels is None:
            return
        if self.vector_size == vector_size:
            return
        try:
            info = self.client.get_collection(self.collection)
            existing_size = info.config.params.vectors.size  # type: ignore[attr-defined]
            if existing_size != vector_size:
                logger.warning(
                    "Recreating Qdrant collection %s with new vector size %s (was %s)",
                    self.collection,
                    vector_size,
                    existing_size,
                )
                self.client.recreate_collection(
                    collection_name=self.collection,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size,
                        distance=self._distance_enum(),
                    ),
                )
        except Exception:
            logger.info(
                "Creating Qdrant collection %s (vector size %s)",
                self.collection,
                vector_size,
            )
            try:
                self.client.recreate_collection(
                    collection_name=self.collection,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size,
                        distance=self._distance_enum(),
                    ),
                )
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Failed to initialize Qdrant collection: %s", exc)
                return
        self.vector_size = vector_size

    def upsert(self, record_id: str, vector: List[float], payload: Dict[str, object]) -> None:
        if not self.client or qmodels is None or not vector:
            return
        self.ensure_collection(len(vector))
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    qmodels.PointStruct(
                        id=str(record_id),
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to upsert vector for %s: %s", record_id, exc)

    def delete(self, record_id: str) -> None:
        """Delete a vector by record ID."""
        if not self.client or qmodels is None:
            return
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=qmodels.PointIdsList(
                    points=[str(record_id)],
                ),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to delete vector for %s: %s", record_id, exc)

    def delete_by_chat(self, chat_name: str) -> int:
        """
        Delete all vectors for a specific chat.

        Args:
            chat_name: Name of the chat to delete vectors for

        Returns:
            Number of deleted vectors
        """
        if not self.client or qmodels is None:
            return 0

        deleted_count = 0
        try:
            # Используем scroll для получения всех точек с фильтром по chat
            filter_condition = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="chat",
                        match=qmodels.MatchValue(value=chat_name),
                    )
                ]
            )

            # Получаем все точки с фильтром
            scroll_result = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=filter_condition,
                limit=10000,  # Максимальное количество за один запрос
                with_payload=True,
                with_vectors=False,
            )

            point_ids = [point.id for point in scroll_result[0]]
            if not point_ids:
                logger.info(f"No vectors found for chat: {chat_name}")
                return 0

            # Удаляем точки батчами (Qdrant может иметь ограничения на размер батча)
            batch_size = 1000
            for i in range(0, len(point_ids), batch_size):
                batch = point_ids[i : i + batch_size]
                try:
                    self.client.delete(
                        collection_name=self.collection,
                        points_selector=qmodels.PointIdsList(points=batch),
                    )
                    deleted_count += len(batch)
                except Exception as exc:
                    logger.warning(
                        f"Failed to delete batch of vectors for chat {chat_name}: {exc}"
                    )

            logger.info(f"Deleted {deleted_count} vectors for chat: {chat_name}")
            return deleted_count

        except Exception as exc:  # pragma: no cover
            logger.warning(
                f"Failed to delete vectors by chat {chat_name}: {exc}", exc_info=True
            )
            return deleted_count

    def search(
        self,
        vector: List[float],
        *,
        limit: int,
        source: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[VectorSearchResult]:
        if not self.client or qmodels is None or not vector:
            return []
        self.ensure_collection(len(vector))

        must: List[qmodels.FilterCondition] = []
        if source:
            must.append(
                qmodels.FieldCondition(
                    key="source",
                    match=qmodels.MatchValue(value=source),
                )
            )
        if tags:
            must.append(
                qmodels.FieldCondition(
                    key="tags",
                    match=qmodels.MatchAny(any=list(tags)),
                )
            )
        if date_from or date_to:
            range_kwargs = {}
            if date_from:
                range_kwargs["gte"] = date_from.timestamp()
            if date_to:
                range_kwargs["lte"] = date_to.timestamp()
            must.append(
                qmodels.FieldCondition(
                    key="timestamp",
                    range=qmodels.Range(**range_kwargs),
                )
            )
        flt = qmodels.Filter(must=must) if must else None

        try:
            # Qdrant API: используем query_filter для фильтров (не filter!)
            if flt:
                result = self.client.search(
                    collection_name=self.collection,
                    query_vector=vector,
                    limit=limit,
                    query_filter=flt,
                )
            else:
                result = self.client.search(
                    collection_name=self.collection,
                    query_vector=vector,
                    limit=limit,
                )
        except Exception as exc:  # pragma: no cover
            logger.warning("Vector search failed: %s", exc)
            return []

        output: List[VectorSearchResult] = []
        for point in result:
            output.append(
                VectorSearchResult(
                    record_id=str(point.id),
                    score=float(point.score),
                    payload=dict(point.payload or {}),
                )
            )
        return output

    def close(self) -> None:
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass

    def _distance_enum(self):
        if qmodels is None:
            return None
        if self.distance.lower() == "cosine":
            return qmodels.Distance.COSINE
        if self.distance.lower() == "dot":
            return qmodels.Distance.DOT
        return qmodels.Distance.EUCLID


def build_vector_store_from_env() -> VectorStore | None:
    settings = get_settings()
    url = settings.get_qdrant_url()
    if not url:
        return None
    store = VectorStore(url=url)
    if not store.available():
        return None
    return store
