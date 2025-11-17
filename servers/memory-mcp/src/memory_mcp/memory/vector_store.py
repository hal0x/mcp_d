"""Vector store integration backed by Qdrant."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency tested at runtime
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:  # pragma: no cover - fallback when qdrant is not installed
    QdrantClient = None  # type: ignore[assignment]
    qmodels = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


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
                self.client = QdrantClient(url=url)
                logger.info("Vector store connected to %s", url)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to connect to Qdrant at %s: %s", url, exc)
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
            result = self.client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=limit,
                filter=flt,
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
    url = os.getenv("QDRANT_URL")
    if not url:
        return None
    store = VectorStore(url=url)
    if not store.available():
        return None
    return store
