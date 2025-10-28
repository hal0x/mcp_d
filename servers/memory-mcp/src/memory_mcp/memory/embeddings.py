"""HTTP client for obtaining text embeddings from external service."""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Simple synchronous client for the text-embeddings inference service."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout = timeout
        self.session = requests.Session()
        self._dimension: Optional[int] = None

    def available(self) -> bool:
        return bool(self.base_url)

    @property
    def dimension(self) -> Optional[int]:
        return self._dimension

    def embed(self, text: str) -> Optional[List[float]]:
        if not self.base_url:
            return None
        payload = text.strip()
        if not payload:
            return None

        try:
            response = self.session.post(
                f"{self.base_url}/embeddings",
                json={"inputs": [payload]},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            vector = data["data"][0]["embedding"]  # type: ignore[index]
            if not isinstance(vector, list):
                raise ValueError("Embedding service returned unexpected payload")
            if self._dimension is None:
                self._dimension = len(vector)
            return vector
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Embedding service error: %s", exc)
            return None

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass


def build_embedding_service_from_env() -> EmbeddingService | None:
    url = os.getenv("EMBEDDINGS_URL")
    if not url:
        return None
    service = EmbeddingService(url)
    # Warm-up to determine embedding dimension; ignore errors
    vector = service.embed("warmup sentence for embeddings")
    if vector is None:
        logger.warning("Embedding service is configured but returned no vector.")
    return service
