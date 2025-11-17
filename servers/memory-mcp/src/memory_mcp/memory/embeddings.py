"""HTTP client for obtaining text embeddings from external service."""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Simple synchronous client for the text-embeddings inference service.
    
    Supports two API formats:
    1. text-embeddings-inference format: /embeddings with {"inputs": [...]}
    2. OpenAI/LM Studio format: /v1/embeddings with {"model": "...", "input": "..."}
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout: float = 10.0,
        model_name: Optional[str] = None,
        api_format: str = "auto",  # "auto", "tei", "openai"
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout = timeout
        self.model_name = model_name
        self.session = requests.Session()
        self._dimension: Optional[int] = None
        
        # Auto-detect API format based on URL
        if api_format == "auto" and self.base_url:
            if "/v1" in self.base_url or ":1234" in self.base_url:
                self.api_format = "openai"
            else:
                self.api_format = "tei"
        else:
            self.api_format = api_format

    def available(self) -> bool:
        return bool(self.base_url)

    @property
    def dimension(self) -> Optional[int]:
        return self._dimension

    def embed(self, text: str) -> Optional[List[float]]:
        if not self.base_url:
            return None
        payload_text = text.strip()
        if not payload_text:
            return None

        try:
            # Determine endpoint and payload format
            if self.api_format == "openai":
                endpoint = f"{self.base_url}/v1/embeddings"
                # LM Studio и OpenAI API ожидают input как массив строк
                payload = {
                    "model": self.model_name or "text-embedding-ada-002",
                    "input": [payload_text],
                }
            else:  # tei format
                endpoint = f"{self.base_url}/embeddings"
                payload = {"inputs": [payload_text]}

            response = self.session.post(
                endpoint,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            
            # Handle both response formats
            if "data" in data and isinstance(data["data"], list):
                vector = data["data"][0]["embedding"]  # OpenAI format
            elif isinstance(data, list) and len(data) > 0:
                vector = data[0]  # TEI format
            else:
                raise ValueError("Embedding service returned unexpected payload format")
            
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
    """Build embedding service from environment variables.
    
    Priority:
    1. EMBEDDINGS_URL (if set, use it directly)
    2. LM Studio variables (LMSTUDIO_HOST, LMSTUDIO_PORT, LMSTUDIO_MODEL)
    
    Returns None if no configuration is found.
    """
    # Priority 1: EMBEDDINGS_URL
    url = os.getenv("EMBEDDINGS_URL")
    model_name = None
    
    if url:
        # Use EMBEDDINGS_URL as-is
        service = EmbeddingService(url)
    else:
        # Priority 2: LM Studio variables
        host = os.getenv("LMSTUDIO_HOST", "127.0.0.1")
        port = os.getenv("LMSTUDIO_PORT", "1234")
        model_name = os.getenv("LMSTUDIO_MODEL")
        
        # Build URL from LM Studio variables
        url = f"http://{host}:{port}"
        service = EmbeddingService(
            url,
            model_name=model_name,
            api_format="openai",  # LM Studio uses OpenAI-compatible API
        )
    
    if not service.available():
        return None
    
    # Warm-up to determine embedding dimension; ignore errors
    vector = service.embed("warmup sentence for embeddings")
    if vector is None:
        logger.warning(
            "Embedding service is configured but returned no vector. "
            f"URL: {url}, Model: {model_name or 'default'}"
        )
    else:
        logger.info(
            f"Embedding service initialized: URL={url}, "
            f"Model={model_name or 'default'}, Dimension={service.dimension}"
        )
    return service
