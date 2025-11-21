"""Модули для работы с эмбеддингами."""

from .service import EmbeddingService, build_embedding_service_from_env

__all__ = [
    "EmbeddingService",
    "build_embedding_service_from_env",
]


