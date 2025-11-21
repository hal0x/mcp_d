"""LangChain embeddings service wrapper."""

from __future__ import annotations

import logging

from ..core.langchain_adapters import (
    LangChainEmbeddingAdapter,
    build_langchain_embeddings_from_env,
)

logger = logging.getLogger(__name__)

# Алиас для удобства
EmbeddingService = LangChainEmbeddingAdapter


def build_embedding_service_from_env() -> LangChainEmbeddingAdapter | None:
    """Build LangChain embedding service from settings.
    
    Priority:
    1. embeddings_url (if set, use it directly)
    2. LM Studio variables (lmstudio_host, lmstudio_port, lmstudio_model)
    
    Returns None if no configuration is found.
    """
    try:
        langchain_service = build_langchain_embeddings_from_env()
        if langchain_service:
            logger.info("Using LangChain embeddings")
            return langchain_service
        else:
            logger.error("Failed to initialize LangChain embeddings")
            return None
    except ImportError as e:
        logger.error(f"LangChain not available: {e}. Install: pip install langchain langchain-community langchain-openai")
        return None
    except Exception as e:
        logger.error(f"Error initializing LangChain embeddings: {e}")
        return None
