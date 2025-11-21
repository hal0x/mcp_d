"""Core functionality for Telegram Dump Manager v2.0"""

from .indexer import TwoLevelIndexer
from .lmstudio_client import LMStudioEmbeddingClient

try:
    from .langchain_adapters import (
        LangChainEmbeddingAdapter,
        LangChainLLMAdapter,
        build_langchain_embeddings_from_env,
        build_langchain_llm_from_env,
        get_llm_client_factory,
    )
    from .langchain_prompts import LangChainPromptManager
    from .langchain_text_splitters import LangChainTextSplitter, create_text_splitter
except ImportError:
    # LangChain не установлен
    LangChainEmbeddingAdapter = None  # type: ignore
    LangChainLLMAdapter = None  # type: ignore
    build_langchain_embeddings_from_env = None  # type: ignore
    build_langchain_llm_from_env = None  # type: ignore
    get_llm_client_factory = None  # type: ignore
    LangChainPromptManager = None  # type: ignore
    LangChainTextSplitter = None  # type: ignore
    create_text_splitter = None  # type: ignore

__all__ = [
    "TwoLevelIndexer",
    "LMStudioEmbeddingClient",
    "LangChainEmbeddingAdapter",
    "LangChainLLMAdapter",
    "build_langchain_embeddings_from_env",
    "build_langchain_llm_from_env",
    "get_llm_client_factory",
    "LangChainPromptManager",
    "LangChainTextSplitter",
    "create_text_splitter",
]
