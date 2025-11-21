"""Core functionality for Telegram Dump Manager v2.0"""

from .indexing import TwoLevelIndexer

try:
    from .adapters.langchain_adapters import (
        LangChainEmbeddingAdapter,
        LangChainLLMAdapter,
        build_langchain_embeddings_from_env,
        build_langchain_llm_from_env,
        get_llm_client_factory,
    )
    from .adapters.langchain_prompts import LangChainPromptManager
    from .adapters.langchain_text_splitters import LangChainTextSplitter, create_text_splitter
except ImportError as e:
    raise ImportError(
        "LangChain не установлен. Установите: pip install langchain langchain-community langchain-openai"
    ) from e

__all__ = [
    "TwoLevelIndexer",
    "LangChainEmbeddingAdapter",
    "LangChainLLMAdapter",
    "build_langchain_embeddings_from_env",
    "build_langchain_llm_from_env",
    "get_llm_client_factory",
    "LangChainPromptManager",
    "LangChainTextSplitter",
    "create_text_splitter",
]
