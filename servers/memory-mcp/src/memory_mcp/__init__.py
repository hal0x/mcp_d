"""
Telegram Dump Manager v2.0

Современная система для управления дампами Telegram чатов:
- Двухуровневая индексация (сессии + сообщения + задачи)
- Умная группировка и саммаризация
- Граф инсайтов для анализа связей
- Извлечение сущностей и задач
"""

__version__ = "2.0.0"
__author__ = "Telegram Dump Team"
__email__ = "team@memory_mcp.dev"

from .analysis.insight_graph import SummaryInsightAnalyzer
from .core.indexer import TwoLevelIndexer
from .core.langchain_adapters import LangChainLLMAdapter, LangChainEmbeddingAdapter
from .indexing import (
    Attachment,
    BaseIndexer,
    MemoryRecord,
    TelegramIndexer,
)

__all__ = [
    "TwoLevelIndexer",
    "LangChainLLMAdapter",
    "LangChainEmbeddingAdapter",
    "SummaryInsightAnalyzer",
    "BaseIndexer",
    "TelegramIndexer",
    "MemoryRecord",
    "Attachment",
]
