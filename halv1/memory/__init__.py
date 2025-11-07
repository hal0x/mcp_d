"""Memory management utilities."""

from .archive import MemoryArchive
from .episode_graph import EpisodeGraph
from .fts5_index import FTS5Index
from .hnsw_index import HNSWIndex
from .memory_store import MemoryStore as _MemoryStore
from .memory_service_adapter import MemoryServiceAdapter as _MemoryServiceAdapter
from .profile_store import ProfileStore
from .schemas import (
    ConsolidationResult,
    Episode,
    Schema,
    consolidate,
    consolidate_graph,
)
from .service import MemoryService
from .unified_memory import UnifiedMemory
from .mcp_memory_adapter import MCPMemoryAdapter
from .write_pipeline import WritePipeline, WriteResult

# Экспортируем унифицированную систему как основную
MemoryServiceAdapter = UnifiedMemory
MemoryStore = UnifiedMemory

__all__ = [
    "EpisodeGraph",
    "FTS5Index",
    "HNSWIndex",
    "MemoryStore",  # Теперь это UnifiedMemory
    "MemoryService",
    "MemoryServiceAdapter",  # Теперь это UnifiedMemory
    "UnifiedMemory",  # Основной класс
    "MCPMemoryAdapter",  # MCP адаптер
    "ProfileStore",
    "MemoryArchive",
    "Episode",
    "Schema",
    "ConsolidationResult",
    "consolidate",
    "consolidate_graph",
    "WritePipeline",
    "WriteResult",
]
