"""
Memory management модуль

Компоненты для управления памятью и оценки важности данных
"""

from .graph_builder import GraphBuilder
from .graph_types import (
    DocChunkNode,
    EdgeType,
    EntityNode,
    EventNode,
    GraphEdge,
    GraphNode,
    NodeType,
    ToolCallNode,
    TopicNode,
)
from .importance_scoring import EvictionScorer, ImportanceScorer, MemoryPruner
from .typed_graph import TypedGraphMemory
from .embeddings import EmbeddingService, build_embedding_service_from_env
from .vector_store import (
    VectorStore,
    EntityVectorStore,
    build_vector_store_from_env,
    build_entity_vector_store_from_env,
)

__all__ = [
    "ImportanceScorer",
    "EvictionScorer",
    "MemoryPruner",
    "TypedGraphMemory",
    "GraphBuilder",
    "NodeType",
    "EdgeType",
    "GraphNode",
    "EntityNode",
    "EventNode",
    "DocChunkNode",
    "TopicNode",
    "ToolCallNode",
    "GraphEdge",
    "EmbeddingService",
    "build_embedding_service_from_env",
    "VectorStore",
    "EntityVectorStore",
    "build_vector_store_from_env",
    "build_entity_vector_store_from_env",
]
