#!/usr/bin/env python3
"""
Типы узлов и рёбер для типизированного графа знаний

Вдохновлено архитектурой HALv1 Memory 2.0
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Типы узлов в графе знаний"""

    ENTITY = "Entity"  # Персоны, организации, проекты, термины
    EVENT = "Event"  # Сессии, действия, факты во времени
    DOC_CHUNK = "DocChunk"  # Фрагменты документов/сообщений
    TOPIC = "Topic"  # Тематические кластеры
    TOOL_CALL = "ToolCall"  # Вызовы инструментов (для будущего расширения)
    TRADING_SIGNAL = "TradingSignal"  # Узлы торговых сигналов


class EdgeType(str, Enum):
    """Типы рёбер (связей) в графе"""

    RELATES_TO = "relates_to"  # Семантические связи
    CAUSES = "causes"  # Причинно-следственные связи
    MENTIONS = "mentions"  # Упоминания в документах
    HAS_TOPIC = "has_topic"  # Тематическая принадлежность
    AUTHORED_BY = "authored_by"  # Авторство
    TEMPORAL = "temporal"  # Временная последовательность
    SIMILAR_TO = "similar_to"  # Семантическое сходство
    PART_OF = "part_of"  # Иерархические связи
    ASSOCIATED_WITH = "associated_with"  # Ассоциации (например, сигнал - актив)


class GraphNode(BaseModel):
    """Базовый узел графа"""

    id: str = Field(..., description="Уникальный идентификатор узла")
    type: NodeType = Field(..., description="Тип узла")
    label: str = Field(..., description="Человекочитаемое название")
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Дополнительные свойства"
    )
    embedding: Optional[List[float]] = Field(
        None, description="Векторное представление"
    )
    created_at: Optional[str] = Field(None, description="Время создания (ISO)")
    updated_at: Optional[str] = Field(None, description="Время обновления (ISO)")

    class Config:
        use_enum_values = True


class EntityNode(GraphNode):
    """Узел сущности (персона, организация, проект, термин)"""

    type: Literal[NodeType.ENTITY] = Field(default=NodeType.ENTITY)
    entity_type: str = Field(..., description="Тип сущности (person/org/project/term)")
    aliases: List[str] = Field(
        default_factory=list, description="Альтернативные названия"
    )
    description: Optional[str] = Field(None, description="Описание сущности")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Важность (0-1)")


class EventNode(GraphNode):
    """Узел события (сессия, действие, факт)"""

    type: Literal[NodeType.EVENT] = Field(default=NodeType.EVENT)
    event_type: str = Field(..., description="Тип события (session/action/fact)")
    timestamp: str = Field(..., description="Время события (ISO)")
    duration_seconds: Optional[int] = Field(None, description="Длительность в секундах")
    participants: List[str] = Field(
        default_factory=list, description="ID участников события"
    )
    location: Optional[str] = Field(None, description="Локация (чат)")
    summary: Optional[str] = Field(None, description="Краткая сводка события")


class DocChunkNode(GraphNode):
    """Узел фрагмента документа/сообщения"""

    type: Literal[NodeType.DOC_CHUNK] = Field(default=NodeType.DOC_CHUNK)
    content: str = Field(..., description="Текст фрагмента")
    source: str = Field(..., description="Источник (чат/файл)")
    timestamp: str = Field(..., description="Время создания (ISO)")
    author: Optional[str] = Field(None, description="Автор")
    message_count: int = Field(default=1, description="Количество сообщений")
    language: str = Field(default="unknown", description="Язык")


class TopicNode(GraphNode):
    """Узел темы/кластера"""

    type: Literal[NodeType.TOPIC] = Field(default=NodeType.TOPIC)
    topic_name: str = Field(..., description="Название темы")
    description: str = Field(..., description="Описание темы")
    keywords: List[str] = Field(default_factory=list, description="Ключевые слова")
    session_count: int = Field(default=0, description="Количество сессий")
    message_count: int = Field(default=0, description="Количество сообщений")
    time_span_days: Optional[int] = Field(None, description="Временной охват в днях")


class ToolCallNode(GraphNode):
    """Узел вызова инструмента (для будущего расширения)"""

    type: Literal[NodeType.TOOL_CALL] = Field(default=NodeType.TOOL_CALL)
    tool_name: str = Field(..., description="Название инструмента")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Параметры вызова"
    )
    result: Optional[Dict[str, Any]] = Field(None, description="Результат выполнения")
    timestamp: str = Field(..., description="Время вызова (ISO)")
    success: bool = Field(default=True, description="Успешность выполнения")


class GraphEdge(BaseModel):
    """Ребро (связь) между узлами"""

    id: str = Field(..., description="Уникальный идентификатор ребра")
    source_id: str = Field(..., description="ID исходного узла")
    target_id: str = Field(..., description="ID целевого узла")
    type: EdgeType = Field(..., description="Тип связи")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Вес связи (0-1)")
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Дополнительные свойства"
    )
    created_at: Optional[str] = Field(None, description="Время создания (ISO)")
    bidirectional: bool = Field(default=False, description="Двунаправленная связь")

    class Config:
        use_enum_values = True


class GraphQuery(BaseModel):
    """Запрос к графу"""

    node_types: Optional[List[NodeType]] = Field(
        None, description="Фильтр по типам узлов"
    )
    edge_types: Optional[List[EdgeType]] = Field(
        None, description="Фильтр по типам рёбер"
    )
    properties_filter: Optional[Dict[str, Any]] = Field(
        None, description="Фильтр по свойствам"
    )
    limit: int = Field(default=100, ge=1, le=10000, description="Лимит результатов")
    include_embeddings: bool = Field(default=False, description="Включить эмбеддинги")


class GraphPath(BaseModel):
    """Путь в графе между узлами"""

    nodes: List[GraphNode] = Field(..., description="Узлы в пути")
    edges: List[GraphEdge] = Field(..., description="Рёбра в пути")
    length: int = Field(..., description="Длина пути")
    total_weight: float = Field(..., description="Суммарный вес пути")


class GraphStats(BaseModel):
    """Статистика графа"""

    node_count: int = Field(..., description="Всего узлов")
    edge_count: int = Field(..., description="Всего рёбер")
    node_types_count: Dict[str, int] = Field(
        default_factory=dict, description="Распределение по типам узлов"
    )
    edge_types_count: Dict[str, int] = Field(
        default_factory=dict, description="Распределение по типам рёбер"
    )
    avg_node_degree: float = Field(..., description="Средняя степень узла")
    density: float = Field(..., description="Плотность графа (0-1)")
    connected_components: int = Field(..., description="Количество компонент связности")
