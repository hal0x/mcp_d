#!/usr/bin/env python3
"""
Построение типизированного графа знаний из данных

Извлекает сущности, события и связи из сессий и сообщений.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from memory_mcp.core.constants import DEFAULT_GRAPH_QUERY_LIMIT
from .graph_types import (
    DocChunkNode,
    EdgeType,
    EntityNode,
    EventNode,
    GraphEdge,
    NodeType,
    TopicNode,
)
from .typed_graph import TypedGraphMemory

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Построение графа знаний из данных tg_dump"""

    def __init__(self, graph: TypedGraphMemory):
        """
        Инициализация

        Args:
            graph: TypedGraphMemory для заполнения
        """
        self.graph = graph
        self.entity_cache: Dict[str, str] = {}  # name -> node_id
        logger.info("Инициализирован GraphBuilder")

    def build_from_sessions(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Построение графа из списка сессий

        Args:
            sessions: Список сессий с метаданными

        Returns:
            Статистика построения
        """
        logger.info(f"Начало построения графа из {len(sessions)} сессий")

        stats = {
            "sessions_processed": 0,
            "entities_created": 0,
            "events_created": 0,
            "doc_chunks_created": 0,
            "topics_created": 0,
            "edges_created": 0,
            "errors": 0,
        }

        for session in sessions:
            try:
                self._process_session(session, stats)
                stats["sessions_processed"] += 1

            except Exception as e:
                logger.error(
                    f"Ошибка обработки сессии {session.get('session_id')}: {e}"
                )
                stats["errors"] += 1

        logger.info(f"Граф построен: {stats}")
        return stats

    def _process_session(self, session: Dict[str, Any], stats: Dict[str, Any]):
        """
        Обработка одной сессии

        Args:
            session: Данные сессии
            stats: Статистика (мутируется)
        """
        session_id = session.get("session_id", "unknown")

        # 1. Создаём узел события для сессии
        event_node = self._create_event_node(session)
        if event_node and self.graph.add_node(event_node):
            stats["events_created"] += 1

        # 2. Извлекаем и создаём узлы сущностей
        entities = session.get("entities", [])
        for entity_name in entities:
            entity_node = self._create_entity_node(entity_name, "term")
            if entity_node and self.graph.add_node(entity_node):
                stats["entities_created"] += 1
                self.entity_cache[entity_name] = entity_node.id

            # Создаём связь: Event -> mentions -> Entity
            if entity_node:
                edge = GraphEdge(
                    id=f"{event_node.id}-mentions-{entity_node.id}",
                    source_id=event_node.id,
                    target_id=entity_node.id,
                    type=EdgeType.MENTIONS,
                    weight=0.8,
                )
                if self.graph.add_edge(edge):
                    stats["edges_created"] += 1

        # 3. Обрабатываем сообщения как DocChunks
        messages = session.get("messages", [])
        for i, msg in enumerate(messages[:10]):  # Ограничиваем первыми 10
            doc_chunk = self._create_doc_chunk_node(msg, session_id, i)
            if doc_chunk and self.graph.add_node(doc_chunk):
                stats["doc_chunks_created"] += 1

                # Связь: Event -> part_of -> DocChunk
                edge = GraphEdge(
                    id=f"{doc_chunk.id}-partof-{event_node.id}",
                    source_id=doc_chunk.id,
                    target_id=event_node.id,
                    type=EdgeType.PART_OF,
                    weight=1.0,
                )
                if self.graph.add_edge(edge):
                    stats["edges_created"] += 1

                # Извлекаем упоминания из сообщения
                text = msg.get("text", "")
                mentioned = self._extract_mentions_from_text(text)
                for mention in mentioned:
                    if mention in self.entity_cache:
                        entity_id = self.entity_cache[mention]
                        edge = GraphEdge(
                            id=f"{doc_chunk.id}-mentions-{entity_id}",
                            source_id=doc_chunk.id,
                            target_id=entity_id,
                            type=EdgeType.MENTIONS,
                            weight=0.5,
                        )
                        if self.graph.add_edge(edge):
                            stats["edges_created"] += 1

        # 4. Обрабатываем темы как Topic узлы
        topics = session.get("topics", [])
        for topic in topics[:3]:  # Ограничиваем первыми 3
            if isinstance(topic, dict):
                topic_node = self._create_topic_node(topic, session_id)
                if topic_node and self.graph.add_node(topic_node):
                    stats["topics_created"] += 1

                    # Связь: Event -> has_topic -> Topic
                    edge = GraphEdge(
                        id=f"{event_node.id}-hastopic-{topic_node.id}",
                        source_id=event_node.id,
                        target_id=topic_node.id,
                        type=EdgeType.HAS_TOPIC,
                        weight=0.9,
                    )
                    if self.graph.add_edge(edge):
                        stats["edges_created"] += 1

    def _create_event_node(self, session: Dict[str, Any]) -> Optional[EventNode]:
        """Создание узла события из сессии"""
        session_id = session.get("session_id")
        if not session_id:
            return None

        meta = session.get("meta", {})

        return EventNode(
            id=f"event-{session_id}",
            label=f"Session {session_id}",
            event_type="session",
            timestamp=meta.get("start_time_utc", datetime.now().isoformat()),
            participants=[],
            location=meta.get("chat_name", "unknown"),
            summary=session.get("summary", ""),
            properties={
                "message_count": meta.get("messages_total", 0),
                "time_span": meta.get("time_span", ""),
                "language": meta.get("dominant_language", "unknown"),
            },
        )

    def _create_entity_node(
        self, entity_name: str, entity_type: str = "term", description: Optional[str] = None
    ) -> Optional[EntityNode]:
        """Создание узла сущности
        
        Args:
            entity_name: Имя сущности
            entity_type: Тип сущности
            description: Описание сущности (опционально)
        """
        if not entity_name or entity_name in self.entity_cache:
            return None

        # Нормализуем имя
        normalized = entity_name.strip().lower()
        entity_id = f"entity-{normalized.replace(' ', '-')}"

        return EntityNode(
            id=entity_id,
            label=entity_name,
            entity_type=entity_type,
            aliases=[normalized],
            description=description,
            importance=0.5,
            properties={"original_name": entity_name},
        )
    
    def _get_entity_description_from_dict(self, entity_name: str, entity_type: str) -> Optional[str]:
        """Получение описания сущности из словаря
        
        Args:
            entity_name: Имя сущности
            entity_type: Тип сущности
            
        Returns:
            Описание сущности или None
        """
        # GraphBuilder не имеет прямого доступа к EntityDictionary
        # Описание будет получено позже при обновлении узла в графе
        return None

    def _create_doc_chunk_node(
        self, message: Dict[str, Any], session_id: str, index: int
    ) -> Optional[DocChunkNode]:
        """Создание узла фрагмента документа из сообщения"""
        text = message.get("text", "")
        if not text or len(text) < 10:
            return None

        msg_id = f"doc-{session_id}-{index}"

        return DocChunkNode(
            id=msg_id,
            label=f"Message {index}",
            content=text[:500],  # Ограничиваем длину
            source=session_id,
            timestamp=message.get("date_utc", datetime.now().isoformat()),
            author=self._extract_author(message),
            message_count=1,
            properties={
                "full_text_length": len(text),
                "has_media": bool(message.get("media_type")),
            },
        )

    def _create_topic_node(
        self, topic: Dict[str, Any], session_id: str
    ) -> Optional[TopicNode]:
        """Создание узла темы из топика сессии"""
        title = topic.get("title", "")
        if not title:
            return None

        topic_id = f"topic-{title.lower().replace(' ', '-')[:50]}"

        return TopicNode(
            id=topic_id,
            label=title,
            topic_name=title,
            description=topic.get("summary", "")[:500],
            keywords=[],
            session_count=1,
            properties={"session_id": session_id},
        )

    def _extract_author(self, message: Dict[str, Any]) -> str:
        """Извлечение автора из сообщения"""
        from_data = message.get("from", {})
        if isinstance(from_data, dict):
            return (
                from_data.get("username")
                or from_data.get("display")
                or from_data.get("id", "unknown")
            )
        return "unknown"

    def _extract_mentions_from_text(self, text: str) -> Set[str]:
        """Извлечение упоминаний из текста"""
        mentions = set()

        # @упоминания
        at_mentions = re.findall(r"@(\w+)", text)
        mentions.update(at_mentions)

        # #хештеги
        hashtags = re.findall(r"#(\w+)", text)
        mentions.update(hashtags)

        return mentions

    def add_temporal_edges(self, time_window_minutes: int = 60) -> int:
        """
        Добавление временных связей между событиями

        Args:
            time_window_minutes: Окно времени для связывания событий

        Returns:
            Количество добавленных связей
        """
        logger.info("Добавление временных связей...")

        # Получаем все события
        events = self.graph.get_nodes_by_type(NodeType.EVENT, limit=DEFAULT_GRAPH_QUERY_LIMIT)

        # Сортируем по времени
        events_with_time = []
        for event in events:
            timestamp_str = event.properties.get("timestamp")
            if timestamp_str:
                try:
                    from ..utils.datetime_utils import parse_datetime_utc

                    timestamp = parse_datetime_utc(timestamp_str, use_zoneinfo=True)
                    events_with_time.append((event, timestamp))
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning(
                        f"Ошибка парсинга timestamp для события {event.id}: {e}"
                    )
                    # Пропускаем события с некорректными timestamp

        events_with_time.sort(key=lambda x: x[1])

        # Создаём временные связи
        edges_added = 0
        for i in range(len(events_with_time) - 1):
            event1, time1 = events_with_time[i]
            event2, time2 = events_with_time[i + 1]

            # Проверяем временное окно
            time_diff = (time2 - time1).total_seconds() / 60
            if time_diff <= time_window_minutes:
                edge = GraphEdge(
                    id=f"temporal-{event1.id}-{event2.id}",
                    source_id=event1.id,
                    target_id=event2.id,
                    type=EdgeType.TEMPORAL,
                    weight=1.0
                    - (time_diff / time_window_minutes),  # Вес убывает с расстоянием
                    properties={"time_diff_minutes": time_diff},
                )

                if self.graph.add_edge(edge):
                    edges_added += 1

        logger.info(f"Добавлено {edges_added} временных связей")
        return edges_added

    def add_semantic_edges(
        self, embeddings_map: Dict[str, List[float]], threshold: float = 0.8
    ) -> int:
        """
        Добавление семантических связей на основе эмбеддингов

        Args:
            embeddings_map: Словарь {node_id: embedding}
            threshold: Порог косинусного сходства

        Returns:
            Количество добавленных связей
        """
        logger.info("Добавление семантических связей...")

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        # Получаем узлы с эмбеддингами
        node_ids = list(embeddings_map.keys())
        embeddings = np.array([embeddings_map[nid] for nid in node_ids])

        # Вычисляем косинусное сходство
        similarity_matrix = cosine_similarity(embeddings)

        edges_added = 0
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                similarity = similarity_matrix[i][j]

                if similarity >= threshold:
                    edge = GraphEdge(
                        id=f"semantic-{node_ids[i]}-{node_ids[j]}",
                        source_id=node_ids[i],
                        target_id=node_ids[j],
                        type=EdgeType.SIMILAR_TO,
                        weight=similarity,
                        bidirectional=True,
                        properties={"similarity": float(similarity)},
                    )

                    if self.graph.add_edge(edge):
                        edges_added += 1

        logger.info(f"Добавлено {edges_added} семантических связей")
        return edges_added
