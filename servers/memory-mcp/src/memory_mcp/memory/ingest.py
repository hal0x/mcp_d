"""Adapters that push MemoryRecord instances into the memory pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional
from uuid import uuid4

from ..indexing import Attachment, MemoryRecord
from .graph_types import DocChunkNode, EdgeType, GraphEdge, NodeType
from .typed_graph import TypedGraphMemory

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestResult:
    records_ingested: int = 0
    attachments_ingested: int = 0


class MemoryIngestor:
    """Write normalized records into TypedGraphMemory."""

    def __init__(self, graph: TypedGraphMemory) -> None:
        self.graph = graph

    def ingest(self, records: Iterable[MemoryRecord]) -> IngestResult:
        stats = IngestResult()
        records_list = list(records)
        added_nodes = []
        
        for record in records_list:
            doc_node = self._build_doc_node(record)
            if self.graph.add_node(doc_node):
                stats.records_ingested += 1
                added_nodes.append((doc_node, record))
            else:
                logger.debug("Запись %s уже существует", record.record_id)
            for attachment in record.attachments:
                attachment_node = self._build_attachment_node(record, attachment)
                if attachment_node and self.graph.add_node(attachment_node):
                    stats.attachments_ingested += 1
                    edge = GraphEdge(
                        id=f"{doc_node.id}-attachment-{attachment_node.id}",
                        source_id=doc_node.id,
                        target_id=attachment_node.id,
                        type=EdgeType.RELATES_TO,
                        weight=0.3,
                    )
                    self.graph.add_edge(edge)
        
        # Создаем связи между записями из одного источника/чата
        # Группируем по источнику и чату
        from collections import defaultdict
        by_source_chat = defaultdict(list)
        for doc_node, record in added_nodes:
            chat = record.metadata.get("chat") or record.source
            key = (record.source, chat)
            by_source_chat[key].append((doc_node, record))
        
        # Создаем связи между соседними записями в одной группе (по timestamp)
        # Также проверяем существующие записи в графе для создания связей с уже проиндексированными
        for (source, chat), nodes_records in by_source_chat.items():
            if not nodes_records:
                continue
            
            # Сортируем по timestamp
            sorted_nodes = sorted(
                nodes_records,
                key=lambda x: x[1].timestamp,
            )
            
            # Создаем связи между новыми записями (если их больше одной)
            # Используем несколько критериев для создания связей
            if len(sorted_nodes) > 1:
                for i in range(len(sorted_nodes) - 1):
                    prev_node, prev_record = sorted_nodes[i]
                    next_node, next_record = sorted_nodes[i + 1]
                    
                    # Вычисляем разницу во времени
                    time_diff = (next_record.timestamp - prev_record.timestamp).total_seconds()
                    
                    # Получаем данные записей
                    prev_entities = set(prev_record.entities or [])
                    next_entities = set(next_record.entities or [])
                    prev_tags = set(prev_record.tags or [])
                    next_tags = set(next_record.tags or [])
                    prev_author = prev_record.author
                    next_author = next_record.author
                    
                    # Определяем тип связи и вес на основе критериев
                    weight = None
                    edge_type = EdgeType.RELATES_TO
                    edge_properties = {}
                    
                    # Критерий 1: Временная близость (расширенный диапазон)
                    if time_diff <= 4 * 3600:  # До 4 часов - сильная связь
                        weight = 0.7
                        edge_type = EdgeType.TEMPORAL
                        edge_properties = {"time_diff_seconds": time_diff, "link_reason": "temporal_close"}
                    elif time_diff <= 24 * 3600:  # До 24 часов - средняя связь
                        weight = 0.5
                        edge_type = EdgeType.TEMPORAL
                        edge_properties = {"time_diff_seconds": time_diff, "link_reason": "temporal_medium"}
                    elif time_diff <= 7 * 24 * 3600:  # До недели - слабая связь
                        weight = 0.3
                        edge_type = EdgeType.RELATES_TO
                        edge_properties = {"time_diff_seconds": time_diff, "link_reason": "temporal_far"}
                    
                    # Критерий 2: Общие сущности (сильная связь)
                    common_entities = prev_entities & next_entities
                    if common_entities and len(common_entities) > 0:
                        entity_weight = min(0.8, 0.5 + len(common_entities) * 0.1)
                        if weight is None or entity_weight > weight:
                            weight = entity_weight
                            edge_type = EdgeType.MENTIONS
                            edge_properties = {
                                "common_entities": list(common_entities),
                                "entity_count": len(common_entities),
                                "link_reason": "shared_entities",
                                "time_diff_seconds": time_diff
                            }
                    
                    # Критерий 3: Общие теги (средняя связь)
                    common_tags = prev_tags & next_tags
                    if common_tags and len(common_tags) > 0:
                        tag_weight = min(0.6, 0.4 + len(common_tags) * 0.1)
                        if weight is None or tag_weight > weight:
                            weight = tag_weight
                            edge_type = EdgeType.HAS_TOPIC
                            edge_properties = {
                                "common_tags": list(common_tags),
                                "tag_count": len(common_tags),
                                "link_reason": "shared_tags",
                                "time_diff_seconds": time_diff
                            }
                    
                    # Критерий 4: Один автор (слабая связь)
                    if prev_author and next_author and prev_author == next_author:
                        author_weight = 0.4
                        if weight is None or (weight < 0.5 and author_weight > weight):
                            weight = author_weight
                            edge_type = EdgeType.AUTHORED_BY
                            edge_properties = {
                                "author": prev_author,
                                "link_reason": "same_author",
                                "time_diff_seconds": time_diff
                            }
                    
                    # Создаем связь, если есть хотя бы один критерий
                    if weight is not None and weight >= 0.3:  # Минимальный порог веса
                        edge = GraphEdge(
                            id=f"{prev_node.id}-relates-{next_node.id}",
                            source_id=prev_node.id,
                            target_id=next_node.id,
                            type=edge_type,
                            weight=weight,
                            properties=edge_properties,
                        )
                        try:
                            if self.graph.add_edge(edge):
                                logger.debug(
                                    f"Создана связь {edge_type.value} между новыми записями {prev_node.id} -> {next_node.id} "
                                    f"(вес={weight:.2f}, причина={edge_properties.get('link_reason', 'unknown')})"
                                )
                        except Exception as e:
                            logger.debug(f"Failed to add edge between {prev_node.id} and {next_node.id}: {e}")
            
            # Связываем ВСЕ новые записи с существующими в графе (важно для создания связей)
            for new_node, new_record in sorted_nodes:
                self._link_to_existing_records(new_node, new_record, source, chat)
        
        return stats

    def _link_to_existing_records(
        self,
        new_node: DocChunkNode,
        new_record: MemoryRecord,
        source: str,
        chat: str,
    ) -> None:
        """Создает связи между новой записью и существующими записями в графе.
        
        Использует несколько критериев для создания связей:
        1. Временная близость (расширенный диапазон)
        2. Общие сущности (entities)
        3. Общие теги
        4. Один автор
        """
        try:
            cursor = self.graph.conn.cursor()
            
            # Получаем существующие записи из того же чата, отсортированные по timestamp
            # Увеличиваем лимит для лучшего покрытия
            query = """
                SELECT id, properties FROM nodes
                WHERE type = 'DocChunk' 
                AND id != ?
                AND properties IS NOT NULL
                AND (
                    json_extract(properties, '$.source') = ? 
                    OR json_extract(properties, '$.chat') = ?
                )
                ORDER BY json_extract(properties, '$.timestamp') DESC
                LIMIT 50
            """
            
            cursor.execute(query, (new_node.id, source, chat))
            existing_rows = cursor.fetchall()
            
            if not existing_rows:
                return
            
            # Парсим timestamp новой записи
            new_timestamp = new_record.timestamp
            new_entities = set(record.entities or [])
            new_tags = set(record.tags or [])
            new_author = record.author
            
            edges_created = 0
            max_edges_per_record = 20  # Ограничиваем количество связей на запись
            
            for row in existing_rows:
                if edges_created >= max_edges_per_record:
                    break
                    
                try:
                    props = json.loads(row["properties"]) if isinstance(row["properties"], str) else row["properties"]
                    if not props:
                        continue
                    
                    existing_id = row["id"]
                    
                    # Получаем timestamp существующей записи
                    existing_timestamp_str = props.get("timestamp") or props.get("created_at")
                    existing_timestamp = None
                    if existing_timestamp_str:
                        from ..utils.datetime_utils import parse_datetime_utc
                        existing_timestamp = parse_datetime_utc(existing_timestamp_str, default=None)
                    
                    # Получаем данные существующей записи
                    existing_entities = set(props.get("entities", []) or [])
                    existing_tags = set(props.get("tags", []) or [])
                    existing_author = props.get("author")
                    
                    # Вычисляем разницу во времени
                    time_diff = None
                    if existing_timestamp:
                        time_diff = abs((new_timestamp - existing_timestamp).total_seconds())
                    
                    # Определяем направление связи (от более старой к более новой)
                    if existing_timestamp and new_timestamp > existing_timestamp:
                        source_id = existing_id
                        target_id = new_node.id
                    else:
                        source_id = new_node.id
                        target_id = existing_id
                    
                    edge_id = f"{source_id}-relates-{target_id}"
                    
                    # Критерий 1: Временная близость (расширенный диапазон с разными весами)
                    weight = None
                    edge_type = EdgeType.RELATES_TO
                    edge_properties = {}
                    
                    if time_diff is not None:
                        if time_diff <= 4 * 3600:  # До 4 часов - сильная связь
                            weight = 0.7
                            edge_type = EdgeType.TEMPORAL
                            edge_properties = {"time_diff_seconds": time_diff, "link_reason": "temporal_close"}
                        elif time_diff <= 24 * 3600:  # До 24 часов - средняя связь
                            weight = 0.5
                            edge_type = EdgeType.TEMPORAL
                            edge_properties = {"time_diff_seconds": time_diff, "link_reason": "temporal_medium"}
                        elif time_diff <= 7 * 24 * 3600:  # До недели - слабая связь
                            weight = 0.3
                            edge_type = EdgeType.RELATES_TO
                            edge_properties = {"time_diff_seconds": time_diff, "link_reason": "temporal_far"}
                    
                    # Критерий 2: Общие сущности (сильная связь)
                    common_entities = new_entities & existing_entities
                    if common_entities and len(common_entities) > 0:
                        entity_weight = min(0.8, 0.5 + len(common_entities) * 0.1)
                        if weight is None or entity_weight > weight:
                            weight = entity_weight
                            edge_type = EdgeType.MENTIONS
                            edge_properties = {
                                "common_entities": list(common_entities),
                                "entity_count": len(common_entities),
                                "link_reason": "shared_entities"
                            }
                            if time_diff is not None:
                                edge_properties["time_diff_seconds"] = time_diff
                    
                    # Критерий 3: Общие теги (средняя связь)
                    common_tags = new_tags & existing_tags
                    if common_tags and len(common_tags) > 0:
                        tag_weight = min(0.6, 0.4 + len(common_tags) * 0.1)
                        if weight is None or tag_weight > weight:
                            weight = tag_weight
                            edge_type = EdgeType.HAS_TOPIC
                            edge_properties = {
                                "common_tags": list(common_tags),
                                "tag_count": len(common_tags),
                                "link_reason": "shared_tags"
                            }
                            if time_diff is not None:
                                edge_properties["time_diff_seconds"] = time_diff
                    
                    # Критерий 4: Один автор (слабая связь, но добавляет контекст)
                    if new_author and existing_author and new_author == existing_author:
                        author_weight = 0.4
                        if weight is None or (weight < 0.5 and author_weight > weight):
                            weight = author_weight
                            edge_type = EdgeType.AUTHORED_BY
                            edge_properties = {
                                "author": new_author,
                                "link_reason": "same_author"
                            }
                            if time_diff is not None:
                                edge_properties["time_diff_seconds"] = time_diff
                    
                    # Создаем связь, если есть хотя бы один критерий
                    if weight is not None and weight >= 0.3:  # Минимальный порог веса
                        edge = GraphEdge(
                            id=edge_id,
                            source_id=source_id,
                            target_id=target_id,
                            type=edge_type,
                            weight=weight,
                            properties=edge_properties,
                        )
                        try:
                            if self.graph.add_edge(edge):
                                edges_created += 1
                                logger.debug(
                                    f"Создана связь {edge_type.value} между {source_id} и {target_id} "
                                    f"(вес={weight:.2f}, причина={edge_properties.get('link_reason', 'unknown')})"
                                )
                        except Exception as e:
                            # Игнорируем ошибки, если связь уже существует
                            logger.debug(f"Failed to add edge between {source_id} and {target_id}: {e}")
                            
                except Exception as e:
                    logger.debug(f"Ошибка при создании связи с существующей записью {row['id']}: {e}")
                    continue
                    
            if edges_created > 0:
                logger.debug(f"Создано {edges_created} связей для записи {new_node.id}")
        except Exception as e:
            logger.debug(f"Ошибка при связывании новой записи {new_node.id} с существующими: {e}")

    def _build_doc_node(self, record: MemoryRecord) -> DocChunkNode:
        created_at = record.timestamp.isoformat()
        props = dict(record.metadata)
        props["content"] = record.content
        props["source"] = record.source
        props["timestamp"] = created_at
        # ВАЖНО: сохраняем теги и сущности в properties для FTS5 индексации
        if record.tags:
            props["tags"] = record.tags if isinstance(record.tags, list) else [record.tags]
        else:
            props.setdefault("tags", [])
        if record.entities:
            props["entities"] = record.entities if isinstance(record.entities, list) else [record.entities]
        else:
            props.setdefault("entities", [])
        if record.author:
            props["author"] = record.author
        return DocChunkNode(
            id=record.record_id,
            label=record.metadata.get("chat") or record.source,
            content=record.content,
            source=record.source,
            timestamp=created_at,
            author=record.author,
            created_at=created_at,
            updated_at=created_at,
            properties=props,
        )

    def _build_attachment_node(
        self,
        record: MemoryRecord,
        attachment: Attachment,
    ) -> Optional[DocChunkNode]:
        if not attachment.uri and not attachment.text:
            return None
        # Используем UUID для генерации уникального ID вложения
        attachment_id = f"{record.record_id}-att-{uuid4().hex[:16]}"
        props = dict(attachment.metadata)
        props["type"] = attachment.type
        if attachment.uri:
            props["uri"] = attachment.uri
        props["content"] = attachment.text or attachment.uri or ""
        props["source"] = f"{record.source}:attachment"
        props["timestamp"] = record.timestamp.isoformat()
        return DocChunkNode(
            id=attachment_id,
            label=attachment.type,
            content=attachment.text or attachment.uri or "",
            source=f"{record.source}:attachment",
            timestamp=record.timestamp.isoformat(),
            author=record.author,
            created_at=record.timestamp.isoformat(),
            updated_at=record.timestamp.isoformat(),
            properties=props,
        )
