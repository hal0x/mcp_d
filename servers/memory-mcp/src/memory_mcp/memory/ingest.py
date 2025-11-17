"""Adapters that push MemoryRecord instances into the memory pipeline."""

from __future__ import annotations

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
        for (source, chat), nodes_records in by_source_chat.items():
            if len(nodes_records) < 2:
                continue
            # Сортируем по timestamp
            sorted_nodes = sorted(
                nodes_records,
                key=lambda x: x[1].timestamp,
            )
            # Создаем связи между соседними записями
            for i in range(len(sorted_nodes) - 1):
                prev_node, prev_record = sorted_nodes[i]
                next_node, next_record = sorted_nodes[i + 1]
                # Создаем связь только если записи близки по времени (в пределах 4 часов)
                time_diff = (next_record.timestamp - prev_record.timestamp).total_seconds()
                if time_diff <= 4 * 3600:  # 4 часа
                    edge = GraphEdge(
                        id=f"{prev_node.id}-next-{next_node.id}",
                        source_id=prev_node.id,
                        target_id=next_node.id,
                        type=EdgeType.RELATES_TO,
                        weight=0.5,  # Средний вес для временных связей
                        properties={"time_diff_seconds": time_diff},
                    )
                    try:
                        self.graph.add_edge(edge)
                    except Exception as e:
                        logger.debug(f"Failed to add edge between {prev_node.id} and {next_node.id}: {e}")
        
        return stats

    def _build_doc_node(self, record: MemoryRecord) -> DocChunkNode:
        created_at = record.timestamp.isoformat()
        props = dict(record.metadata)
        props["content"] = record.content
        props["source"] = record.source
        props["timestamp"] = created_at
        props.setdefault("tags", record.tags)
        props.setdefault("entities", record.entities)
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
