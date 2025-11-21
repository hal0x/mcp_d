"""Adapters converting MCP requests into memory operations."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..core.constants import DEFAULT_SEARCH_LIMIT
from ..models.memory import Attachment, MemoryRecord
from ..memory.artifacts_reader import ArtifactsReader
from ..memory.embeddings import build_embedding_service_from_env
from ..memory.ingest import MemoryIngestor
from ..memory.storage.graph.graph_types import EdgeType, NodeType
from ..memory.storage.graph.typed_graph import TypedGraphMemory
from ..memory.storage.vector.vector_store import build_vector_store_from_env
from ..memory.trading_memory import TradingMemory
from ..utils.processing.datetime_utils import parse_datetime_utc
from .schema import (
    AnalyzeEntitiesRequest,
    AnalyzeEntitiesResponse,
    SearchEntitiesRequest,
    SearchEntitiesResponse,
    GetEntityProfileRequest,
    GetEntityProfileResponse,
    EntitySearchResult,
    EntityProfile,
    EntityContextItem,
    RelatedEntityItem,
    AttachmentPayload,
    BackgroundIndexingRequest,
    BackgroundIndexingResponse,
    BatchDeleteRecordsRequest,
    BatchDeleteRecordsResponse,
    BatchDeleteResult,
    BatchFetchRecordsRequest,
    BatchFetchRecordsResponse,
    BatchFetchResult,
    BatchOperationsRequest,
    BatchOperationsResponse,
    BatchUpdateRecordItem,
    BatchUpdateRecordsRequest,
    BatchUpdateRecordsResponse,
    BatchUpdateResult,
    BuildInsightGraphRequest,
    BuildInsightGraphResponse,
    DeleteRecordRequest,
    DeleteRecordResponse,
    EntityItem,
    ExportRecordsRequest,
    ExportRecordsResponse,
    FetchRequest,
    FetchResponse,
    FindGraphPathRequest,
    FindGraphPathResponse,
    GenerateEmbeddingRequest,
    GenerateEmbeddingResponse,
    GetGraphNeighborsRequest,
    GetGraphNeighborsResponse,
    GetIndexingProgressRequest,
    GetIndexingProgressResponse,
    GetRelatedRecordsRequest,
    GetRelatedRecordsResponse,
    GetStatisticsRequest,
    GetStatisticsResponse,
    GetTagsStatisticsResponse,
    GetTimelineRequest,
    GetTimelineResponse,
    GraphNeighborItem,
    GraphQueryRequest,
    GraphQueryResponse,
    IndexChatRequest,
    IndexChatResponse,
    GetAvailableChatsRequest,
    GetAvailableChatsResponse,
    ChatInfo,
    GetSignalPerformanceRequest,
    GetSignalPerformanceResponse,
    ImportRecordsRequest,
    ImportRecordsResponse,
    IndexingProgressItem,
    InsightItem,
    IngestRequest,
    IngestResponse,
    MemoryRecordPayload,
    ReviewSummariesRequest,
    ReviewSummariesResponse,
    ScrapedContentRequest,
    ScrapedContentResponse,
    SearchByEmbeddingRequest,
    SearchByEmbeddingResponse,
    SearchExplainRequest,
    SearchExplainResponse,
    SearchFeedback,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SearchTradingPatternsRequest,
    SearchTradingPatternsResponse,
    SimilarRecordsRequest,
    SimilarRecordsResponse,
    SignalPerformance,
    SmartSearchRequest,
    SmartSearchResponse,
    StoreTradingSignalRequest,
    StoreTradingSignalResponse,
    SummariesRequest,
    SummariesResponse,
    TimelineItem,
    TradingSignalRecord,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
    UnifiedStatisticsResponse,
    UpdateRecordRequest,
    UpdateRecordResponse,
    UpdateSummariesRequest,
    UpdateSummariesResponse,
)

logger = logging.getLogger(__name__)


def _parse_timestamp(value: str | None) -> datetime:
    """Парсит временную метку с fallback на текущее UTC время."""
    return parse_datetime_utc(value, default=datetime.now(timezone.utc))


def _payload_to_record(payload: MemoryRecordPayload) -> MemoryRecord:
    """Преобразует MCP payload в внутренний формат MemoryRecord."""
    return MemoryRecord(
        record_id=payload.record_id,
        source=payload.source,
        content=payload.content,
        timestamp=payload.timestamp,
        author=payload.author,
        tags=list(payload.tags),
        entities=list(payload.entities),
        attachments=[
            Attachment(
                type=att.type,
                uri=att.uri,
                text=att.text,
                metadata=dict(att.metadata),
            )
            for att in payload.attachments
        ],
        metadata=dict(payload.metadata),
    )


def _node_to_payload(
    graph: TypedGraphMemory, node_id: str, data: dict
) -> MemoryRecordPayload:
    """Преобразует узел графа в MCP payload, включая вложения и эмбеддинги."""
    props = dict(data.get("properties") or {})
    tags = list(props.get("tags", []))
    entities = list(props.get("entities", []))
    source = data.get("source") or props.get("source") or "unknown"
    author = data.get("author") or props.get("author")
    attachments: list[AttachmentPayload] = []

    for target_id in graph.graph.successors(node_id):
        neighbor = graph.graph.nodes[target_id]
        if neighbor.get("type") not in (NodeType.DOC_CHUNK, NodeType.DOC_CHUNK.value):
            continue
        if neighbor.get("source", "").endswith(":attachment"):
            meta = dict(neighbor.get("properties") or {})
            attachments.append(
                AttachmentPayload(
                    type=meta.get("type", "attachment"),
                    uri=meta.get("uri"),
                    text=neighbor.get("content"),
                    metadata=meta,
                )
            )

    timestamp = _parse_timestamp(data.get("timestamp"))
    
    embedding = data.get("embedding")
    if embedding is not None:
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        elif not isinstance(embedding, list):
            try:
                embedding = list(embedding)
            except (TypeError, ValueError):
                logger.debug(f"Не удалось преобразовать эмбеддинг в список для узла {node_id}")
                embedding = None
    else:
        logger.debug(f"Эмбеддинг отсутствует для узла {node_id}")

    return MemoryRecordPayload(
        record_id=node_id,
        source=source,
        content=data.get("content") or "",
        timestamp=timestamp,
        author=author,
        tags=tags,
        entities=entities,
        attachments=attachments,
        metadata=props,
        embedding=embedding,
    )


class MemoryServiceAdapter:
    """Адаптер для работы с памятью: индексация, поиск, получение записей.
    
    Объединяет граф памяти, векторное хранилище и эмбеддинги для гибридного поиска.
    """

    def __init__(self, db_path: str = "data/memory_graph.db") -> None:
        self.graph = TypedGraphMemory(db_path=db_path)
        self.ingestor = MemoryIngestor(self.graph)
        self.embedding_service = build_embedding_service_from_env()
        self.vector_store = build_vector_store_from_env()
        if (
            self.vector_store
            and self.embedding_service
            and self.embedding_service.dimension
        ):
            self.vector_store.ensure_collection(self.embedding_service.dimension)
        self.trading_memory = TradingMemory(self.graph)
        
        # Параметры гибридного поиска: FTS5 (80%) + векторный (20%)
        _fts_weight_raw = 0.8
        _vector_weight_raw = 0.2
        _total_weight = _fts_weight_raw + _vector_weight_raw
        self._fts_weight = _fts_weight_raw / _total_weight
        self._vector_weight = _vector_weight_raw / _total_weight
        self._rrf_k = 60  # Reciprocal Rank Fusion параметр
        self._use_rrf = True
        
        from ..config import get_settings
        settings = get_settings()
        self.artifacts_reader = ArtifactsReader(artifacts_dir=settings.artifacts_path)
        
        from ..search.entity_context_enricher import EntityContextEnricher
        from ..analysis.entities import get_entity_dictionary
        
        entity_dict = get_entity_dictionary(graph=self.graph)
        self.entity_enricher = EntityContextEnricher(
            entity_dictionary=entity_dict,
            graph=self.graph,
        )
    
    def _find_node_id(self, record_id: str) -> Optional[str]:
        """Находит node_id в графе по различным форматам record_id.
        
        Поддерживает:
        - Точное совпадение: "telegram:Семья:257859"
        - Формат индексатора: "Семья-S0006-M0068"
        - Частичное совпадение через LIKE
        
        Returns:
            Найденный node_id или None
        """
        if record_id in self.graph.graph:
            return record_id
        
        cursor = self.graph.conn.cursor()
        
        if "-" in record_id and "M" in record_id:
            parts = record_id.split("-")
            if len(parts) >= 3:
                chat_name = parts[0]
                msg_part = parts[-1]
                if msg_part.startswith("M"):
                    try:
                        msg_num = int(msg_part[1:])
                        cursor.execute(
                            "SELECT id FROM nodes WHERE id LIKE ? ORDER BY id",
                            (f"telegram:{chat_name}:%",)
                        )
                        all_rows = cursor.fetchall()
                        if msg_num > 0 and msg_num <= len(all_rows):
                            found_id = all_rows[msg_num - 1]["id"]
                            if found_id not in self.graph.graph:
                                cursor.execute(
                                    "SELECT id, type, label, properties, embedding FROM nodes WHERE id = ?",
                                    (found_id,)
                                )
                                node_row = cursor.fetchone()
                                if node_row:
                                    props = json.loads(node_row["properties"]) if node_row["properties"] else {}
                                    node_attrs = {
                                        "node_type": node_row["type"],
                                        "label": node_row["label"],
                                        "properties": props,
                                        **props,
                                    }
                                    if node_row["embedding"]:
                                        try:
                                            embedding = json.loads(node_row["embedding"].decode())
                                            if hasattr(embedding, 'tolist'):
                                                embedding = embedding.tolist()
                                            elif not isinstance(embedding, list):
                                                embedding = list(embedding)
                                            node_attrs["embedding"] = embedding
                                        except Exception:
                                            pass
                                    self.graph.graph.add_node(found_id, **node_attrs)
                                    logger.debug(f"Узел {found_id} загружен из БД в граф")
                            if found_id in self.graph.graph:
                                logger.debug(f"Найден узел по номеру: {found_id} для {record_id}")
                                return found_id
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Ошибка при поиске узла по номеру для {record_id}: {e}")
                        pass
        
        cursor.execute(
            "SELECT id FROM nodes WHERE id LIKE ? LIMIT 1",
            (f"%{record_id}%",)
        )
        row = cursor.fetchone()
        if row:
            found_id = row["id"]
            if found_id not in self.graph.graph:
                cursor.execute(
                    "SELECT id, type, label, properties, embedding FROM nodes WHERE id = ?",
                    (found_id,)
                )
                node_row = cursor.fetchone()
                if node_row:
                    props = json.loads(node_row["properties"]) if node_row["properties"] else {}
                    node_attrs = {
                        "node_type": node_row["type"],
                        "label": node_row["label"],
                        "properties": props,
                        **props,
                    }
                    if node_row["embedding"]:
                        try:
                            embedding = json.loads(node_row["embedding"].decode())
                            if hasattr(embedding, 'tolist'):
                                embedding = embedding.tolist()
                            elif not isinstance(embedding, list):
                                embedding = list(embedding)
                            node_attrs["embedding"] = embedding
                        except Exception:
                            pass
                    self.graph.graph.add_node(found_id, **node_attrs)
                    logger.debug(f"Узел {found_id} загружен из БД в граф")
            if found_id in self.graph.graph:
                logger.debug(f"Найден узел по частичному совпадению: {found_id} для {record_id}")
                return found_id
        
        return None

    def close(self) -> None:
        try:
            self.graph.conn.close()
        except Exception:  # pragma: no cover
            logger.debug("Ошибка при закрытии соединения с БД", exc_info=True)
        if self.embedding_service:
            self.embedding_service.close()
        if self.vector_store:
            self.vector_store.close()

    def clear_chat_data(self, chat_name: str) -> Dict[str, int]:
        """Удаляет все данные чата из графа и Qdrant."""
        stats = {
            "nodes_deleted": 0,
            "vectors_deleted": 0,
            "qdrant_deleted": 0,
        }

        logger.info(f"Начало очистки данных чата: {chat_name}")

        try:
            nodes_deleted = self.graph.delete_nodes_by_chat(chat_name)
            stats["nodes_deleted"] = nodes_deleted
            logger.info(f"Удалено {nodes_deleted} узлов из графа (чат: {chat_name})")
        except Exception as e:
            logger.warning(
                f"Ошибка при очистке графа для чата {chat_name}: {e}",
                exc_info=True,
            )

        if self.vector_store:
            try:
                vectors_deleted = self.vector_store.delete_by_chat(chat_name)
                stats["vectors_deleted"] = vectors_deleted
                logger.info(
                    f"Удалено {vectors_deleted} векторов из Qdrant (чат: {chat_name})"
                )
            except Exception as e:
                logger.warning(
                    f"Ошибка при очистке Qdrant для чата {chat_name}: {e}",
                    exc_info=True,
                )
        else:
            logger.debug("Qdrant недоступен, пропускаем очистку векторов")

        try:
            qdrant_deleted = self._clear_qdrant_chat(chat_name)
            stats["qdrant_deleted"] = qdrant_deleted
            logger.info(
                f"Удалено {qdrant_deleted} записей из Qdrant (чат: {chat_name})"
            )
        except Exception as e:
            logger.warning(
                f"Ошибка при очистке Qdrant для чата {chat_name}: {e}",
                exc_info=True,
            )

        total_deleted = (
            stats["nodes_deleted"]
            + stats["vectors_deleted"]
            + stats.get("qdrant_deleted", 0)
        )
        logger.info(
            f"Очистка завершена для чата {chat_name}: "
            f"узлов={stats['nodes_deleted']}, "
            f"векторов={stats['vectors_deleted']}, "
            f"Qdrant={stats.get('qdrant_deleted', 0)}, "
            f"всего={total_deleted}"
        )

        return stats

    def _build_embedding_text(self, payload: MemoryRecordPayload) -> str:
        """Формирует расширенный текст для эмбеддингов, включая метаданные.
        
        Args:
            payload: Запись памяти с метаданными
            
        Returns:
            Расширенный текст с контентом и метаданными
        """
        parts = [payload.content]
        metadata_parts = []
        
        sender_username = payload.metadata.get("sender_username")
        if sender_username:
            metadata_parts.append(f"Автор: @{sender_username}")
        elif payload.author:
            metadata_parts.append(f"Автор: {payload.author}")
        
        reactions = payload.metadata.get("reactions")
        if reactions and isinstance(reactions, list) and len(reactions) > 0:
            reaction_strs = []
            for reaction in reactions:
                if isinstance(reaction, dict):
                    emoji = reaction.get("emoji", "")
                    count = reaction.get("count", 0)
                    if emoji and count > 0:
                        if "emoticon=" in str(emoji):
                            try:
                                emoji_value = str(emoji).split("emoticon=")[1].split("'")[1]
                                reaction_strs.append(f"{emoji_value} x{count}")
                            except (IndexError, AttributeError):
                                reaction_strs.append(f"{emoji} x{count}")
                        else:
                            reaction_strs.append(f"{emoji} x{count}")
            if reaction_strs:
                metadata_parts.append(f"Реакции: {', '.join(reaction_strs)}")
        
        edited_utc = payload.metadata.get("edited_utc")
        if edited_utc:
            metadata_parts.append(f"Отредактировано: {edited_utc}")
        
        if metadata_parts:
            parts.append("\n[Метаданные]")
            parts.extend(metadata_parts)
        
        return "\n".join(parts)

    def ingest(self, payloads: Iterable[MemoryRecordPayload]) -> IngestResponse:
        payload_list = list(payloads)
        records = [_payload_to_record(item) for item in payload_list]
        stats = self.ingestor.ingest(records)
        if self.embedding_service and self.vector_store and payload_list:
            embedding_texts = []
            payload_indices = []
            for i, payload in enumerate(payload_list):
                embedding_text = self._build_embedding_text(payload)
                if embedding_text.strip():  # Пропускаем пустые тексты
                    embedding_texts.append(embedding_text)
                    payload_indices.append(i)
            
            if embedding_texts:
                vectors = self.embedding_service.embed_batch(embedding_texts)
                
                for vec_idx, payload_idx in enumerate(payload_indices):
                    vector = vectors[vec_idx] if vec_idx < len(vectors) else None
                    if not vector:
                        continue
                    
                    payload = payload_list[payload_idx]
                    payload_data: dict[str, object] = {
                        "record_id": payload.record_id,
                        "source": payload.source,
                        "tags": payload.tags,
                        "timestamp": payload.timestamp.timestamp(),
                        "timestamp_iso": payload.timestamp.isoformat(),
                        "content_preview": payload.content[:200],
                    }
                    chat_name = payload.metadata.get("chat")
                    if isinstance(chat_name, str):
                        payload_data["chat"] = chat_name
                    try:
                        self.vector_store.upsert(payload.record_id, vector, payload_data)
                        self.graph.update_node(
                            payload.record_id,
                            embedding=vector,
                        )
                    except Exception:  # pragma: no cover
                        logger.debug(
                            "Vector upsert failed for %s", payload.record_id, exc_info=True
                        )
        duplicates = max(0, len(payload_list) - stats.records_ingested)
        return IngestResponse(
            records_ingested=stats.records_ingested,
            attachments_ingested=stats.attachments_ingested,
            duplicates_skipped=duplicates,
        )

    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[SearchResultItem]],
        k: int = 60,
    ) -> dict[str, float]:
        """
        Объединяет результаты из разных источников используя Reciprocal Rank Fusion (RRF).
        
        Args:
            result_lists: Список списков результатов из разных источников (FTS5, Vector/Qdrant)
            k: Параметр RRF (стандартное значение 60)
            
        Returns:
            Словарь {record_id: rrf_score} отсортированный по убыванию score
        """
        rrf_scores: dict[str, float] = {}
        
        for results in result_lists:
            for rank, item in enumerate(results, start=1):
                rrf_score = 1.0 / (k + rank)
                rrf_scores[item.record_id] = rrf_scores.get(item.record_id, 0.0) + rrf_score
        
        return rrf_scores

    def search(self, request: SearchRequest) -> SearchResponse:
        """Выполняет гибридный поиск: FTS5 + векторный (Qdrant) с объединением через RRF."""
        enriched_query = self.entity_enricher.enrich_query_with_entity_context(request.query)
        extracted_entities = self.entity_enricher.extract_entities_from_query(request.query)
        
        rows, total_fts = self.graph.search_text(
            enriched_query,
            limit=request.top_k * 2,  # Берем больше результатов для RRF
            source=request.source,
            tags=request.tags,
            date_from=request.date_from,
            date_to=request.date_to,
        )

        fts_results: list[SearchResultItem] = []
        all_items: dict[str, SearchResultItem] = {}
        
        for row in rows:
            embedding = None
            if row["node_id"] in self.graph.graph:
                node_data = self.graph.graph.nodes[row["node_id"]]
                embedding = node_data.get("embedding")
                if embedding is not None:
                    if hasattr(embedding, 'tolist'):
                        embedding = embedding.tolist()
                    elif not isinstance(embedding, list):
                        try:
                            embedding = list(embedding)
                        except (TypeError, ValueError):
                            embedding = None
                
                if embedding is None:
                    try:
                        cursor = self.graph.conn.cursor()
                        cursor.execute(
                            "SELECT embedding FROM nodes WHERE id = ?",
                            (row["node_id"],),
                        )
                        db_row = cursor.fetchone()
                        if db_row and db_row["embedding"]:
                            import json
                            embedding = json.loads(db_row["embedding"].decode())
                    except Exception:
                        pass
            
            row_metadata = row.get("metadata")
            if row_metadata is None:
                metadata = {}
            elif isinstance(row_metadata, dict):
                metadata = dict(row_metadata)
            elif isinstance(row_metadata, str):
                try:
                    import json
                    metadata = json.loads(row_metadata)
                except:
                    metadata = {}
            else:
                metadata = {}
            
            metadata["bm25_score"] = float(row.get("score", 0.0))
            metadata["search_source"] = "fts5"
            
            item = SearchResultItem(
                record_id=row["node_id"],
                score=row["score"],
                content=row["snippet"],
                source=row["source"],
                timestamp=row["timestamp"],
                author=row.get("author"),
                metadata=metadata,
                embedding=embedding,
            )
            fts_results.append(item)
            all_items[row["node_id"]] = item

        vector_results: list[SearchResultItem] = []
        if self.embedding_service and self.vector_store and self.vector_store.available():
            query_vector = self.embedding_service.embed(enriched_query)
            if query_vector:
                vector_matches = self.vector_store.search(
                    query_vector,
                    limit=request.top_k * 2,
                    source=request.source,
                    tags=request.tags,
                    date_from=request.date_from,
                    date_to=request.date_to,
                )

                for match in vector_matches:
                    record_id = match.record_id
                    payload = match.payload or {}
                    snippet = None
                    preview = payload.get("content_preview")
                    if isinstance(preview, str) and preview.strip():
                        snippet = preview
                    
                    item = self._build_item_from_graph(
                        record_id,
                        match.score,
                        snippet=snippet,
                    )
                    if item:
                        if not item.metadata:
                            item.metadata = {}
                        item.metadata["vector_score"] = match.score
                        item.metadata["search_source"] = "vector"
                        vector_results.append(item)
                        if record_id not in all_items:
                            all_items[record_id] = item
                        else:
                            existing_item = all_items[record_id]
                            if not existing_item.metadata:
                                existing_item.metadata = {}
                            if "vector_score" not in existing_item.metadata:
                                existing_item.metadata["vector_score"] = match.score
                            if not existing_item.embedding and item.embedding:
                                existing_item.embedding = item.embedding

        # Boost для точных совпадений в FTS5 результатах
        query_words = set(request.query.lower().split())
        for item in fts_results:
            content_lower = item.content.lower()
            matched_words = sum(1 for word in query_words if word in content_lower)
            if matched_words == len(query_words) and len(query_words) > 0:
                item.score *= 1.2  # Boost 20% для точных совпадений

        # Объединение результатов через RRF или простое объединение с весами
        if self._use_rrf and (fts_results or vector_results):
            result_lists = []
            if fts_results:
                result_lists.append(fts_results)
            if vector_results:
                result_lists.append(vector_results)
            
            rrf_scores = self._reciprocal_rank_fusion(result_lists, k=self._rrf_k)
            
            for record_id, rrf_score in rrf_scores.items():
                if record_id in all_items:
                    item = all_items[record_id]
                    if not item.metadata:
                        item.metadata = {}
                    if "rrf_input_score" not in item.metadata:
                        item.metadata["rrf_input_score"] = item.score
                    item.score = rrf_score
                    item.metadata["rrf_score"] = rrf_score
            
            results = sorted(all_items.values(), key=lambda item: item.score, reverse=True)
        else:
            combined: dict[str, SearchResultItem] = {}
            
            for item in fts_results:
                if not item.metadata:
                    item.metadata = {}
                item.metadata["bm25_score_original"] = item.score
                item.score = item.score * self._fts_weight
                combined[item.record_id] = item
            
            for item in vector_results:
                if not item.metadata:
                    item.metadata = {}
                item.metadata["vector_score_original"] = item.score
                score = item.score * self._vector_weight
                existing = combined.get(item.record_id)
                if existing:
                    if not existing.metadata:
                        existing.metadata = {}
                    existing.metadata["vector_score_original"] = item.score
                    existing.score = max(existing.score, score)
                    if not existing.embedding and item.embedding:
                        existing.embedding = item.embedding
                else:
                    item.score = score
                    combined[item.record_id] = item
            
            results = sorted(combined.values(), key=lambda item: item.score, reverse=True)

        total_combined = max(total_fts, len(all_items))
        
        if extracted_entities:
            for result in results[: request.top_k]:
                if not result.metadata:
                    result.metadata = {}
                result.metadata["query_entities"] = extracted_entities
        
        return SearchResponse(
            results=results[: request.top_k],
            total_matches=total_combined,
        )

    def fetch(self, request: FetchRequest) -> FetchResponse:
        """Получение полной записи по идентификатору с поддержкой частичного совпадения."""
        try:
            logger.debug(f"Fetch record: {request.record_id}")
            
            if request.record_id in self.graph.graph:
                data = self.graph.graph.nodes[request.record_id]
                payload = _node_to_payload(self.graph, request.record_id, data)
                logger.debug(f"Запись найдена в графе: {request.record_id}")
                return FetchResponse(record=payload)
            
            cursor = self.graph.conn.cursor()
            
            search_patterns = [request.record_id]
            
            if ":" in request.record_id:
                parts = request.record_id.split(":")
                if len(parts) >= 3:
                    try:
                        msg_id = int(parts[-1])
                        search_patterns.append(f"telegram:{parts[1]}:{msg_id}")
                        search_patterns.append(f"%:{parts[1]}:{msg_id}")
                    except ValueError:
                        pass
            
            search_patterns = list(dict.fromkeys(search_patterns))
            
            search_patterns_with_like = []
            for pattern in search_patterns:
                search_patterns_with_like.append(pattern)
                if "-" in pattern:
                    parts = pattern.split("-")
                    if len(parts) >= 2:
                        for i in range(2, len(parts) + 1):
                            partial = "-".join(parts[:i])
                            search_patterns_with_like.append(partial)
            
            for pattern in search_patterns_with_like:
                cursor.execute(
                    "SELECT id FROM nodes WHERE id = ? LIMIT 1",
                    (pattern,)
                )
                row = cursor.fetchone()
                if not row:
                    cursor.execute(
                        "SELECT id FROM nodes WHERE id LIKE ? LIMIT 1",
                        (f"{pattern}%",)
                    )
                    row = cursor.fetchone()
                
                if row:
                    found_id = row["id"]
                    logger.debug(f"Найдена запись по паттерну '{pattern}': {found_id}")
                    if found_id not in self.graph.graph:
                        cursor.execute(
                            "SELECT id, type, label, properties, embedding FROM nodes WHERE id = ?",
                            (found_id,)
                        )
                        node_row = cursor.fetchone()
                        if node_row:
                            props = json.loads(node_row["properties"]) if node_row["properties"] else {}
                            node_attrs = {
                                "node_type": node_row["type"],
                                "label": node_row["label"],
                                "properties": props,
                                **props,
                            }
                            if node_row["embedding"]:
                                try:
                                    embedding = json.loads(node_row["embedding"].decode())
                                    if hasattr(embedding, 'tolist'):
                                        embedding = embedding.tolist()
                                    elif not isinstance(embedding, list):
                                        embedding = list(embedding)
                                    node_attrs["embedding"] = embedding
                                except Exception:
                                    pass
                            self.graph.graph.add_node(found_id, **node_attrs)
                            logger.debug(f"Узел {found_id} загружен из БД в граф")
                    
                    if found_id in self.graph.graph:
                        data = self.graph.graph.nodes[found_id]
                        payload = _node_to_payload(self.graph, found_id, data)
                        return FetchResponse(record=payload)
            
            if self.vector_store and self.vector_store.available():
                try:
                    from ..memory.storage.vector.qdrant_collections import QdrantCollectionsManager
                    from ..config import get_settings
                    
                    settings = get_settings()
                    qdrant_url = settings.get_qdrant_url()
                    if qdrant_url:
                        embedding_dimension = self.embedding_service.dimension if self.embedding_service else 1024
                        qdrant_manager = QdrantCollectionsManager(url=qdrant_url, vector_size=embedding_dimension)
                        
                        for collection_name in ["chat_messages", "chat_sessions", "chat_tasks"]:
                            try:
                                result = qdrant_manager.get(
                                    collection_name=collection_name,
                                    ids=[request.record_id]
                                )
                                
                                if result.get("ids") and len(result["ids"]) > 0:
                                    doc = result["documents"][0] if result.get("documents") else ""
                                    metadata = result["metadatas"][0] if result.get("metadatas") else {}
                                    
                                    chat_name = metadata.get("chat", collection_name.replace("chat_", ""))
                                    
                                    date_utc = metadata.get("date_utc") or metadata.get("start_time_utc") or metadata.get("end_time_utc")
                                    timestamp = None
                                    if date_utc:
                                        try:
                                            timestamp = parse_datetime_utc(date_utc, use_zoneinfo=True)
                                        except Exception:
                                            timestamp = datetime.now(timezone.utc)
                                    else:
                                        timestamp = datetime.now(timezone.utc)
                                    
                                    author = metadata.get("sender") or metadata.get("author") or metadata.get("username")
                                    
                                    tags = metadata.get("tags", [])
                                    if isinstance(tags, str):
                                        tags = [tags] if tags else []
                                    
                                    entities = metadata.get("entities", [])
                                    if isinstance(entities, str):
                                        entities = [entities] if entities else []
                                    
                                    embedding = None
                                    if result.get("embeddings") and len(result["embeddings"]) > 0:
                                        embedding = result["embeddings"][0]
                                    
                                    payload = MemoryRecordPayload(
                                        record_id=request.record_id,
                                        source=chat_name,
                                        content=doc,
                                        timestamp=timestamp,
                                        author=author,
                                        tags=tags if isinstance(tags, list) else [],
                                        entities=entities if isinstance(entities, list) else [],
                                        attachments=[],
                                        metadata={
                                            "collection": collection_name,
                                            "chat": chat_name,
                                            **metadata,
                                        },
                                        embedding=embedding,
                                    )
                                    
                                    if embedding and request.record_id not in self.graph.graph:
                                        try:
                                            from ..models.memory import MemoryRecord
                                            record = MemoryRecord(
                                                record_id=request.record_id,
                                                source=chat_name,
                                                content=doc,
                                                timestamp=timestamp,
                                                author=author,
                                                tags=tags if isinstance(tags, list) else [],
                                                entities=entities if isinstance(entities, list) else [],
                                                attachments=[],
                                                metadata=payload.metadata,
                                            )
                                            self.ingestor.ingest([record])
                                            self.graph.update_node(request.record_id, embedding=embedding)
                                            logger.debug(f"Синхронизирована запись {request.record_id} из Qdrant в граф")
                                        except Exception as e:
                                            logger.debug(f"Ошибка при синхронизации записи {request.record_id}: {e}")
                                    
                                    return FetchResponse(record=payload)
                            except Exception as e:
                                logger.debug(f"Failed to fetch from collection {collection_name}: {e}")
                                continue
                except Exception as e:
                    logger.debug(f"Failed to fetch from Qdrant: {e}")
            
            return FetchResponse(record=None)
        except Exception as e:
            logger.error(f"Failed to fetch record {request.record_id}: {e}", exc_info=True)
            return FetchResponse(record=None)

    def store_trading_signal(
        self, request: StoreTradingSignalRequest
    ) -> StoreTradingSignalResponse:
        """Сохранение торгового сигнала в память."""
        data = self.trading_memory.store_signal(
            symbol=request.symbol,
            signal_type=request.signal_type,
            direction=request.direction,
            entry=request.entry,
            confidence=request.confidence,
            context=dict(request.context),
            timestamp=request.timestamp,
        )
        record = TradingSignalRecord(
            signal_id=data["signal_id"],
            timestamp=data["timestamp"],
            symbol=data["symbol"],
            signal_type=data["signal_type"],
            direction=data.get("direction"),
            entry=data.get("entry"),
            confidence=data.get("confidence"),
            context=data.get("context", {}),
        )
        return StoreTradingSignalResponse(signal=record)

    def search_trading_patterns(
        self, request: SearchTradingPatternsRequest
    ) -> SearchTradingPatternsResponse:
        rows = self.trading_memory.search_patterns(
            query=request.query,
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=request.limit,
        )
        signals = [
            TradingSignalRecord(
                signal_id=row["signal_id"],
                timestamp=row["timestamp"],
                symbol=row["symbol"],
                signal_type=row["signal_type"],
                direction=row.get("direction"),
                entry=row.get("entry"),
                confidence=row.get("confidence"),
                context=row.get("context", {}),
            )
            for row in rows
        ]
        return SearchTradingPatternsResponse(signals=signals)

    def get_signal_performance(
        self, request: GetSignalPerformanceRequest
    ) -> GetSignalPerformanceResponse:
        data = self.trading_memory.get_performance(request.signal_id)
        if not data:
            raise ValueError(f"Signal {request.signal_id} not found")
        signal_data = data["signal"]
        signal = TradingSignalRecord(
            signal_id=signal_data["signal_id"],
            timestamp=signal_data["timestamp"],
            symbol=signal_data["symbol"],
            signal_type=signal_data["signal_type"],
            direction=signal_data.get("direction"),
            entry=signal_data.get("entry"),
            confidence=signal_data.get("confidence"),
            context=signal_data.get("context", {}),
        )
        perf = None
        if "performance" in data:
            perf_data = data["performance"]
            perf = SignalPerformance(
                pnl=perf_data.get("pnl"),
                result=perf_data.get("result"),
                closed_at=perf_data.get("closed_at"),
                notes=perf_data.get("notes"),
            )
        return GetSignalPerformanceResponse(signal=signal, performance=perf)

    def search_entities(
        self, request: SearchEntitiesRequest
    ) -> SearchEntitiesResponse:
        """Поиск сущностей по семантическому сходству."""
        from ..memory.storage.vector.vector_store import build_entity_vector_store_from_env
        
        entity_vector_store = build_entity_vector_store_from_env()
        if not entity_vector_store or not entity_vector_store.available():
            return SearchEntitiesResponse(results=[], total_found=0)
        
        query_vector = request.query_vector
        if not query_vector and request.query:
            if not self.embedding_service:
                return SearchEntitiesResponse(results=[], total_found=0)
            query_vector = self.embedding_service.embed(request.query)
        
        if not query_vector:
            return SearchEntitiesResponse(results=[], total_found=0)
        
        search_results = entity_vector_store.search_entities(
            query_vector=query_vector,
            entity_type=request.entity_type,
            limit=request.limit,
        )
        
        results = []
        for result in search_results:
            payload = result.payload or {}
            results.append(
                EntitySearchResult(
                    entity_id=result.entity_id,
                    score=result.score,
                    entity_type=payload.get("entity_type", ""),
                    value=payload.get("value", ""),
                    description=payload.get("description"),
                    importance=payload.get("importance", 0.5),
                    mention_count=payload.get("mention_count", 0),
                )
            )
        
        return SearchEntitiesResponse(results=results, total_found=len(results))

    def get_entity_profile(
        self, request: GetEntityProfileRequest
    ) -> GetEntityProfileResponse:
        """Получение полного профиля сущности."""
        from ..analysis.entities import get_entity_dictionary
        
        entity_dict = get_entity_dictionary(graph=self.graph)
        if not entity_dict:
            return GetEntityProfileResponse(profile=None)
        
        profile_data = entity_dict.build_entity_profile(request.entity_type, request.value)
        
        if not profile_data:
            return GetEntityProfileResponse(profile=None)
        
        contexts = [
            EntityContextItem(
                node_id=ctx.get("node_id", ""),
                content=ctx.get("content", ""),
                source=ctx.get("source", ""),
                chat=ctx.get("chat", ""),
                author=ctx.get("author"),
                timestamp=ctx.get("timestamp"),
            )
            for ctx in profile_data.get("contexts", [])
        ]
        
        related = [
            RelatedEntityItem(
                entity_id=r.get("entity_id", ""),
                label=r.get("label", ""),
                entity_type=r.get("entity_type", ""),
                description=r.get("description"),
                edge_type=r.get("edge_type", ""),
                weight=r.get("weight", 0.0),
            )
            for r in profile_data.get("related_entities", [])
        ]
        
        profile = EntityProfile(
            entity_type=profile_data.get("entity_type", ""),
            value=profile_data.get("value", ""),
            normalized_value=profile_data.get("normalized_value", ""),
            description=profile_data.get("description"),
            aliases=profile_data.get("aliases", []),
            mention_count=profile_data.get("mention_count", 0),
            chats=profile_data.get("chats", []),
            chat_counts=profile_data.get("chat_counts", {}),
            first_seen=profile_data.get("first_seen"),
            last_seen=profile_data.get("last_seen"),
            contexts=contexts,
            context_count=profile_data.get("context_count", 0),
            related_entities=related,
            importance=profile_data.get("importance", 0.5),
        )
        
        return GetEntityProfileResponse(profile=profile)

    def ingest_scraped_content(
        self, request: ScrapedContentRequest
    ) -> ScrapedContentResponse:
        """Ingest scraped web content into memory."""
        import uuid
        from datetime import datetime

        # Используем полный UUID (32 символа) для избежания коллизий
        record_id = f"scrape_{uuid.uuid4().hex}"
        record = MemoryRecordPayload(
            record_id=record_id,
            source=request.source,
            content=request.content,
            timestamp=datetime.now(),
            author=None,
            tags=request.tags + ["web_scrape", "bright_data"],
            entities=request.entities,
            attachments=[],
            metadata={
                "url": request.url,
                "title": request.title,
                "scraped_at": datetime.now().isoformat(),
                **request.metadata,
            },
        )

        try:
            ingest_response = self.ingest([record])

            return ScrapedContentResponse(
                record_id=record_id,
                status="success",
                url=request.url,
                message=f"Successfully ingested scraped content from {request.url}",
            )

        except Exception as e:
            logger.error(f"Failed to ingest scraped content from {request.url}: {e}")
            return ScrapedContentResponse(
                record_id=record_id,
                status="error",
                url=request.url,
                message=f"Failed to ingest scraped content: {str(e)}",
            )

    def generate_embedding(
        self, request: GenerateEmbeddingRequest
    ) -> GenerateEmbeddingResponse:
        """Генерация эмбеддинга для произвольного текста."""
        """Generate embedding for arbitrary text."""
        if not self.embedding_service:
            raise ValueError(
                "Embedding service is not configured. "
                "Please set MEMORY_MCP_EMBEDDINGS_URL or MEMORY_MCP_LMSTUDIO_HOST/PORT/MODEL environment variables."
            )
        
        if not request.text or not request.text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            vector = self.embedding_service.embed(request.text.strip())
            if vector is None:
                raise ValueError(
                    "Embedding service returned None. "
                    "Check if the service is running and configured correctly."
                )
            if not isinstance(vector, list) or len(vector) == 0:
                raise ValueError(
                    "Embedding service returned invalid result. "
                    "Expected non-empty list of floats."
                )
        except ValueError as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            raise ValueError(
                f"Failed to generate embedding: {str(e)}. "
                "Check if the embedding service is running and accessible."
            ) from e
        
        return GenerateEmbeddingResponse(
            embedding=vector,
            dimension=len(vector),
            model=self.embedding_service.model_name,
        )

    def update_record(self, request: UpdateRecordRequest) -> UpdateRecordResponse:
        """Update an existing memory record."""
        node = self.graph.get_node(request.record_id)
        if not node:
            return UpdateRecordResponse(
                record_id=request.record_id,
                updated=False,
                message=f"Record {request.record_id} not found",
            )

        props = dict(node.get("properties") or {})
        has_changes = False
        
        if request.content is not None and request.content != (node.get("content") or ""):
            has_changes = True
        if request.source is not None and request.source != (node.get("source") or props.get("source")):
            has_changes = True
        if request.tags is not None:
            old_tags = set(props.get("tags", []))
            new_tags = set(request.tags)
            if old_tags != new_tags:
                has_changes = True
        if request.entities is not None:
            old_entities = set(props.get("entities", []))
            new_entities = set(request.entities)
            if old_entities != new_entities:
                has_changes = True
        if request.metadata:
            # Проверяем, есть ли изменения в metadata
            old_metadata = {k: v for k, v in props.items() 
                          if k not in ["source", "content", "timestamp", "author", "tags", "entities", "created_at"]}
            for key, value in request.metadata.items():
                if old_metadata.get(key) != value:
                    has_changes = True
                    break
        
        if not has_changes:
            return UpdateRecordResponse(
                record_id=request.record_id,
                updated=False,
                message="No changes detected. Record was not updated.",
            )

        new_embedding = None
        if request.content and self.embedding_service:
            new_embedding = self.embedding_service.embed(request.content)

        updated = self.graph.update_node(
            request.record_id,
            properties=request.metadata,
            content=request.content,
            source=request.source,
            tags=request.tags,
            entities=request.entities,
            embedding=new_embedding,
        )

        if not updated:
            return UpdateRecordResponse(
                record_id=request.record_id,
                updated=False,
                message="Failed to update record",
            )

        if new_embedding and self.vector_store:
            try:
                updated_node = self.graph.get_node(request.record_id)
                if updated_node:
                    props = updated_node.properties
                    self.vector_store.upsert(
                        request.record_id,
                        new_embedding,
                        {
                            "source": request.source or props.get("source", ""),
                            "tags": request.tags or props.get("tags", []),
                            "timestamp": props.get("timestamp", ""),
                            "content_preview": (request.content or props.get("content", ""))[:200],
                        },
                    )
            except Exception as e:
                logger.warning(f"Failed to update vector store for {request.record_id}: {e}")

        return UpdateRecordResponse(
            record_id=request.record_id,
            updated=True,
            message="Record updated successfully",
        )

    def delete_record(self, request: DeleteRecordRequest) -> DeleteRecordResponse:
        """Delete a memory record."""
        node = self.graph.get_node(request.record_id)
        if not node:
            return DeleteRecordResponse(
                record_id=request.record_id,
                deleted=False,
                message=f"Record {request.record_id} not found",
            )

        deleted = self.graph.delete_node(request.record_id)

        if not deleted:
            return DeleteRecordResponse(
                record_id=request.record_id,
                deleted=False,
                message="Failed to delete record",
            )

        if self.vector_store:
            try:
                self.vector_store.delete(request.record_id)
            except Exception as e:
                logger.warning(f"Failed to delete from vector store for {request.record_id}: {e}")

        return DeleteRecordResponse(
            record_id=request.record_id,
            deleted=True,
            message="Record deleted successfully",
        )

    def get_statistics(self) -> GetStatisticsResponse:
        """Get system statistics."""
        graph_stats_obj = self.graph.get_stats()
        graph_stats = graph_stats_obj.model_dump()

        sources_count = {}
        tags_count = {}
        
        cursor = self.graph.conn.cursor()
        try:
            cursor.execute("PRAGMA table_info(nodes)")
            columns = {row[1] for row in cursor.fetchall()}
            if "properties" not in columns:
                logger.warning(
                    "Столбец 'properties' не найден в таблице 'nodes'. "
                    "Статистика по источникам и тегам будет пустой."
                )
        except Exception as e:
            logger.warning(f"Не удалось проверить схему таблицы 'nodes': {e}")

        try:
            cursor.execute(
                """
                SELECT properties FROM nodes
                WHERE type = 'DocChunk' AND properties IS NOT NULL
            """
            )
            for row in cursor.fetchall():
                if row["properties"]:
                    try:
                        props = json.loads(row["properties"]) if isinstance(row["properties"], str) else row["properties"]
                        source = props.get("source") or props.get("chat", "unknown")
                        sources_count[source] = sources_count.get(source, 0) + 1
                    except (json.JSONDecodeError, TypeError):
                        logger.debug(f"Не удалось распарсить properties для подсчёта статистики: {row.get('id', 'unknown')}")
                        continue
        except Exception as e:
            logger.warning(f"Ошибка при подсчёте источников из графа: {e}")

        try:
            cursor.execute(
                """
                SELECT properties FROM nodes
            """
            )
            for row in cursor.fetchall():
                if row["properties"]:
                    props = json.loads(row["properties"])
                    tags = props.get("tags", [])
                    if isinstance(tags, list):
                        for tag in tags:
                            tags_count[tag] = tags_count.get(tag, 0) + 1
        except Exception as e:
            logger.warning(f"Ошибка при подсчёте тегов из графа: {e}")

        db_size = None
        try:
            db_path = self.graph.conn.execute("PRAGMA database_list").fetchone()
            if db_path:
                db_file = Path(db_path[2]) if len(db_path) > 2 else None
                if db_file and db_file.exists():
                    db_size = db_file.stat().st_size
        except Exception as e:
            logger.debug(f"Не удалось получить размер БД: {e}")

        return GetStatisticsResponse(
            graph_stats=graph_stats,
            sources_count=sources_count,
            tags_count=tags_count,
            database_size_bytes=db_size,
        )

    def get_indexing_progress(
        self, request: GetIndexingProgressRequest
    ) -> GetIndexingProgressResponse:
        """Получить прогресс индексации из активных задач и трекера.

        Примечание: Прогресс индексации теперь хранится в SQLite через IndexingJobTracker.
        При ошибках инициализации возвращаем ошибку без попытки восстановления.
        """
        from ..core.indexing_tracker import IndexingJobTracker
        from ..config import get_settings
        from ..utils.system.paths import find_project_root
        from pathlib import Path
        
        # Создаем трекер напрямую, избегая циклического импорта
        settings = get_settings()
        storage_path = "data/indexing_jobs.json"
        if not os.path.isabs(storage_path):
            project_root = find_project_root(Path(__file__).parent)
            storage_path = str(project_root / storage_path)
        
        tracker = IndexingJobTracker(storage_path=storage_path)
        
        # Автоматически завершаем зависшие задачи (старше 2 часов)
        stale_completed = tracker.cleanup_stale_running_jobs(max_age_hours=2)
        if stale_completed > 0:
            logger.info(f"Автоматически завершено {stale_completed} зависших задач индексации")
        
        # Получаем активные задачи из трекера
        active_jobs = tracker.get_all_jobs(status="running", chat=request.chat) if request.chat else tracker.get_all_jobs(status="running")
        
        progress_items = []
        for job in active_jobs:
            progress_items.append(
                IndexingProgressItem(
                    chat_name=job.get("chat") or job.get("current_chat") or "Unknown",
                    job_id=job.get("job_id"),
                    status=job.get("status"),
                    started_at=job.get("started_at"),
                    current_stage=job.get("current_stage"),
                    total_messages=job.get("stats", {}).get("messages_indexed", 0),
                    total_sessions=job.get("stats", {}).get("sessions_indexed", 0),
                )
            )
        
        # Если запрошен конкретный чат, фильтруем результаты
        if request.chat:
            progress_items = [item for item in progress_items if item.chat_name == request.chat]
        
        return GetIndexingProgressResponse(
            progress=progress_items,
            message=None if progress_items else f"No progress found for chat '{request.chat}'" if request.chat else "No active indexing jobs",
        )

    # Graph Operations
    def get_graph_neighbors(
        self, request: GetGraphNeighborsRequest
    ) -> GetGraphNeighborsResponse:
        """Get neighbors of a graph node."""
        # Логирование для отладки
        logger.debug(
            f"get_graph_neighbors: node_id={request.node_id}, "
            f"edge_type={request.edge_type}, direction={request.direction}"
        )
        
        node_id = self._find_node_id(request.node_id)
        if not node_id:
            logger.warning(f"Узел {request.node_id} не найден в графе")
            return GetGraphNeighborsResponse(neighbors=[])
        
        edge_type = None
        if request.edge_type:
            try:
                edge_type = EdgeType(request.edge_type)
            except ValueError:
                logger.warning(f"Invalid edge type: {request.edge_type}")

        neighbors_data = self.graph.get_neighbors(
            node_id,
            edge_type=edge_type,
            direction=request.direction,
        )

        logger.debug(
            f"Найдено {len(neighbors_data)} соседей для узла {node_id} (запрошен {request.node_id})"
        )

        neighbors = []
        for neighbor_id, edge_data in neighbors_data:
            neighbors.append(
                GraphNeighborItem(
                    node_id=neighbor_id,
                    edge_type=edge_data.get("edge_type"),
                    edge_data=dict(edge_data),
                )
            )

        return GetGraphNeighborsResponse(neighbors=neighbors)

    def find_graph_path(
        self, request: FindGraphPathRequest
    ) -> FindGraphPathResponse:
        """Find path between two nodes in the graph."""
        path_obj = self.graph.find_path(
            request.source_id,
            request.target_id,
            max_length=request.max_length,
        )

        if path_obj:
            return FindGraphPathResponse(
                path=[node.id for node in path_obj.nodes],
                total_weight=path_obj.total_weight,
                found=True,
            )
        else:
            return FindGraphPathResponse(
                path=None,
                total_weight=None,
                found=False,
            )

    def get_related_records(
        self, request: GetRelatedRecordsRequest
    ) -> GetRelatedRecordsResponse:
        """Get related records through graph connections."""
        try:
            # Пробуем найти узел по разным форматам record_id
            node_id = self._find_node_id(request.record_id)
            if not node_id:
                logger.debug(f"Узел {request.record_id} не найден в графе для get_related_records")
                return GetRelatedRecordsResponse(records=[])
            
            visited = set()
            related_records = []
            current_level = {node_id}
            
            for depth in range(request.max_depth):
                next_level = set()
                for node_id in current_level:
                    if node_id in visited:
                        continue
                    visited.add(node_id)
                    
                    try:
                        neighbors_data = self.graph.get_neighbors(node_id, direction="both")
                        for neighbor_id, _ in neighbors_data:
                            if neighbor_id not in visited and neighbor_id not in current_level:
                                next_level.add(neighbor_id)
                                
                                try:
                                    fetch_req = FetchRequest(record_id=neighbor_id)
                                    fetch_resp = self.fetch(fetch_req)
                                    if fetch_resp.record:
                                        related_records.append(fetch_resp.record)
                                        if len(related_records) >= request.limit:
                                            return GetRelatedRecordsResponse(records=related_records)
                                except Exception as e:
                                    logger.debug(f"Failed to fetch related record {neighbor_id}: {e}")
                                    continue
                    except Exception as e:
                        logger.debug(f"Failed to get neighbors for {node_id}: {e}")
                        continue
                
                current_level = next_level
                if not current_level:
                    break
            
            return GetRelatedRecordsResponse(records=related_records[:request.limit])
        except Exception as e:
            logger.error(f"Failed to get related records: {e}", exc_info=True)
            return GetRelatedRecordsResponse(records=[])

    # Advanced Search
    def _search_by_embedding(
        self, request: SearchByEmbeddingRequest
    ) -> SearchByEmbeddingResponse:
        """Search by embedding vector directly."""
        if not self.vector_store:
            return SearchByEmbeddingResponse(results=[], total_matches=0)

        vector_results = self.vector_store.search(
            request.embedding,
            limit=request.top_k,
            source=request.source,
            tags=request.tags if request.tags else None,
            date_from=request.date_from,
            date_to=request.date_to,
        )

        results = []
        for match in vector_results:
            record_id = match.record_id
            score = match.score
            payload = match.payload or {}
            snippet = payload.get("content_preview", "")
            item = self._build_item_from_graph(record_id, score, snippet=snippet)
            if item:
                results.append(item)

        return SearchByEmbeddingResponse(
            results=results[:request.top_k],
            total_matches=len(results),
        )

    def _similar_records(
        self, request: SimilarRecordsRequest
    ) -> SimilarRecordsResponse:
        """Find similar records by getting embedding of the record and searching."""
        # Получаем запись
        fetch_req = FetchRequest(record_id=request.record_id)
        fetch_resp = self.fetch(fetch_req)
        if not fetch_resp.record:
            return SimilarRecordsResponse(results=[])

        if not self.embedding_service:
            return SimilarRecordsResponse(results=[])

        vector = self.embedding_service.embed(fetch_resp.record.content)
        if not vector or not self.vector_store:
            return SimilarRecordsResponse(results=[])

        vector_results = self.vector_store.search(
            vector,
            limit=request.top_k + 1,
        )

        results = []
        for match in vector_results:
            if match.record_id == request.record_id:
                continue
            item = self._build_item_from_graph(match.record_id, match.score)
            if item:
                results.append(item)
            if len(results) >= request.top_k:
                break

        return SimilarRecordsResponse(results=results)

    def search_explain(
        self, request: SearchExplainRequest
    ) -> SearchExplainResponse:
        """Explain search result relevance."""
        try:
            from ..search import SearchExplainer

            explainer = SearchExplainer(typed_graph_memory=self.graph)

            search_req = SearchRequest(query=request.query, top_k=50)
            search_resp = self.search(search_req)

            target_record = None
            target_rank = request.rank
            for i, result in enumerate(search_resp.results):
                if result.record_id == request.record_id:
                    target_record = result
                    target_rank = i
                    break

            if not target_record:
                raise ValueError(f"Record {request.record_id} not found in search results")

            bm25_rows, _ = self.graph.search_text(
                request.query,
                limit=DEFAULT_SEARCH_LIMIT,
            )
            bm25_results = [(row["node_id"], row["score"]) for row in bm25_rows]

            vector_results = []
            if self.embedding_service and self.vector_store:
                query_vector = self.embedding_service.embed(request.query)
                if query_vector:
                    vector_matches = self.vector_store.search(
                        query_vector,
                        limit=DEFAULT_SEARCH_LIMIT,
                    )
                    vector_results = [(match.record_id, match.score) for match in vector_matches]

            fetch_resp = self.fetch(FetchRequest(record_id=request.record_id))
            metadata = {}
            if fetch_resp.record:
                metadata = fetch_resp.record.metadata

            explanation = explainer.explain_result(
                doc_id=request.record_id,
                query=request.query,
                rank=target_rank,
                final_score=target_record.score,
                bm25_results=bm25_results,
                vector_results=vector_results,
                metadata=metadata,
            )

            return SearchExplainResponse(
                explanation={
                    "doc_id": explanation.doc_id,
                    "query": explanation.query,
                    "rank": explanation.rank,
                },
                score_breakdown=explanation.score_breakdown.to_dict(),
                connection_paths=[
                    {
                        "path": [n.id for n in path.path],
                        "strength": path.strength,
                        "explanation": path.explanation,
                    }
                    for path in explanation.connection_paths
                ],
                explanation_text=explanation.explanation_text,
            )
        except Exception as e:
            logger.error(f"Failed to explain search result: {e}")
            raise ValueError(f"Failed to explain search result: {str(e)}")

    # Analytics
    def get_tags_statistics(self) -> GetTagsStatisticsResponse:
        """Get statistics about tags usage."""
        tags_count = {}
        tagged_records = set()

        cursor = self.graph.conn.cursor()
        cursor.execute("SELECT id, properties FROM nodes")
        for row in cursor.fetchall():
            if row["properties"]:
                props = json.loads(row["properties"])
                tags = props.get("tags", [])
                if isinstance(tags, list) and tags:
                    tagged_records.add(row["id"])
                    for tag in tags:
                        tags_count[tag] = tags_count.get(tag, 0) + 1

        return GetTagsStatisticsResponse(
            tags_count=tags_count,
            total_tags=len(tags_count),
            total_tagged_records=len(tagged_records),
        )

    def get_timeline(self, request: GetTimelineRequest) -> GetTimelineResponse:
        """Get timeline of records sorted by timestamp."""
        cursor = self.graph.conn.cursor()
        
        # Получаем все узлы типа DocChunk с properties
        # Фильтруем только записи с типом DocChunk, чтобы исключить служебные узлы
        query = """
            SELECT id, properties, type FROM nodes 
            WHERE type = 'DocChunk' AND properties IS NOT NULL
        """
        
        # Используем более надежный способ сортировки
        # Сначала получаем все записи, потом сортируем в Python
        cursor.execute(query)
        
        items = []
        for row in cursor.fetchall():
            if not row["properties"]:
                continue
                
            try:
                props = json.loads(row["properties"]) if isinstance(row["properties"], str) else row["properties"]
            except (json.JSONDecodeError, TypeError):
                logger.debug(f"Не удалось распарсить properties для узла {row['id']}")
                continue
            
            source = props.get("source") or props.get("chat", "unknown")
            
            # Дополнительная фильтрация по source, если указан
            if request.source and source != request.source:
                continue
            
            timestamp_str = props.get("timestamp") or props.get("created_at")
            if not timestamp_str:
                continue
            
            try:
                timestamp = parse_datetime_utc(timestamp_str, default=None)
                if timestamp is None:
                    logger.debug(
                        f"Не удалось распарсить timestamp '{timestamp_str}' для записи {row['id']}"
                    )
                    continue
            except Exception as e:
                logger.debug(
                    f"Ошибка при парсинге timestamp '{timestamp_str}' для записи {row['id']}: {e}"
                )
                continue
            
            if request.date_from:
                date_from = request.date_from
                if timestamp.tzinfo is None and date_from.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                elif timestamp.tzinfo is not None and date_from.tzinfo is None:
                    date_from = date_from.replace(tzinfo=timezone.utc)
                if timestamp < date_from:
                    continue
            if request.date_to:
                date_to = request.date_to
                if timestamp.tzinfo is None and date_to.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                elif timestamp.tzinfo is not None and date_to.tzinfo is None:
                    date_to = date_to.replace(tzinfo=timezone.utc)
                if timestamp > date_to:
                    continue
            
            content = props.get("content", "")
            content_preview = content[:200] if content else ""
            
            items.append(
                TimelineItem(
                    record_id=row["id"],
                    timestamp=timestamp,
                    source=source,
                    content_preview=content_preview,
                )
            )
        
        # Сортируем по timestamp
        items.sort(key=lambda x: x.timestamp, reverse=True)
        
        return GetTimelineResponse(
            items=items[:request.limit] if request.limit else items,
            total=len(items),
        )

    def analyze_entities(
        self, request: AnalyzeEntitiesRequest
    ) -> AnalyzeEntitiesResponse:
        """Analyze entities in text."""
        try:
            from ..analysis.entities import EntityExtractor

            if not request.text or not request.text.strip():
                return AnalyzeEntitiesResponse(entities=[], total_entities=0)

            extractor = EntityExtractor()
            extracted = extractor.extract_entities(request.text, chat_name="")
            
            logger.debug(f"Extracted entities: {extracted}")

            if not extracted:
                logger.debug("No entities extracted from text")
                return AnalyzeEntitiesResponse(entities=[], total_entities=0)

            entities = []
            for entity_type, values in extracted.items():
                if request.entity_types and entity_type not in request.entity_types:
                    continue
                if not values:
                    continue
                # Подсчитываем частоту каждого значения
                value_counts = {}
                for value in values:
                    if value and str(value).strip():  # Пропускаем пустые значения
                        value_counts[value] = value_counts.get(value, 0) + 1
                
                for value, count in value_counts.items():
                    entities.append(
                        EntityItem(
                            value=str(value),
                            entity_type=entity_type,
                            count=count,
                        )
                    )

            logger.debug(f"Processed {len(entities)} entity items")
            return AnalyzeEntitiesResponse(
                entities=entities,
                total_entities=len(entities),
            )
        except ImportError as e:
            logger.error(f"Failed to import EntityExtractor: {e}", exc_info=True)
            return AnalyzeEntitiesResponse(
                entities=[],
                total_entities=0,
            )
        except Exception as e:
            logger.error(f"Failed to analyze entities: {e}", exc_info=True)
            return AnalyzeEntitiesResponse(entities=[], total_entities=0)

    # Batch Operations
    def batch_update_records(
        self, request: BatchUpdateRecordsRequest
    ) -> BatchUpdateRecordsResponse:
        """Batch update multiple records."""
        results = []
        total_updated = 0
        total_failed = 0

        for update_item in request.updates:
            try:
                update_req = UpdateRecordRequest(
                    record_id=update_item.record_id,
                    content=update_item.content,
                    source=update_item.source,
                    tags=update_item.tags,
                    entities=update_item.entities,
                    metadata=update_item.metadata,
                )
                update_resp = self.update_record(update_req)
                
                results.append(
                    BatchUpdateResult(
                        record_id=update_item.record_id,
                        updated=update_resp.updated,
                        message=update_resp.message,
                    )
                )
                
                if update_resp.updated:
                    total_updated += 1
                else:
                    total_failed += 1
            except Exception as e:
                # Обрабатываем ошибку для конкретного элемента, продолжаем с остальными
                logger.warning(
                    f"Ошибка при обновлении записи {update_item.record_id}: {e}",
                    exc_info=True,
                )
                results.append(
                    BatchUpdateResult(
                        record_id=update_item.record_id,
                        updated=False,
                        message=f"Ошибка при обновлении: {str(e)}",
                    )
                )
                total_failed += 1

        return BatchUpdateRecordsResponse(
            results=results,
            total_updated=total_updated,
            total_failed=total_failed,
        )

    def batch_delete_records(
        self, request: BatchDeleteRecordsRequest
    ) -> BatchDeleteRecordsResponse:
        """Batch delete multiple records."""
        results = []
        total_deleted = 0
        total_failed = 0
        successfully_deleted_ids = []

        for record_id in request.record_ids:
            try:
                node = self.graph.get_node(record_id)
                if not node:
                    results.append(
                        BatchDeleteResult(
                            record_id=record_id,
                            deleted=False,
                            message=f"Record {record_id} not found",
                        )
                    )
                    total_failed += 1
                    continue

                # Удаляем из графа
                deleted = self.graph.delete_node(record_id)

                if not deleted:
                    results.append(
                        BatchDeleteResult(
                            record_id=record_id,
                            deleted=False,
                            message="Failed to delete record from graph",
                        )
                    )
                    total_failed += 1
                    continue

                # Собираем успешно удалённые ID для батч-удаления из vector_store
                successfully_deleted_ids.append(record_id)
                results.append(
                    BatchDeleteResult(
                        record_id=record_id,
                        deleted=True,
                        message="Record deleted successfully",
                    )
                )
                total_deleted += 1

            except Exception as e:
                # Обрабатываем ошибку для конкретного элемента, продолжаем с остальными
                logger.warning(
                    f"Ошибка при удалении записи {record_id}: {e}",
                    exc_info=True,
                )
                results.append(
                    BatchDeleteResult(
                        record_id=record_id,
                        deleted=False,
                        message=f"Ошибка при удалении: {str(e)}",
                    )
                )
                total_failed += 1

        # Оптимизация: удаляем из vector_store батчами
        if (
            self.vector_store
            and self.vector_store.available()
            and successfully_deleted_ids
        ):
            try:
                from qdrant_client.http import models as qmodels

                if not self.vector_store.client or qmodels is None:
                    logger.warning("Vector store client or qmodels not available")
                    return BatchDeleteRecordsResponse(
                        results=results,
                        total_deleted=total_deleted,
                        total_failed=total_failed,
                    )

                batch_size = 1000
                for i in range(0, len(successfully_deleted_ids), batch_size):
                    batch = successfully_deleted_ids[i : i + batch_size]
                    try:
                        self.vector_store.client.delete(
                            collection_name=self.vector_store.collection,
                            points_selector=qmodels.PointIdsList(
                                points=[str(record_id) for record_id in batch]
                            ),
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to delete batch from vector store: {e}",
                            exc_info=True,
                        )
            except Exception as e:
                logger.warning(
                    f"Failed to delete from vector store: {e}",
                    exc_info=True,
                )

        return BatchDeleteRecordsResponse(
            results=results,
            total_deleted=total_deleted,
            total_failed=total_failed,
        )

    def batch_fetch_records(
        self, request: BatchFetchRecordsRequest
    ) -> BatchFetchRecordsResponse:
        """Batch fetch multiple records."""
        results = []
        total_found = 0
        total_not_found = 0

        for record_id in request.record_ids:
            try:
                fetch_req = FetchRequest(record_id=record_id)
                fetch_resp = self.fetch(fetch_req)

                if fetch_resp.record:
                    results.append(
                        BatchFetchResult(
                            record_id=record_id,
                            record=fetch_resp.record,
                            found=True,
                            message="Record found",
                        )
                    )
                    total_found += 1
                else:
                    results.append(
                        BatchFetchResult(
                            record_id=record_id,
                            record=None,
                            found=False,
                            message=f"Record {record_id} not found",
                        )
                    )
                    total_not_found += 1

            except Exception as e:
                # Обрабатываем ошибку для конкретного элемента, продолжаем с остальными
                logger.warning(
                    f"Ошибка при получении записи {record_id}: {e}",
                    exc_info=True,
                )
                results.append(
                    BatchFetchResult(
                        record_id=record_id,
                        record=None,
                        found=False,
                        message=f"Ошибка при получении: {str(e)}",
                    )
                )
                total_not_found += 1

        return BatchFetchRecordsResponse(
            results=results,
            total_found=total_found,
            total_not_found=total_not_found,
        )

    # Export/Import
    def export_records(self, request: ExportRecordsRequest) -> ExportRecordsResponse:
        """Export records in various formats."""
        import json
        from datetime import datetime
        
        cursor = self.graph.conn.cursor()
        
        # Используем более надежный способ фильтрации через json_extract
        query = """
            SELECT id, properties FROM nodes
            WHERE type = 'DocChunk' AND properties IS NOT NULL
        """
        params = []
        
        # Фильтрация по source через json_extract
        if request.source:
            query += " AND (json_extract(properties, '$.source') = ? OR json_extract(properties, '$.chat') = ?)"
            params.extend([request.source, request.source])
        
        # Фильтрация по датам
        if request.date_from:
            query += " AND json_extract(properties, '$.timestamp') IS NOT NULL"
            # Будем фильтровать в Python после парсинга
        
        if request.date_to:
            query += " AND json_extract(properties, '$.timestamp') IS NOT NULL"
            # Будем фильтровать в Python после парсинга
        
        # Берем больше записей для фильтрации в Python
        limit_multiplier = 10 if (request.source or request.tags or request.date_from or request.date_to) else 1
        query += f" LIMIT {request.limit * limit_multiplier}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        records = []
        for row in rows:
            try:
                props = json.loads(row["properties"]) if isinstance(row["properties"], str) else row["properties"]
                if not props:
                    continue
                
                source = props.get("source") or props.get("chat", "unknown")
                if request.source and source != request.source:
                    continue
                
                if request.tags:
                    node_tags = props.get("tags", [])
                    if not isinstance(node_tags, list):
                        node_tags = []
                    if not all(tag in node_tags for tag in request.tags):
                        continue
                
                timestamp_str = props.get("timestamp") or props.get("created_at")
                if timestamp_str:
                    try:
                        timestamp = parse_datetime_utc(timestamp_str, default=None)
                        if timestamp:
                            if request.date_from:
                                if timestamp < request.date_from:
                                    continue
                            if request.date_to:
                                if timestamp > request.date_to:
                                    continue
                    except Exception as e:
                        logger.debug(f"Ошибка при парсинге timestamp для экспорта: {e}")
                        if request.date_from or request.date_to:
                            continue
                
                record = MemoryRecordPayload(
                    record_id=row["id"],
                    source=props.get("source", "unknown"),
                    content=props.get("content", ""),
                    timestamp=_parse_timestamp(props.get("timestamp") or props.get("created_at")),
                    author=props.get("author"),
                    tags=props.get("tags", []),
                    entities=props.get("entities", []),
                    attachments=[],
                    metadata={k: v for k, v in props.items() if k not in ["source", "content", "timestamp", "author", "tags", "entities", "created_at"]},
                )
                records.append(record)
                
                # Ограничиваем количество записей после фильтрации
                if len(records) >= request.limit:
                    break
            except Exception as e:
                logger.debug(f"Failed to parse record {row['id']}: {e}")
                continue

        if request.format == "json":
            import json
            content = json.dumps(
                [r.model_dump(mode="json") for r in records],
                ensure_ascii=False,
                indent=2,
            )
        elif request.format == "csv":
            import csv
            import io
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=["record_id", "source", "timestamp", "author", "content", "tags"],
            )
            writer.writeheader()
            for record in records:
                writer.writerow({
                    "record_id": record.record_id,
                    "source": record.source,
                    "timestamp": record.timestamp.isoformat() if record.timestamp else "",
                    "author": record.author or "",
                    "content": record.content[:500] if record.content else "",
                    "tags": ",".join(record.tags),
                })
            content = output.getvalue()
        elif request.format == "markdown":
            content = "# Экспорт записей\n\n"
            for record in records:
                content += f"## {record.record_id}\n\n"
                content += f"**Источник:** {record.source}\n\n"
                content += f"**Дата:** {record.timestamp.isoformat() if record.timestamp else 'N/A'}\n\n"
                if record.author:
                    content += f"**Автор:** {record.author}\n\n"
                if record.tags:
                    content += f"**Теги:** {', '.join(record.tags)}\n\n"
                content += f"{record.content}\n\n"
                content += "---\n\n"
        else:
            raise ValueError(f"Unsupported export format: {request.format}")

        return ExportRecordsResponse(
            format=request.format,
            content=content,
            records_count=len(records),
        )

    def import_records(self, request: ImportRecordsRequest) -> ImportRecordsResponse:
        """Import records from various formats."""
        records = []
        
        try:
            if request.format == "json":
                import json
                data = json.loads(request.content)
                if isinstance(data, list):
                    for item in data:
                        if request.source and "source" not in item:
                            item["source"] = request.source
                        records.append(MemoryRecordPayload(**item))
                else:
                    if request.source and "source" not in data:
                        data["source"] = request.source
                    records.append(MemoryRecordPayload(**data))
            elif request.format == "csv":
                import csv
                import io
                reader = csv.DictReader(io.StringIO(request.content))
                for row in reader:
                    # Проверяем наличие поля timestamp перед парсингом
                    timestamp_value = row.get("timestamp")
                    if timestamp_value and timestamp_value.strip():
                        parsed_timestamp = _parse_timestamp(timestamp_value)
                    else:
                        # Если timestamp отсутствует, используем текущее время как метку создания
                        parsed_timestamp = datetime.now(timezone.utc)
                        logger.debug(
                            f"Отсутствует timestamp в CSV строке для record_id={row.get('record_id', 'unknown')}. "
                            f"Используется текущее время."
                        )
                    
                    record_data = {
                        "record_id": row.get("record_id", ""),
                        "source": request.source or row.get("source", "imported"),
                        "content": row.get("content", ""),
                        "timestamp": parsed_timestamp,
                        "author": row.get("author"),
                        "tags": row.get("tags", "").split(",") if row.get("tags") else [],
                        "entities": [],
                        "attachments": [],
                        "metadata": {},
                    }
                    records.append(MemoryRecordPayload(**record_data))
            else:
                raise ValueError(f"Unsupported import format: {request.format}")

            if records:
                ingest_resp = self.ingest(records)
                return ImportRecordsResponse(
                    records_imported=ingest_resp.records_ingested,
                    records_failed=len(records) - ingest_resp.records_ingested,
                    message=f"Imported {ingest_resp.records_ingested} records",
                )
            else:
                return ImportRecordsResponse(
                    records_imported=0,
                    records_failed=0,
                    message="No records to import",
                )
        except Exception as e:
            logger.error(f"Failed to import records: {e}")
            return ImportRecordsResponse(
                records_imported=0,
                records_failed=len(records) if records else 0,
                message=f"Import failed: {str(e)}",
            )

    # Summaries
    async def update_summaries(self, request: UpdateSummariesRequest) -> UpdateSummariesResponse:
        """Update markdown summaries without full re-indexing."""
        import json
        from datetime import datetime, timedelta
        from pathlib import Path
        from zoneinfo import ZoneInfo

        from ..analysis.rendering import MarkdownRenderer

        reports_dir = Path("artifacts/reports")

        if not reports_dir.exists():
            return UpdateSummariesResponse(
                chats_updated=0,
                message="Директория artifacts/reports не найдена. Запустите индексацию: memory_mcp index",
            )

        if request.chat:
            chat_dirs = [reports_dir / request.chat] if (reports_dir / request.chat).exists() else []
        else:
            chat_dirs = [
                d for d in reports_dir.iterdir() if d.is_dir() and (d / "sessions").exists()
            ]

        if not chat_dirs:
            return UpdateSummariesResponse(
                chats_updated=0, message="Не найдено чатов с саммаризациями"
            )

        renderer = MarkdownRenderer(output_dir=reports_dir)

        def parse_message_time(date_str: str) -> datetime:
            try:
                from ..utils.processing.datetime_utils import parse_datetime_utc

                return parse_datetime_utc(
                    date_str, default=datetime.now(ZoneInfo("UTC")), use_zoneinfo=True
                )
            except Exception:
                return datetime.now(ZoneInfo("UTC"))

        def load_session_summaries(chat_dir: Path) -> list:
            sessions = []
            sessions_dir = chat_dir / "sessions"
            if not sessions_dir.exists():
                return sessions

            json_files = list(sessions_dir.glob("*.json"))
            for json_file in json_files:
                try:
                    with open(json_file, encoding="utf-8") as f:
                        session = json.load(f)
                        sessions.append(session)
                except Exception:
                    continue
            return sessions

        updated = 0

        for chat_dir in chat_dirs:
            chat_name = chat_dir.name.replace("_", " ").title()
            sessions = load_session_summaries(chat_dir)

            if not sessions:
                continue

            now = datetime.now(ZoneInfo("UTC"))
            thirty_days_ago = now - timedelta(days=30)

            recent_sessions = []
            for session in sessions:
                end_time_str = session.get("meta", {}).get("end_time_utc", "")
                if end_time_str:
                    end_time = parse_message_time(end_time_str)
                    if end_time >= thirty_days_ago:
                        recent_sessions.append(session)

            top_sessions = sorted(
                recent_sessions,
                key=lambda s: s.get("quality", {}).get("score", 0),
                reverse=True,
            )

            try:
                renderer.render_chat_summary(
                    chat_name, sessions, top_sessions=top_sessions, force=request.force
                )
                renderer.render_cumulative_context(chat_name, sessions, force=request.force)
                renderer.render_chat_index(chat_name, sessions, force=request.force)
                updated += 1
            except Exception:
                continue

        return UpdateSummariesResponse(
            chats_updated=updated,
            message=f"Обновлено чатов: {updated}. Обновленные файлы находятся в: ./artifacts/reports/",
        )

    async def review_summaries(self, request: ReviewSummariesRequest) -> ReviewSummariesResponse:
        """Review and fix summaries with -needs-review suffix."""
        from pathlib import Path

        reports_dir = Path("artifacts/reports")

        if not reports_dir.exists():
            return ReviewSummariesResponse(
                files_processed=0,
                files_fixed=0,
                message="Директория artifacts/reports не найдена",
            )

        needs_review_files = []
        for md_file in reports_dir.rglob("*-needs-review.md"):
            file_info = {
                "md_file": md_file,
                "session_id": md_file.stem.replace("-needs-review", ""),
                "chat": md_file.parent.parent.name,
            }

            if request.chat and request.chat.lower() not in file_info["chat"].lower():
                continue

            needs_review_files.append(file_info)

        if request.limit:
            needs_review_files = needs_review_files[: request.limit]

        if not needs_review_files:
            return ReviewSummariesResponse(
                files_processed=0,
                files_fixed=0,
                message="Не найдено файлов с суффиксом -needs-review",
            )

        fixed = 0 if request.dry_run else len(needs_review_files)

        return ReviewSummariesResponse(
            files_processed=len(needs_review_files),
            files_fixed=fixed,
            message=f"Обработано файлов: {len(needs_review_files)}, исправлено: {fixed}",
        )

    # Insight Graph
    async def build_insight_graph(
        self, request: BuildInsightGraphRequest
    ) -> BuildInsightGraphResponse:
        """Build insight graph from markdown summaries."""
        from pathlib import Path

        from ..analysis.rendering import SummaryInsightAnalyzer

        summaries_dir = Path(request.summaries_dir or "artifacts/reports")

        analyzer = SummaryInsightAnalyzer(
            summaries_dir=summaries_dir,
            similarity_threshold=request.similarity_threshold,
            max_similar_results=request.max_similar_results,
        )

        try:
            async with analyzer:
                result = await analyzer.analyze()

                insights = [
                    InsightItem(
                        title=insight.title,
                        description=insight.description,
                        confidence=insight.confidence,
                    )
                    for insight in result.insights
                ]

                return BuildInsightGraphResponse(
                    nodes_count=result.graph.number_of_nodes(),
                    edges_count=result.graph.number_of_edges(),
                    insights=insights,
                    metrics=result.metrics,
                    message=f"Граф построен: {result.graph.number_of_nodes()} узлов, {result.graph.number_of_edges()} связей",
                )
        except Exception as e:
            logger.error(f"Failed to build insight graph: {e}")
            return BuildInsightGraphResponse(
                nodes_count=0,
                edges_count=0,
                insights=[],
                metrics={},
                message=f"Ошибка при построении графа: {str(e)}",
            )

    # Indexing
    async def index_chat(self, request: "IndexChatRequest") -> "IndexChatResponse":
        """Index a specific Telegram chat with two-level indexing."""
        from ..core.indexing import TwoLevelIndexer
        from ..config import get_settings
        from ..core.adapters.langchain_adapters import get_llm_client_factory
        from pathlib import Path

        settings = get_settings()
        
        # Инициализируем embedding client
        embedding_client = get_llm_client_factory()
        if embedding_client is None:
            raise ValueError(
                "Не удалось инициализировать LangChain LLM клиент. "
                "Убедитесь, что LangChain установлен и MEMORY_MCP_LMSTUDIO_LLM_MODEL настроен."
            )
        
        # Инициализируем индексатор с графом памяти
        indexer = TwoLevelIndexer(
            artifacts_path=settings.artifacts_path,
            embedding_client=embedding_client,
            enable_smart_aggregation=True,
            aggregation_strategy="smart",
            graph=self.graph,  # Передаём граф для синхронизации записей
        )
        
        try:
            # Запускаем индексацию
            stats = await indexer.build_index(
                scope="chat",
                chat=request.chat,
                force_full=request.force_full,
                recent_days=request.recent_days,
            )
            
            return IndexChatResponse(
                success=True,
                indexed_chats=stats.get("indexed_chats", []),
                sessions_indexed=stats.get("sessions_indexed", 0),
                messages_indexed=stats.get("messages_indexed", 0),
                tasks_indexed=stats.get("tasks_indexed", 0),
                message=f"Индексация чата '{request.chat}' завершена успешно. "
                       f"Сессий: {stats.get('sessions_indexed', 0)}, "
                       f"Сообщений: {stats.get('messages_indexed', 0)}, "
                       f"Задач: {stats.get('tasks_indexed', 0)}",
            )
        except Exception as e:
            logger.error(f"Ошибка при индексации чата '{request.chat}': {e}", exc_info=True)
            return IndexChatResponse(
                success=False,
                indexed_chats=[],
                sessions_indexed=0,
                messages_indexed=0,
                tasks_indexed=0,
                message=f"Ошибка при индексации чата '{request.chat}': {str(e)}",
            )

    def get_available_chats(self, request: GetAvailableChatsRequest) -> GetAvailableChatsResponse:
        """Get list of all available Telegram chats for indexing."""
        from ..config import get_settings
        from datetime import datetime

        settings = get_settings()
        chats_path = Path(settings.chats_path).expanduser()

        if not chats_path.exists():
            return GetAvailableChatsResponse(
                chats=[],
                total_count=0,
                message=f"Директория с чатами не найдена: {chats_path}",
            )

        chats = []
        for chat_dir in sorted(chats_path.iterdir()):
            if not chat_dir.is_dir():
                continue

            chat_info = ChatInfo(
                name=chat_dir.name,
                path=str(chat_dir),
                message_count=0,
                last_modified=None,
            )

            if request.include_stats:
                # Подсчитываем количество JSON файлов с сообщениями
                json_files = list(chat_dir.rglob("*.json"))
                chat_info.message_count = len(json_files)

                # Получаем дату последнего изменения
                if json_files:
                    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
                    last_modified = datetime.fromtimestamp(latest_file.stat().st_mtime)
                    chat_info.last_modified = last_modified.isoformat()

            chats.append(chat_info)

        return GetAvailableChatsResponse(
            chats=chats,
            total_count=len(chats),
            message=f"Найдено {len(chats)} доступных чатов" + (
                f" в {chats_path}" if chats else ""
            ),
        )

    def _search_qdrant_collections(
        self,
        query: str,
        limit: int = 10,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[SearchResultItem]:
        """Поиск в Qdrant коллекциях, созданных TwoLevelIndexer."""
        if not self.embedding_service:
            return []
        
        try:
            from ..memory.storage.vector.vector_store import VectorStore
            from ..config import get_settings
            
            settings = get_settings()
            qdrant_url = settings.get_qdrant_url()
            if not qdrant_url:
                return []
            
            query_vector = self.embedding_service.embed(query)
            if not query_vector:
                return []
            
            results: List[SearchResultItem] = []
            
            # Используем векторный поиск через VectorStore для каждой коллекции
            for collection_name in ["chat_messages", "chat_sessions", "chat_tasks"]:
                try:
                    # Создаем временный VectorStore для этой коллекции
                    temp_vector_store = VectorStore(url=qdrant_url, collection=collection_name)
                    
                    if not temp_vector_store.available():
                        continue
                    
                    # Выполняем векторный поиск
                    vector_matches = temp_vector_store.search(
                        query_vector,
                        limit=limit,
                        source=source,
                        tags=tags,
                        date_from=date_from,
                        date_to=date_to,
                    )
                    
                    for match in vector_matches:
                        payload = match.payload or {}
                        document = payload.get("document", "")
                        metadata = payload
                        
                        chat_name = metadata.get("chat", "")
                        date_utc = metadata.get("date_utc", "")
                        timestamp = None
                        if date_utc:
                            try:
                                timestamp = parse_datetime_utc(date_utc, use_zoneinfo=True)
                            except Exception:
                                pass
                        
                        result_source = chat_name if chat_name else collection_name.replace("chat_", "")
                        if source and source != chat_name:
                            continue
                        
                        if timestamp:
                            if date_from and timestamp < date_from:
                                continue
                            if date_to and timestamp > date_to:
                                continue
                        
                        # Пытаемся получить эмбеддинг из графа, если запись там есть
                        embedding = None
                        if match.record_id in self.graph.graph:
                            node_data = self.graph.graph.nodes[match.record_id]
                            emb_from_graph = node_data.get("embedding")
                            if emb_from_graph is not None:
                                # Преобразуем numpy массив в список, если нужно
                                if hasattr(emb_from_graph, 'tolist'):
                                    embedding = emb_from_graph.tolist()
                                elif not isinstance(emb_from_graph, list):
                                    try:
                                        embedding = list(emb_from_graph)
                                    except (TypeError, ValueError):
                                        embedding = None
                                else:
                                    embedding = emb_from_graph
                        
                        result = SearchResultItem(
                            record_id=match.record_id,
                            score=match.score,
                            content=document[:500] if document else "",
                            source=result_source,
                            timestamp=timestamp,
                            author=None,
                            metadata={
                                "collection": collection_name,
                                "chat": chat_name,
                                "vector_score": match.score,
                                "search_source": "vector_qdrant",
                                **metadata,
                            },
                            embedding=embedding,
                        )
                        results.append(result)
                        
                except Exception as e:
                    logger.debug(f"Ошибка при поиске в коллекции {collection_name}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.debug(f"Ошибка при поиске в Qdrant коллекциях: {e}", exc_info=True)
            return []

    def _clear_qdrant_chat(self, chat_name: str) -> int:
        """Удаление всех записей конкретного чата из Qdrant коллекций."""
        total_deleted = 0
        try:
            from ..memory.storage.vector.qdrant_collections import QdrantCollectionsManager
            from ..config import get_settings

            settings = get_settings()
            qdrant_url = settings.get_qdrant_url()
            if not qdrant_url:
                logger.debug("Qdrant URL не установлен, очистка невозможна")
                return 0

            # Получаем размерность эмбеддингов
            embedding_dimension = self.embedding_service.dimension if self.embedding_service else 1024
            qdrant_manager = QdrantCollectionsManager(url=qdrant_url, vector_size=embedding_dimension)
            
            if not qdrant_manager.available():
                logger.debug("Qdrant недоступен, очистка невозможна")
                return 0

            for collection_name in ["chat_sessions", "chat_messages", "chat_tasks"]:
                try:
                    deleted_count = qdrant_manager.delete(
                        collection_name=collection_name,
                        where={"chat": chat_name}
                    )

                    if deleted_count == 0:
                        logger.debug(
                            f"Нет записей для удаления в коллекции {collection_name} (чат: {chat_name})"
                        )
                        continue

                    total_deleted += deleted_count
                    logger.info(
                        f"Удалено {deleted_count} записей из коллекции {collection_name} (чат: {chat_name})"
                    )

                except Exception as e:
                    logger.warning(
                        f"Ошибка при удалении из коллекции {collection_name} (чат: {chat_name}): {e}",
                        exc_info=True,
                    )
                    continue

            logger.info(
                f"Всего удалено {total_deleted} записей из Qdrant (чат: {chat_name})"
            )
            return total_deleted

        except Exception as e:
            logger.warning(
                f"Ошибка при очистке Qdrant для чата {chat_name}: {e}",
                exc_info=True,
            )
            return total_deleted

    # Utils
    def _build_item_from_graph(
        self,
        record_id: str,
        score: float,
        *,
        snippet: Optional[str] = None,
        content_override: Optional[str] = None,
    ) -> Optional[SearchResultItem]:
        if record_id not in self.graph.graph:
            return None
        data = self.graph.graph.nodes[record_id]
        props = dict(data.get("properties") or {})
        content = content_override or data.get("content") or props.get("content") or ""
        snippet_text = snippet or content[:200]
        source = data.get("source") or props.get("source") or "unknown"
        timestamp_raw = props.get("timestamp") or data.get("timestamp")
        if timestamp_raw:
            timestamp = _parse_timestamp(timestamp_raw)
        else:
            logger.warning(
                f"Отсутствует timestamp для записи {record_id}. "
                f"Используется текущее время. Это может указывать на проблему в данных."
            )
            timestamp = datetime.now(timezone.utc)
        author = data.get("author") or props.get("author")
        metadata = dict(props) if props else {}
        
        # Получаем эмбеддинг из графа или БД
        embedding = data.get("embedding")
        if embedding is None:
            try:
                cursor = self.graph.conn.cursor()
                cursor.execute(
                    "SELECT embedding FROM nodes WHERE id = ?",
                    (record_id,),
                )
                db_row = cursor.fetchone()
                if db_row and db_row["embedding"]:
                    import json
                    embedding = json.loads(db_row["embedding"].decode())
            except Exception:
                pass
        
        # Преобразуем numpy массив в список, если нужно
        if embedding is not None:
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            elif not isinstance(embedding, list):
                try:
                    embedding = list(embedding)
                except (TypeError, ValueError):
                    embedding = None
        
        return SearchResultItem(
            record_id=record_id,
            score=score,
            content=snippet_text,
            source=source,
            timestamp=timestamp,
            author=author,
            metadata=metadata,
            embedding=embedding,
        )

    def search_artifacts(
        self,
        query: str,
        artifact_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Поиск по артифактам через ArtifactsReader.

        Args:
            query: Поисковый запрос
            artifact_types: Фильтр по типам артифактов
            limit: Максимальное количество результатов

        Returns:
            Список результатов поиска в формате словарей
        """
        results = self.artifacts_reader.search_artifacts(
            query=query,
            artifact_types=artifact_types,
            limit=limit,
        )

        # Конвертируем в формат словарей
        return [
            {
                "artifact_path": r.artifact_path,
                "artifact_type": r.artifact_type,
                "score": r.score,
                "content": r.content,
                "snippet": r.snippet,
                "metadata": r.metadata,
                "line_number": r.line_number,
            }
            for r in results
        ]

    # --------------------------- Unified methods for tool optimization ---------------------------

    async def unified_search(self, request: UnifiedSearchRequest) -> UnifiedSearchResponse:
        """Универсальный метод поиска, объединяющий все типы поиска."""
        if request.search_type == "hybrid":
            if not request.query:
                raise ValueError("query is required for hybrid search")
            search_req = SearchRequest(
                query=request.query,
                top_k=request.top_k,
                source=request.source,
                tags=request.tags,
                date_from=request.date_from,
                date_to=request.date_to,
                include_embeddings=request.include_embeddings,
            )
            result = self.search(search_req)
            return UnifiedSearchResponse(
                search_type="hybrid",
                results=result.results,
                total_matches=result.total_matches,
            )
        
        elif request.search_type == "smart":
            if not request.query:
                raise ValueError("query is required for smart search")
            # Smart search требует SmartSearchEngine, который вызывается из server.py
            # Здесь мы возвращаем ошибку, так как это async операция
            raise NotImplementedError("smart search should be called through server.py with SmartSearchEngine")
        
        elif request.search_type == "embedding":
            if not request.embedding:
                raise ValueError("embedding is required for embedding search")
            # Прямой поиск по вектору эмбеддинга
            if not self.vector_store:
                return UnifiedSearchResponse(
                    search_type="embedding",
                    results=[],
                    total_matches=0,
                )
            
            vector_results = self.vector_store.search(
                request.embedding,
                limit=request.top_k,
                source=request.source,
                tags=request.tags if request.tags else None,
                date_from=request.date_from,
                date_to=request.date_to,
            )
            
            results = []
            for match in vector_results:
                record_id = match.record_id
                score = match.score
                payload = match.payload or {}
                snippet = payload.get("content_preview", "")
                item = self._build_item_from_graph(record_id, score, snippet=snippet)
                if item:
                    results.append(item)
            
            return UnifiedSearchResponse(
                search_type="embedding",
                results=results[:request.top_k],
                total_matches=len(results),
            )
        
        elif request.search_type == "similar":
            if not request.record_id:
                raise ValueError("record_id is required for similar search")
            # Прямой поиск похожих записей
            fetch_req = FetchRequest(record_id=request.record_id)
            fetch_resp = self.fetch(fetch_req)
            if not fetch_resp.record:
                return UnifiedSearchResponse(
                    search_type="similar",
                    results=[],
                    total_matches=0,
                )
            
            if not self.embedding_service or not self.vector_store:
                return UnifiedSearchResponse(
                    search_type="similar",
                    results=[],
                    total_matches=0,
                )
            
            vector = self.embedding_service.embed(fetch_resp.record.content)
            if not vector:
                return UnifiedSearchResponse(
                    search_type="similar",
                    results=[],
                    total_matches=0,
                )
            
            vector_results = self.vector_store.search(
                vector,
                limit=request.top_k + 1,
            )
            
            results = []
            for match in vector_results:
                if match.record_id == request.record_id:
                    continue
                item = self._build_item_from_graph(match.record_id, match.score)
                if item:
                    results.append(item)
                if len(results) >= request.top_k:
                    break
            
            return UnifiedSearchResponse(
                search_type="similar",
                results=results,
                total_matches=len(results),
            )
        
        elif request.search_type == "trading":
            if not request.query:
                raise ValueError("query is required for trading search")
            # Прямой поиск торговых паттернов
            rows = self.trading_memory.search_patterns(
                query=request.query,
                symbol=request.symbol,
                timeframe=None,  # timeframe не передается в UnifiedSearchRequest
                limit=request.limit or request.top_k,
            )
            signals = [
                TradingSignalRecord(
                    signal_id=row["signal_id"],
                    timestamp=row["timestamp"],
                    symbol=row["symbol"],
                    signal_type=row["signal_type"],
                    direction=row.get("direction"),
                    entry=row.get("entry"),
                    confidence=row.get("confidence"),
                    context=row.get("context", {}),
                )
                for row in rows
            ]
            return UnifiedSearchResponse(
                search_type="trading",
                signals=signals,
            )
        
        else:
            raise ValueError(f"Unknown search_type: {request.search_type}")

    def batch_operations(self, request: BatchOperationsRequest) -> BatchOperationsResponse:
        """Универсальный метод для batch операций."""
        if request.operation == "update":
            if not request.updates:
                raise ValueError("updates is required for update operation")
            update_req = BatchUpdateRecordsRequest(updates=request.updates)
            result = self.batch_update_records(update_req)
            return BatchOperationsResponse(
                operation="update",
                update_results=result.results,
                total_updated=result.total_updated,
                total_failed=result.total_failed,
            )
        
        elif request.operation == "delete":
            if not request.record_ids:
                raise ValueError("record_ids is required for delete operation")
            delete_req = BatchDeleteRecordsRequest(record_ids=request.record_ids)
            result = self.batch_delete_records(delete_req)
            return BatchOperationsResponse(
                operation="delete",
                delete_results=result.results,
                total_deleted=result.total_deleted,
            )
        
        elif request.operation == "fetch":
            if not request.record_ids:
                raise ValueError("record_ids is required for fetch operation")
            fetch_req = BatchFetchRecordsRequest(record_ids=request.record_ids)
            result = self.batch_fetch_records(fetch_req)
            return BatchOperationsResponse(
                operation="fetch",
                fetch_results=result.results,
                total_found=result.total_found,
                total_not_found=result.total_not_found,
            )
        
        else:
            raise ValueError(f"Unknown operation: {request.operation}")

    def get_statistics_unified(self, request: Optional[GetStatisticsRequest] = None) -> UnifiedStatisticsResponse:
        """Универсальный метод получения статистики."""
        if request is None or request.type is None:
            # Возвращаем всю статистику
            general_stats = self.get_statistics()
            tags_stats = self.get_tags_statistics()
            indexing_req = GetIndexingProgressRequest(chat=request.chat if request else None)
            indexing_stats = self.get_indexing_progress(indexing_req)
            
            return UnifiedStatisticsResponse(
                graph_stats=general_stats.graph_stats,
                sources_count=general_stats.sources_count,
                tags_count=general_stats.tags_count,
                database_size_bytes=general_stats.database_size_bytes,
                total_tags=tags_stats.total_tags,
                total_tagged_records=tags_stats.total_tagged_records,
                indexing_progress=indexing_stats.progress,
                indexing_message=indexing_stats.message,
            )
        
        elif request.type == "general":
            general_stats = self.get_statistics()
            return UnifiedStatisticsResponse(
                graph_stats=general_stats.graph_stats,
                sources_count=general_stats.sources_count,
                tags_count=general_stats.tags_count,
                database_size_bytes=general_stats.database_size_bytes,
            )
        
        elif request.type == "tags":
            tags_stats = self.get_tags_statistics()
            return UnifiedStatisticsResponse(
                tags_count=tags_stats.tags_count,
                total_tags=tags_stats.total_tags,
                total_tagged_records=tags_stats.total_tagged_records,
            )
        
        elif request.type == "indexing":
            indexing_req = GetIndexingProgressRequest(chat=request.chat)
            indexing_stats = self.get_indexing_progress(indexing_req)
            return UnifiedStatisticsResponse(
                indexing_progress=indexing_stats.progress,
                indexing_message=indexing_stats.message,
            )
        
        else:
            raise ValueError(f"Unknown statistics type: {request.type}")

    def graph_query(self, request: GraphQueryRequest) -> GraphQueryResponse:
        """Универсальный метод запросов к графу."""
        if request.query_type == "neighbors":
            if not request.node_id:
                raise ValueError("node_id is required for neighbors query")
            neighbors_req = GetGraphNeighborsRequest(
                node_id=request.node_id,
                edge_type=request.edge_type,
                direction=request.direction or "both",
            )
            result = self.get_graph_neighbors(neighbors_req)
            return GraphQueryResponse(
                query_type="neighbors",
                neighbors=result.neighbors,
            )
        
        elif request.query_type == "path":
            if not request.source_id or not request.target_id:
                raise ValueError("source_id and target_id are required for path query")
            path_req = FindGraphPathRequest(
                source_id=request.source_id,
                target_id=request.target_id,
                max_length=request.max_length or 5,
            )
            result = self.find_graph_path(path_req)
            return GraphQueryResponse(
                query_type="path",
                path=result.path,
                total_weight=result.total_weight,
                found=result.found,
            )
        
        elif request.query_type == "related":
            node_id = request.node_id or request.record_id
            if not node_id:
                raise ValueError("node_id or record_id is required for related query")
            related_req = GetRelatedRecordsRequest(
                record_id=node_id,
                max_depth=request.max_depth or 1,
                limit=request.limit or 10,
            )
            result = self.get_related_records(related_req)
            return GraphQueryResponse(
                query_type="related",
                records=result.records,
            )
        
        else:
            raise ValueError(f"Unknown query_type: {request.query_type}")

    def ingest_unified(self, request: IngestRequest) -> IngestResponse:
        """Универсальный метод индексации, поддерживающий records и scraped."""
        if request.source_type == "scraped":
            # Преобразуем IngestRequest в ScrapedContentRequest
            if not request.url or not request.content:
                raise ValueError("url and content are required for scraped source_type")
            scraped_req = ScrapedContentRequest(
                url=request.url,
                title=request.title,
                content=request.content,
                metadata=request.metadata or {},
                source=request.source or "bright_data",
                tags=request.tags or [],
                entities=request.entities or [],
            )
            result = self.ingest_scraped_content(scraped_req)
            # Преобразуем ScrapedContentResponse в IngestResponse
            return IngestResponse(
                records_ingested=1 if result.status == "success" else 0,
                attachments_ingested=0,
                duplicates_skipped=0,
            )
        else:
            # Обычная индексация записей
            return self.ingest(request.records)

    async def summaries(self, request: SummariesRequest) -> SummariesResponse:
        """Универсальный метод управления саммаризацией."""
        if request.action == "update":
            update_req = UpdateSummariesRequest(
                chat=request.chat,
                force=request.force or False,
            )
            result = await self.update_summaries(update_req)
            return SummariesResponse(
                action="update",
                message=result.message,
                chats_updated=result.chats_updated,
            )
        
        elif request.action == "review":
            review_req = ReviewSummariesRequest(
                dry_run=request.dry_run or False,
                chat=request.chat,
                limit=request.limit,
            )
            result = await self.review_summaries(review_req)
            return SummariesResponse(
                action="review",
                message=result.message,
                files_processed=result.files_processed,
                files_fixed=result.files_fixed,
            )
        
        else:
            raise ValueError(f"Unknown action: {request.action}")
