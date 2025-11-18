"""Adapters converting MCP requests into memory operations."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..core.constants import DEFAULT_SEARCH_LIMIT
from ..indexing import Attachment, MemoryRecord
from ..memory.artifacts_reader import ArtifactsReader
from ..memory.embeddings import build_embedding_service_from_env
from ..memory.graph_types import EdgeType, NodeType
from ..memory.ingest import MemoryIngestor
from ..memory.trading_memory import TradingMemory
from ..memory.typed_graph import TypedGraphMemory
from ..memory.vector_store import build_vector_store_from_env
from ..utils.datetime_utils import parse_datetime_utc
from .schema import (
    AnalyzeEntitiesRequest,
    AnalyzeEntitiesResponse,
    AttachmentPayload,
    BatchDeleteRecordsRequest,
    BatchDeleteRecordsResponse,
    BatchDeleteResult,
    BatchFetchRecordsRequest,
    BatchFetchRecordsResponse,
    BatchFetchResult,
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
    IndexChatRequest,
    IndexChatResponse,
    GetAvailableChatsRequest,
    GetAvailableChatsResponse,
    ChatInfo,
    GetSignalPerformanceRequest,
    GetSignalPerformanceResponse,
    GetStatisticsResponse,
    GetTagsStatisticsResponse,
    GetTimelineRequest,
    GetTimelineResponse,
    GraphNeighborItem,
    ImportRecordsRequest,
    ImportRecordsResponse,
    IndexingProgressItem,
    InsightItem,
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
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SearchTradingPatternsRequest,
    SearchTradingPatternsResponse,
    SimilarRecordsRequest,
    SimilarRecordsResponse,
    SignalPerformance,
    StoreTradingSignalRequest,
    StoreTradingSignalResponse,
    TimelineItem,
    TradingSignalRecord,
    UpdateRecordRequest,
    UpdateRecordResponse,
    UpdateSummariesRequest,
    UpdateSummariesResponse,
)

logger = logging.getLogger(__name__)


def _parse_timestamp(value: str | None) -> datetime:
    """Парсинг временной метки."""
    return parse_datetime_utc(value, default=datetime.now(timezone.utc))


def _payload_to_record(payload: MemoryRecordPayload) -> MemoryRecord:
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
    
    # Получаем эмбеддинг из данных узла
    embedding = data.get("embedding")
    if embedding is not None:
        # Преобразуем numpy массив в список, если нужно
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        elif not isinstance(embedding, list):
            try:
                embedding = list(embedding)
            except (TypeError, ValueError):
                embedding = None

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
    """High-level wrapper exposing ingest/search/fetch for MCP tools."""

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
        # Веса для гибридного поиска (FTS + векторный)
        # Увеличиваем приоритет FTS5 результатов для лучшего ранжирования
        _fts_weight_raw = 0.8
        _vector_weight_raw = 0.2
        _total_weight = _fts_weight_raw + _vector_weight_raw
        self._fts_weight = _fts_weight_raw / _total_weight
        self._vector_weight = _vector_weight_raw / _total_weight
        # Параметр для Reciprocal Rank Fusion (стандартное значение 60)
        self._rrf_k = 60
        # Использовать RRF для объединения результатов (True) или простое объединение с весами (False)
        self._use_rrf = True
        # Инициализация читателя артифактов
        from ..config import get_settings
        settings = get_settings()
        self.artifacts_reader = ArtifactsReader(artifacts_dir=settings.artifacts_path)

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
        """Очистка всех данных конкретного чата из всех хранилищ."""
        stats = {
            "nodes_deleted": 0,
            "vectors_deleted": 0,
            "chromadb_deleted": 0,
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
            chromadb_deleted = self._clear_chromadb_chat(chat_name)
            stats["chromadb_deleted"] = chromadb_deleted
            logger.info(
                f"Удалено {chromadb_deleted} записей из ChromaDB (чат: {chat_name})"
            )
        except Exception as e:
            logger.warning(
                f"Ошибка при очистке ChromaDB для чата {chat_name}: {e}",
                exc_info=True,
            )

        total_deleted = (
            stats["nodes_deleted"]
            + stats["vectors_deleted"]
            + stats["chromadb_deleted"]
        )
        logger.info(
            f"Очистка завершена для чата {chat_name}: "
            f"узлов={stats['nodes_deleted']}, "
            f"векторов={stats['vectors_deleted']}, "
            f"ChromaDB={stats['chromadb_deleted']}, "
            f"всего={total_deleted}"
        )

        return stats

    # Ingest
    def _build_embedding_text(self, payload: MemoryRecordPayload) -> str:
        """
        Формирует расширенный текст для эмбеддингов, включая метаданные.
        
        Args:
            payload: Запись памяти с метаданными
            
        Returns:
            Расширенный текст с контентом и метаданными
        """
        parts = [payload.content]
        metadata_parts = []
        
        # Добавляем username отправителя, если есть
        sender_username = payload.metadata.get("sender_username")
        if sender_username:
            metadata_parts.append(f"Автор: @{sender_username}")
        elif payload.author:
            metadata_parts.append(f"Автор: {payload.author}")
        
        # Добавляем реакции, если есть
        reactions = payload.metadata.get("reactions")
        if reactions and isinstance(reactions, list) and len(reactions) > 0:
            reaction_strs = []
            for reaction in reactions:
                if isinstance(reaction, dict):
                    emoji = reaction.get("emoji", "")
                    count = reaction.get("count", 0)
                    if emoji and count > 0:
                        # Извлекаем emoji из строки вида "ReactionEmoji(emoticon='👍')"
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
        
        # Добавляем информацию о редактировании, если есть
        edited_utc = payload.metadata.get("edited_utc")
        if edited_utc:
            metadata_parts.append(f"Отредактировано: {edited_utc}")
        
        # Объединяем все части
        if metadata_parts:
            parts.append("\n[Метаданные]")
            parts.extend(metadata_parts)
        
        return "\n".join(parts)

    def ingest(self, payloads: Iterable[MemoryRecordPayload]) -> IngestResponse:
        payload_list = list(payloads)
        records = [_payload_to_record(item) for item in payload_list]
        stats = self.ingestor.ingest(records)
        if self.embedding_service and self.vector_store and payload_list:
            for payload in payload_list:
                embedding_text = self._build_embedding_text(payload)
                vector = self.embedding_service.embed(embedding_text)
                if not vector:
                    continue
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
                    # Сохраняем эмбеддинг в Qdrant
                    self.vector_store.upsert(payload.record_id, vector, payload_data)
                    # Сохраняем эмбеддинг в граф
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
            result_lists: Список списков результатов из разных источников (FTS5, Vector, ChromaDB)
            k: Параметр RRF (стандартное значение 60)
            
        Returns:
            Словарь {record_id: rrf_score} отсортированный по убыванию score
        """
        rrf_scores: dict[str, float] = {}
        
        for results in result_lists:
            for rank, item in enumerate(results, start=1):
                # RRF score = 1 / (k + rank)
                rrf_score = 1.0 / (k + rank)
                # Суммируем scores для одинаковых record_id
                rrf_scores[item.record_id] = rrf_scores.get(item.record_id, 0.0) + rrf_score
        
        return rrf_scores

    # Search
    def search(self, request: SearchRequest) -> SearchResponse:
        # 1. FTS5 поиск
        rows, total_fts = self.graph.search_text(
            request.query,
            limit=request.top_k * 2,  # Берем больше для RRF
            source=request.source,
            tags=request.tags,
            date_from=request.date_from,
            date_to=request.date_to,
        )

        fts_results: list[SearchResultItem] = []
        all_items: dict[str, SearchResultItem] = {}
        
        for row in rows:
            # Получаем эмбеддинг из графа, если доступен
            embedding = None
            if row["node_id"] in self.graph.graph:
                node_data = self.graph.graph.nodes[row["node_id"]]
                embedding = node_data.get("embedding")
                # Преобразуем numpy массив в список, если нужно
                if embedding is not None:
                    if hasattr(embedding, 'tolist'):
                        embedding = embedding.tolist()
                    elif not isinstance(embedding, list):
                        try:
                            embedding = list(embedding)
                        except (TypeError, ValueError):
                            embedding = None
                
                # Если эмбеддинг не в графе, пытаемся получить из БД
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
            
            item = SearchResultItem(
                record_id=row["node_id"],
                score=row["score"],  # Оригинальный score, RRF пересчитает
                content=row["snippet"],
                source=row["source"],
                timestamp=row["timestamp"],
                author=row.get("author"),
                metadata=row.get("metadata", {}),
                embedding=embedding,
            )
            fts_results.append(item)
            all_items[row["node_id"]] = item

        # 2. Векторный поиск (Qdrant)
        vector_results: list[SearchResultItem] = []
        if self.embedding_service and self.vector_store:
            query_vector = self.embedding_service.embed(request.query)
            if query_vector:
                vector_matches = self.vector_store.search(
                    query_vector,
                    limit=request.top_k * 2,  # Берем больше для RRF
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
                    
                    # Пытаемся получить полные данные из графа или используем данные из payload
                    item = self._build_item_from_graph(
                        record_id,
                        match.score,  # Оригинальный score, RRF пересчитает
                        snippet=snippet,
                    )
                    if item:
                        vector_results.append(item)
                        # Сохраняем в all_items, если еще нет (или обновляем, если векторный результат лучше)
                        if record_id not in all_items:
                            all_items[record_id] = item
                        elif not all_items[record_id].embedding and item.embedding:
                            # Обновляем эмбеддинг, если его не было
                            all_items[record_id].embedding = item.embedding

        # 3. Поиск в ChromaDB коллекциях
        chromadb_results: list[SearchResultItem] = []
        chromadb_matches = self._search_chromadb_collections(
            request.query,
            limit=request.top_k,
            source=request.source,
            tags=request.tags,
            date_from=request.date_from,
            date_to=request.date_to,
        )
        
        for match in chromadb_matches:
            record_id = match.record_id
            # Пытаемся получить эмбеддинг из графа, если его нет в ChromaDB результате
            if not match.embedding and record_id in self.graph.graph:
                node_data = self.graph.graph.nodes[record_id]
                emb_from_graph = node_data.get("embedding")
                if emb_from_graph is not None:
                    # Преобразуем numpy массив в список, если нужно
                    if hasattr(emb_from_graph, 'tolist'):
                        emb_from_graph = emb_from_graph.tolist()
                    elif not isinstance(emb_from_graph, list):
                        try:
                            emb_from_graph = list(emb_from_graph)
                        except (TypeError, ValueError):
                            emb_from_graph = None
                    if emb_from_graph:
                        # Создаем новый SearchResultItem с эмбеддингом из графа
                        match = SearchResultItem(
                            record_id=match.record_id,
                            score=match.score,  # Оригинальный score, RRF пересчитает
                            content=match.content,
                            source=match.source,
                            timestamp=match.timestamp,
                            author=match.author,
                            metadata=match.metadata,
                            embedding=emb_from_graph,
                        )
            chromadb_results.append(match)
            # Добавляем в all_items, если еще нет
            if record_id not in all_items:
                all_items[record_id] = match

        # 4. Boost для точных совпадений в FTS5 результатах
        query_words = set(request.query.lower().split())
        for item in fts_results:
            # Проверяем, содержатся ли все слова запроса в контенте
            content_lower = item.content.lower()
            matched_words = sum(1 for word in query_words if word in content_lower)
            if matched_words == len(query_words) and len(query_words) > 0:
                # Все слова найдены - добавляем boost 20%
                item.score *= 1.2

        # 5. Объединение результатов с помощью RRF или простого объединения
        if self._use_rrf and (fts_results or vector_results or chromadb_results):
            # Применяем RRF для объединения результатов
            result_lists = []
            if fts_results:
                result_lists.append(fts_results)
            if vector_results:
                result_lists.append(vector_results)
            if chromadb_results:
                result_lists.append(chromadb_results)
            
            rrf_scores = self._reciprocal_rank_fusion(result_lists, k=self._rrf_k)
            
            # Обновляем scores в all_items на основе RRF
            for record_id, rrf_score in rrf_scores.items():
                if record_id in all_items:
                    all_items[record_id].score = rrf_score
            
            # Сортируем по RRF score
            results = sorted(all_items.values(), key=lambda item: item.score, reverse=True)
        else:
            # Простое объединение с весами (старая логика)
            combined: dict[str, SearchResultItem] = {}
            
            # FTS5 результаты с весом
            for item in fts_results:
                item.score = item.score * self._fts_weight
                combined[item.record_id] = item
            
            # Vector результаты с весом
            for item in vector_results:
                score = item.score * self._vector_weight
                existing = combined.get(item.record_id)
                if existing:
                    existing.score = max(existing.score, score)
                    # Обновляем эмбеддинг, если его не было
                    if not existing.embedding and item.embedding:
                        existing.embedding = item.embedding
                else:
                    item.score = score
                    combined[item.record_id] = item
            
            # ChromaDB результаты с низким весом
            for item in chromadb_results:
                score = item.score * 0.05
                existing = combined.get(item.record_id)
                if existing:
                    if existing.score < 0.2 and score > existing.score * 2.0:
                        existing.score = score
                    if not existing.embedding and item.embedding:
                        existing.embedding = item.embedding
                else:
                    item.score = score
                    combined[item.record_id] = item
            
            results = sorted(combined.values(), key=lambda item: item.score, reverse=True)

        total_combined = max(total_fts, len(all_items))
        return SearchResponse(
            results=results[: request.top_k],
            total_matches=total_combined,
        )

    # Fetch
    def fetch(self, request: FetchRequest) -> FetchResponse:
        try:
            # Сначала пытаемся найти в графе
            if request.record_id in self.graph.graph:
                data = self.graph.graph.nodes[request.record_id]
                payload = _node_to_payload(self.graph, request.record_id, data)
                return FetchResponse(record=payload)
            
            # Если не найдено в графе, ищем в ChromaDB коллекциях
            try:
                import chromadb
                from ..config import get_settings
                
                settings = get_settings()
                chroma_path = settings.chroma_path
                
                if not os.path.isabs(chroma_path):
                    current_dir = Path(__file__).parent
                    project_root = current_dir
                    while project_root.parent != project_root:
                        if (project_root / "pyproject.toml").exists():
                            break
                        project_root = project_root.parent
                    if not (project_root / "pyproject.toml").exists():
                        project_root = Path.cwd()
                    chroma_path = str(project_root / chroma_path)
                
                if os.path.exists(chroma_path):
                    chroma_client = chromadb.PersistentClient(path=chroma_path)
                    
                    # Ищем в коллекциях chat_messages, chat_sessions, chat_tasks
                    for collection_name in ["chat_messages", "chat_sessions", "chat_tasks"]:
                        try:
                            collection = chroma_client.get_collection(collection_name)
                            result = collection.get(ids=[request.record_id], include=["documents", "metadatas", "embeddings"])
                            
                            if result.get("ids") and len(result["ids"]) > 0:
                                doc = result["documents"][0] if result.get("documents") else ""
                                metadata = result["metadatas"][0] if result.get("metadatas") else {}
                                
                                # Преобразуем ChromaDB запись в MemoryRecordPayload
                                chat_name = metadata.get("chat", collection_name.replace("chat_", ""))
                                
                                # Парсим timestamp из разных полей
                                date_utc = metadata.get("date_utc") or metadata.get("start_time_utc") or metadata.get("end_time_utc")
                                timestamp = None
                                if date_utc:
                                    try:
                                        timestamp = parse_datetime_utc(date_utc, use_zoneinfo=True)
                                    except Exception:
                                        timestamp = datetime.now(timezone.utc)
                                else:
                                    timestamp = datetime.now(timezone.utc)
                                
                                # Извлекаем автора из разных полей
                                author = metadata.get("sender") or metadata.get("author") or metadata.get("username")
                                
                                # Извлекаем теги и сущности
                                tags = metadata.get("tags", [])
                                if isinstance(tags, str):
                                    tags = [tags] if tags else []
                                
                                entities = metadata.get("entities", [])
                                if isinstance(entities, str):
                                    entities = [entities] if entities else []
                                
                                # Получаем эмбеддинг, если доступен
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
                                
                                # Если эмбеддинг есть, но записи нет в графе, синхронизируем
                                if embedding and request.record_id not in self.graph.graph:
                                    try:
                                        # Создаём узел в графе для синхронизации
                                        from ..indexing import MemoryRecord
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
                                        # Сохраняем эмбеддинг
                                        self.graph.update_node(request.record_id, embedding=embedding)
                                        logger.debug(f"Синхронизирована запись {request.record_id} из ChromaDB в граф")
                                    except Exception as e:
                                        logger.debug(f"Ошибка при синхронизации записи {request.record_id}: {e}")
                                
                                return FetchResponse(record=payload)
                        except Exception as e:
                            logger.debug(f"Failed to fetch from collection {collection_name}: {e}")
                            continue
            except Exception as e:
                logger.debug(f"Failed to fetch from ChromaDB: {e}")
            
            return FetchResponse(record=None)
        except Exception as e:
            logger.error(f"Failed to fetch record {request.record_id}: {e}", exc_info=True)
            return FetchResponse(record=None)

    # Trading API
    def store_trading_signal(
        self, request: StoreTradingSignalRequest
    ) -> StoreTradingSignalResponse:
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

    # Embeddings
    def generate_embedding(
        self, request: GenerateEmbeddingRequest
    ) -> GenerateEmbeddingResponse:
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

    # Record Management
    def update_record(self, request: UpdateRecordRequest) -> UpdateRecordResponse:
        """Update an existing memory record."""
        node = self.graph.get_node(request.record_id)
        if not node:
            return UpdateRecordResponse(
                record_id=request.record_id,
                updated=False,
                message=f"Record {request.record_id} not found",
            )

        # Проверяем, есть ли изменения
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

    # Statistics
    def get_statistics(self) -> GetStatisticsResponse:
        """Get system statistics."""
        graph_stats_obj = self.graph.get_stats()
        graph_stats = graph_stats_obj.model_dump()

        sources_count = {}
        tags_count = {}
        
        # Проверяем наличие столбца properties в таблице nodes
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

        # Подсчитываем источники из графа
        try:
            cursor.execute(
                """
                SELECT properties FROM nodes
                WHERE type = 'DocChunk'
            """
            )
            for row in cursor.fetchall():
                if row["properties"]:
                    props = json.loads(row["properties"])
                    source = props.get("source", "unknown")
                    sources_count[source] = sources_count.get(source, 0) + 1
        except Exception as e:
            logger.warning(f"Ошибка при подсчёте источников из графа: {e}")

        # Подсчитываем теги из графа
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

        # Добавляем статистику из ChromaDB коллекций
        try:
            import chromadb
            from ..config import get_settings
            
            settings = get_settings()
            chroma_path = settings.chroma_path
            
            if not os.path.isabs(chroma_path):
                current_dir = Path(__file__).parent
                project_root = current_dir
                while project_root.parent != project_root:
                    if (project_root / "pyproject.toml").exists():
                        break
                    project_root = project_root.parent
                if not (project_root / "pyproject.toml").exists():
                    project_root = Path.cwd()
                chroma_path = str(project_root / chroma_path)
            
            if os.path.exists(chroma_path):
                chroma_client = chromadb.PersistentClient(path=chroma_path)
                
                for collection_name in ["chat_messages", "chat_sessions", "chat_tasks"]:
                    try:
                        collection = chroma_client.get_collection(collection_name)
                        # Используем count() для получения общего количества записей
                        total_count = collection.count()
                        
                        # Получаем все метаданные для подсчёта источников и тегов
                        # Используем scroll для больших коллекций
                        all_metadatas = []
                        offset = 0
                        batch_size = 1000
                        
                        while offset < total_count:
                            try:
                                result = collection.get(
                                    limit=batch_size,
                                    offset=offset,
                                    include=["metadatas"]
                                )
                                batch_metadatas = result.get("metadatas", [])
                                if not batch_metadatas:
                                    break
                                all_metadatas.extend(batch_metadatas)
                                offset += len(batch_metadatas)
                                if len(batch_metadatas) < batch_size:
                                    break
                            except Exception as e:
                                logger.debug(f"Ошибка при получении метаданных из {collection_name} (offset={offset}): {e}")
                                break
                        
                        # Подсчитываем источники и теги
                        for metadata in all_metadatas:
                            if not isinstance(metadata, dict):
                                continue
                            
                            # Подсчитываем источники (чаты)
                            chat = metadata.get("chat")
                            if chat:
                                sources_count[chat] = sources_count.get(chat, 0) + 1
                            else:
                                # Если chat не указан, используем имя коллекции
                                source_name = collection_name.replace("chat_", "")
                                sources_count[source_name] = sources_count.get(source_name, 0) + 1
                            
                            # Подсчитываем теги
                            tags = metadata.get("tags", [])
                            if isinstance(tags, str):
                                tags = [tags] if tags else []
                            if isinstance(tags, list):
                                for tag in tags:
                                    if tag:  # Пропускаем пустые теги
                                        tags_count[tag] = tags_count.get(tag, 0) + 1
                    except Exception as e:
                        logger.debug(f"Ошибка при подсчёте статистики из коллекции {collection_name}: {e}")
                        continue
        except Exception as e:
            logger.debug(f"Ошибка при подсчёте статистики из ChromaDB: {e}")

        # Получаем размер БД
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
        """Получить прогресс индексации из ChromaDB и активных задач.
        
        ВАЖНО: ChromaDB может паниковать (Rust panic), что убьет процесс Python.
        При ошибках инициализации возвращаем ошибку без попытки восстановления.
        """
        from ..mcp.server import _get_indexing_tracker
        
        # Получаем трекер задач
        tracker = _get_indexing_tracker()
        
        # Получаем активные задачи из трекера
        active_jobs = tracker.get_all_jobs(status="running", chat=request.chat) if request.chat else tracker.get_all_jobs(status="running")
        
        try:
            import chromadb
        except ImportError:
            logger.warning("chromadb is not installed. Cannot get indexing progress.")
            # Возвращаем только данные из трекера, если ChromaDB недоступен
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
            return GetIndexingProgressResponse(
                progress=progress_items,
                message="ChromaDB is not installed. Showing only active jobs from tracker.",
            )
        
        try:
            from ..utils.naming import slugify
        except ImportError:
            logger.warning("Failed to import slugify utility")
            return GetIndexingProgressResponse(
                progress=[],
                message="Internal error: failed to import required utilities",
            )
        
        chroma_path = os.getenv("MEMORY_MCP_CHROMA_PATH")
        if not chroma_path:
            try:
                from ..config import get_settings
                settings = get_settings()
                chroma_path = settings.chroma_path
            except Exception:
                chroma_path = "/app/chroma_db" if os.path.exists("/app") else "./chroma_db"
        
        if not os.path.isabs(chroma_path):
            current_dir = Path(__file__).parent
            project_root = current_dir
            while project_root.parent != project_root:
                if (project_root / "pyproject.toml").exists():
                    break
                project_root = project_root.parent
            if not (project_root / "pyproject.toml").exists():
                project_root = Path.cwd()
            chroma_path = str(project_root / chroma_path)
        
        chroma_path_obj = Path(chroma_path)
        
        if not chroma_path_obj.exists():
            try:
                chroma_path_obj.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Cannot create ChromaDB directory {chroma_path_obj}: {e}")
                return GetIndexingProgressResponse(
                    progress=[],
                    message=f"Cannot access ChromaDB directory: {str(e)}",
                )
        
        try:
            chroma_client = chromadb.PersistentClient(path=str(chroma_path_obj))
        except Exception as e:
            logger.error(
                f"Failed to initialize ChromaDB client at {chroma_path_obj}: {e}. "
                "The database may be corrupted. Consider removing the ChromaDB directory and re-indexing.",
                exc_info=True
            )
            return GetIndexingProgressResponse(
                progress=[],
                message=(
                    f"ChromaDB is not available: {str(e)}. "
                    "The database may be corrupted. "
                    "To fix: remove the ChromaDB directory and re-run indexing."
                ),
            )
        
        try:
            progress_collection = chroma_client.get_collection("indexing_progress")
        except Exception as e:
            logger.debug(f"Indexing progress collection not found: {e}")
            # Возвращаем только данные из трекера, если коллекция не найдена
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
            return GetIndexingProgressResponse(
                progress=progress_items,
                message="Indexing progress collection not found. Showing only active jobs from tracker.",
            )

        if request.chat:
            progress_id = f"progress_{slugify(request.chat)}"
            try:
                result = progress_collection.get(
                    ids=[progress_id], include=["metadatas"]
                )
                progress_item = None
                if result.get("ids") and len(result["ids"]) > 0:
                    metadata = result["metadatas"][0] if result.get("metadatas") else {}
                    progress_item = IndexingProgressItem(
                        chat_name=metadata.get("chat_name", request.chat),
                        last_indexed_date=metadata.get("last_indexed_date"),
                        last_indexing_time=metadata.get("last_indexing_time"),
                        total_messages=metadata.get("total_messages", 0),
                        total_sessions=metadata.get("total_sessions", 0),
                    )
                
                # Объединяем с данными из трекера
                active_job = None
                for job in active_jobs:
                    if job.get("chat") == request.chat or job.get("current_chat") == request.chat:
                        active_job = job
                        break
                
                if active_job:
                    if progress_item:
                        progress_item.job_id = active_job.get("job_id")
                        progress_item.status = active_job.get("status")
                        progress_item.started_at = active_job.get("started_at")
                        progress_item.current_stage = active_job.get("current_stage")
                    else:
                        progress_item = IndexingProgressItem(
                            chat_name=request.chat,
                            job_id=active_job.get("job_id"),
                            status=active_job.get("status"),
                            started_at=active_job.get("started_at"),
                            current_stage=active_job.get("current_stage"),
                            total_messages=active_job.get("stats", {}).get("messages_indexed", 0),
                            total_sessions=active_job.get("stats", {}).get("sessions_indexed", 0),
                        )
                
                if progress_item:
                    return GetIndexingProgressResponse(
                        progress=[progress_item],
                        message=None,
                    )
                else:
                    return GetIndexingProgressResponse(
                        progress=[],
                        message=f"No progress found for chat '{request.chat}'",
                    )
            except Exception as e:
                logger.warning(f"Failed to get progress for chat '{request.chat}': {e}")
                return GetIndexingProgressResponse(
                    progress=[],
                    message=f"Failed to get progress for chat '{request.chat}': {str(e)}",
                )
        else:
            try:
                result = progress_collection.get(include=["metadatas"])
                progress_items = []
                metadatas = result.get("metadatas", [])
                
                progress_by_chat = {}
                for metadata in metadatas:
                    if not isinstance(metadata, dict):
                        continue
                    chat_name = metadata.get("chat_name", "Unknown")
                    progress_by_chat[chat_name] = IndexingProgressItem(
                        chat_name=chat_name,
                        last_indexed_date=metadata.get("last_indexed_date"),
                        last_indexing_time=metadata.get("last_indexing_time"),
                        total_messages=metadata.get("total_messages", 0),
                        total_sessions=metadata.get("total_sessions", 0),
                    )
                
                # Объединяем с данными из трекера
                for job in active_jobs:
                    chat_name = job.get("chat") or job.get("current_chat") or "Unknown"
                    if chat_name in progress_by_chat:
                        progress_by_chat[chat_name].job_id = job.get("job_id")
                        progress_by_chat[chat_name].status = job.get("status")
                        progress_by_chat[chat_name].started_at = job.get("started_at")
                        progress_by_chat[chat_name].current_stage = job.get("current_stage")
                    else:
                        progress_by_chat[chat_name] = IndexingProgressItem(
                            chat_name=chat_name,
                            job_id=job.get("job_id"),
                            status=job.get("status"),
                            started_at=job.get("started_at"),
                            current_stage=job.get("current_stage"),
                            total_messages=job.get("stats", {}).get("messages_indexed", 0),
                            total_sessions=job.get("stats", {}).get("sessions_indexed", 0),
                        )
                
                return GetIndexingProgressResponse(
                    progress=list(progress_by_chat.values()),
                    message=None,
                )
            except Exception as e:
                logger.warning(f"Failed to get all progress: {e}")
                return GetIndexingProgressResponse(
                    progress=[],
                    message=f"Failed to get indexing progress: {str(e)}",
                )

    # Graph Operations
    def get_graph_neighbors(
        self, request: GetGraphNeighborsRequest
    ) -> GetGraphNeighborsResponse:
        """Get neighbors of a graph node."""
        edge_type = None
        if request.edge_type:
            try:
                edge_type = EdgeType(request.edge_type)
            except ValueError:
                logger.warning(f"Invalid edge type: {request.edge_type}")

        neighbors_data = self.graph.get_neighbors(
            request.node_id,
            edge_type=edge_type,
            direction=request.direction,
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
            if request.record_id not in self.graph.graph:
                return GetRelatedRecordsResponse(records=[])
            
            visited = set()
            related_records = []
            current_level = {request.record_id}
            
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
    def search_by_embedding(
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

    def similar_records(
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
        
        # Получаем все узлы с properties, фильтруем в Python для более гибкого поиска
        query = "SELECT id, properties FROM nodes WHERE properties IS NOT NULL"
        
        query += " ORDER BY json_extract(properties, '$.timestamp') DESC"
        if request.limit:
            # Берем больше записей для фильтрации
            query += f" LIMIT {request.limit * 10 if request.source else request.limit}"
        
        cursor.execute(query)
        
        items = []
        for row in cursor.fetchall():
            if row["properties"]:
                props = json.loads(row["properties"])
                # Используем source из properties, или chat, или "unknown"
                source = props.get("source") or props.get("chat", "unknown")
                
                # Дополнительная фильтрация по source, если указан
                if request.source and source != request.source:
                    continue
                
                timestamp_str = props.get("timestamp") or props.get("created_at")
                if not timestamp_str:
                    continue
                
                # Используем parse_datetime_utc для более гибкой обработки дат
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
                
                # Фильтруем по датам
                if request.date_from and timestamp < request.date_from:
                    continue
                if request.date_to and timestamp > request.date_to:
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
            from ..analysis.entity_extraction import EntityExtractor

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
                # Проверяем существование записи
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
        
        query = """
            SELECT id, properties FROM nodes
            WHERE type = 'DocChunk' AND properties IS NOT NULL
        """
        params = []
        
        if request.source:
            query += " AND properties LIKE ?"
            params.append(f'%"source": "{request.source}"%')
        
        if request.tags:
            for tag in request.tags:
                query += " AND properties LIKE ?"
                params.append(f'%"tags":%"{tag}"%')
        
        if request.date_from:
            query += " AND json_extract(properties, '$.timestamp') >= ?"
            if isinstance(request.date_from, datetime):
                params.append(request.date_from.timestamp())
            else:
                params.append(request.date_from)
        
        if request.date_to:
            query += " AND json_extract(properties, '$.timestamp') <= ?"
            if isinstance(request.date_to, datetime):
                params.append(request.date_to.timestamp())
            else:
                params.append(request.date_to)
        
        query += f" LIMIT {request.limit}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        records = []
        for row in rows:
            try:
                props = json.loads(row["properties"]) if isinstance(row["properties"], str) else row["properties"]
                if not props:
                    continue
                
                if request.source and props.get("source") != request.source:
                    continue
                
                if request.tags:
                    node_tags = props.get("tags", [])
                    if not isinstance(node_tags, list):
                        node_tags = []
                    if not all(tag in node_tags for tag in request.tags):
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

        from ..analysis.markdown_renderer import MarkdownRenderer

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
                from ..utils.datetime_utils import parse_datetime_utc

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

        from ..analysis.insight_graph import SummaryInsightAnalyzer

        summaries_dir = Path(request.summaries_dir or "artifacts/reports")
        chroma_path = Path(request.chroma_path or "./chroma_db")

        analyzer = SummaryInsightAnalyzer(
            summaries_dir=summaries_dir,
            chroma_path=chroma_path,
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
        from ..core.indexer import TwoLevelIndexer
        from ..config import get_settings
        from ..core.lmstudio_client import LMStudioEmbeddingClient
        from pathlib import Path

        settings = get_settings()
        
        # Инициализируем embedding client
        embedding_client = LMStudioEmbeddingClient(
            model_name=settings.lmstudio_model,
            base_url=f"http://{settings.lmstudio_host}:{settings.lmstudio_port}"
        )
        
        # Инициализируем индексатор с графом памяти
        indexer = TwoLevelIndexer(
            chroma_path=settings.chroma_path,
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

    def _search_chromadb_collections(
        self,
        query: str,
        limit: int = 10,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[SearchResultItem]:
        """Поиск в ChromaDB коллекциях, созданных TwoLevelIndexer."""
        if not self.embedding_service:
            return []
        
        try:
            import chromadb
            from ..config import get_settings
            
            settings = get_settings()
            chroma_path = settings.chroma_path
            
            if not os.path.isabs(chroma_path):
                current_dir = Path(__file__).parent
                project_root = current_dir
                while project_root.parent != project_root:
                    if (project_root / "pyproject.toml").exists():
                        break
                    project_root = project_root.parent
                if not (project_root / "pyproject.toml").exists():
                    project_root = Path.cwd()
                chroma_path = str(project_root / chroma_path)
            
            if not os.path.exists(chroma_path):
                return []
            
            chroma_client = chromadb.PersistentClient(path=chroma_path)
            
            query_vector = self.embedding_service.embed(query)
            if not query_vector:
                return []
            
            results: List[SearchResultItem] = []
            
            for collection_name in ["chat_messages", "chat_sessions", "chat_tasks"]:
                try:
                    collection = chroma_client.get_collection(collection_name)
                    
                    where_filter = {}
                    if source:
                        where_filter["chat"] = source
                    
                    search_results = collection.query(
                        query_embeddings=[query_vector],
                        n_results=limit,
                        where=where_filter if where_filter else None,
                    )
                    
                    if not search_results.get("ids") or not search_results["ids"][0]:
                        continue
                    
                    for i, doc_id in enumerate(search_results["ids"][0]):
                        distance = search_results["distances"][0][i] if search_results.get("distances") else 0.0
                        document = search_results["documents"][0][i] if search_results.get("documents") else ""
                        metadata = search_results["metadatas"][0][i] if search_results.get("metadatas") else {}
                        
                        similarity = 1.0 / (1.0 + distance)
                        
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
                        if doc_id in self.graph.graph:
                            node_data = self.graph.graph.nodes[doc_id]
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
                            record_id=doc_id,
                            score=similarity,
                            content=document[:500] if document else "",
                            source=result_source,
                            timestamp=timestamp,
                            author=None,
                            metadata={
                                "collection": collection_name,
                                "chat": chat_name,
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
            logger.debug(f"Ошибка при поиске в ChromaDB коллекциях: {e}", exc_info=True)
            return []

    def _clear_chromadb_chat(self, chat_name: str) -> int:
        """Удаление всех записей конкретного чата из ChromaDB коллекций."""
        total_deleted = 0
        try:
            import chromadb
            from ..config import get_settings

            settings = get_settings()
            chroma_path = settings.chroma_path

            if not os.path.isabs(chroma_path):
                current_dir = Path(__file__).parent
                project_root = current_dir
                while project_root.parent != project_root:
                    if (project_root / "pyproject.toml").exists():
                        break
                    project_root = project_root.parent
                if not (project_root / "pyproject.toml").exists():
                    project_root = Path.cwd()
                chroma_path = str(project_root / chroma_path)

            if not os.path.exists(chroma_path):
                logger.debug(f"ChromaDB путь не существует: {chroma_path}")
                return 0

            chroma_client = chromadb.PersistentClient(path=chroma_path)

            for collection_name in ["chat_sessions", "chat_messages", "chat_tasks"]:
                try:
                    collection = chroma_client.get_collection(collection_name)

                    result = collection.get(where={"chat": chat_name})
                    ids_to_delete = result.get("ids", [])

                    if not ids_to_delete:
                        logger.debug(
                            f"Нет записей для удаления в коллекции {collection_name} (чат: {chat_name})"
                        )
                        continue

                    collection.delete(ids=ids_to_delete)
                    deleted_count = len(ids_to_delete)
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
                f"Всего удалено {total_deleted} записей из ChromaDB (чат: {chat_name})"
            )
            return total_deleted

        except Exception as e:
            logger.warning(
                f"Ошибка при очистке ChromaDB для чата {chat_name}: {e}",
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
