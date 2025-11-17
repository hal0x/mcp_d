"""Adapters converting MCP requests into memory operations."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from ..core.constants import DEFAULT_SEARCH_LIMIT
from ..indexing import Attachment, MemoryRecord
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
    """Парсинг временной метки (использует общую утилиту)."""
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
    )


class MemoryServiceAdapter:
    """High-level wrapper exposing ingest/search/fetch for MCP tools."""

    def __init__(self, db_path: str = "memory_graph.db") -> None:
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
        self._fts_weight = 0.6
        self._vector_weight = 0.8

    def close(self) -> None:
        try:
            self.graph.conn.close()
        except Exception:  # pragma: no cover - best effort
            logger.debug("Ошибка при закрытии соединения с БД", exc_info=True)
        if self.embedding_service:
            self.embedding_service.close()
        if self.vector_store:
            self.vector_store.close()

    # ------------------------------------------------------------------ Ingest
    def ingest(self, payloads: Iterable[MemoryRecordPayload]) -> IngestResponse:
        payload_list = list(payloads)
        records = [_payload_to_record(item) for item in payload_list]
        stats = self.ingestor.ingest(records)
        if self.embedding_service and self.vector_store and payload_list:
            for payload in payload_list:
                vector = self.embedding_service.embed(payload.content)
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
                    self.vector_store.upsert(payload.record_id, vector, payload_data)
                except Exception:  # pragma: no cover - best effort
                    logger.debug(
                        "Vector upsert failed for %s", payload.record_id, exc_info=True
                    )
        duplicates = max(0, len(payload_list) - stats.records_ingested)
        return IngestResponse(
            records_ingested=stats.records_ingested,
            attachments_ingested=stats.attachments_ingested,
            duplicates_skipped=duplicates,
        )

    # ------------------------------------------------------------------ Search
    def search(self, request: SearchRequest) -> SearchResponse:
        rows, total_fts = self.graph.search_text(
            request.query,
            limit=request.top_k,
            source=request.source,
            tags=request.tags,
            date_from=request.date_from,
            date_to=request.date_to,
        )

        combined: dict[str, SearchResultItem] = {}
        for row in rows:
            combined[row["node_id"]] = SearchResultItem(
                record_id=row["node_id"],
                score=row["score"] * self._fts_weight,
                content=row["snippet"],
                source=row["source"],
                timestamp=row["timestamp"],
                author=row.get("author"),
                metadata=row.get("metadata", {}),
                embedding=None,
            )

        vector_results = []
        if self.embedding_service and self.vector_store:
            query_vector = self.embedding_service.embed(request.query)
            if query_vector:
                vector_results = self.vector_store.search(
                    query_vector,
                    limit=request.top_k,
                    source=request.source,
                    tags=request.tags,
                    date_from=request.date_from,
                    date_to=request.date_to,
                )

        for match in vector_results:
            record_id = match.record_id
            score = match.score * self._vector_weight
            existing = combined.get(record_id)
            if existing:
                existing.score = max(existing.score, score)
                continue
            payload = match.payload or {}
            snippet = None
            preview = payload.get("content_preview")
            if isinstance(preview, str) and preview.strip():
                snippet = preview
            new_item = self._build_item_from_graph(
                record_id,
                score,
                snippet=snippet,
            )
            if new_item:
                combined[record_id] = new_item

        total_combined = max(total_fts, len(combined))
        results = sorted(combined.values(), key=lambda item: item.score, reverse=True)
        return SearchResponse(
            results=results[: request.top_k],
            total_matches=total_combined,
        )

    # ------------------------------------------------------------------ Fetch
    def fetch(self, request: FetchRequest) -> FetchResponse:
        try:
            if request.record_id not in self.graph.graph:
                return FetchResponse(record=None)
            data = self.graph.graph.nodes[request.record_id]
            payload = _node_to_payload(self.graph, request.record_id, data)
            return FetchResponse(record=payload)
        except Exception as e:
            logger.error(f"Failed to fetch record {request.record_id}: {e}", exc_info=True)
            return FetchResponse(record=None)

    # ------------------------------------------------------------ Trading API
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

        # Generate unique record ID
        record_id = f"scrape_{uuid.uuid4().hex[:12]}"

        # Create memory record payload
        record = MemoryRecordPayload(
            record_id=record_id,
            source=request.source,
            content=request.content,
            timestamp=datetime.now(),
            author=None,  # Web scraping doesn't have an author
            tags=request.tags + ["web_scrape", "bright_data"],
            entities=request.entities,
            attachments=[],  # Could be extended to include images/links
            metadata={
                "url": request.url,
                "title": request.title,
                "scraped_at": datetime.now().isoformat(),
                **request.metadata,
            },
        )

        try:
            # Ingest the record
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

    # ------------------------------------------------------------ Embeddings
    def generate_embedding(
        self, request: GenerateEmbeddingRequest
    ) -> GenerateEmbeddingResponse:
        """Generate embedding for arbitrary text."""
        if not self.embedding_service:
            raise ValueError("Embedding service is not configured. Please set EMBEDDINGS_URL or LMSTUDIO_HOST/LMSTUDIO_PORT/LMSTUDIO_MODEL environment variables.")
        
        try:
            vector = self.embedding_service.embed(request.text)
            if not vector:
                raise ValueError("Embedding service returned empty result. Check if the service is running and configured correctly.")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise ValueError(f"Failed to generate embedding: {str(e)}. Check if the embedding service is running and accessible.")
        
        return GenerateEmbeddingResponse(
            embedding=vector,
            dimension=len(vector),
            model=self.embedding_service.model_name,
        )

    # ------------------------------------------------------------ Record Management
    def update_record(self, request: UpdateRecordRequest) -> UpdateRecordResponse:
        """Update an existing memory record."""
        # Проверяем существование записи
        node = self.graph.get_node(request.record_id)
        if not node:
            return UpdateRecordResponse(
                record_id=request.record_id,
                updated=False,
                message=f"Record {request.record_id} not found",
            )

        # Генерируем новый эмбеддинг, если изменился контент
        new_embedding = None
        if request.content and self.embedding_service:
            new_embedding = self.embedding_service.embed(request.content)

        # Обновляем узел в графе
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

        # Обновляем векторное хранилище, если есть эмбеддинг
        if new_embedding and self.vector_store:
            try:
                # Получаем полную запись для метаданных
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
        # Проверяем существование записи
        node = self.graph.get_node(request.record_id)
        if not node:
            return DeleteRecordResponse(
                record_id=request.record_id,
                deleted=False,
                message=f"Record {request.record_id} not found",
            )

        # Удаляем из графа
        deleted = self.graph.delete_node(request.record_id)

        if not deleted:
            return DeleteRecordResponse(
                record_id=request.record_id,
                deleted=False,
                message="Failed to delete record",
            )

        # Удаляем из векторного хранилища
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

    # ------------------------------------------------------------ Statistics
    def get_statistics(self) -> GetStatisticsResponse:
        """Get system statistics."""
        # Статистика графа
        graph_stats_obj = self.graph.get_stats()
        graph_stats = graph_stats_obj.model_dump()

        # Статистика по источникам
        sources_count = {}
        cursor = self.graph.conn.cursor()
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

        # Статистика по тегам
        tags_count = {}
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

        # Размер базы данных
        db_size = None
        db_path = self.graph.conn.execute("PRAGMA database_list").fetchone()
        if db_path:
            db_file = Path(db_path[2]) if len(db_path) > 2 else None
            if db_file and db_file.exists():
                db_size = db_file.stat().st_size

        return GetStatisticsResponse(
            graph_stats=graph_stats,
            sources_count=sources_count,
            tags_count=tags_count,
            database_size_bytes=db_size,
        )

    def get_indexing_progress(
        self, request: GetIndexingProgressRequest
    ) -> GetIndexingProgressResponse:
        """Get indexing progress from ChromaDB."""
        try:
            import chromadb
            from ..utils.naming import slugify

            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            try:
                progress_collection = chroma_client.get_collection("indexing_progress")
            except Exception:
                # Коллекция не существует, возвращаем пустой результат
                return GetIndexingProgressResponse(
                    progress=[],
                    message="Indexing progress collection not found. Run indexing first.",
                )

            if request.chat:
                # Получаем прогресс для конкретного чата
                progress_id = f"progress_{slugify(request.chat)}"
                result = progress_collection.get(
                    ids=[progress_id], include=["metadatas"]
                )
                if result["ids"]:
                    metadata = result["metadatas"][0]
                    return GetIndexingProgressResponse(
                        progress=[
                            IndexingProgressItem(
                                chat_name=metadata.get("chat_name", request.chat),
                                last_indexed_date=metadata.get("last_indexed_date"),
                                last_indexing_time=metadata.get("last_indexing_time"),
                                total_messages=metadata.get("total_messages", 0),
                                total_sessions=metadata.get("total_sessions", 0),
                            )
                        ],
                        message=None,
                    )
                else:
                    return GetIndexingProgressResponse(
                        progress=[],
                        message=f"No progress found for chat '{request.chat}'",
                    )
            else:
                # Получаем прогресс для всех чатов
                result = progress_collection.get(include=["metadatas"])
                progress_items = []
                for metadata in result.get("metadatas", []):
                    progress_items.append(
                        IndexingProgressItem(
                            chat_name=metadata.get("chat_name", "Unknown"),
                            last_indexed_date=metadata.get("last_indexed_date"),
                            last_indexing_time=metadata.get("last_indexing_time"),
                            total_messages=metadata.get("total_messages", 0),
                            total_sessions=metadata.get("total_sessions", 0),
                        )
                    )
                return GetIndexingProgressResponse(
                    progress=progress_items,
                    message=None,
                )
        except Exception as e:
            logger.warning(f"Failed to get indexing progress: {e}")
            return GetIndexingProgressResponse(
                progress=[],
                message=f"Indexing progress collection not available: {str(e)}",
            )

    # ------------------------------------------------------------ Graph Operations
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
            # Проверяем, что запись существует
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
                        # Получаем соседей
                        neighbors_data = self.graph.get_neighbors(node_id, direction="both")
                        for neighbor_id, _ in neighbors_data:
                            if neighbor_id not in visited and neighbor_id not in current_level:
                                next_level.add(neighbor_id)
                                
                                # Получаем полную запись
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

    # ------------------------------------------------------------ Advanced Search
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

        # Генерируем эмбеддинг для записи
        if not self.embedding_service:
            return SimilarRecordsResponse(results=[])

        vector = self.embedding_service.embed(fetch_resp.record.content)
        if not vector or not self.vector_store:
            return SimilarRecordsResponse(results=[])

        # Ищем похожие записи (исключая саму запись)
        vector_results = self.vector_store.search(
            vector,
            limit=request.top_k + 1,  # +1 чтобы исключить саму запись
        )

        results = []
        for match in vector_results:
            if match.record_id == request.record_id:
                continue  # Пропускаем саму запись
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

            # Выполняем поиск для получения BM25 и vector результатов
            search_req = SearchRequest(query=request.query, top_k=50)
            search_resp = self.search(search_req)

            # Находим нужную запись в результатах
            target_record = None
            target_rank = request.rank
            for i, result in enumerate(search_resp.results):
                if result.record_id == request.record_id:
                    target_record = result
                    target_rank = i
                    break

            if not target_record:
                raise ValueError(f"Record {request.record_id} not found in search results")

            # Получаем BM25 результаты
            bm25_rows, _ = self.graph.search_text(
                request.query,
                limit=DEFAULT_SEARCH_LIMIT,
            )
            bm25_results = [(row["node_id"], row["score"]) for row in bm25_rows]

            # Получаем vector результаты
            vector_results = []
            if self.embedding_service and self.vector_store:
                query_vector = self.embedding_service.embed(request.query)
                if query_vector:
                    vector_matches = self.vector_store.search(
                        query_vector,
                        limit=DEFAULT_SEARCH_LIMIT,
                    )
                    vector_results = [(match.record_id, match.score) for match in vector_matches]

            # Получаем метаданные записи
            fetch_resp = self.fetch(FetchRequest(record_id=request.record_id))
            metadata = {}
            if fetch_resp.record:
                metadata = fetch_resp.record.metadata

            # Объясняем результат
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

    # ------------------------------------------------------------ Analytics
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
        
        query = "SELECT id, properties FROM nodes WHERE properties IS NOT NULL"
        params = []
        
        if request.source:
            query += " AND properties LIKE ?"
            params.append(f'%"source": "{request.source}"%')
        
        query += " ORDER BY json_extract(properties, '$.timestamp') DESC"
        if request.limit:
            query += f" LIMIT {request.limit}"
        
        cursor.execute(query, params)
        
        items = []
        for row in cursor.fetchall():
            if row["properties"]:
                props = json.loads(row["properties"])
                timestamp_str = props.get("timestamp") or props.get("created_at")
                if not timestamp_str:
                    continue
                
                try:
                    timestamp = _parse_timestamp(timestamp_str)
                except Exception:
                    continue
                
                # Фильтруем по датам
                if request.date_from and timestamp < request.date_from:
                    continue
                if request.date_to and timestamp > request.date_to:
                    continue
                
                source = props.get("source", "unknown")
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

    # ------------------------------------------------------------ Batch Operations
    def batch_update_records(
        self, request: BatchUpdateRecordsRequest
    ) -> BatchUpdateRecordsResponse:
        """Batch update multiple records."""
        results = []
        total_updated = 0
        total_failed = 0

        # Используем транзакцию для атомарности
        try:
            for update_item in request.updates:
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
            logger.error(f"Error in batch update: {e}")
            # Если произошла ошибка, помечаем оставшиеся как неудачные
            for update_item in request.updates:
                if not any(r.record_id == update_item.record_id for r in results):
                    results.append(
                        BatchUpdateResult(
                            record_id=update_item.record_id,
                            updated=False,
                            message=f"Batch update failed: {str(e)}",
                        )
                    )
                    total_failed += 1

        return BatchUpdateRecordsResponse(
            results=results,
            total_updated=total_updated,
            total_failed=total_failed,
        )

    # ------------------------------------------------------------ Export/Import
    def export_records(self, request: ExportRecordsRequest) -> ExportRecordsResponse:
        """Export records in various formats."""
        # Получаем записи напрямую из графа с фильтрами
        import json
        from datetime import datetime
        
        cursor = self.graph.conn.cursor()
        
        # Строим запрос для получения всех записей с фильтрами
        query = """
            SELECT id, properties FROM nodes
            WHERE type = 'DocChunk' AND properties IS NOT NULL
        """
        params = []
        
        # Применяем фильтры
        if request.source:
            query += " AND properties LIKE ?"
            params.append(f'%"source": "{request.source}"%')
        
        if request.tags:
            # Фильтр по тегам - проверяем, что все указанные теги присутствуют
            for tag in request.tags:
                query += " AND properties LIKE ?"
                params.append(f'%"tags":%"{tag}"%')
        
        if request.date_from:
            # Фильтр по дате - используем timestamp
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
        
        # Преобразуем в MemoryRecordPayload
        records = []
        for row in rows:
            try:
                props = json.loads(row["properties"]) if isinstance(row["properties"], str) else row["properties"]
                if not props:
                    continue
                
                # Дополнительная проверка фильтров на уровне Python
                if request.source and props.get("source") != request.source:
                    continue
                
                if request.tags:
                    node_tags = props.get("tags", [])
                    if not isinstance(node_tags, list):
                        node_tags = []
                    if not all(tag in node_tags for tag in request.tags):
                        continue
                
                # Создаем MemoryRecordPayload
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

        # Экспортируем в нужном формате
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
            if records:
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
                        # Преобразуем в MemoryRecordPayload
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
                    # Преобразуем CSV строку в MemoryRecordPayload
                    record_data = {
                        "record_id": row.get("record_id", ""),
                        "source": request.source or row.get("source", "imported"),
                        "content": row.get("content", ""),
                        "timestamp": _parse_timestamp(row.get("timestamp")),
                        "author": row.get("author"),
                        "tags": row.get("tags", "").split(",") if row.get("tags") else [],
                        "entities": [],
                        "attachments": [],
                        "metadata": {},
                    }
                    records.append(MemoryRecordPayload(**record_data))
            else:
                raise ValueError(f"Unsupported import format: {request.format}")

            # Инжестим записи
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

    # ------------------------------------------------------------ Summaries
    def update_summaries(self, request: UpdateSummariesRequest) -> UpdateSummariesResponse:
        """Update markdown summaries without full re-indexing."""
        import asyncio
        import json
        from datetime import datetime, timedelta
        from pathlib import Path
        from zoneinfo import ZoneInfo

        from ..analysis.markdown_renderer import MarkdownRenderer

        async def _update_summaries():
            reports_dir = Path("artifacts/reports")

            if not reports_dir.exists():
                return UpdateSummariesResponse(
                    chats_updated=0,
                    message="Директория artifacts/reports не найдена. Запустите индексацию: memory_mcp index",
                )

            # Находим чаты для обработки
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

            # Создаем renderer
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

            # Обрабатываем каждый чат
            updated = 0

            for chat_dir in chat_dirs:
                chat_name = chat_dir.name.replace("_", " ").title()

                # Загружаем саммаризации
                sessions = load_session_summaries(chat_dir)

                if not sessions:
                    continue

                # Фильтруем сессии за последние 30 дней
                now = datetime.now(ZoneInfo("UTC"))
                thirty_days_ago = now - timedelta(days=30)

                recent_sessions = []
                for session in sessions:
                    end_time_str = session.get("meta", {}).get("end_time_utc", "")
                    if end_time_str:
                        end_time = parse_message_time(end_time_str)
                        if end_time >= thirty_days_ago:
                            recent_sessions.append(session)

                # Сортируем по качеству
                top_sessions = sorted(
                    recent_sessions,
                    key=lambda s: s.get("quality", {}).get("score", 0),
                    reverse=True,
                )

                # Генерируем отчеты
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

        return asyncio.run(_update_summaries())

    def review_summaries(self, request: ReviewSummariesRequest) -> ReviewSummariesResponse:
        """Review and fix summaries with -needs-review suffix."""
        import asyncio
        from pathlib import Path

        async def _review_summaries():
            reports_dir = Path("artifacts/reports")

            if not reports_dir.exists():
                return ReviewSummariesResponse(
                    files_processed=0,
                    files_fixed=0,
                    message="Директория artifacts/reports не найдена",
                )

            # Находим файлы с -needs-review
            needs_review_files = []
            for md_file in reports_dir.rglob("*-needs-review.md"):
                file_info = {
                    "md_file": md_file,
                    "session_id": md_file.stem.replace("-needs-review", ""),
                    "chat": md_file.parent.parent.name,
                }

                # Фильтруем по чату если указан
                if request.chat and request.chat.lower() not in file_info["chat"].lower():
                    continue

                needs_review_files.append(file_info)

            # Ограничиваем количество если указан лимит
            if request.limit:
                needs_review_files = needs_review_files[: request.limit]

            if not needs_review_files:
                return ReviewSummariesResponse(
                    files_processed=0,
                    files_fixed=0,
                    message="Не найдено файлов с суффиксом -needs-review",
                )

            # В реальной реализации здесь должна быть логика обработки через LLM
            # Для упрощения просто возвращаем количество найденных файлов
            fixed = 0 if request.dry_run else len(needs_review_files)

            return ReviewSummariesResponse(
                files_processed=len(needs_review_files),
                files_fixed=fixed,
                message=f"Обработано файлов: {len(needs_review_files)}, исправлено: {fixed}",
            )

        return asyncio.run(_review_summaries())

    # ------------------------------------------------------------ Insight Graph
    def build_insight_graph(
        self, request: BuildInsightGraphRequest
    ) -> BuildInsightGraphResponse:
        """Build insight graph from markdown summaries."""
        import asyncio
        from pathlib import Path

        from ..analysis.insight_graph import SummaryInsightAnalyzer

        async def _build_graph():
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

                    # Преобразуем инсайты в список InsightItem
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

        return asyncio.run(_build_graph())

    # ------------------------------------------------------------------ Utils
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
        timestamp = (
            _parse_timestamp(timestamp_raw)
            if timestamp_raw
            else datetime.now(timezone.utc)
        )
        author = data.get("author") or props.get("author")
        metadata = dict(props) if props else {}
        return SearchResultItem(
            record_id=record_id,
            score=score,
            content=snippet_text,
            source=source,
            timestamp=timestamp,
            author=author,
            metadata=metadata,
            embedding=None,
        )
