"""Adapters converting MCP requests into memory operations."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable, Optional

from ..indexing import Attachment, MemoryRecord
from ..memory.embeddings import build_embedding_service_from_env
from ..memory.graph_types import NodeType
from ..memory.ingest import MemoryIngestor
from ..memory.trading_memory import TradingMemory
from ..memory.typed_graph import TypedGraphMemory
from ..memory.vector_store import build_vector_store_from_env
from ..utils.datetime_utils import parse_datetime_utc
from .schema import (
    AttachmentPayload,
    FetchRequest,
    FetchResponse,
    GetSignalPerformanceRequest,
    GetSignalPerformanceResponse,
    IngestResponse,
    MemoryRecordPayload,
    ScrapedContentRequest,
    ScrapedContentResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SearchTradingPatternsRequest,
    SearchTradingPatternsResponse,
    SignalPerformance,
    StoreTradingSignalRequest,
    StoreTradingSignalResponse,
    TradingSignalRecord,
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
        if request.record_id not in self.graph.graph:
            return FetchResponse(record=None)
        data = self.graph.graph.nodes[request.record_id]
        payload = _node_to_payload(self.graph, request.record_id, data)
        return FetchResponse(record=payload)

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
