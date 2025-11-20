"""MCP server entrypoint exposing the unified memory service."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import asyncio
import uuid

from mcp.server import Server
from mcp.types import Tool, TextContent
from pydantic import BaseModel

from ..quality_analyzer.utils.error_handler import format_error_message
from ..config import get_settings

from .adapters import MemoryServiceAdapter
from ..memory.artifacts_reader import ArtifactsReader
from .schema import (
    StartBackgroundIndexingResponse,
    StopBackgroundIndexingResponse,
    GetBackgroundIndexingStatusResponse,
    AnalyzeEntitiesRequest,
    AnalyzeEntitiesResponse,
    BackgroundIndexingRequest,
    BackgroundIndexingResponse,
    BatchOperationsRequest,
    BatchOperationsResponse,
    BatchUpdateRecordsRequest,
    BatchUpdateRecordsResponse,
    BuildInsightGraphRequest,
    BuildInsightGraphResponse,
    ExportRecordsRequest,
    ExportRecordsResponse,
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
    GraphQueryRequest,
    GraphQueryResponse,
    ImportRecordsRequest,
    ImportRecordsResponse,
    IndexChatRequest,
    IndexChatResponse,
    IngestRequest,
    IngestResponse,
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
    SearchTradingPatternsRequest,
    SearchTradingPatternsResponse,
    SimilarRecordsRequest,
    SimilarRecordsResponse,
    SmartSearchRequest,
    SmartSearchResponse,
    SearchFeedback,
    StoreTradingSignalRequest,
    StoreTradingSignalResponse,
    SummariesRequest,
    SummariesResponse,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
    UnifiedStatisticsResponse,
    UpdateRecordRequest,
    UpdateRecordResponse,
    UpdateSummariesRequest,
    UpdateSummariesResponse,
    SearchEntitiesRequest,
    SearchEntitiesResponse,
    GetEntityProfileRequest,
    GetEntityProfileResponse,
)


def configure_logging() -> None:
    """Configure logging for the MCP server."""
    if not logging.getLogger().handlers:
        settings = get_settings()
        logging.basicConfig(
            level=settings.log_level.upper(),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


# Настраиваем логирование при импорте модуля
configure_logging()

logger = logging.getLogger(__name__)

server = Server("memory-mcp")
logger.info(f"MCP сервер '{server.name}' создан, начинаем регистрацию инструментов...")

ToolResponse = Tuple[List[TextContent], Dict[str, Any]]

# Глобальные сервисы с ленивой инициализацией
_adapter: MemoryServiceAdapter | None = None
_smart_search_engine: Any | None = None
_indexing_tracker: Any | None = None
_background_indexing_service = None


def _get_adapter() -> MemoryServiceAdapter:
    """Инициализирует адаптер памяти при первом обращении.
    
    Резолвит относительный путь к БД относительно корня проекта (pyproject.toml).
    """
    global _adapter
    if _adapter is None:
        settings = get_settings()
        db_path = settings.db_path
        if not os.path.isabs(db_path):
            current_dir = Path(__file__).parent
            project_root = current_dir
            while project_root.parent != project_root:
                if (project_root / "pyproject.toml").exists():
                    break
                project_root = project_root.parent
            if not (project_root / "pyproject.toml").exists():
                project_root = Path.cwd()
            db_path = str(project_root / db_path)
        db_path_obj = Path(db_path)
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Используется путь к БД: {db_path}")
        _adapter = MemoryServiceAdapter(db_path=db_path)
    return _adapter


def _get_indexing_tracker():
    """Инициализирует трекер задач индексации при первом обращении."""
    global _indexing_tracker
    if _indexing_tracker is None:
        from ..core.indexing_tracker import IndexingJobTracker
        from ..config import get_settings
        
        settings = get_settings()
        storage_path = "data/indexing_jobs.json"
        _indexing_tracker = IndexingJobTracker(storage_path=storage_path)
    return _indexing_tracker


def _get_smart_search_engine():
    """Инициализирует интерактивный поисковый движок при первом обращении.
    
    Использует ленивый импорт для избежания циклических зависимостей.
    """
    global _smart_search_engine
    if _smart_search_engine is None:
        from ..search import SmartSearchEngine, SearchSessionStore
        
        settings = get_settings()
        adapter = _get_adapter()
        artifacts_reader = ArtifactsReader(artifacts_dir=settings.artifacts_path)
        
        session_db_path = os.getenv("SMART_SEARCH_SESSION_STORE_PATH", "data/search_sessions.db")
        if not os.path.isabs(session_db_path):
            current_dir = Path(__file__).parent
            project_root = current_dir
            while project_root.parent != project_root:
                if (project_root / "pyproject.toml").exists():
                    break
                project_root = project_root.parent
            if not (project_root / "pyproject.toml").exists():
                project_root = Path.cwd()
            session_db_path = str(project_root / session_db_path)
        
        session_store = SearchSessionStore(db_path=session_db_path)
        min_confidence = float(os.getenv("SMART_SEARCH_MIN_CONFIDENCE", "0.5"))
        
        _smart_search_engine = SmartSearchEngine(
            adapter=adapter,
            artifacts_reader=artifacts_reader,
            session_store=session_store,
            min_confidence=min_confidence,
        )
    return _smart_search_engine


def _to_serializable(value: Any) -> Any:
    """Рекурсивно конвертирует Pydantic модели и datetime в JSON-сериализуемые типы."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, BaseModel):
        try:
            return value.model_dump(mode="json")
        except AttributeError:
            return value.dict()
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    return value


def _format_tool_response(payload: Any, *, root_key: str = "result") -> ToolResponse:
    """Форматирует ответ инструмента в текстовый и структурированный формат."""
    serialized = _to_serializable(payload)
    if isinstance(serialized, dict):
        structured = serialized
        text_payload = serialized
    else:
        structured = {root_key: serialized}
        text_payload = {root_key: serialized}
    text = json.dumps(text_payload, indent=2, ensure_ascii=False)
    return ([TextContent(type="text", text=text)], structured)


def _parse_date_safe(date_str: str | None) -> datetime | None:
    """Парсит ISO 8601 дату с обработкой ошибок.
    
    Поддерживает формат с 'Z' (UTC), заменяя его на '+00:00' для fromisoformat.
    """
    if not date_str:
        return None
    try:
        normalized = date_str.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except (ValueError, AttributeError) as e:
        logger.warning(f"Не удалось распарсить дату '{date_str}': {e}")
        return None


@server.list_tools()  # type: ignore[misc]
async def list_tools() -> List[Tool]:
    """Возвращает список доступных инструментов MCP."""
    logger.info("list_tools() вызвана, регистрируем инструменты...")
    tools = [
        Tool(
            name="health",
            description="Check MCP server health and configuration status.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="version",
            description="Return server version and enabled capabilities.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        # Новые универсальные инструменты
        Tool(
            name="search",
            description="Universal search tool supporting hybrid, smart, embedding, similar, and trading search types. Use search_type parameter to specify the search mode.",
            inputSchema={
                "type": "object",
                "properties": {
                    "search_type": {
                        "type": "string",
                        "enum": ["hybrid", "smart", "embedding", "similar", "trading"],
                        "default": "hybrid",
                        "description": "Тип поиска: hybrid (FTS+вектор), smart (LLM-assisted), embedding (по вектору), similar (похожие записи), trading (торговые паттерны)",
                    },
                    "query": {"type": "string", "description": "Поисковый запрос (требуется для hybrid, smart, trading)"},
                    "embedding": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Вектор эмбеддинга (требуется для embedding)",
                    },
                    "record_id": {"type": "string", "description": "ID записи (требуется для similar)"},
                    "top_k": {"type": "integer", "description": "Максимальное количество результатов", "default": 5},
                    "source": {"type": "string", "description": "Фильтр по источнику"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Фильтр по тегам"},
                    "date_from": {"type": "string", "description": "Фильтр: дата не раньше (ISO format)"},
                    "date_to": {"type": "string", "description": "Фильтр: дата не позже (ISO format)"},
                    "include_embeddings": {"type": "boolean", "description": "Возвращать ли эмбеддинги", "default": False},
                    "session_id": {"type": "string", "description": "ID сессии для smart search"},
                    "feedback": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Обратная связь для smart search",
                    },
                    "clarify": {"type": "boolean", "description": "Запросить уточняющие вопросы для smart search", "default": False},
                    "artifact_types": {"type": "array", "items": {"type": "string"}, "description": "Фильтр по типам артифактов для smart search"},
                    "symbol": {"type": "string", "description": "Торговая пара для trading search"},
                    "limit": {"type": "integer", "description": "Лимит результатов для trading search"},
                },
                "required": [],
            },
        ),
        Tool(
            name="batch_operations",
            description="Universal batch operations tool supporting update, delete, and fetch operations. Use operation parameter to specify the operation type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["update", "delete", "fetch"],
                        "description": "Тип операции: update, delete, fetch",
                    },
                    "updates": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Список обновлений (требуется для update)",
                    },
                    "record_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список ID записей (требуется для delete и fetch)",
                    },
                },
                "required": ["operation"],
            },
        ),
        Tool(
            name="graph_query",
            description="Universal graph query tool supporting neighbors, path, and related queries. Use query_type parameter to specify the query type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["neighbors", "path", "related"],
                        "description": "Тип запроса: neighbors (соседи узла), path (путь между узлами), related (связанные записи)",
                    },
                    "node_id": {"type": "string", "description": "ID узла (требуется для neighbors и related)"},
                    "edge_type": {"type": "string", "description": "Фильтр по типу рёбер (для neighbors)"},
                    "direction": {"type": "string", "enum": ["out", "in", "both"], "default": "both", "description": "Направление (для neighbors)"},
                    "source_id": {"type": "string", "description": "ID исходного узла (требуется для path)"},
                    "target_id": {"type": "string", "description": "ID целевого узла (требуется для path)"},
                    "max_length": {"type": "integer", "description": "Максимальная длина пути (для path)", "default": 5},
                    "record_id": {"type": "string", "description": "ID записи (альтернатива node_id для related)"},
                    "max_depth": {"type": "integer", "description": "Максимальная глубина поиска (для related)", "default": 1},
                    "limit": {"type": "integer", "description": "Максимальное количество результатов (для related)", "default": 10},
                },
                "required": ["query_type"],
            },
        ),
        Tool(
            name="background_indexing",
            description="Universal background indexing management tool supporting start, stop, and status actions. Use action parameter to specify the action.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop", "status"],
                        "description": "Действие: start (запустить), stop (остановить), status (получить статус)",
                    },
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="summaries",
            description="Universal summaries management tool supporting update and review actions. Use action parameter to specify the action.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["update", "review"],
                        "description": "Действие: update (обновить), review (проверить)",
                    },
                    "chat": {"type": "string", "description": "Обработать только конкретный чат"},
                    "force": {"type": "boolean", "description": "Принудительно пересоздать существующие артефакты (для update)", "default": False},
                    "dry_run": {"type": "boolean", "description": "Только анализ, без изменения файлов (для review)", "default": False},
                    "limit": {"type": "integer", "description": "Максимальное количество файлов для обработки (для review)"},
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="ingest",
            description="Universal ingest tool supporting records and scraped content. Use source_type parameter to specify the source type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "records": {
                        "type": "array",
                        "description": "Список записей для загрузки (требуется для source_type=records)",
                        "items": {"type": "object"},
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["records", "scraped"],
                        "default": "records",
                        "description": "Тип источника: records (обычные записи), scraped (скрапленный контент)",
                    },
                    "url": {"type": "string", "description": "URL скрапленной страницы (для scraped)"},
                    "title": {"type": "string", "description": "Заголовок страницы (для scraped)"},
                    "content": {"type": "string", "description": "Основной текстовый контент (для scraped)"},
                    "metadata": {"type": "object", "description": "Дополнительные метаданные (для scraped)"},
                    "source": {"type": "string", "description": "Источник скраппинга (для scraped)", "default": "bright_data"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Теги для классификации (для scraped)"},
                    "entities": {"type": "array", "items": {"type": "string"}, "description": "Извлеченные сущности (для scraped)"},
                },
                "required": [],
            },
        ),
        Tool(
            name="store_trading_signal",
            description="Persist a trading signal in the memory store.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Торговый инструмент (например, BTCUSDT, ETHUSDT)",
                    },
                    "signal_type": {
                        "type": "string",
                        "description": "Тип сигнала (momentum, breakout, mean_revert, volume_profile и т.д.)",
                    },
                    "direction": {
                        "type": "string",
                        "description": "Направление торговли: 'long' или 'short' (опционально)",
                        "enum": ["long", "short"],
                    },
                    "entry": {
                        "type": "number",
                        "description": "Цена входа (опционально)",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Уверенность в сигнале от 0 до 100 (опционально)",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "context": {
                        "type": "object",
                        "description": "Дополнительный контекст сигнала (стратегия, таймфрейм и т.д.)",
                        "additionalProperties": True,
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Временная метка сигнала в ISO формате (опционально, по умолчанию текущее время)",
                    },
                },
                "required": ["symbol", "signal_type"],
            },
        ),
        Tool(
            name="search_trading_patterns",
            description="Search for stored trading patterns and signals.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"},
                    "symbol": {
                        "type": "string",
                        "description": "Торговая пара (опционально)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Максимальное количество результатов",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_signal_performance",
            description="Return performance metrics for a trading signal.",
            inputSchema={
                "type": "object",
                "properties": {
                    "signal_id": {
                        "type": "string",
                        "description": "Идентификатор сигнала",
                    }
                },
                "required": ["signal_id"],
            },
        ),
        Tool(
            name="search_entities",
            description="Search entities by semantic similarity using vector search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Текстовый запрос для поиска сущностей",
                    },
                    "query_vector": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Вектор запроса (если query не указан)",
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "Фильтр по типу сущности (опционально)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Максимальное количество результатов",
                        "default": 10,
                    },
                },
            },
        ),
        Tool(
            name="get_entity_profile",
            description="Get full profile of an entity including description, statistics, and related entities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "description": "Тип сущности",
                    },
                    "value": {
                        "type": "string",
                        "description": "Значение сущности",
                    },
                },
                "required": ["entity_type", "value"],
            },
        ),
        Tool(
            name="generate_embedding",
            description="Generate embedding vector for arbitrary text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Текст для генерации эмбеддинга",
                    },
                    "model": {
                        "type": "string",
                        "description": "Модель для эмбеддингов (опционально, используется из конфигурации)",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="update_record",
            description="Update an existing memory record (content, metadata, tags, entities).",
            inputSchema={
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "Идентификатор записи для обновления",
                    },
                    "content": {
                        "type": "string",
                        "description": "Новый контент записи (опционально)",
                    },
                    "source": {
                        "type": "string",
                        "description": "Новый источник (опционально)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Новые теги (опционально)",
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Новые сущности (опционально)",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Дополнительные метаданные (объединяются с существующими)",
                        "additionalProperties": True,
                    },
                },
                "required": ["record_id"],
            },
        ),
        Tool(
            name="get_statistics",
            description="Get system statistics (graph, sources, tags, database size). Supports filtering by type: general, tags, or indexing. If type is not specified, returns all statistics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["general", "tags", "indexing"],
                        "description": "Тип статистики: general (общая), tags (по тегам), indexing (прогресс индексации). Если не указан, возвращается вся статистика.",
                    },
                    "chat": {
                        "type": "string",
                        "description": "Фильтр по конкретному чату (для type='indexing')",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="search_explain",
            description="Explain search result relevance with score breakdown and graph connections.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос",
                    },
                    "record_id": {
                        "type": "string",
                        "description": "ID записи для объяснения",
                    },
                    "rank": {
                        "type": "integer",
                        "description": "Позиция в результатах (0-based)",
                        "default": 0,
                    },
                },
                "required": ["query", "record_id"],
            },
        ),
        Tool(
            name="get_timeline",
            description="Get timeline of records sorted by timestamp.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Фильтр по источнику (опционально)",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Начало периода (ISO format)",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Конец периода (ISO format)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Максимальное количество записей",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 500,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="analyze_entities",
            description="Extract and analyze entities from text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Текст для анализа сущностей",
                    },
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Типы сущностей для извлечения (опционально)",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="export_records",
            description="Export records in JSON, CSV, or Markdown format.",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "Формат экспорта: json, csv, markdown",
                        "enum": ["json", "csv", "markdown"],
                        "default": "json",
                    },
                    "source": {
                        "type": "string",
                        "description": "Фильтр по источнику (опционально)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Фильтр по тегам",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Начало периода (ISO format)",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Конец периода (ISO format)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Максимальное количество записей",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 10000,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="import_records",
            description="Import records from JSON or CSV format.",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "Формат импорта: json, csv",
                        "enum": ["json", "csv"],
                    },
                    "content": {
                        "type": "string",
                        "description": "Содержимое для импорта",
                    },
                    "source": {
                        "type": "string",
                        "description": "Источник для импортируемых записей (опционально)",
                    },
                },
                "required": ["format", "content"],
            },
        ),
        Tool(
            name="build_insight_graph",
            description="Build insight graph from markdown summaries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "summaries_dir": {
                        "type": "string",
                        "description": "Директория с саммаризациями (опционально, по умолчанию artifacts/reports)",
                    },
                    "chroma_path": {
                        "type": "string",
                        "description": "Deprecated: не используется, заменено на Qdrant",
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Порог схожести",
                        "default": 0.76,
                    },
                    "max_similar_results": {
                        "type": "integer",
                        "description": "Максимальное количество похожих результатов",
                        "default": 8,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="index_chat",
            description="Index a specific Telegram chat with two-level indexing (L1: sessions, L2: messages, L3: tasks). Индексация выполняется в фоновом режиме и не блокирует вызов. Используйте get_indexing_progress для отслеживания прогресса. Поддерживает все параметры индексации: качество, кластеризация, группировка, умная агрегация и др.",
            inputSchema={
                "type": "object",
                "properties": {
                    "chat": {
                        "type": "string",
                        "description": "Название чата для индексации (как папка в /app/chats/)",
                    },
                    "force_full": {
                        "type": "boolean",
                        "description": "Полная пересборка индекса",
                        "default": False,
                    },
                    "recent_days": {
                        "type": "integer",
                        "description": "Пересаммаризировать последние N дней",
                        "default": 7,
                    },
                    "enable_quality_check": {
                        "type": "boolean",
                        "description": "Включить проверку качества саммаризации",
                        "default": True,
                    },
                    "enable_iterative_refinement": {
                        "type": "boolean",
                        "description": "Включить автоматическое улучшение саммаризаций",
                        "default": True,
                    },
                    "min_quality_score": {
                        "type": "number",
                        "description": "Минимальный приемлемый балл качества",
                        "default": 80.0,
                    },
                    "enable_clustering": {
                        "type": "boolean",
                        "description": "Включить автоматическую кластеризацию сессий",
                        "default": True,
                    },
                    "clustering_threshold": {
                        "type": "number",
                        "description": "Порог сходства для кластеризации",
                        "default": 0.8,
                    },
                    "min_cluster_size": {
                        "type": "integer",
                        "description": "Минимальный размер кластера",
                        "default": 2,
                    },
                    "max_messages_per_group": {
                        "type": "integer",
                        "description": "Максимальное количество сообщений в группе",
                        "default": 100,
                    },
                    "max_session_hours": {
                        "type": "integer",
                        "description": "Максимальная длительность сессии в часах",
                        "default": 6,
                    },
                    "gap_minutes": {
                        "type": "integer",
                        "description": "Максимальный разрыв между сообщениями в минутах",
                        "default": 60,
                    },
                    "enable_smart_aggregation": {
                        "type": "boolean",
                        "description": "Включить умную группировку с скользящими окнами",
                        "default": True,
                    },
                    "aggregation_strategy": {
                        "type": "string",
                        "description": "Стратегия группировки: 'smart', 'channel', 'legacy'",
                        "default": "smart",
                        "enum": ["smart", "channel", "legacy"],
                    },
                    "now_window_hours": {
                        "type": "integer",
                        "description": "Размер NOW окна в часах",
                        "default": 24,
                    },
                    "fresh_window_days": {
                        "type": "integer",
                        "description": "Размер FRESH окна в днях",
                        "default": 14,
                    },
                    "recent_window_days": {
                        "type": "integer",
                        "description": "Размер RECENT окна в днях",
                        "default": 30,
                    },
                    "strategy_threshold": {
                        "type": "integer",
                        "description": "Порог количества сообщений для перехода между стратегиями",
                        "default": 1000,
                    },
                    "enable_entity_learning": {
                        "type": "boolean",
                        "description": "Включить автоматическое обучение словарей сущностей",
                        "default": True,
                    },
                    "enable_time_analysis": {
                        "type": "boolean",
                        "description": "Включить анализ временных паттернов",
                        "default": True,
                    },
                    "run_optimizations": {
                        "type": "boolean",
                        "description": "Выполнить оптимизации после индексации (VACUUM, ANALYZE, пересчёт важности, проверка целостности)",
                        "default": True,
                    },
                },
                "required": ["chat"],
            },
        ),
        Tool(
            name="get_available_chats",
            description="Get list of all available Telegram chats for indexing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_stats": {
                        "type": "boolean",
                        "description": "Включить статистику (количество сообщений, дата изменения)",
                        "default": False,
                    },
                },
                "required": [],
            },
        ),
    ]
    logger.info(f"Зарегистрировано {len(tools)} инструментов: {[t.name for t in tools]}")
    return tools


@server.call_tool()  # type: ignore[misc]
async def call_tool(name: str, arguments: Dict[str, Any]) -> ToolResponse:
    """Обрабатывает вызов MCP инструмента и возвращает форматированный ответ."""
    try:
        logger.info(f"Вызов инструмента: {name} с аргументами: {arguments}")

        if name == "health":
            result = get_health_payload()
            return _format_tool_response(result)

        if name == "version":
            result = get_version_payload()
            return _format_tool_response(result)

        adapter = _get_adapter()

        # Новые универсальные инструменты
        if name == "search":
            try:
                # Парсим feedback для smart search
                feedback_data = arguments.get("feedback")
                feedback_list = None
                if feedback_data:
                    feedback_list = [SearchFeedback(**item) for item in feedback_data]

                # Парсим даты
                date_from = _parse_date_safe(arguments.get("date_from"))
                date_to = _parse_date_safe(arguments.get("date_to"))

                search_type = arguments.get("search_type", "hybrid")
                
                if search_type == "smart":
                    # Smart search требует SmartSearchEngine
                    request = SmartSearchRequest(
                        query=arguments.get("query", ""),
                        session_id=arguments.get("session_id"),
                        feedback=feedback_list,
                        clarify=arguments.get("clarify", False),
                        top_k=arguments.get("top_k", 10),
                        source=arguments.get("source"),
                        tags=arguments.get("tags", []),
                        date_from=date_from,
                        date_to=date_to,
                        artifact_types=arguments.get("artifact_types"),
                    )
                    engine = _get_smart_search_engine()
                    result = await engine.search(request)
                    # Преобразуем SmartSearchResponse в UnifiedSearchResponse
                    unified_result = UnifiedSearchResponse(
                        search_type="smart",
                        results=result.results,
                        total_matches=result.total_matches,
                        clarifying_questions=result.clarifying_questions,
                        suggested_refinements=result.suggested_refinements,
                        session_id=result.session_id,
                        confidence_score=result.confidence_score,
                        artifacts_found=result.artifacts_found,
                        db_records_found=result.db_records_found,
                    )
                    return _format_tool_response(unified_result.model_dump())
                else:
                    # Остальные типы поиска через адаптер
                    unified_request = UnifiedSearchRequest(
                        search_type=search_type,
                        query=arguments.get("query"),
                        embedding=arguments.get("embedding"),
                        record_id=arguments.get("record_id"),
                        top_k=arguments.get("top_k", 5),
                        source=arguments.get("source"),
                        tags=arguments.get("tags", []),
                        date_from=date_from,
                        date_to=date_to,
                        include_embeddings=arguments.get("include_embeddings", False),
                        session_id=arguments.get("session_id"),
                        feedback=feedback_list,
                        clarify=arguments.get("clarify", False),
                        artifact_types=arguments.get("artifact_types"),
                        symbol=arguments.get("symbol"),
                        limit=arguments.get("limit"),
                    )
                    result = await adapter.unified_search(unified_request)
                    return _format_tool_response(result.model_dump())
            except Exception as e:
                logger.exception(f"search failed: {e}")
                raise RuntimeError(format_error_message(e)) from e

        elif name == "batch_operations":
            try:
                from .schema import BatchUpdateRecordItem
                
                operation = arguments.get("operation")
                if operation == "update":
                    updates_data = arguments.get("updates", [])
                    updates = [BatchUpdateRecordItem(**item) for item in updates_data]
                    request = BatchOperationsRequest(operation="update", updates=updates)
                elif operation in ("delete", "fetch"):
                    record_ids = arguments.get("record_ids", [])
                    request = BatchOperationsRequest(operation=operation, record_ids=record_ids)
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                
                result = adapter.batch_operations(request)
                return _format_tool_response(result.model_dump())
            except Exception as e:
                logger.exception(f"batch_operations failed: {e}")
                raise RuntimeError(format_error_message(e)) from e

        elif name == "graph_query":
            try:
                request = GraphQueryRequest(**arguments)
                result = adapter.graph_query(request)
                return _format_tool_response(result.model_dump())
            except Exception as e:
                logger.exception(f"graph_query failed: {e}")
                raise RuntimeError(format_error_message(e)) from e

        elif name == "background_indexing":
            try:
                request = BackgroundIndexingRequest(**arguments)
                action = request.action
                
                if action == "start":
                    result_obj = await _start_background_indexing(adapter)
                    return _format_tool_response(BackgroundIndexingResponse(
                        action="start",
                        success=result_obj.success,
                        message=result_obj.message,
                    ).model_dump())
                elif action == "stop":
                    result_obj = await _stop_background_indexing()
                    return _format_tool_response(BackgroundIndexingResponse(
                        action="stop",
                        success=result_obj.success,
                        message=result_obj.message,
                    ).model_dump())
                elif action == "status":
                    result_obj = _get_background_indexing_status()
                    return _format_tool_response(BackgroundIndexingResponse(
                        action="status",
                        message=result_obj.message,
                        running=result_obj.running,
                        check_interval=result_obj.check_interval,
                        last_check_time=result_obj.last_check_time,
                        input_path=result_obj.input_path,
                        chats_path=result_obj.chats_path,
                    ).model_dump())
                else:
                    raise ValueError(f"Unknown action: {action}")
            except Exception as e:
                logger.exception(f"background_indexing failed: {e}")
                raise RuntimeError(format_error_message(e)) from e

        elif name == "summaries":
            try:
                request = SummariesRequest(**arguments)
                result = await adapter.summaries(request)
                return _format_tool_response(result.model_dump())
            except Exception as e:
                logger.exception(f"summaries failed: {e}")
                raise RuntimeError(format_error_message(e)) from e

        elif name == "ingest":
            try:
                from .schema import MemoryRecordPayload
                
                source_type = arguments.get("source_type", "records")
                if source_type == "scraped":
                    # Преобразуем аргументы в IngestRequest для scraped
                    request = IngestRequest(
                        records=[],
                        source_type="scraped",
                        url=arguments.get("url"),
                        title=arguments.get("title"),
                        content=arguments.get("content"),
                        metadata=arguments.get("metadata"),
                        source=arguments.get("source", "bright_data"),
                        tags=arguments.get("tags", []),
                        entities=arguments.get("entities", []),
                    )
                else:
                    # Обычная индексация записей
                    records_data = arguments.get("records", [])
                    records = [MemoryRecordPayload(**item) for item in records_data]
                    request = IngestRequest(records=records, source_type="records")
                
                result = adapter.ingest_unified(request)
                return _format_tool_response(result.model_dump())
            except Exception as e:
                logger.exception(f"ingest failed: {e}")
                raise RuntimeError(format_error_message(e)) from e

        elif name == "store_trading_signal":
            request = StoreTradingSignalRequest(**arguments)
            result = adapter.store_trading_signal(request)
            return _format_tool_response(result.model_dump())

        elif name == "search_trading_patterns":
            try:
                request = SearchTradingPatternsRequest(**arguments)
                result = adapter.search_trading_patterns(request)
                return _format_tool_response(result.model_dump())
            except Exception as e:
                logger.exception(f"search_trading_patterns failed: {e}")
                raise RuntimeError(format_error_message(e)) from e

        elif name == "get_signal_performance":
            request = GetSignalPerformanceRequest(**arguments)
            result = adapter.get_signal_performance(request)
            return _format_tool_response(result.model_dump())

        elif name == "search_entities":
            try:
                request = SearchEntitiesRequest(**arguments)
                result = adapter.search_entities(request)
                return _format_tool_response(result.model_dump())
            except Exception as e:
                logger.exception(f"search_entities failed: {e}")
                raise RuntimeError(format_error_message(e)) from e

        elif name == "get_entity_profile":
            try:
                request = GetEntityProfileRequest(**arguments)
                result = adapter.get_entity_profile(request)
                return _format_tool_response(result.model_dump())
            except Exception as e:
                logger.exception(f"get_entity_profile failed: {e}")
                raise RuntimeError(format_error_message(e)) from e

        elif name == "generate_embedding":
            try:
                request = GenerateEmbeddingRequest(**arguments)
                result = adapter.generate_embedding(request)
                return _format_tool_response(result.model_dump())
            except ValueError as e:
                original_msg = str(e)
                error_msg = format_error_message(e)
                logger.warning(
                    f"generate_embedding failed: {error_msg} (original: {original_msg})",
                    exc_info=True,
                )
                raise RuntimeError(f"{error_msg} (original: {original_msg})") from e

        elif name == "update_record":
            request = UpdateRecordRequest(**arguments)
            result = adapter.update_record(request)
            return _format_tool_response(result.model_dump())

        elif name == "get_statistics":
            # Поддержка нового универсального формата
            request = GetStatisticsRequest(**arguments) if arguments else GetStatisticsRequest()
            result = adapter.get_statistics_unified(request)
            return _format_tool_response(result.model_dump())

        elif name == "search_explain":
            request = SearchExplainRequest(**arguments)
            result = adapter.search_explain(request)
            return _format_tool_response(result.model_dump())

        elif name == "get_timeline":
            if "date_from" in arguments and arguments["date_from"]:
                parsed_date = _parse_date_safe(arguments["date_from"])
                if parsed_date is not None:
                    arguments["date_from"] = parsed_date
                else:
                    del arguments["date_from"]
            if "date_to" in arguments and arguments["date_to"]:
                parsed_date = _parse_date_safe(arguments["date_to"])
                if parsed_date is not None:
                    arguments["date_to"] = parsed_date
                else:
                    del arguments["date_to"]
            request = GetTimelineRequest(**arguments)
            result = adapter.get_timeline(request)
            return _format_tool_response(result.model_dump())

        elif name == "analyze_entities":
            request = AnalyzeEntitiesRequest(**arguments)
            result = adapter.analyze_entities(request)
            return _format_tool_response(result.model_dump())

        elif name == "export_records":
            if "date_from" in arguments and arguments["date_from"]:
                parsed_date = _parse_date_safe(arguments["date_from"])
                if parsed_date is not None:
                    arguments["date_from"] = parsed_date
                else:
                    del arguments["date_from"]
            if "date_to" in arguments and arguments["date_to"]:
                parsed_date = _parse_date_safe(arguments["date_to"])
                if parsed_date is not None:
                    arguments["date_to"] = parsed_date
                else:
                    del arguments["date_to"]
            request = ExportRecordsRequest(**arguments)
            result = adapter.export_records(request)
            return _format_tool_response(result.model_dump())

        elif name == "import_records":
            request = ImportRecordsRequest(**arguments)
            result = adapter.import_records(request)
            return _format_tool_response(result.model_dump())

        elif name == "build_insight_graph":
            request = BuildInsightGraphRequest(**arguments)
            result = await adapter.build_insight_graph(request)
            return _format_tool_response(result.model_dump())

        elif name == "index_chat":
            request = IndexChatRequest(**arguments)
            result = await _start_indexing_job(request, adapter)
            return _format_tool_response(result.model_dump())

        elif name == "get_available_chats":
            request = GetAvailableChatsRequest(**arguments)
            result = adapter.get_available_chats(request)
            return _format_tool_response(result.model_dump())

        else:
            raise ValueError(f"Неизвестный инструмент: {name}")

    except ValueError as exc:
        logger.warning(f"Ошибка при выполнении инструмента {name}: {exc}")
        raise RuntimeError(format_error_message(exc)) from exc
    except Exception as exc:
        logger.exception(f"Ошибка при выполнении инструмента {name}: {exc}")
        raise RuntimeError(format_error_message(exc)) from exc


async def _run_indexing_job(
    job_id: str,
    request: "IndexChatRequest",
    adapter: MemoryServiceAdapter,
) -> None:
    """Выполняет индексацию чата в фоновом режиме с отслеживанием прогресса."""
    tracker = _get_indexing_tracker()
    
    from ..core.indexer import TwoLevelIndexer
    from ..config import get_settings
    from ..core.lmstudio_client import LMStudioEmbeddingClient
    from datetime import timezone
    
    try:
        tracker.update_job(
            job_id=job_id,
            status="running",
            current_stage="Инициализация",
        )
        
        settings = get_settings()
        
        embedding_client = LMStudioEmbeddingClient(
            model_name=settings.lmstudio_model,
            base_url=f"http://{settings.lmstudio_host}:{settings.lmstudio_port}"
        )
        
        indexer = TwoLevelIndexer(
            chroma_path=settings.chroma_path,
            artifacts_path=settings.artifacts_path,
            embedding_client=embedding_client,
            enable_quality_check=request.enable_quality_check if request.enable_quality_check is not None else True,
            enable_iterative_refinement=request.enable_iterative_refinement if request.enable_iterative_refinement is not None else True,
            min_quality_score=request.min_quality_score if request.min_quality_score is not None else 80.0,
            enable_clustering=request.enable_clustering if request.enable_clustering is not None else True,
            clustering_threshold=request.clustering_threshold if request.clustering_threshold is not None else 0.8,
            min_cluster_size=request.min_cluster_size if request.min_cluster_size is not None else 2,
            max_messages_per_group=request.max_messages_per_group if request.max_messages_per_group is not None else 100,
            max_session_hours=request.max_session_hours if request.max_session_hours is not None else 6,
            gap_minutes=request.gap_minutes if request.gap_minutes is not None else 60,
            enable_smart_aggregation=request.enable_smart_aggregation if request.enable_smart_aggregation is not None else True,
            aggregation_strategy=request.aggregation_strategy if request.aggregation_strategy is not None else "smart",
            now_window_hours=request.now_window_hours if request.now_window_hours is not None else 24,
            fresh_window_days=request.fresh_window_days if request.fresh_window_days is not None else 14,
            recent_window_days=request.recent_window_days if request.recent_window_days is not None else 30,
            strategy_threshold=request.strategy_threshold if request.strategy_threshold is not None else 1000,
            force=request.force_full,
            enable_entity_learning=request.enable_entity_learning if request.enable_entity_learning is not None else True,
            enable_time_analysis=request.enable_time_analysis if request.enable_time_analysis is not None else True,
        )
        
        if request.force_full:
            tracker.update_job(job_id=job_id, current_stage="Очистка старых данных")
            logger.info(f"🧹 Очистка старых данных чата '{request.chat}' перед переиндексацией...")
            try:
                cleanup_stats = adapter.clear_chat_data(request.chat)
                logger.info(
                    f"✅ Очистка завершена для чата '{request.chat}': "
                    f"узлов={cleanup_stats.get('nodes_deleted', 0)}, "
                    f"векторов={cleanup_stats.get('vectors_deleted', 0)}, "
                    f"ChromaDB={cleanup_stats.get('chromadb_deleted', 0)}"
                )
                tracker.update_job(job_id=job_id, cleanup_stats=cleanup_stats)
            except Exception as e:
                logger.warning(
                    f"⚠️ Ошибка при очистке данных чата '{request.chat}': {e}. "
                    f"Продолжаем индексацию...",
                    exc_info=True,
                )
        
        tracker.update_job(job_id=job_id, current_stage="Загрузка сообщений")
        
        def progress_callback(job_id: str, event: str, data: Dict[str, Any]) -> None:
            """Обновляет статус задачи индексации на основе событий от индексатора."""
            try:
                if event == "chat_started":
                    tracker.update_job(
                        job_id=job_id,
                        status="running",
                        current_stage=f"Обработка чата '{data.get('chat')}'",
                        current_chat=data.get("chat"),
                    )
                elif event == "sessions_processing":
                    tracker.update_job(
                        job_id=job_id,
                        current_stage=f"Обработка сессий чата '{data.get('chat')}' ({data.get('session_index')}/{data.get('total_sessions')})",
                        current_chat=data.get("chat"),
                        progress={
                            "current_chat_sessions": data.get("sessions_count", 0),
                            "current_chat_messages": data.get("messages_count", 0),
                        },
                    )
                elif event == "chat_completed":
                    chat_stats = data.get("stats", {})
                    tracker.update_job(
                        job_id=job_id,
                        current_stage=f"Завершена обработка чата '{data.get('chat')}'",
                        stats={
                            "sessions_indexed": chat_stats.get("sessions_indexed", 0),
                            "messages_indexed": chat_stats.get("messages_indexed", 0),
                            "tasks_indexed": chat_stats.get("tasks_indexed", 0),
                        },
                    )
                elif event == "error":
                    tracker.update_job(
                        job_id=job_id,
                        status="failed",
                        error=f"Ошибка в чате '{data.get('chat')}': {data.get('error')}",
                    )
                elif event == "completed":
                    final_stats = data.get("stats", {})
                    tracker.update_job(
                        job_id=job_id,
                        status="completed",
                        current_stage="Индексация завершена",
                        stats={
                            "sessions_indexed": final_stats.get("sessions_indexed", 0),
                            "messages_indexed": final_stats.get("messages_indexed", 0),
                            "tasks_indexed": final_stats.get("tasks_indexed", 0),
                        },
                    )
            except Exception as e:
                logger.warning(f"Ошибка при обновлении прогресса: {e}")
        
        indexer.progress_callback = progress_callback
        
        stats = await indexer.build_index(
            scope="chat",
            chat=request.chat,
            force_full=request.force_full,
            recent_days=request.recent_days,
            adapter=adapter,
            job_id=job_id,
        )
        
        logger.info(f"Индексация чата '{request.chat}' завершена успешно (job_id: {job_id})")
        
        run_optimizations = request.run_optimizations if request.run_optimizations is not None else True
        if run_optimizations:
            try:
                tracker.update_job(job_id=job_id, current_stage="Оптимизация базы данных")
                logger.info(f"⚡ Запуск оптимизаций после индексации чата '{request.chat}'...")
                
                optimization_results = {}
                
                db_path = str(adapter.graph.db_path) if hasattr(adapter.graph, 'db_path') else str(adapter.graph.conn.execute("PRAGMA database_list").fetchone()[2])
                
                # Оптимизация базы данных
                try:
                    tracker.update_job(job_id=job_id, current_stage="Оптимизация: VACUUM/ANALYZE")
                    opt_result = _optimize_database(db_path)
                    optimization_results["optimize_database"] = opt_result
                    logger.info(f"✅ Оптимизация БД завершена: {opt_result}")
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка при оптимизации БД: {e}", exc_info=True)
                    optimization_results["optimize_database"] = {"error": str(e)}
                
                # Пересчёт важности записей
                try:
                    tracker.update_job(job_id=job_id, current_stage="Обновление важности записей")
                    importance_result = _update_importance_scores(adapter.graph)
                    optimization_results["update_importance"] = importance_result
                    logger.info(f"✅ Пересчёт важности завершён: {importance_result}")
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка при пересчёте важности: {e}", exc_info=True)
                    optimization_results["update_importance"] = {"error": str(e)}
                
                # Проверка целостности
                try:
                    tracker.update_job(job_id=job_id, current_stage="Проверка целостности")
                    validation_result = _validate_database(db_path)
                    optimization_results["validate_database"] = validation_result
                    if validation_result.get("valid"):
                        logger.info(f"✅ Проверка целостности пройдена")
                    else:
                        logger.warning(f"⚠️ Найдены проблемы при проверке целостности: {validation_result.get('total_issues', 0)}")
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка при проверке целостности: {e}", exc_info=True)
                    optimization_results["validate_database"] = {"error": str(e)}
                
                # Проверка необходимости очистки памяти
                try:
                    tracker.update_job(job_id=job_id, current_stage="Проверка очистки памяти")
                    prune_result = _check_prune_memory(adapter.graph)
                    optimization_results["prune_check"] = prune_result
                    if prune_result.get("prune_needed"):
                        logger.info(f"⚠️ Требуется очистка памяти: {prune_result.get('candidates_count', 0)} кандидатов")
                    else:
                        logger.info(f"✅ Очистка памяти не требуется")
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка при проверке очистки памяти: {e}", exc_info=True)
                    optimization_results["prune_check"] = {"error": str(e)}
                
                tracker.update_job(job_id=job_id, optimization_results=optimization_results)
                logger.info(f"✅ Все оптимизации завершены для чата '{request.chat}'")
                
            except Exception as e:
                logger.error(f"Ошибка при выполнении оптимизаций: {e}", exc_info=True)
                tracker.update_job(job_id=job_id, optimization_error=str(e))
        
    except Exception as e:
        logger.error(f"Ошибка при индексации чата '{request.chat}' (job_id: {job_id}): {e}", exc_info=True)
        job = tracker.get_job(job_id)
        started_at = job.get("started_at") if job else None
        tracker.update_job(
            job_id=job_id,
            status="failed",
            failed_at=datetime.now(timezone.utc).isoformat(),
            error=str(e),
        )


async def _start_indexing_job(
    request: "IndexChatRequest",
    adapter: MemoryServiceAdapter,
) -> "IndexChatResponse":
    """Создаёт и запускает задачу индексации в фоновом режиме.
    
    Предотвращает параллельные индексации одного чата.
    """
    tracker = _get_indexing_tracker()
    
    from datetime import timezone
    
    job_id = f"index_{uuid.uuid4().hex[:12]}"
    
    existing_jobs = tracker.get_all_jobs(status="running", chat=request.chat)
    if existing_jobs:
        existing_job = existing_jobs[0]
        return IndexChatResponse(
            job_id=existing_job["job_id"],
            status="running",
            chat=request.chat,
            message=f"Индексация чата '{request.chat}' уже выполняется (job_id: {existing_job['job_id']})",
        )
    
    tracker.create_job(
        job_id=job_id,
        scope="chat",
        chat=request.chat,
        force_full=request.force_full,
        recent_days=request.recent_days,
    )
    
    asyncio.create_task(
        _run_indexing_job(
            job_id=job_id,
            request=request,
            adapter=adapter,
        )
    )
    
    return IndexChatResponse(
        job_id=job_id,
        status="started",
        chat=request.chat,
        message=f"Индексация чата '{request.chat}' запущена в фоновом режиме. Используйте get_indexing_progress для отслеживания прогресса.",
    )


def _optimize_database(db_path: str) -> Dict[str, Any]:
    """Выполняет оптимизацию SQLite БД: VACUUM, ANALYZE и оптимизацию FTS5 индекса."""
    import sqlite3
    import time
    from pathlib import Path
    
    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        return {"error": f"База данных не найдена: {db_path}"}
    
    size_before = db_path_obj.stat().st_size
    operations_performed = []
    start_time = time.time()
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # VACUUM
        cursor.execute("VACUUM")
        conn.commit()
        operations_performed.append("VACUUM")
        
        # ANALYZE
        cursor.execute("ANALYZE")
        conn.commit()
        operations_performed.append("ANALYZE")
        
        # FTS5 оптимизация
        try:
            cursor.execute("INSERT INTO node_search(node_search) VALUES('optimize')")
            conn.commit()
            operations_performed.append("FTS5_optimize")
        except sqlite3.OperationalError:
            pass  # FTS5 таблица может отсутствовать
        
        conn.close()
        
        size_after = db_path_obj.stat().st_size
        space_freed = size_before - size_after
        duration = time.time() - start_time
        
        return {
            "success": True,
            "operations": operations_performed,
            "size_before_mb": round(size_before / 1024 / 1024, 2),
            "size_after_mb": round(size_after / 1024 / 1024, 2),
            "space_freed_mb": round(space_freed / 1024 / 1024, 2) if space_freed > 0 else 0,
            "duration_sec": round(duration, 2),
        }
    except Exception as e:
        return {"error": str(e)}


def _update_importance_scores(graph) -> Dict[str, Any]:
    """Пересчитывает важность всех узлов графа на основе метрик использования."""
    import json
    import sqlite3
    from ..memory.importance_scoring import ImportanceScorer
    
    try:
        scorer = ImportanceScorer()
        db_path = graph.db_path
        
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM nodes")
        nodes = cursor.fetchall()
        
        updated_count = 0
        importance_scores = []
        
        for node in nodes:
            try:
                node_dict = dict(node)
                properties = json.loads(node_dict.get("properties", "{}") or "{}")
                node_dict.update(properties)
                
                metadata = {
                    "_search_hits": properties.get("_search_hits", 0)
                }
                
                importance_score = scorer.compute_importance(node_dict, metadata)
                importance_scores.append(importance_score)
                
                properties["_importance_score"] = importance_score
                
                graph.update_node(
                    node_id=node_dict["id"],
                    properties=properties
                )
                
                updated_count += 1
            except Exception:
                continue
        
        conn.close()
        
        avg_importance = sum(importance_scores) / len(importance_scores) if importance_scores else 0
        min_importance = min(importance_scores) if importance_scores else 0
        max_importance = max(importance_scores) if importance_scores else 0
        
        return {
            "success": True,
            "updated_count": updated_count,
            "avg_importance": round(avg_importance, 3),
            "min_importance": round(min_importance, 3),
            "max_importance": round(max_importance, 3),
        }
    except Exception as e:
        return {"error": str(e)}


def _validate_database(db_path: str) -> Dict[str, Any]:
    """Проверяет целостность БД: SQLite integrity, внешние ключи, сиротские узлы/рёбра."""
    import sqlite3
    from pathlib import Path
    
    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        return {"error": f"База данных не найдена: {db_path}"}
    
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        issues = []
        checks_performed = []
        
        # Проверка целостности SQLite
        cursor.execute("PRAGMA integrity_check")
        integrity_result = cursor.fetchone()[0]
        checks_performed.append("integrity_check")
        if integrity_result != "ok":
            issues.append({
                "type": "integrity",
                "severity": "error",
                "message": f"SQLite integrity check failed: {integrity_result}",
            })
        
        # Проверка внешних ключей
        cursor.execute("PRAGMA foreign_key_check")
        fk_issues = cursor.fetchall()
        checks_performed.append("foreign_key_check")
        if fk_issues:
            for issue in fk_issues[:10]:  # Первые 10
                issues.append({
                    "type": "foreign_key",
                    "severity": "error",
                    "message": f"Foreign key violation: {dict(issue)}",
                })
        
        # Проверка графа знаний
        from ..memory.typed_graph import TypedGraphMemory
        graph = TypedGraphMemory(db_path=str(db_path))
        
        # Сиротские узлы
        cursor.execute("""
            SELECT id, type 
            FROM nodes 
            WHERE id NOT IN (
                SELECT DISTINCT source_id FROM edges
                UNION
                SELECT DISTINCT target_id FROM edges
            )
        """)
        orphaned_nodes = cursor.fetchall()
        checks_performed.append("orphaned_nodes")
        if orphaned_nodes:
            for node in orphaned_nodes[:10]:
                issues.append({
                    "type": "orphaned_node",
                    "severity": "warning",
                    "message": f"Node '{node['id']}' has no connections",
                })
        
        # Сиротские рёбра
        cursor.execute("""
            SELECT e.id, e.source_id, e.target_id, e.type
            FROM edges e
            LEFT JOIN nodes n1 ON e.source_id = n1.id
            LEFT JOIN nodes n2 ON e.target_id = n2.id
            WHERE n1.id IS NULL OR n2.id IS NULL
        """)
        orphaned_edges = cursor.fetchall()
        checks_performed.append("orphaned_edges")
        if orphaned_edges:
            for edge in orphaned_edges[:10]:
                issues.append({
                    "type": "orphaned_edge",
                    "severity": "error",
                    "message": f"Edge '{edge['id']}' references non-existent node",
                })
        
        conn.close()
        
        return {
            "valid": len([i for i in issues if i["severity"] == "error"]) == 0,
            "total_issues": len(issues),
            "error_count": len([i for i in issues if i["severity"] == "error"]),
            "warning_count": len([i for i in issues if i["severity"] == "warning"]),
            "checks_performed": checks_performed,
        }
    except Exception as e:
        return {"error": str(e)}


def _check_prune_memory(graph) -> Dict[str, Any]:
    """Проверяет, требуется ли очистка памяти на основе порога количества записей."""
    import json
    import sqlite3
    from ..memory.importance_scoring import MemoryPruner, EvictionScorer
    
    try:
        eviction_scorer = EvictionScorer()
        pruner = MemoryPruner(
            eviction_scorer=eviction_scorer,
            max_messages=100000,
            eviction_threshold=0.7
        )
        
        db_path = graph.db_path
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM nodes")
        nodes = cursor.fetchall()
        
        current_count = len(nodes)
        prune_needed = pruner.should_prune(current_count)
        
        if not prune_needed:
            conn.close()
            return {
                "prune_needed": False,
                "current_count": current_count,
                "max_records": 100000,
            }
        
        # Получаем кандидатов
        messages = []
        for node in nodes:
            node_dict = dict(node)
            properties = json.loads(node_dict.get("properties", "{}") or "{}")
            node_dict.update(properties)
            if "id" not in node_dict:
                node_dict["id"] = node["id"]
            messages.append(node_dict)
        
        candidates = pruner.get_eviction_candidates(messages, threshold=0.7)
        
        conn.close()
        
        return {
            "prune_needed": True,
            "current_count": current_count,
            "max_records": 100000,
            "candidates_count": len(candidates),
        }
    except Exception as e:
        return {"error": str(e)}


def get_health_payload() -> Dict[str, Any]:
    """Возвращает статус здоровья сервера и доступности сервисов."""
    from datetime import datetime

    status = "healthy"
    error: str | None = None
    adapter: MemoryServiceAdapter | None = None

    try:
        adapter = _get_adapter()
    except Exception as exc:
        status = "degraded"
        error = str(exc)

    settings = get_settings()
    db_path = settings.db_path

    payload: Dict[str, Any] = {
        "status": status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "services": {
            "memory": adapter is not None,
            "vector_search": adapter.vector_store is not None if adapter else False,
            "embeddings": adapter.embedding_service is not None if adapter else False,
        },
        "config": {
            "db_path": db_path,
        },
    }

    if error:
        payload["error"] = error

    return payload


def get_version_payload() -> Dict[str, Any]:
    """Возвращает версию сервера и список доступных функций."""
    from importlib import metadata

    try:
        version = metadata.version("memory-mcp")
    except metadata.PackageNotFoundError:
        version = "0.0.0"

    features = [
        "health",
        "version",
        # Универсальные инструменты
        "search",
        "batch_operations",
        "graph_query",
        "background_indexing",
        "summaries",
        "ingest",
        # Специализированные инструменты
        "store_trading_signal",
        "search_trading_patterns",
        "get_signal_performance",
        "generate_embedding",
        "update_record",
        "get_statistics",
        "search_explain",
        "get_timeline",
        "analyze_entities",
        "export_records",
        "import_records",
        "build_insight_graph",
        "index_chat",
        "get_available_chats",
    ]

    return {
        "name": "memory-mcp",
        "version": version,
        "features": features,
    }


async def _start_background_indexing(adapter: MemoryServiceAdapter) -> "StartBackgroundIndexingResponse":
    """Запускает фоновый сервис периодической индексации новых сообщений."""
    global _background_indexing_service
    from .schema import StartBackgroundIndexingResponse
    from ..config import get_settings
    
    try:
        settings = get_settings()
        
        if _background_indexing_service and _background_indexing_service.is_running():
            return StartBackgroundIndexingResponse(
                success=False,
                message="Фоновая индексация уже запущена"
            )
        
        # Инициализируем сервис
        from ..core.background_indexing import BackgroundIndexingService
        
        if not _background_indexing_service:
            _background_indexing_service = BackgroundIndexingService(
                input_path=settings.input_path,
                chats_path=settings.chats_path,
                chroma_path=settings.chroma_path,
                check_interval=settings.background_indexing_interval,
            )
            
            async def index_chat_callback(request: "IndexChatRequest"):
                return await _start_indexing_job(request, adapter)
            
            _background_indexing_service.set_index_chat_callback(index_chat_callback)
        
        _background_indexing_service.start()
        logger.info("Фоновая индексация запущена")
        
        return StartBackgroundIndexingResponse(
            success=True,
            message="Фоновая индексация успешно запущена"
        )
    except Exception as e:
        logger.error(f"Ошибка при запуске фоновой индексации: {e}", exc_info=True)
        return StartBackgroundIndexingResponse(
            success=False,
            message=f"Ошибка при запуске: {str(e)}"
        )


async def _stop_background_indexing() -> "StopBackgroundIndexingResponse":
    """Остановить фоновую индексацию."""
    global _background_indexing_service
    from .schema import StopBackgroundIndexingResponse
    
    try:
        if not _background_indexing_service:
            return StopBackgroundIndexingResponse(
                success=False,
                message="Фоновая индексация не запущена"
            )
        
        if not _background_indexing_service.is_running():
            return StopBackgroundIndexingResponse(
                success=False,
                message="Фоновая индексация уже остановлена"
            )
        
        # Останавливаем сервис
        await _background_indexing_service.stop_async()
        logger.info("Фоновая индексация остановлена")
        
        return StopBackgroundIndexingResponse(
            success=True,
            message="Фоновая индексация успешно остановлена"
        )
    except Exception as e:
        logger.error(f"Ошибка при остановке фоновой индексации: {e}", exc_info=True)
        return StopBackgroundIndexingResponse(
            success=False,
            message=f"Ошибка при остановке: {str(e)}"
        )


def _get_background_indexing_status() -> "GetBackgroundIndexingStatusResponse":
    """Получить статус фоновой индексации."""
    global _background_indexing_service
    from .schema import GetBackgroundIndexingStatusResponse
    from ..config import get_settings
    
    try:
        settings = get_settings()
        
        if not _background_indexing_service:
            return GetBackgroundIndexingStatusResponse(
                running=False,
                check_interval=settings.background_indexing_interval,
                last_check_time=None,
                input_path=settings.input_path,
                chats_path=settings.chats_path,
                message="Фоновая индексация не инициализирована"
            )
        
        status = _background_indexing_service.get_status()
        is_running = _background_indexing_service.is_running()
        
        return GetBackgroundIndexingStatusResponse(
            running=is_running,
            check_interval=settings.background_indexing_interval,
            last_check_time=status.get("last_check_time"),
            input_path=settings.input_path,
            chats_path=settings.chats_path,
            message=status.get("message", "Фоновая индексация работает" if is_running else "Фоновая индексация остановлена")
        )
    except Exception as e:
        logger.error(f"Ошибка при получении статуса фоновой индексации: {e}", exc_info=True)
        return GetBackgroundIndexingStatusResponse(
            running=False,
            check_interval=60,
            last_check_time=None,
            input_path="",
            chats_path="",
            message=f"Ошибка: {str(e)}"
        )


def _start_background_indexing_if_enabled():
    """Запустить фоновую индексацию если она включена в конфигурации."""
    global _background_indexing_service
    
    try:
        from ..config import get_settings
        settings = get_settings()
        
        if settings.background_indexing_enabled:
            logger.info("Автозапуск фоновой индексации (настроено в конфигурации)")
            
            # Инициализируем сервис синхронно
            from ..core.background_indexing import BackgroundIndexingService
            
            if not _background_indexing_service:
                _background_indexing_service = BackgroundIndexingService(
                    input_path=settings.input_path,
                    chats_path=settings.chats_path,
                    chroma_path=settings.chroma_path,
                    check_interval=settings.background_indexing_interval,
                )
                
                # Устанавливаем callback для запуска индексации
                adapter = _get_adapter()
                
                async def index_chat_callback(request: "IndexChatRequest"):
                    return await _start_indexing_job(request, adapter)
                
                _background_indexing_service.set_index_chat_callback(index_chat_callback)
            
            # Запускаем сервис
            _background_indexing_service.start()
            logger.info("Фоновая индексация запущена при старте сервера")
    except Exception as e:
        logger.warning(f"Не удалось запустить фоновую индексацию при старте: {e}")


async def _stop_background_indexing_on_shutdown():
    """Остановить фоновую индексацию при завершении сервера."""
    global _background_indexing_service
    
    if _background_indexing_service and _background_indexing_service.is_running():
        logger.info("Остановка фоновой индексации при завершении сервера...")
        try:
            await _stop_background_indexing()
        except Exception as e:
            logger.error(f"Ошибка при остановке фоновой индексации: {e}")


async def run_stdio_server() -> None:
    """Run the MCP server in stdio mode."""
    from mcp.server.stdio import stdio_server

    configure_logging()
    logger.info("Запуск Memory MCP сервера в stdio режиме")

    try:
        async with stdio_server() as (read_stream, write_stream):
            init_options = server.create_initialization_options()
            await server.run(read_stream, write_stream, init_options)
    except Exception as e:
        logger.error(f"Ошибка запуска MCP сервера: {e}")
        raise


def main() -> None:
    """Main entry point for stdio mode."""
    import asyncio

    try:
        asyncio.run(run_stdio_server())
    except Exception as exc:
        logging.exception("Critical error in MCP server: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
