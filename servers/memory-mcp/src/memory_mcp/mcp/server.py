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

from .adapters import MemoryServiceAdapter
from .schema import (
    StartBackgroundIndexingRequest,
    StartBackgroundIndexingResponse,
    StopBackgroundIndexingRequest,
    StopBackgroundIndexingResponse,
    GetBackgroundIndexingStatusRequest,
    GetBackgroundIndexingStatusResponse,
    AnalyzeEntitiesRequest,
    AnalyzeEntitiesResponse,
    BatchUpdateRecordsRequest,
    BatchUpdateRecordsResponse,
    BuildInsightGraphRequest,
    BuildInsightGraphResponse,
    DeleteRecordRequest,
    DeleteRecordResponse,
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
    GetSignalPerformanceRequest,
    GetSignalPerformanceResponse,
    GetStatisticsResponse,
    GetTagsStatisticsResponse,
    GetTimelineRequest,
    GetTimelineResponse,
    ImportRecordsRequest,
    ImportRecordsResponse,
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
    StoreTradingSignalRequest,
    StoreTradingSignalResponse,
    UpdateRecordRequest,
    UpdateRecordResponse,
    UpdateSummariesRequest,
    UpdateSummariesResponse,
)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("MEMORY_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

logger = logging.getLogger(__name__)

server = Server("memory-mcp")
logger.info(f"MCP сервер '{server.name}' создан, начинаем регистрацию инструментов...")

ToolResponse = Tuple[List[TextContent], Dict[str, Any]]

_adapter: MemoryServiceAdapter | None = None

# Глобальный словарь для хранения активных задач индексации
_active_indexing_jobs: Dict[str, Dict[str, Any]] = {}

# Глобальный сервис фоновой индексации
_background_indexing_service = None


def _get_adapter() -> MemoryServiceAdapter:
    """Получить или создать адаптер памяти."""
    global _adapter
    if _adapter is None:
        db_path = os.getenv("MEMORY_DB_PATH", "data/memory_graph.db")
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


def _to_serializable(value: Any) -> Any:
    """Recursively convert Pydantic models and iterables into plain Python data."""
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
    """Return combined textual and structured content for tool responses."""
    serialized = _to_serializable(payload)
    if isinstance(serialized, dict):
        structured = serialized
        text_payload = serialized
    else:
        structured = {root_key: serialized}
        text_payload = serialized
    text = json.dumps(text_payload, indent=2, ensure_ascii=False)
    return ([TextContent(type="text", text=text)], structured)


@server.list_tools()  # type: ignore[misc]
async def list_tools() -> List[Tool]:
    """Returns a list of available memory MCP tools."""
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
        Tool(
            name="ingest_records",
            description="Ingest a batch of memory records into the unified memory store.",
            inputSchema={
                "type": "object",
                "properties": {
                    "records": {
                        "type": "array",
                        "description": "Список записей для загрузки",
                        "items": {"type": "object"},
                    }
                },
                "required": ["records"],
            },
        ),
        Tool(
            name="search_memory",
            description="Search memory for relevant records using hybrid FTS + vector search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"},
                    "top_k": {
                        "type": "integer",
                        "description": "Максимальное количество результатов",
                        "default": 5,
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
                        "description": "Фильтр: дата не раньше (ISO format)",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Фильтр: дата не позже (ISO format)",
                    },
                    "include_embeddings": {
                        "type": "boolean",
                        "description": "Возвращать ли эмбеддинги",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="fetch_record",
            description="Fetch a full memory record by identifier.",
            inputSchema={
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "Идентификатор записи",
                    }
                },
                "required": ["record_id"],
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
            name="ingest_scraped_content",
            description="Ingest scraped web content into memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL скрапленной страницы",
                    },
                    "title": {
                        "type": "string",
                        "description": "Заголовок страницы (опционально)",
                    },
                    "content": {
                        "type": "string",
                        "description": "Основной текстовый контент страницы",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Дополнительные метаданные (автор, дата публикации и т.д.)",
                        "additionalProperties": True,
                    },
                    "source": {
                        "type": "string",
                        "description": "Источник скраппинга (по умолчанию 'bright_data')",
                        "default": "bright_data",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Теги для классификации контента",
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Извлеченные сущности (имена, организации и т.д.)",
                    },
                },
                "required": ["url", "content"],
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
            name="delete_record",
            description="Delete a memory record from storage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "Идентификатор записи для удаления",
                    },
                },
                "required": ["record_id"],
            },
        ),
        Tool(
            name="get_statistics",
            description="Get system statistics (graph, sources, tags, database size).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="get_indexing_progress",
            description="Get indexing progress from ChromaDB collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "chat": {
                        "type": "string",
                        "description": "Фильтр по конкретному чату (опционально)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get_graph_neighbors",
            description="Get neighboring nodes for a graph node.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "ID узла",
                    },
                    "edge_type": {
                        "type": "string",
                        "description": "Фильтр по типу рёбер (опционально)",
                    },
                    "direction": {
                        "type": "string",
                        "description": "Направление: 'out', 'in', 'both'",
                        "enum": ["out", "in", "both"],
                        "default": "both",
                    },
                },
                "required": ["node_id"],
            },
        ),
        Tool(
            name="find_graph_path",
            description="Find shortest path between two nodes in the graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": "ID исходного узла",
                    },
                    "target_id": {
                        "type": "string",
                        "description": "ID целевого узла",
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Максимальная длина пути",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["source_id", "target_id"],
            },
        ),
        Tool(
            name="get_related_records",
            description="Get related records through graph connections.",
            inputSchema={
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "ID записи",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Максимальная глубина поиска",
                        "default": 1,
                        "minimum": 1,
                        "maximum": 3,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Максимальное количество результатов",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["record_id"],
            },
        ),
        Tool(
            name="search_by_embedding",
            description="Search memory by embedding vector directly (without text query).",
            inputSchema={
                "type": "object",
                "properties": {
                    "embedding": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Вектор эмбеддинга для поиска",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Максимальное количество результатов",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 50,
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
                        "description": "Фильтр: дата не раньше (ISO format)",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Фильтр: дата не позже (ISO format)",
                    },
                },
                "required": ["embedding"],
            },
        ),
        Tool(
            name="similar_records",
            description="Find similar records to a given record by embedding similarity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "ID записи для поиска похожих",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Максимальное количество результатов",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["record_id"],
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
            name="get_tags_statistics",
            description="Get statistics about tags usage in records.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
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
            name="batch_update_records",
            description="Update multiple records in a single batch operation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "updates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "record_id": {
                                    "type": "string",
                                    "description": "ID записи для обновления",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Новый контент (опционально)",
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
                                    "description": "Новые метаданные (опционально)",
                                    "additionalProperties": True,
                                },
                            },
                            "required": ["record_id"],
                        },
                        "description": "Список обновлений записей",
                    },
                },
                "required": ["updates"],
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
            name="update_summaries",
            description="Update markdown summaries without full re-indexing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "chat": {
                        "type": "string",
                        "description": "Обновить отчеты только для конкретного чата (опционально)",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Принудительно пересоздать существующие артефакты",
                        "default": False,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="review_summaries",
            description="Review and fix summaries with -needs-review suffix.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dry_run": {
                        "type": "boolean",
                        "description": "Только анализ, без изменения файлов",
                        "default": False,
                    },
                    "chat": {
                        "type": "string",
                        "description": "Обработать только конкретный чат (опционально)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Максимальное количество файлов для обработки (опционально)",
                    },
                },
                "required": [],
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
                        "description": "Путь к ChromaDB (опционально, по умолчанию ./chroma_db)",
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
            description="Index a specific Telegram chat with two-level indexing (L1: sessions, L2: messages, L3: tasks). Индексация выполняется в фоновом режиме и не блокирует вызов. Используйте get_indexing_progress для отслеживания прогресса.",
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
                    "progress": {
                        "type": "boolean",
                        "description": "Показать прогресс-бар (не используется, индексация всегда в фоне)",
                        "default": False,
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
        Tool(
            name="start_background_indexing",
            description="Start background indexing service that periodically checks input directory for new messages and indexes them.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="stop_background_indexing",
            description="Stop background indexing service.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="get_background_indexing_status",
            description="Get status of background indexing service.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]
    logger.info(f"Зарегистрировано {len(tools)} инструментов: {[t.name for t in tools]}")
    return tools


@server.call_tool()  # type: ignore[misc]
async def call_tool(name: str, arguments: Dict[str, Any]) -> ToolResponse:
    """Execute a tool call and format the result."""
    try:
        logger.info(f"Вызов инструмента: {name} с аргументами: {arguments}")

        if name == "health":
            result = get_health_payload()
            return _format_tool_response(result)

        if name == "version":
            result = get_version_payload()
            return _format_tool_response(result)

        adapter = _get_adapter()

        if name == "ingest_records":
            from .schema import MemoryRecordPayload

            records_data = arguments.get("records", [])
            records = [MemoryRecordPayload(**item) for item in records_data]
            result = adapter.ingest(records)
            return _format_tool_response(result.model_dump())

        elif name == "search_memory":
            request = SearchRequest(**arguments)
            result = adapter.search(request)
            return _format_tool_response(result.model_dump())

        elif name == "fetch_record":
            request = FetchRequest(**arguments)
            result = adapter.fetch(request)
            return _format_tool_response(result.model_dump())

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

        elif name == "ingest_scraped_content":
            request = ScrapedContentRequest(**arguments)
            result = adapter.ingest_scraped_content(request)
            return _format_tool_response(result.model_dump())

        elif name == "generate_embedding":
            try:
                request = GenerateEmbeddingRequest(**arguments)
                result = adapter.generate_embedding(request)
                return _format_tool_response(result.model_dump())
            except ValueError as e:
                error_msg = format_error_message(e)
                logger.warning(f"generate_embedding failed: {error_msg}")
                raise RuntimeError(error_msg) from e

        elif name == "update_record":
            request = UpdateRecordRequest(**arguments)
            result = adapter.update_record(request)
            return _format_tool_response(result.model_dump())

        elif name == "delete_record":
            request = DeleteRecordRequest(**arguments)
            result = adapter.delete_record(request)
            return _format_tool_response(result.model_dump())

        elif name == "get_statistics":
            result = adapter.get_statistics()
            return _format_tool_response(result.model_dump())

        elif name == "get_indexing_progress":
            try:
                request = GetIndexingProgressRequest(**arguments)
                result = adapter.get_indexing_progress(request)
                return _format_tool_response(result.model_dump())
            except Exception as e:
                logger.exception(f"get_indexing_progress failed: {e}")
                raise RuntimeError(format_error_message(e)) from e

        elif name == "get_graph_neighbors":
            request = GetGraphNeighborsRequest(**arguments)
            result = adapter.get_graph_neighbors(request)
            return _format_tool_response(result.model_dump())

        elif name == "find_graph_path":
            request = FindGraphPathRequest(**arguments)
            result = adapter.find_graph_path(request)
            return _format_tool_response(result.model_dump())

        elif name == "get_related_records":
            request = GetRelatedRecordsRequest(**arguments)
            result = adapter.get_related_records(request)
            return _format_tool_response(result.model_dump())

        elif name == "search_by_embedding":
            if "date_from" in arguments and arguments["date_from"]:
                from datetime import datetime
                arguments["date_from"] = datetime.fromisoformat(arguments["date_from"].replace("Z", "+00:00"))
            if "date_to" in arguments and arguments["date_to"]:
                from datetime import datetime
                arguments["date_to"] = datetime.fromisoformat(arguments["date_to"].replace("Z", "+00:00"))
            request = SearchByEmbeddingRequest(**arguments)
            result = adapter.search_by_embedding(request)
            return _format_tool_response(result.model_dump())

        elif name == "similar_records":
            request = SimilarRecordsRequest(**arguments)
            result = adapter.similar_records(request)
            return _format_tool_response(result.model_dump())

        elif name == "search_explain":
            request = SearchExplainRequest(**arguments)
            result = adapter.search_explain(request)
            return _format_tool_response(result.model_dump())

        elif name == "get_tags_statistics":
            result = adapter.get_tags_statistics()
            return _format_tool_response(result.model_dump())

        elif name == "get_timeline":
            if "date_from" in arguments and arguments["date_from"]:
                from datetime import datetime
                arguments["date_from"] = datetime.fromisoformat(arguments["date_from"].replace("Z", "+00:00"))
            if "date_to" in arguments and arguments["date_to"]:
                from datetime import datetime
                arguments["date_to"] = datetime.fromisoformat(arguments["date_to"].replace("Z", "+00:00"))
            request = GetTimelineRequest(**arguments)
            result = adapter.get_timeline(request)
            return _format_tool_response(result.model_dump())

        elif name == "analyze_entities":
            request = AnalyzeEntitiesRequest(**arguments)
            result = adapter.analyze_entities(request)
            return _format_tool_response(result.model_dump())

        elif name == "batch_update_records":
            updates_data = arguments.get("updates", [])
            from .schema import BatchUpdateRecordItem
            updates = [BatchUpdateRecordItem(**item) for item in updates_data]
            request = BatchUpdateRecordsRequest(updates=updates)
            result = adapter.batch_update_records(request)
            return _format_tool_response(result.model_dump())

        elif name == "export_records":
            if "date_from" in arguments and arguments["date_from"]:
                from datetime import datetime
                arguments["date_from"] = datetime.fromisoformat(arguments["date_from"].replace("Z", "+00:00"))
            if "date_to" in arguments and arguments["date_to"]:
                from datetime import datetime
                arguments["date_to"] = datetime.fromisoformat(arguments["date_to"].replace("Z", "+00:00"))
            request = ExportRecordsRequest(**arguments)
            result = adapter.export_records(request)
            return _format_tool_response(result.model_dump())

        elif name == "import_records":
            request = ImportRecordsRequest(**arguments)
            result = adapter.import_records(request)
            return _format_tool_response(result.model_dump())

        elif name == "update_summaries":
            request = UpdateSummariesRequest(**arguments)
            result = await adapter.update_summaries(request)
            return _format_tool_response(result.model_dump())

        elif name == "review_summaries":
            request = ReviewSummariesRequest(**arguments)
            result = await adapter.review_summaries(request)
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

        elif name == "start_background_indexing":
            request = StartBackgroundIndexingRequest(**arguments)
            result = await _start_background_indexing(adapter)
            return _format_tool_response(result.model_dump())

        elif name == "stop_background_indexing":
            request = StopBackgroundIndexingRequest(**arguments)
            result = await _stop_background_indexing()
            return _format_tool_response(result.model_dump())

        elif name == "get_background_indexing_status":
            request = GetBackgroundIndexingStatusRequest(**arguments)
            result = _get_background_indexing_status()
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
    chat: str,
    force_full: bool,
    recent_days: int,
    adapter: MemoryServiceAdapter,
) -> None:
    """Запуск индексации в фоновом режиме."""
    global _active_indexing_jobs
    
    from ..core.indexer import TwoLevelIndexer
    from ..config import get_settings
    from ..core.lmstudio_client import LMStudioEmbeddingClient
    from datetime import timezone
    
    try:
        # Обновляем статус на "running"
        _active_indexing_jobs[job_id] = {
            "status": "running",
            "chat": chat,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "current_stage": "Инициализация",
        }
        
        settings = get_settings()
        
        # Инициализируем embedding client
        embedding_client = LMStudioEmbeddingClient(
            model_name=settings.lmstudio_model,
            base_url=f"http://{settings.lmstudio_host}:{settings.lmstudio_port}"
        )
        
        # Инициализируем индексатор
        indexer = TwoLevelIndexer(
            chroma_path=settings.chroma_path,
            artifacts_path=settings.artifacts_path,
            embedding_client=embedding_client,
            enable_smart_aggregation=True,
            aggregation_strategy="smart",
        )
        
        # Обновляем статус
        _active_indexing_jobs[job_id]["current_stage"] = "Загрузка сообщений"
        
        # Запускаем индексацию
        stats = await indexer.build_index(
            scope="chat",
            chat=chat,
            force_full=force_full,
            recent_days=recent_days,
        )
        
        # Обновляем статус на "completed"
        _active_indexing_jobs[job_id] = {
            "status": "completed",
            "chat": chat,
            "started_at": _active_indexing_jobs[job_id].get("started_at"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "sessions_indexed": stats.get("sessions_indexed", 0),
            "messages_indexed": stats.get("messages_indexed", 0),
            "tasks_indexed": stats.get("tasks_indexed", 0),
        }
        
        logger.info(f"Индексация чата '{chat}' завершена успешно (job_id: {job_id})")
        
    except Exception as e:
        logger.error(f"Ошибка при индексации чата '{chat}' (job_id: {job_id}): {e}", exc_info=True)
        # Обновляем статус на "failed"
        _active_indexing_jobs[job_id] = {
            "status": "failed",
            "chat": chat,
            "started_at": _active_indexing_jobs.get(job_id, {}).get("started_at"),
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }
    finally:
        # Удаляем задачу через 1 час после завершения (чтобы можно было посмотреть результат)
        # В реальности можно использовать более сложную логику очистки
        pass


async def _start_indexing_job(
    request: "IndexChatRequest",
    adapter: MemoryServiceAdapter,
) -> "IndexChatResponse":
    """Запустить задачу индексации в фоновом режиме."""
    global _active_indexing_jobs
    
    from datetime import timezone
    
    # Генерируем уникальный ID задачи
    job_id = f"index_{uuid.uuid4().hex[:12]}"
    
    # Проверяем, не запущена ли уже индексация для этого чата
    for existing_job_id, job_info in _active_indexing_jobs.items():
        if job_info.get("chat") == request.chat and job_info.get("status") == "running":
            return IndexChatResponse(
                job_id=existing_job_id,
                status="running",
                chat=request.chat,
                message=f"Индексация чата '{request.chat}' уже выполняется (job_id: {existing_job_id})",
            )
    
    # Создаем задачу
    _active_indexing_jobs[job_id] = {
        "status": "started",
        "chat": request.chat,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "force_full": request.force_full,
        "recent_days": request.recent_days,
    }
    
    # Запускаем индексацию в фоне
    asyncio.create_task(
        _run_indexing_job(
            job_id=job_id,
            chat=request.chat,
            force_full=request.force_full,
            recent_days=request.recent_days,
            adapter=adapter,
        )
    )
    
    return IndexChatResponse(
        job_id=job_id,
        status="started",
        chat=request.chat,
        message=f"Индексация чата '{request.chat}' запущена в фоновом режиме. Используйте get_indexing_progress для отслеживания прогресса.",
    )


def get_health_payload() -> Dict[str, Any]:
    """Возвращает информацию о состоянии сервера."""
    from datetime import datetime

    status = "healthy"
    error: str | None = None
    adapter: MemoryServiceAdapter | None = None

    try:
        adapter = _get_adapter()
    except Exception as exc:
        status = "degraded"
        error = str(exc)

    db_path = os.getenv("MEMORY_DB_PATH", "memory_graph.db")

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
    """Возвращает информацию о версии сервера."""
    from importlib import metadata

    try:
        version = metadata.version("memory-mcp")
    except metadata.PackageNotFoundError:
        version = "0.0.0"

    features = [
        "health",
        "version",
        "ingest_records",
        "search_memory",
        "fetch_record",
        "store_trading_signal",
        "search_trading_patterns",
        "get_signal_performance",
        "ingest_scraped_content",
        "generate_embedding",
        "update_record",
        "delete_record",
        "get_statistics",
        "get_indexing_progress",
        "get_graph_neighbors",
        "find_graph_path",
        "get_related_records",
        "search_by_embedding",
        "similar_records",
        "search_explain",
        "get_tags_statistics",
        "get_timeline",
        "analyze_entities",
        "batch_update_records",
        "export_records",
        "import_records",
        "update_summaries",
        "review_summaries",
        "build_insight_graph",
        "index_chat",
        "get_available_chats",
        "start_background_indexing",
        "stop_background_indexing",
        "get_background_indexing_status",
    ]

    return {
        "name": "memory-mcp",
        "version": version,
        "features": features,
    }


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


def configure_logging() -> None:
    """Configure logging for the MCP server."""
    logging.basicConfig(
        level=os.getenv("MEMORY_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


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
