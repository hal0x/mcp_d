"""MCP server entrypoint exposing the unified memory service."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

from mcp.server import Server
from mcp.types import Tool, TextContent
from pydantic import BaseModel

from ..quality_analyzer.utils.error_handler import format_error_message

from .adapters import MemoryServiceAdapter
from .schema import (
    FetchRequest,
    FetchResponse,
    GetSignalPerformanceRequest,
    GetSignalPerformanceResponse,
    IngestRequest,
    IngestResponse,
    ScrapedContentRequest,
    ScrapedContentResponse,
    SearchRequest,
    SearchResponse,
    SearchTradingPatternsRequest,
    SearchTradingPatternsResponse,
    StoreTradingSignalRequest,
    StoreTradingSignalResponse,
)

logger = logging.getLogger(__name__)

# Создаем MCP сервер
server = Server("memory-mcp")

ToolResponse = Tuple[List[TextContent], Dict[str, Any]]

# Глобальный адаптер (инициализируется при первом использовании)
_adapter: MemoryServiceAdapter | None = None


def _get_adapter() -> MemoryServiceAdapter:
    """Получить или создать адаптер памяти."""
    global _adapter
    if _adapter is None:
        db_path = os.getenv("MEMORY_DB_PATH", "memory_graph.db")
        _adapter = MemoryServiceAdapter(db_path=db_path)
    return _adapter


def _to_serializable(value: Any) -> Any:
    """Recursively convert Pydantic models and iterables into plain Python data."""
    if isinstance(value, BaseModel):
        try:
            return value.model_dump()
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


# Используем общую функцию форматирования ошибок из error_handler
# _format_error_message удалена, используется format_error_message из error_handler


@server.list_tools()  # type: ignore[misc]
async def list_tools() -> List[Tool]:
    """Returns a list of available memory MCP tools."""
    return [
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
                    "signal": {
                        "type": "object",
                        "description": "Данные торгового сигнала",
                    }
                },
                "required": ["signal"],
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
                    "content": {
                        "type": "object",
                        "description": "Данные скрапленного контента",
                    }
                },
                "required": ["content"],
            },
        ),
    ]


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
            request = SearchTradingPatternsRequest(**arguments)
            result = adapter.search_trading_patterns(request)
            return _format_tool_response(result.model_dump())

        elif name == "get_signal_performance":
            request = GetSignalPerformanceRequest(**arguments)
            result = adapter.get_signal_performance(request)
            return _format_tool_response(result.model_dump())

        elif name == "ingest_scraped_content":
            request = ScrapedContentRequest(**arguments)
            result = adapter.ingest_scraped_content(request)
            return _format_tool_response(result.model_dump())

        else:
            raise ValueError(f"Неизвестный инструмент: {name}")

    except ValueError as exc:
        logger.warning(f"Ошибка при выполнении инструмента {name}: {exc}")
        raise RuntimeError(format_error_message(exc)) from exc
    except Exception as exc:
        logger.exception(f"Ошибка при выполнении инструмента {name}: {exc}")
        raise RuntimeError(format_error_message(exc)) from exc


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

    return {
        "name": "memory-mcp",
        "version": version,
        "features": [
            "ingest_records",
            "search_memory",
            "fetch_record",
            "store_trading_signal",
            "search_trading_patterns",
            "get_signal_performance",
            "ingest_scraped_content",
        ],
    }


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
