#!/usr/bin/env python3
"""
🚀 MCP Сервер для поиска по Telegram дампам

Предоставляет доступ к индексированным данным через Model Context Protocol (MCP).
Поддерживает поиск по сообщениям, сессиям и задачам.
"""
# mypy: ignore-errors

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi_mcp.transport.http import FastApiHttpSessionManager
from mcp.types import Tool, TextContent
import uvicorn

# Добавляем src и scripts в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Импорты после изменения PYTHONPATH
from memory_mcp.config import get_settings, Settings  # noqa: E402
from mcp_server_base import TelegramDumpMCPBase  # noqa: E402

logger = logging.getLogger(__name__)

FEATURES: List[str] = [
    "health",
    "version",
    "search_messages",
    "search_sessions",
    "search_tasks",
    "get_chat_info",
    "get_stats",
    "tokenize_text",
    "search_numeric_data",
    "analyze_chat_content",
    "read_session_report",
    "read_chat_context",
    "list_chat_artifacts",
    "get_chats_list",
    "index_chat",
]


def _safe_repr(value: Any, max_length: int = 120) -> str:
    """Create log-friendly representation."""
    try:
        rendered = repr(value)
    except Exception:
        rendered = f"<unrepresentable {type(value).__name__}>"
    if len(rendered) > max_length:
        return rendered[: max_length - 3] + "..."
    return rendered


def _summarize_text_result(text: str) -> str:
    """Summarize textual result for logs."""
    try:
        size = len(text.encode("utf-8"))
    except Exception:
        size = len(text)
    return f"{len(text)} chars (~{size} bytes)"


class TelegramDumpMCP(TelegramDumpMCPBase):
    """MCP сервер для поиска по Telegram дампам"""

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        chats_path: Optional[str] = None,
        artifacts_path: Optional[str] = None,
    ):
        """
        Инициализация MCP сервера

        Args:
            chroma_path: Путь к базе ChromaDB
            chats_path: Путь к папке с чатами
            artifacts_path: Путь к папке с артефактами
        """
        settings = get_settings()
        resolved_chroma = chroma_path or settings.chroma_path
        resolved_chats = chats_path or settings.chats_path
        resolved_artifacts = artifacts_path or settings.artifacts_path

        Path(resolved_artifacts).mkdir(parents=True, exist_ok=True)
        super().__init__(resolved_chroma, resolved_chats, resolved_artifacts)
    
    async def _health_payload(self) -> Dict[str, Any]:
        """Собирает информацию о состоянии сервиса."""
        try:
            stats_raw = await self._get_stats()
            stats: Dict[str, Any] = json.loads(stats_raw) if isinstance(stats_raw, str) else stats_raw
        except Exception as exc:
            logger.exception("Не удалось получить статистику для health: %s", exc)
            stats = {"error": str(exc)}
        
        return {
            "status": "healthy" if "error" not in stats else "degraded",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "paths": {
                "chroma": getattr(self, "chroma_path", None),
                "chats": getattr(self, "chats_path", None),
                "artifacts": getattr(self, "artifacts_path", None),
            },
            "collections": stats.get("collections"),
            "stats": {
                "total_records": stats.get("total_records"),
                "total_chats": stats.get("total_chats"),
            },
            "config": self._config_snapshot(),
            "features": FEATURES,
        }
    
    def _version_payload(self) -> Dict[str, Any]:
        """Возвращает информацию о версии пакета."""
        try:
            version = metadata.version("memory_mcp")
        except metadata.PackageNotFoundError:
            version = "0.0.0"
        
        return {
            "name": "memory-mcp",
            "version": version,
            "features": FEATURES,
        }
    
    def _config_snapshot(self) -> Dict[str, Any]:
        """Возвращает текущие настройки сервиса."""
        return {
            "chroma_path": getattr(self, "chroma_path", None),
            "chats_path": getattr(self, "chats_path", None),
            "artifacts_path": getattr(self, "artifacts_path", None),
        }

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> List[TextContent]:
            """Вызов инструмента"""
            logger.info("Вызов инструмента %s с аргументами %s", name, {k: _safe_repr(v) for k, v in arguments.items()})
            try:
                if name == "health":
                    result_dict = await self._health_payload()
                    result_text = json.dumps(result_dict, ensure_ascii=False, indent=2)
                elif name == "version":
                    result_dict = self._version_payload()
                    result_text = json.dumps(result_dict, ensure_ascii=False, indent=2)
                elif name == "search_messages":
                    result = await self._search_collection(
                        collection_name="chat_messages",
                        query=arguments["query"],
                        chat_filter=arguments.get("chat_filter"),
                        limit=arguments.get("limit", 10),
                        depth=arguments.get("depth", "shallow"),
                        include_metadata=arguments.get("include_metadata", False),
                    )
                    result_text = result
                elif name == "search_sessions":
                    result = await self._search_collection(
                        collection_name="chat_sessions",
                        query=arguments["query"],
                        chat_filter=arguments.get("chat_filter"),
                        limit=arguments.get("limit", 5),
                        depth=arguments.get("depth", "shallow"),
                        include_metadata=arguments.get("include_metadata", False),
                    )
                    result_text = result
                elif name == "search_tasks":
                    result = await self._search_collection(
                        collection_name="chat_tasks",
                        query=arguments["query"],
                        chat_filter=arguments.get("chat_filter"),
                        limit=arguments.get("limit", 5),
                    )
                    result_text = result
                elif name == "get_stats":
                    result = await self._get_stats()
                    result_text = result
                elif name == "get_chats_list":
                    result = await self._get_chats_list()
                    result_text = result
                else:
                    result_text = json.dumps({"error": f"Неизвестный инструмент: {name}"}, ensure_ascii=False)

                logger.info("Инструмент %s выполнен (%s)", name, _summarize_text_result(result_text))
                return [TextContent(type="text", text=result_text)]

            except Exception as e:
                logger.exception("Ошибка выполнения инструмента %s", name)
                error_result = json.dumps({"error": str(e)})
                return [TextContent(type="text", text=error_result)]

def _configure_logging(level_name: str) -> int:
    """Настраивает логирование для сервиса."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger.setLevel(level)
    return level


def _create_http_app(server: TelegramDumpMCP) -> FastAPI:
    """Создает FastAPI приложение с поддержкой streamable-http транспорта."""
    version_info = server._version_payload()
    app = FastAPI(
        title="Memory MCP",
        version=version_info.get("version", "0.0.0"),
        description="Streamable HTTP интерфейс для Memory MCP",
    )

    transport = FastApiHttpSessionManager(mcp_server=server.server)

    @app.on_event("shutdown")
    async def shutdown_transport() -> None:
        await transport.shutdown()

    @app.get("/healthz", tags=["internal"], include_in_schema=False)
    async def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    @app.api_route("/mcp", methods=["GET", "POST", "DELETE"], include_in_schema=False)
    async def handle_mcp(request: Request):
        return await transport.handle_fastapi_request(request)

    return app


async def _run_stdio(server: TelegramDumpMCP) -> None:
    """Запускает сервер в stdio режиме."""
    await server.run()


def _run_http(server: TelegramDumpMCP, host: str, port: int, log_level: str) -> None:
    """Запускает сервер в streamable-http режиме."""
    app = _create_http_app(server)
    logger.info(
        "Запуск Memory MCP в режиме streamable-http (host=%s, port=%s, log_level=%s)",
        host,
        port,
        log_level,
    )
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory MCP server")
    parser.add_argument(
        "transport",
        choices=["stdio", "streamable-http"],
        nargs="?",
        help="Транспорт для запуска (по умолчанию из переменной TRANSPORT или stdio)",
    )
    parser.add_argument("--host", help="Хост для HTTP транспорта")
    parser.add_argument("--port", type=int, help="Порт для HTTP транспорта")
    parser.add_argument("--print-config", action="store_true", help="Показать текущую конфигурацию и выйти")
    parser.add_argument("--log-level", help="Уровень логирования (INFO, DEBUG, WARNING, ...)")
    args = parser.parse_args()

    base_settings = get_settings()

    transport_candidate = (args.transport or base_settings.transport).lower()
    transport = transport_candidate if transport_candidate in {"stdio", "streamable-http"} else base_settings.transport
    transport = transport.lower()

    host = args.host or base_settings.host
    port = args.port or base_settings.port
    log_level_name = (args.log_level or base_settings.log_level).upper()

    _configure_logging(log_level_name)
    if base_settings.debug:
        logger.setLevel(logging.DEBUG)

    server = TelegramDumpMCP(
        chroma_path=base_settings.chroma_path,
        chats_path=base_settings.chats_path,
        artifacts_path=base_settings.artifacts_path,
    )

    effective_config = {
        "chroma_path": base_settings.chroma_path,
        "chats_path": base_settings.chats_path,
        "artifacts_path": base_settings.artifacts_path,
        "host": host,
        "port": port,
        "log_level": log_level_name,
        "transport": transport,
        "debug": base_settings.debug,
    }

    if args.print_config:
        snapshot = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "transport": transport,
            "config": effective_config,
            "features": FEATURES,
        }
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
        return

    if transport == "stdio":
        asyncio.run(_run_stdio(server))
    else:
        _run_http(server, host, port, log_level_name)


if __name__ == "__main__":
    main()
