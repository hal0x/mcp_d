#!/usr/bin/env python3
"""MCP сервер для TradingView анализа."""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from importlib import metadata
from typing import Any, Dict, List, Optional, Tuple

import uvicorn

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from pydantic import BaseModel

# Добавляем src к пути
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tradingview_mcp.config import Settings, get_settings
from tradingview_mcp.pro_scanners import ScannerProfiles, ScannerService
from tradingview_mcp.pro_scanners.models import BacktestRequest

# Настройка логирования (конфигурируется в main)
logger = logging.getLogger(__name__)

# Создаем MCP сервер
server = Server("tradingview-mcp")

ToolResponse = Tuple[List[TextContent], Dict[str, Any]]


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


def _format_error_message(exc: Exception) -> str:
    """Ensure tool errors propagate with a consistent prefix for clients."""
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if not lowered.startswith("ошибка") and not lowered.startswith("error"):
        message = f"Ошибка: {message}"
    return message


@server.list_tools()  # type: ignore[misc]
async def list_tools() -> List[Tool]:
    """Returns a curated list of available TradingView tools."""
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
            name="exchanges_list",
            description="List available cryptocurrency exchanges.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        # TODO: Добавить остальные tools из старого server.py
    ]


@server.call_tool()  # type: ignore[misc]
async def call_tool(name: str, arguments: Dict[str, Any]) -> ToolResponse:
    """Execute a tool call and format the result."""
    try:
        logger.info(f"Вызов инструмента: {name} с аргументами: {arguments}")

        if name == "health":
            result = get_health_payload()
            return _format_tool_response(result)

        elif name == "version":
            result = get_version_payload()
            return _format_tool_response(result)

        elif name == "exchanges_list":
            # TODO: Реализовать
            result = {"exchanges": ["BINANCE", "KUCOIN", "BYBIT", "OKX"]}
            return _format_tool_response(result)

        else:
            raise ValueError(f"Неизвестный инструмент: {name}")

    except ValueError as exc:
        logger.warning(f"Ошибка при выполнении инструмента {name}: {exc}")
        raise RuntimeError(_format_error_message(exc)) from exc
    except Exception as exc:
        logger.exception(f"Ошибка при выполнении инструмента {name}: {exc}")
        raise RuntimeError(_format_error_message(exc)) from exc


def get_health_payload() -> Dict[str, Any]:
    """Возвращает информацию о состоянии сервера."""
    status = "healthy"
    error: Optional[str] = None
    settings: Optional[Settings] = None

    try:
        settings = get_settings()
    except Exception as exc:
        status = "degraded"
        error = str(exc)

    payload: Dict[str, Any] = {
        "status": status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "services": {
            "tradingview": True,
            "screener": True,
            "pro_scanners": True,  # TODO: Проверить статус
        },
    }

    if settings:
        payload["config"] = {
            "host": settings.host,
            "port": settings.port,
            "log_level": settings.log_level,
            "default_transport": settings.default_transport,
            "debug": settings.debug,
        }
    else:
        payload["config"] = None

    if error:
        payload["error"] = error

    return payload


def get_version_payload() -> Dict[str, Any]:
    """Возвращает информацию о версии сервера."""
    try:
        version = metadata.version("tradingview-mcp")
    except metadata.PackageNotFoundError:
        version = "0.1.0"

    return {
        "name": "tradingview-mcp",
        "version": version,
        "features": [
            "market_analysis",
            "technical_screening", 
            "strategy_tools",
            "professional_scanners",
            "meta_tools",
        ],
    }


async def run_stdio_server() -> None:
    """Запускает MCP сервер в stdio режиме."""
    try:
        # Проверяем конфигурацию
        get_settings()

        logger.info("Запуск TradingView MCP сервера (stdio режим)")

        # Запускаем сервер
        async with stdio_server() as (read_stream, write_stream):
            init_options = server.create_initialization_options()
            await server.run(
                read_stream,
                write_stream,
                init_options,
            )

    except Exception as e:
        logger.error(f"Ошибка запуска MCP сервера: {e}")
        raise


def run_http_server(host: str, port: int, log_level: str) -> None:
    """Запускает MCP сервер в streamable-http режиме через FastAPI."""
    logger.info(
        "Запуск TradingView MCP сервера в режиме streamable-http (host=%s, port=%s, log_level=%s)",
        host,
        port,
        log_level,
    )
    
    from src.api import create_app
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


def build_config_snapshot() -> Dict[str, Any]:
    """Формирует человекочитаемое представление конфигурации."""
    snapshot: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    try:
        settings = get_settings()
        snapshot["config"] = {
            "host": settings.host,
            "port": settings.port,
            "log_level": settings.log_level,
            "default_transport": settings.default_transport,
            "debug": settings.debug,
        }
    except Exception as exc:
        snapshot["config"] = {"error": str(exc)}

    return snapshot


def main() -> None:
    """CLI точка входа для TradingView MCP сервера."""
    parser = argparse.ArgumentParser(description="TradingView MCP server")
    parser.add_argument("--host", help="HTTP хост для streamable-http транспорта")
    parser.add_argument(
        "--port", type=int, help="HTTP порт для streamable-http транспорта"
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Показать текущую конфигурацию и выйти",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Уровень логирования (например, INFO, DEBUG)",
    )
    args = parser.parse_args()

    log_level_name = args.log_level.upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger.setLevel(log_level)

    # Применяем CLI флаги к окружению
    env_overridden = False
    if args.host:
        os.environ["HOST"] = args.host
        env_overridden = True
    if args.port is not None:
        os.environ["PORT"] = str(args.port)
        env_overridden = True

    if args.print_config:
        snapshot = build_config_snapshot()
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
        return

    if (args.host is None) ^ (args.port is None):
        parser.error(
            "Для запуска HTTP режима необходимо указать оба флага --host и --port"
        )

    if args.host and args.port:
        run_http_server(args.host, args.port, log_level_name)
        return

    asyncio.run(run_stdio_server())


if __name__ == "__main__":
    main()
