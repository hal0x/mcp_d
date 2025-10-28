#!/usr/bin/env python3
"""MCP сервер для Binance API."""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from importlib import metadata
from typing import Any, Dict, List, Optional, Tuple

import uvicorn

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from pydantic import BaseModel

from src.config import Config, get_config
from src.models import (
    CreateOrderRequest,
    CancelOrderRequest,
    CreateMarginOrderRequest,
    CancelMarginOrderRequest,
    CreatePositionRequest,
    SafetyRule,
    StopLossConfig,
    AlertConfig,
    RiskManagementRule,
)
from src.services import (
    AccountService,
    MarketService,
    OrderService,
    ExchangeService,
    PortfolioService,
    FuturesService,
    MarginService,
    OCOService,
    BatchService,
    RiskManagementService,
    AlertService,
    TelegramService,
)

# Настройка логирования (конфигурируется в main)
logger = logging.getLogger(__name__)

# Создаем MCP сервер
server = Server("binance-mcp")

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
    """Returns a curated list of available Binance tools."""
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
            name="get_account_info",
            description="Fetch Binance account information.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_trade_fee_batch",
            description="Retrieve trading fees for multiple symbols.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "description": 'Список торговых пар, например ["BTCUSDT", "ETHUSDT"]',
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 20,
                    }
                },
                "required": ["symbols"],
            },
        ),
        Tool(
            name="get_order_book_batch",
            description="Fetch order book depth for multiple symbols.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "description": 'Список торговых пар, например ["BTCUSDT", "ETHUSDT"]',
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 20,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Глубина книги ордеров (5-5000)",
                        "minimum": 5,
                        "maximum": 5000,
                        "default": 100,
                    },
                },
                "required": ["symbols"],
            },
        ),
        Tool(
            name="get_klines_batch",
            description="Retrieve candlestick data for multiple symbols.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "description": 'Список торговых пар, например ["BTCUSDT", "ETHUSDT"]',
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 20,
                    },
                    "interval": {
                        "type": "string",
                        "description": "Интервал свечей (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)",
                        "default": "1h",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Количество свечей (1-1000)",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 100,
                    },
                    "start_time": {
                        "type": "integer",
                        "description": "Начальная метка времени (мс с эпохи)",
                    },
                    "end_time": {
                        "type": "integer",
                        "description": "Конечная метка времени (мс с эпохи)",
                    },
                },
                "required": ["symbols"],
            },
        ),
        Tool(
            name="get_avg_price_batch",
            description="Return average prices for multiple symbols.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "description": 'Список торговых пар, например ["BTCUSDT", "ETHUSDT"]',
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 50,
                    }
                },
                "required": ["symbols"],
            },
        ),
        Tool(
            name="get_recent_trades_batch",
            description="List recent trades for multiple symbols.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "description": 'Список торговых пар, например ["BTCUSDT", "ETHUSDT"]',
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 20,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Максимум сделок на символ (1-1000)",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 20,
                    },
                },
                "required": ["symbols"],
            },
        ),
        Tool(
            name="get_open_orders",
            description="List open orders, optionally filtered by symbol.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Опциональная торговая пара",
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="get_order_history",
            description="Return historical orders for a trading pair.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Торговая пара"},
                    "limit": {
                        "type": "integer",
                        "description": "Количество ордеров (1-1000)",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 10,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="get_trade_history",
            description="Return trade history for a trading pair.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Торговая пара"},
                    "limit": {
                        "type": "integer",
                        "description": "Количество сделок (1-1000)",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 10,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="get_margin_account",
            description="Fetch information about the margin account.",
            inputSchema={
                "type": "object",
                "properties": {
                    "isolated": {
                        "type": "boolean",
                        "description": "Использовать изолированную маржу",
                        "default": False,
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Символ для изолированной маржи",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get_margin_orders",
            description="Return margin order history for a symbol.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Торговая пара"},
                    "limit": {
                        "type": "integer",
                        "description": "Количество записей",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 10,
                    },
                    "is_isolated": {
                        "type": "boolean",
                        "description": "Изолированная маржа",
                        "default": False,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="create_margin_order_batch",
            description="Create multiple margin orders in a single request.",
            inputSchema={
                "type": "object",
                "properties": {
                    "orders": {
                        "type": "array",
                        "description": "Список маржинальных ордеров для создания",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {
                                    "type": "string",
                                    "description": "Торговая пара",
                                },
                                "side": {
                                    "type": "string",
                                    "description": "BUY или SELL",
                                },
                                "type": {"type": "string", "description": "Тип ордера"},
                                "quantity": {
                                    "type": "number",
                                    "description": "Количество базового актива",
                                },
                                "quote_order_qty": {
                                    "type": "number",
                                    "description": "Количество в котируемой валюте",
                                },
                                "price": {"type": "number", "description": "Цена"},
                                "stop_price": {
                                    "type": "number",
                                    "description": "Стоп-цена",
                                },
                                "time_in_force": {
                                    "type": "string",
                                    "description": "GTC, IOC, FOK",
                                },
                                "is_isolated": {
                                    "type": "boolean",
                                    "description": "Изолированная маржа",
                                },
                                "side_effect_type": {
                                    "type": "string",
                                    "description": "NO_SIDE_EFFECT, MARGIN_BUY, AUTO_REPAY",
                                },
                                "new_client_order_id": {
                                    "type": "string",
                                    "description": "Идентификатор клиента",
                                },
                                "new_order_resp_type": {
                                    "type": "string",
                                    "description": "ACK, RESULT, FULL",
                                },
                            },
                            "required": ["symbol", "side", "type"],
                        },
                    }
                },
                "required": ["orders"],
            },
        ),
        Tool(
            name="cancel_margin_order_batch",
            description="Cancel multiple margin orders in one request.",
            inputSchema={
                "type": "object",
                "properties": {
                    "orders": {
                        "type": "array",
                        "description": "Список маржинальных ордеров для отмены",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {
                                    "type": "string",
                                    "description": "Торговая пара",
                                },
                                "order_id": {
                                    "type": "integer",
                                    "description": "Идентификатор ордера (опционально)",
                                },
                                "client_order_id": {
                                    "type": "string",
                                    "description": "Клиентский идентификатор ордера (опционально)",
                                },
                                "new_client_order_id": {
                                    "type": "string",
                                    "description": "Новый идентификатор отмены (опционально)",
                                },
                                "is_isolated": {
                                    "type": "boolean",
                                    "description": "Является ли ордер изолированным",
                                },
                            },
                            "required": ["symbol"],
                        },
                    }
                },
                "required": ["orders"],
            },
        ),
        Tool(
            name="get_margin_trades",
            description="Return historical margin trades for a symbol.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Торговая пара"},
                    "is_isolated": {
                        "type": "boolean",
                        "description": "Изолированная маржа",
                        "default": False,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Количество записей",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 10,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="create_margin_oco_order",
            description="Create a margin OCO order (limit + stop pair).",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Торговая пара"},
                    "side": {"type": "string", "description": "BUY или SELL"},
                    "quantity": {"type": "number", "description": "Количество"},
                    "price": {
                        "type": "number",
                        "description": "Лимит-цена для лимитного плеча",
                    },
                    "stop_price": {"type": "number", "description": "Стоп-цена"},
                    "stop_limit_price": {
                        "type": "number",
                        "description": "Лимит-цена стоп-лимит ноги",
                    },
                    "stop_limit_time_in_force": {
                        "type": "string",
                        "description": "GTC, IOC, FOK",
                    },
                    "is_isolated": {
                        "type": "boolean",
                        "description": "Изолированная маржа",
                    },
                    "limit_client_order_id": {"type": "string"},
                    "stop_client_order_id": {"type": "string"},
                    "list_client_order_id": {"type": "string"},
                },
                "required": ["symbol", "side", "quantity", "price", "stop_price"],
            },
        ),
        Tool(
            name="cancel_margin_oco_order",
            description="Cancel an existing margin OCO order.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Торговая пара"},
                    "order_list_id": {
                        "type": "integer",
                        "description": "ID OCO списка",
                    },
                    "list_client_order_id": {
                        "type": "string",
                        "description": "Клиентский ID списка",
                    },
                    "is_isolated": {
                        "type": "boolean",
                        "description": "Изолированная маржа",
                    },
                    "new_client_order_id": {
                        "type": "string",
                        "description": "Новый ID отмены",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="get_open_margin_oco_orders",
            description="List open margin OCO orders.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Торговая пара"},
                    "is_isolated": {
                        "type": "boolean",
                        "description": "Изолированная маржа",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get_margin_oco_order",
            description="Fetch a margin OCO order by identifiers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Торговая пара (для изолированной маржи)",
                    },
                    "order_list_id": {
                        "type": "integer",
                        "description": "ID OCO списка",
                    },
                    "list_client_order_id": {
                        "type": "string",
                        "description": "Клиентский ID списка",
                    },
                    "is_isolated": {
                        "type": "boolean",
                        "description": "Изолированная маржа",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="create_oco_order",
            description="Create a spot OCO order.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Торговая пара"},
                    "side": {"type": "string", "description": "BUY или SELL"},
                    "quantity": {"type": "number", "description": "Количество"},
                    "price": {"type": "number", "description": "Лимит-цена"},
                    "stop_price": {"type": "number", "description": "Стоп-цена"},
                    "stop_limit_price": {
                        "type": "number",
                        "description": "Стоп-лимит цена",
                    },
                    "stop_limit_time_in_force": {
                        "type": "string",
                        "description": "GTC, IOC, FOK",
                    },
                    "limit_client_order_id": {"type": "string"},
                    "stop_client_order_id": {"type": "string"},
                    "list_client_order_id": {"type": "string"},
                },
                "required": ["symbol", "side", "quantity", "price", "stop_price"],
            },
        ),
        Tool(
            name="cancel_oco_order",
            description="Cancel a spot OCO order.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Торговая пара"},
                    "order_list_id": {
                        "type": "integer",
                        "description": "ID OCO списка",
                    },
                    "list_client_order_id": {
                        "type": "string",
                        "description": "Клиентский ID списка",
                    },
                    "new_client_order_id": {
                        "type": "string",
                        "description": "Идентификатор отмены",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="get_open_oco_orders",
            description="List open spot OCO orders.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="change_futures_margin_type",
            description="Change futures margin type (ISOLATED or CROSSED).",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Торговая пара, например BTCUSDT",
                    },
                    "margin_type": {
                        "type": "string",
                        "description": "Новый тип маржи",
                        "enum": ["ISOLATED", "CROSSED"],
                    },
                },
                "required": ["symbol", "margin_type"],
            },
        ),
        Tool(
            name="create_order_batch",
            description="Create multiple spot or futures orders at once.",
            inputSchema={
                "type": "object",
                "properties": {
                    "orders": {
                        "type": "array",
                        "description": "Список ордеров для создания",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {
                                    "type": "string",
                                    "description": "Торговая пара",
                                },
                                "side": {
                                    "type": "string",
                                    "description": "BUY или SELL",
                                },
                                "type": {
                                    "type": "string",
                                    "description": "Тип ордера (MARKET/LIMIT)",
                                },
                                "quantity": {
                                    "type": "number",
                                    "description": "Количество базового актива",
                                },
                                "price": {
                                    "type": "number",
                                    "description": "Цена (для лимитных ордеров)",
                                },
                                "time_in_force": {
                                    "type": "string",
                                    "description": "GTC, IOC, FOK (для лимитных ордеров)",
                                },
                                "new_client_order_id": {
                                    "type": "string",
                                    "description": "Клиентский идентификатор",
                                },
                                "iceberg_qty": {
                                    "type": "number",
                                    "description": "Количество для айсберг-ордера",
                                },
                                "reduce_only": {
                                    "type": "boolean",
                                    "description": "Только уменьшаем позицию (фьючерсы)",
                                },
                                "close_position": {
                                    "type": "boolean",
                                    "description": "Закрыть позицию полностью (фьючерсы)",
                                },
                                "price_protect": {
                                    "type": "boolean",
                                    "description": "Защитить цену (фьючерсы)",
                                },
                                "new_order_resp_type": {
                                    "type": "string",
                                    "description": "ACK, RESULT, FULL",
                                },
                            },
                            "required": ["symbol", "side", "type"],
                        },
                    }
                },
                "required": ["orders"],
            },
        ),
        Tool(
            name="get_order",
            description="Fetch order details by ID or client ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Торговая пара, например BTCUSDT",
                    },
                    "order_id": {
                        "type": "integer",
                        "description": "Идентификатор ордера",
                    },
                    "client_order_id": {
                        "type": "string",
                        "description": "Клиентский идентификатор ордера",
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="cancel_order_batch",
            description="Cancel multiple orders by ID or client order ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "orders": {
                        "type": "array",
                        "description": "Список ордеров для отмены",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {
                                    "type": "string",
                                    "description": "Торговая пара",
                                },
                                "order_id": {
                                    "type": "integer",
                                    "description": "Идентификатор ордера (опционально)",
                                },
                                "client_order_id": {
                                    "type": "string",
                                    "description": "Клиентский идентификатор ордера (опционально)",
                                },
                            },
                            "required": ["symbol"],
                        },
                    }
                },
                "required": ["orders"],
            },
        ),
        Tool(
            name="get_exchange_info",
            description="Return exchange information and symbol metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Опциональная торговая пара",
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="get_server_time",
            description="Return the current Binance server time.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_simple_balance",
            description="Return a simplified balance summary for the account.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_futures_positions_batch",
            description="Fetch futures positions for a batch of symbols.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список торговых пар (максимум 20)",
                        "maxItems": 20,
                    }
                },
                "required": ["symbols"],
            },
        ),
        Tool(
            name="get_portfolio_overview",
            description="Return consolidated portfolio overview with balances and P&L.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_tickers_batch",
            description="Retrieve price and 24h stats for multiple symbols.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список торговых пар (максимум 20)",
                        "maxItems": 20,
                    }
                },
                "required": ["symbols"],
            },
        ),
        Tool(
            name="change_leverage_batch",
            description="Update leverage for multiple symbols in one call.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol_leverage_map": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 125,
                        },
                        "description": "Словарь {символ: плечо} (максимум 20 пар)",
                        "maxProperties": 20,
                    }
                },
                "required": ["symbol_leverage_map"],
            },
        ),
        Tool(
            name="create_positions_batch",
            description="Open multiple futures positions with auto leverage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "positions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {
                                    "type": "string",
                                    "description": "Торговая пара",
                                },
                                "side": {
                                    "type": "string",
                                    "enum": ["BUY", "SELL"],
                                    "description": "Направление",
                                },
                                "quantity": {
                                    "type": "number",
                                    "description": "Количество",
                                },
                                "leverage": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 125,
                                    "description": "Плечо",
                                },
                            },
                            "required": ["symbol", "side", "quantity", "leverage"],
                        },
                        "description": "Список позиций для создания (максимум 20)",
                        "maxItems": 20,
                    }
                },
                "required": ["positions"],
            },
        ),
        Tool(
            name="close_positions_batch",
            description="Close multiple positions, auto-detecting side per symbol.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список торговых пар для закрытия (максимум 20)",
                        "maxItems": 20,
                    },
                    "percentage": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 100,
                        "description": "Процент закрытия позиции",
                    },
                },
                "required": ["symbols"],
            },
        ),
        Tool(
            name="get_available_pairs",
            description="List tradable symbol pairs with optional filters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "description": "Статус торговли (например, TRADING)",
                            },
                            "quoteAsset": {
                                "type": "string",
                                "description": "Котируемая валюта (например, USDT)",
                            },
                        },
                        "description": "Фильтры для поиска пар",
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="check_trading_limits",
            description="Check trading limits and availability for symbols.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список торговых пар для проверки (максимум 20)",
                        "maxItems": 20,
                    }
                },
                "required": ["symbols"],
            },
        ),
        Tool(
            name="portfolio_safety_check",
            description="Evaluate portfolio safety using technical indicators.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список торговых пар для проверки (максимум 20)",
                        "maxItems": 20,
                    },
                    "safety_rules": {
                        "type": "object",
                        "properties": {
                            "max_rsi_short": {"type": "number", "default": 30.0},
                            "min_rsi_long": {"type": "number", "default": 75.0},
                            "min_adx": {"type": "number", "default": 18.0},
                            "max_drawdown": {"type": "number", "default": 5.0},
                        },
                        "description": "Правила безопасности",
                    },
                    "auto_close_unsafe": {
                        "type": "boolean",
                        "default": False,
                        "description": "Автоматически закрывать небезопасные позиции",
                    },
                },
                "required": ["symbols"],
            },
        ),
        Tool(
            name="manage_stop_losses",
            description="Automate stop-loss management for active positions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список торговых пар (максимум 20)",
                        "maxItems": 20,
                    },
                    "stop_loss_config": {
                        "type": "object",
                        "properties": {
                            "stop_loss_type": {
                                "type": "string",
                                "enum": ["fixed", "trailing"],
                                "default": "trailing",
                            },
                            "trail_percentage": {"type": "number", "default": 2.0},
                            "update_frequency": {"type": "string", "default": "1h"},
                            "max_loss_percent": {"type": "number", "default": 5.0},
                        },
                        "description": "Конфигурация стоп-лоссов",
                    },
                },
                "required": ["symbols"],
            },
        ),
        Tool(
            name="setup_portfolio_alerts",
            description="Configure portfolio monitoring alerts (Telegram only).",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список торговых пар (максимум 20)",
                        "maxItems": 20,
                    },
                    "alerts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "alert_type": {
                                    "type": "string",
                                    "enum": [
                                        "drawdown",
                                        "profit",
                                        "volume_spike",
                                        "price_level",
                                        "rsi_extreme",
                                    ],
                                },
                                "threshold": {"type": "number"},
                                "notification_method": {
                                    "type": "string",
                                    "enum": ["telegram"],
                                    "default": "telegram",
                                },
                                "message_template": {
                                    "type": "string",
                                    "description": "Кастомный шаблон сообщения (опционально)",
                                },
                            },
                            "required": ["alert_type", "threshold"],
                        },
                        "description": "Список алертов для настройки",
                    },
                    "telegram_chat_id": {
                        "type": "string",
                        "description": "ID чата Telegram (опционально, используется из конфигурации если не указан)",
                    },
                },
                "required": ["symbols", "alerts"],
            },
        ),
        Tool(
            name="test_telegram_notification",
            description="Send a test notification to Telegram.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Текст сообщения для отправки",
                    },
                    "chat_id": {
                        "type": "string",
                        "description": "ID чата Telegram (опционально)",
                    },
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="auto_risk_management",
            description="Run automated risk management across the portfolio.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список торговых пар (максимум 20)",
                        "maxItems": 20,
                    },
                    "rules": {
                        "type": "object",
                        "properties": {
                            "max_portfolio_loss": {"type": "number", "default": -10.0},
                            "max_position_loss": {"type": "number", "default": -5.0},
                            "profit_taking": {"type": "number", "default": 15.0},
                            "auto_close_on_loss": {"type": "boolean", "default": True},
                        },
                        "description": "Правила управления рисками",
                    },
                },
                "required": ["symbols"],
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

        if name == "get_account_info":
            account_info = await AccountService.get_account_info()
            return _format_tool_response(account_info.model_dump())

        elif name == "get_trade_fee_batch":
            symbols = arguments.get("symbols")
            if not isinstance(symbols, list) or not symbols:
                raise ValueError("Parameter symbols must be a non-empty list")
            trade_fees = await AccountService.get_trade_fee_batch(symbols)
            payload = [item.model_dump() for item in trade_fees]
            return _format_tool_response(payload)

        elif name == "get_order_book_batch":
            symbols = arguments.get("symbols")
            if not isinstance(symbols, list) or not symbols:
                raise ValueError("Parameter symbols must be a non-empty list")
            limit = arguments.get("limit", 100)
            order_books = await MarketService.get_order_book_batch(symbols, limit)
            payload = [item.model_dump() for item in order_books]
            return _format_tool_response(payload)

        elif name == "get_klines_batch":
            symbols = arguments.get("symbols")
            if not isinstance(symbols, list) or not symbols:
                raise ValueError("Parameter symbols must be a non-empty list")
            interval = arguments.get("interval", "1h")
            limit = arguments.get("limit", 100)
            start_time = arguments.get("start_time")
            end_time = arguments.get("end_time")
            klines_data = await MarketService.get_klines_batch(
                symbols,
                interval,
                limit,
                start_time=start_time,
                end_time=end_time,
            )
            payload = [item.model_dump() for item in klines_data]
            return _format_tool_response(payload)
        elif name == "get_avg_price_batch":
            symbols = arguments.get("symbols")
            if not isinstance(symbols, list) or not symbols:
                raise ValueError("Parameter symbols must be a non-empty list")
            avg_prices = await MarketService.get_avg_price_batch(symbols)
            payload = [item.model_dump() for item in avg_prices]
            return _format_tool_response(payload)
        elif name == "get_recent_trades_batch":
            symbols = arguments.get("symbols")
            if not isinstance(symbols, list) or not symbols:
                raise ValueError("Parameter symbols must be a non-empty list")
            limit = arguments.get("limit", 20)
            trades_data = await MarketService.get_public_trades_batch(symbols, limit)
            payload = [
                {
                    "symbol": item.symbol,
                    "trades": [trade.model_dump() for trade in item.trades],
                }
                for item in trades_data
            ]
            return _format_tool_response(payload)

        elif name == "get_open_orders":
            symbol = arguments.get("symbol")
            orders = await OrderService.get_open_orders(symbol)
            payload = [item.model_dump() for item in orders]
            return _format_tool_response(payload)

        elif name == "get_order_history":
            symbol = arguments.get("symbol")
            if not symbol:
                raise ValueError("Параметр symbol обязателен")
            limit = arguments.get("limit", 10)
            orders = await OrderService.get_order_history(symbol, limit)
            payload = [item.model_dump() for item in orders]
            return _format_tool_response(payload)

        elif name == "get_trade_history":
            symbol = arguments.get("symbol")
            if not symbol:
                raise ValueError("Параметр symbol обязателен")
            limit = arguments.get("limit", 10)
            trades = await OrderService.get_trade_history(symbol, limit)
            payload = [item.model_dump() for item in trades]
            return _format_tool_response(payload)
        elif name == "create_order_batch":
            orders_raw = arguments.get("orders")
            if not isinstance(orders_raw, list) or not orders_raw:
                raise ValueError("Parameter orders must be a non-empty list")
            spot_orders: List[CreateOrderRequest] = [
                CreateOrderRequest(**item) for item in orders_raw
            ]
            batch_results = await OrderService.create_order_batch(spot_orders)
            payload = [item.model_dump() for item in batch_results]
            return _format_tool_response(payload)
        elif name == "cancel_order_batch":
            orders_raw = arguments.get("orders")
            if not isinstance(orders_raw, list) or not orders_raw:
                raise ValueError("Parameter orders must be a non-empty list")
            requests: List[CancelOrderRequest] = [
                CancelOrderRequest(**item) for item in orders_raw
            ]
            batch_results = await OrderService.cancel_orders_batch(requests)
            payload = [item.model_dump() for item in batch_results]
            return _format_tool_response(payload)
        elif name == "get_margin_account":
            isolated = arguments.get("isolated", False)
            symbol = arguments.get("symbol")
            margin_account = await MarginService.get_margin_account(
                isolated=isolated, symbol=symbol
            )
            return _format_tool_response(margin_account.model_dump())
        elif name == "get_margin_orders":
            symbol = arguments.get("symbol")
            if not symbol:
                raise ValueError("Параметр symbol обязателен")
            limit = arguments.get("limit", 10)
            is_isolated = arguments.get("is_isolated")
            margin_orders_list = await MarginService.get_margin_orders(
                symbol, limit=limit, is_isolated=is_isolated
            )
            payload = [item.model_dump() for item in margin_orders_list]
            return _format_tool_response(payload)
        elif name == "create_margin_order_batch":
            orders_raw = arguments.get("orders")
            if not isinstance(orders_raw, list) or not orders_raw:
                raise ValueError("Parameter orders must be a non-empty list")
            margin_orders: List[CreateMarginOrderRequest] = [
                CreateMarginOrderRequest(**item) for item in orders_raw
            ]
            batch_results = await MarginService.create_margin_order_batch(margin_orders)
            payload = [item.model_dump() for item in batch_results]
            return _format_tool_response(payload)
        elif name == "cancel_margin_order_batch":
            orders_raw = arguments.get("orders")
            if not isinstance(orders_raw, list) or not orders_raw:
                raise ValueError("Parameter orders must be a non-empty list")
            margin_requests = [CancelMarginOrderRequest(**item) for item in orders_raw]
            batch_results = await MarginService.cancel_margin_order_batch(
                margin_requests
            )
            payload = [item.model_dump() for item in batch_results]
            return _format_tool_response(payload)
        elif name == "get_margin_trades":
            symbol = arguments.get("symbol")
            if not symbol:
                raise ValueError("Параметр symbol обязателен")
            is_isolated = arguments.get("is_isolated")
            limit = arguments.get("limit", 10)
            margin_trades = await MarginService.get_margin_trades(
                symbol, is_isolated=is_isolated, limit=limit
            )
            payload = [item.model_dump() for item in margin_trades]
            return _format_tool_response(payload)
        elif name == "create_margin_oco_order":
            oco_order = await MarginService.create_margin_oco_order(arguments)
            return _format_tool_response(oco_order.model_dump())
        elif name == "cancel_margin_oco_order":
            oco_order = await MarginService.cancel_margin_oco_order(arguments)
            return _format_tool_response(oco_order.model_dump())
        elif name == "get_open_margin_oco_orders":
            symbol = arguments.get("symbol")
            is_isolated = arguments.get("is_isolated")
            oco_orders = await MarginService.get_open_margin_oco_orders(
                symbol=symbol, is_isolated=is_isolated
            )
            payload = [item.model_dump() for item in oco_orders]
            return _format_tool_response(payload)
        elif name == "get_margin_oco_order":
            oco_order = await MarginService.get_margin_oco_order(arguments)
            return _format_tool_response(oco_order.model_dump())
        elif name == "create_oco_order":
            oco_order = await OCOService.create_oco_order(arguments)
            return _format_tool_response(oco_order.model_dump())
        elif name == "cancel_oco_order":
            oco_order = await OCOService.cancel_oco_order(arguments)
            return _format_tool_response(oco_order.model_dump())
        elif name == "get_open_oco_orders":
            oco_orders = await OCOService.get_open_oco_orders()
            payload = [item.model_dump() for item in oco_orders]
            return _format_tool_response(payload)
        elif name == "change_futures_margin_type":
            symbol = arguments.get("symbol")
            margin_type = arguments.get("margin_type")
            if not symbol or not margin_type:
                raise ValueError("Параметры symbol и margin_type обязательны")
            margin_type_change = await FuturesService.change_margin_type(
                symbol, margin_type
            )
            return _format_tool_response(margin_type_change.model_dump())
        elif name == "get_order":
            symbol = arguments.get("symbol")
            if not symbol:
                raise ValueError("Параметр symbol обязателен")
            order_id = arguments.get("order_id")
            client_order_id = arguments.get("client_order_id")
            order_details = await OrderService.get_order(
                symbol, order_id, client_order_id
            )
            return _format_tool_response(order_details.model_dump())
        elif name == "get_exchange_info":
            symbol = arguments.get("symbol")
            exchange_info = await ExchangeService.get_exchange_info(symbol)
            return _format_tool_response(exchange_info.model_dump())

        elif name == "get_server_time":
            server_time = await ExchangeService.get_server_time()
            return _format_tool_response(server_time.model_dump())

        elif name == "get_simple_balance":
            result = await PortfolioService.get_simple_balance()
            return _format_tool_response(result)

        elif name == "get_futures_positions_batch":
            symbols = arguments.get("symbols", [])
            if not symbols:
                raise ValueError("Параметр symbols обязателен")
            futures_positions = await BatchService.get_futures_positions_batch(symbols)
            payload = [item.model_dump() for item in futures_positions]
            return _format_tool_response(payload)

        elif name == "get_portfolio_overview":
            portfolio_overview = await BatchService.get_portfolio_overview()
            return _format_tool_response(portfolio_overview.model_dump())

        elif name == "get_tickers_batch":
            symbols = arguments.get("symbols", [])
            if not symbols:
                raise ValueError("Параметр symbols обязателен")
            tickers = await BatchService.get_tickers_batch(symbols)
            payload = [item.model_dump() for item in tickers]
            return _format_tool_response(payload)

        elif name == "change_leverage_batch":
            symbol_leverage_map = arguments.get("symbol_leverage_map", {})
            if not symbol_leverage_map:
                raise ValueError("Параметр symbol_leverage_map обязателен")
            leverage_changes = await BatchService.change_leverage_batch(
                symbol_leverage_map
            )
            payload = [item.model_dump() for item in leverage_changes]
            return _format_tool_response(payload)

        elif name == "create_positions_batch":
            positions_data = arguments.get("positions", [])
            if not positions_data:
                raise ValueError("Параметр positions обязателен")
            positions = [CreatePositionRequest(**pos) for pos in positions_data]
            batch_results = await BatchService.create_positions_batch(positions)
            payload = [item.model_dump() for item in batch_results]
            return _format_tool_response(payload)

        elif name == "close_positions_batch":
            symbols = arguments.get("symbols", [])
            percentage = arguments.get("percentage", 100.0)
            if not symbols:
                raise ValueError("Параметр symbols обязателен")
            batch_results = await BatchService.close_positions_batch(
                symbols, percentage
            )
            payload = [item.model_dump() for item in batch_results]
            return _format_tool_response(payload)

        elif name == "get_available_pairs":
            filters = arguments.get("filters")
            available_pairs = await ExchangeService.get_available_pairs(filters)
            payload = [item.model_dump() for item in available_pairs]
            return _format_tool_response(payload)

        elif name == "check_trading_limits":
            symbols = arguments.get("symbols", [])
            if not symbols:
                raise ValueError("Параметр symbols обязателен")
            trading_limits = await ExchangeService.check_trading_limits(symbols)
            payload = [item.model_dump() for item in trading_limits]
            return _format_tool_response(payload)

        elif name == "portfolio_safety_check":
            symbols = arguments.get("symbols", [])
            if not symbols:
                raise ValueError("Параметр symbols обязателен")
            safety_rules = arguments.get("safety_rules")
            auto_close_unsafe = arguments.get("auto_close_unsafe", False)

            safety_rules_obj = SafetyRule(**safety_rules) if safety_rules else None
            safety_results = await RiskManagementService.portfolio_safety_check(
                symbols, safety_rules_obj, auto_close_unsafe
            )
            payload = [item.model_dump() for item in safety_results]
            return _format_tool_response(payload)

        elif name == "manage_stop_losses":
            symbols = arguments.get("symbols", [])
            if not symbols:
                raise ValueError("Параметр symbols обязателен")
            stop_loss_config = arguments.get("stop_loss_config")

            stop_loss_config_obj = (
                StopLossConfig(**stop_loss_config) if stop_loss_config else None
            )
            stop_loss_results = await RiskManagementService.manage_stop_losses(
                symbols, stop_loss_config_obj
            )
            payload = [item.model_dump() for item in stop_loss_results]
            return _format_tool_response(payload)

        elif name == "setup_portfolio_alerts":
            symbols = arguments.get("symbols", [])
            alerts = arguments.get("alerts", [])
            telegram_chat_id = arguments.get("telegram_chat_id")
            if not symbols or not alerts:
                raise ValueError("Параметры symbols и alerts обязательны")

            alerts_obj = [AlertConfig(**alert) for alert in alerts]
            alert_results = await AlertService.setup_portfolio_alerts(
                symbols, alerts_obj, telegram_chat_id
            )
            payload = [item.model_dump() for item in alert_results]
            return _format_tool_response(payload)

        elif name == "test_telegram_notification":
            message = arguments.get("message", "")
            chat_id = arguments.get("chat_id")
            if not message:
                raise ValueError("Параметр message обязателен")

            from src.models import TelegramNotification

            notification = TelegramNotification(
                chat_id=chat_id or "", message=message, parse_mode="Markdown"
            )
            success = await TelegramService.send_notification(notification)
            payload = [{"success": success, "message": message}]
            return _format_tool_response(payload)

        elif name == "auto_risk_management":
            symbols = arguments.get("symbols", [])
            if not symbols:
                raise ValueError("Параметр symbols обязателен")
            rules = arguments.get("rules")

            rules_obj = RiskManagementRule(**rules) if rules else None
            risk_results = await AlertService.auto_risk_management(symbols, rules_obj)
            return _format_tool_response(risk_results)

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
    config: Optional[Config] = None

    try:
        config = get_config()
    except Exception as exc:  # pragma: no cover - конфигурация проверяется при запуске
        status = "degraded"
        error = str(exc)

    client_info = get_client_info()

    payload: Dict[str, Any] = {
        "status": status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": client_info.get("mode", "UNKNOWN"),
        "services": {
            "account": True,
            "market": True,
            "orders": True,
            "portfolio": True,
        },
    }

    if config:
        payload["config"] = {
            "demo_trading": config.demo_trading,
            "host": config.host,
            "port": config.port,
            "api_key_present": bool(config.api_key),
            "demo_api_key_present": bool(config.demo_api_key),
        }
    else:
        payload["config"] = None

    if error:
        payload["error"] = error

    return payload


def get_version_payload() -> Dict[str, Any]:
    """Возвращает информацию о версии сервера."""
    try:
        version = metadata.version("binance-mcp")
    except metadata.PackageNotFoundError:
        version = "0.0.0"

    client_info = get_client_info()

    return {
        "name": "binance-mcp",
        "version": version,
        "mode": client_info.get("mode", "UNKNOWN"),
        "features": [
            "account",
            "market",
            "orders",
            "portfolio",
            "futures",
            "margin",
            "oco",
            "batch",
        ],
    }


async def run_stdio_server() -> None:
    """Запускает MCP сервер в stdio режиме."""
    try:
        # Проверяем конфигурацию
        get_config()
        client_info = get_client_info()

        mode = client_info["mode"]
        logger.info(f"Запуск Binance MCP сервера (режим={mode})")
        logger.info(f"API ключ: {client_info['api_key_prefix']}")

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
        "Запуск Binance MCP сервера в режиме streamable-http (host=%s, port=%s, log_level=%s)",
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
        "client": get_client_info(),
    }

    try:
        config = get_config()
    except Exception as exc:
        snapshot["config"] = {"error": str(exc)}
    else:
        snapshot["config"] = {
            "demo_trading": config.demo_trading,
            "host": config.host,
            "port": config.port,
            "api_key_present": bool(config.api_key),
            "demo_api_key_present": bool(config.demo_api_key),
        }

    return snapshot


def main() -> None:
    """CLI точка входа для Binance Native MCP сервера."""
    parser = argparse.ArgumentParser(description="Binance Native MCP server")
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
        default=os.environ.get("BINANCE_MCP_LOG_LEVEL")
        or os.environ.get("LOG_LEVEL", "INFO"),
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

    if os.environ.get("BINANCE_MCP_LOG_LEVEL") != log_level_name:
        os.environ["BINANCE_MCP_LOG_LEVEL"] = log_level_name
    if os.environ.get("LOG_LEVEL") != log_level_name:
        os.environ["LOG_LEVEL"] = log_level_name

    # Применяем CLI флаги к окружению, чтобы конфигурация читала актуальные значения
    env_overridden = False
    if args.host:
        os.environ["BINANCE_MCP_HOST"] = args.host
        os.environ["HOST"] = args.host
        env_overridden = True
    if args.port is not None:
        os.environ["BINANCE_MCP_PORT"] = str(args.port)
        os.environ["PORT"] = str(args.port)
        env_overridden = True
    if env_overridden:
        get_config.cache_clear()

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


def get_client_info() -> dict:
    """Получает информацию о клиенте."""
    try:
        from src.client import get_client_info as _get_client_info

        return _get_client_info()
    except Exception:
        return {"mode": "UNKNOWN", "api_key_prefix": "..."}


if __name__ == "__main__":
    main()
