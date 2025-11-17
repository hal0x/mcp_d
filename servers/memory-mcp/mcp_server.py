#!/usr/bin/env python3
"""MCP сервер для Memory MCP - точка входа для HTTP режима."""

# Импортируем server из модуля, чтобы он был доступен глобально
# Это нужно для того, чтобы FastApiMCP мог найти зарегистрированные инструменты
from memory_mcp.mcp.server import server  # noqa: F401

# Экспортируем server для использования в run_server.py
__all__ = ["server"]


