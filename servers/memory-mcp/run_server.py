#!/usr/bin/env python3
"""
HTTP MCP сервер для Memory MCP.
Использует FastAPI с FastApiMCP для интеграции с FastMCP сервером.
"""

import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp.transport.http import FastApiHttpSessionManager

# Добавляем src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Импортируем server ДО создания FastAPI приложения, чтобы декораторы зарегистрировались
# Это важно для того, чтобы FastApiMCP мог найти зарегистрированные инструменты
# Импортируем через mcp_server.py модуль, чтобы server был в глобальном пространстве имен
try:
    import mcp_server  # noqa: F401
    # Убеждаемся, что server доступен
    server = mcp_server.server
    logger.info(f"MCP server импортирован: {server.name}, инструменты должны быть зарегистрированы")
except Exception as e:
    logger.warning(f"Не удалось импортировать MCP server: {e}", exc_info=True)

PORT = int(os.getenv("PORT", os.getenv("TG_DUMP_PORT", "8050")))
HOST = os.getenv("HOST", os.getenv("TG_DUMP_HOST", "0.0.0.0"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
SERVER_NAME = "memory-mcp"


def create_app() -> FastAPI:
    """Создает и настраивает FastAPI приложение с MCP интеграцией."""
    app = FastAPI(
        title="Memory MCP",
        version="1.0.0",
        description="MCP сервер для индексации и поиска по унифицированной памяти",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {"message": f"Welcome to {SERVER_NAME}", "status": "running"}

    @app.get("/health")
    async def health():
        return {"status": "ok", "server_name": SERVER_NAME}

    @app.get("/healthz")
    async def healthz():
        """Health check endpoint для Docker/K8s."""
        return {"status": "ok"}

    # Настраиваем MCP интеграцию
    # Используем FastApiHttpSessionManager для интеграции стандартного MCP Server с FastAPI
    try:
        # Импортируем server из mcp_server (уже импортирован в начале файла)
        from mcp_server import server as mcp_server_instance
        
        # Создаем HTTP transport для стандартного MCP Server
        http_transport = FastApiHttpSessionManager(mcp_server=mcp_server_instance)
        
        # Регистрируем MCP endpoint
        @app.post("/mcp")
        async def mcp_endpoint(request: Request):
            """MCP endpoint для обработки MCP запросов."""
            return await http_transport.handle_request(request)
        
        logger.info("MCP сервер успешно подключен к FastAPI через FastApiHttpSessionManager")
        
    except Exception as e:
        logger.error(f"Ошибка при подключении MCP сервера: {e}", exc_info=True)
        # Продолжаем работу даже если MCP не подключен

    @app.on_event("shutdown")
    async def shutdown():
        """Cleanup при остановке сервера."""
        try:
            from memory_mcp.mcp.server import _adapter
            if _adapter is not None:
                _adapter.close()
                logger.info("Memory adapter закрыт")
        except Exception as e:
            logger.error(f"Ошибка при закрытии adapter: {e}")

    return app


if __name__ == "__main__":
    logger.info(f"Starting {SERVER_NAME} HTTP server on {HOST}:{PORT}")
    app = create_app()
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())
