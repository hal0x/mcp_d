#!/usr/bin/env python3
"""
HTTP MCP сервер для Memory MCP.
Использует FastAPI с FastApiMCP для интеграции с FastMCP сервером.
"""

import logging
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp.transport.http import FastApiHttpSessionManager

# Добавляем src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Импортируем настройки
from memory_mcp.config import get_settings

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

# Загружаем настройки
settings = get_settings()
PORT = settings.port
HOST = settings.host
LOG_LEVEL = settings.log_level
SERVER_NAME = "memory-mcp"


def create_app() -> FastAPI:
    """Создает и настраивает FastAPI приложение с MCP интеграцией."""
    # Импортируем функцию для запуска фоновой индексации
    from memory_mcp.mcp.server import _start_background_indexing_if_enabled, _stop_background_indexing_on_shutdown
    import asyncio
    
    # Запускаем фоновую индексацию если включена
    try:
        _start_background_indexing_if_enabled()
    except Exception as e:
        logger.warning(f"Не удалось запустить фоновую индексацию при старте: {e}")
    
    # Регистрируем обработчик завершения
    import atexit
    def stop_on_exit():
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_stop_background_indexing_on_shutdown())
            else:
                asyncio.run(_stop_background_indexing_on_shutdown())
        except Exception as e:
            logger.error(f"Ошибка при остановке фоновой индексации: {e}")
    atexit.register(stop_on_exit)
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

    @app.get("/version")
    async def version():
        """Возвращает информацию о версии сервера и доступных возможностях."""
        from memory_mcp.mcp.server import get_version_payload
        return get_version_payload()

    # Настраиваем MCP интеграцию
    # Используем FastApiHttpSessionManager для интеграции стандартного MCP Server с FastAPI
    try:
        # Импортируем server из mcp_server (уже импортирован в начале файла)
        from mcp_server import server as mcp_server_instance
        
        # Проверяем, что инструменты зарегистрированы
        # Проверяем наличие обработчика list_tools
        if hasattr(mcp_server_instance, 'list_tools'):
            logger.info("Обработчик list_tools найден в сервере")
        else:
            logger.warning("Обработчик list_tools НЕ найден в сервере!")
        
        # Создаем HTTP transport для стандартного MCP Server
        http_transport = FastApiHttpSessionManager(mcp_server=mcp_server_instance)
        
        # Регистрируем MCP endpoint
        @app.post("/mcp")
        async def mcp_endpoint(request: Request):
            """MCP endpoint для обработки MCP запросов."""
            return await http_transport.handle_fastapi_request(request)
        
        logger.info("MCP сервер успешно подключен к FastAPI через FastApiHttpSessionManager")
        
    except Exception as e:
        logger.error(f"Ошибка при подключении MCP сервера: {e}", exc_info=True)
        # Продолжаем работу даже если MCP не подключен

    @app.on_event("shutdown")
    async def shutdown():
        """Cleanup при остановке сервера."""
        try:
            from memory_mcp.mcp.server import _adapter, _stop_background_indexing_on_shutdown
            if _adapter is not None:
                _adapter.close()
                logger.info("Memory adapter закрыт")
            # Останавливаем фоновую индексацию
            await _stop_background_indexing_on_shutdown()
        except Exception as e:
            logger.error(f"Ошибка при закрытии adapter: {e}")

    return app


if __name__ == "__main__":
    logger.info(f"Starting {SERVER_NAME} HTTP server on {HOST}:{PORT}")
    app = create_app()
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())
