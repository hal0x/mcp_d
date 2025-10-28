#!/usr/bin/env python3
"""
Универсальный ASGI сервер для MCP серверов.
Поддерживает HTTP/WS транспорт с неблокирующим планировщиком.
"""

import os
import asyncio
import logging
import signal
import sys
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Попытка импорта FastMCP HTTP транспорта
try:
    from fastmcp import FastMCP
    from fastmcp.transport.http import create_app as create_fastmcp_app
    USE_FASTMCP_HTTP = True
except ImportError:
    try:
        from mcp.server.fastapi import create_fastapi_app
        USE_FASTMCP_HTTP = False
    except ImportError:
        print("ERROR: Neither fastmcp nor mcp.server.fastapi available")
        sys.exit(1)

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Конфигурация
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
SCHEDULER_ENABLED = os.getenv("SCHEDULER_ENABLED", "false").lower() == "true"
INTERVAL_SEC = int(os.getenv("INTERVAL_SEC", "30"))
SERVER_NAME = os.getenv("SERVER_NAME", "mcp-server")
CORS_ENABLED = os.getenv("CORS_ENABLED", "false").lower() == "true"

# Глобальные переменные
mcp_server: Optional[Any] = None
scheduler_task: Optional[asyncio.Task] = None
start_time = asyncio.get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else 0


class SchedulerManager:
    """Менеджер неблокирующего планировщика."""
    
    def __init__(self, server_name: str):
        self.server_name = server_name
        self.running = False
        self.task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Запуск планировщика."""
        if self.running:
            logger.warning(f"Scheduler for {self.server_name} already running")
            return
        
        self.running = True
        self.task = asyncio.create_task(self._scheduler_loop())
        logger.info(f"Scheduler started for {self.server_name}")
    
    async def stop(self):
        """Остановка планировщика."""
        if not self.running:
            return
        
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info(f"Scheduler stopped for {self.server_name}")
    
    async def _scheduler_loop(self):
        """Основной цикл планировщика."""
        while self.running:
            try:
                await self._run_scheduled_tasks()
            except Exception as e:
                logger.error(f"Scheduler error in {self.server_name}: {e}", exc_info=True)
            
            try:
                await asyncio.sleep(INTERVAL_SEC)
            except asyncio.CancelledError:
                break
    
    async def _run_scheduled_tasks(self):
        """Выполнение запланированных задач."""
        # Переопределяется в конкретных серверах
        logger.debug(f"Running scheduled tasks for {self.server_name}")


def create_mcp_server() -> Any:
    """Создание MCP сервера в зависимости от типа."""
    if USE_FASTMCP_HTTP:
        return FastMCP(SERVER_NAME)
    else:
        # Для совместимости с mcp.server.fastapi
        from mcp.server import Server
        return Server(SERVER_NAME)


def register_tools(server: Any):
    """Регистрация инструментов MCP сервера."""
    # Этот метод должен быть переопределен в каждом сервере
    logger.warning("No tools registered - override register_tools()")


def create_scheduler_manager() -> SchedulerManager:
    """Создание менеджера планировщика."""
    return SchedulerManager(SERVER_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    global scheduler_task
    
    # Startup
    logger.info(f"Starting {SERVER_NAME} on {HOST}:{PORT}")
    
    if SCHEDULER_ENABLED:
        scheduler_manager = create_scheduler_manager()
        await scheduler_manager.start()
        scheduler_task = scheduler_manager.task
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {SERVER_NAME}")
    
    if scheduler_task:
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    """Создание FastAPI приложения."""
    global mcp_server
    
    # Создание MCP сервера
    mcp_server = create_mcp_server()
    
    # Регистрация инструментов
    register_tools(mcp_server)
    
    # Создание FastAPI приложения
    if USE_FASTMCP_HTTP:
        app = create_fastmcp_app(mcp_server)
    else:
        app = create_fastapi_app(mcp_server)
    
    # Добавление lifespan
    app.router.lifespan_context = lifespan
    
    # CORS middleware
    if CORS_ENABLED:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        uptime = asyncio.get_event_loop().time() - start_time
        
        health_data = {
            "status": "healthy",
            "server": SERVER_NAME,
            "uptime": uptime,
            "scheduler_enabled": SCHEDULER_ENABLED,
            "transport": "http",
            "version": "1.0.0"
        }
        
        # Проверка зависимостей
        dependencies = {}
        
        # Проверка Redis (если используется)
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                import redis.asyncio as redis
                client = redis.from_url(redis_url)
                await client.ping()
                dependencies["redis"] = {"status": "healthy"}
                await client.aclose()
            except Exception as e:
                dependencies["redis"] = {"status": "unhealthy", "error": str(e)}
                health_data["status"] = "degraded"
        
        # Проверка PostgreSQL (если используется)
        postgres_url = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")
        if postgres_url:
            try:
                import asyncpg
                conn = await asyncpg.connect(postgres_url)
                await conn.execute("SELECT 1")
                await conn.close()
                dependencies["postgres"] = {"status": "healthy"}
            except Exception as e:
                dependencies["postgres"] = {"status": "unhealthy", "error": str(e)}
                health_data["status"] = "degraded"
        
        health_data["dependencies"] = dependencies
        
        return health_data
    
    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Basic metrics endpoint."""
        return {
            "server": SERVER_NAME,
            "uptime": asyncio.get_event_loop().time() - start_time,
            "scheduler_running": scheduler_task is not None and not scheduler_task.done(),
            "transport": "http"
        }
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with server info."""
        return {
            "server": SERVER_NAME,
            "transport": "http",
            "endpoints": {
                "health": "/health",
                "metrics": "/metrics",
                "mcp": "/mcp",
                "ws": "/ws" if USE_FASTMCP_HTTP else None
            },
            "version": "1.0.0"
        }
    
    return app


def setup_signal_handlers():
    """Настройка обработчиков сигналов для graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Основная функция запуска сервера."""
    setup_signal_handlers()
    
    app = create_app()
    
    logger.info(f"Starting {SERVER_NAME} server")
    logger.info(f"Transport: HTTP/WS")
    logger.info(f"Host: {HOST}")
    logger.info(f"Port: {PORT}")
    logger.info(f"Scheduler enabled: {SCHEDULER_ENABLED}")
    logger.info(f"CORS enabled: {CORS_ENABLED}")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=os.getenv("LOG_LEVEL", "INFO").lower(),
        access_log=True,
        reload=False
    )


if __name__ == "__main__":
    main()
