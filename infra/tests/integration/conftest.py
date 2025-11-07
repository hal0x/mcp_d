"""Фикстуры для integration тестов"""
import pytest
import asyncio
import os
import sys
import logging
from typing import AsyncGenerator

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(__file__))

from utils.mcp_client import MCPClient

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def event_loop():
    """Создание event loop для async тестов"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def binance_client() -> AsyncGenerator[MCPClient, None]:
    """Клиент для binance-mcp"""
    url = os.getenv("BINANCE_MCP_URL", "http://binance-mcp:8000")
    async with MCPClient(url) as client:
        # Ждем готовности сервиса
        await wait_for_service(client, timeout=60)
        yield client

@pytest.fixture(scope="session")
async def tradingview_client() -> AsyncGenerator[MCPClient, None]:
    """Клиент для tradingview-mcp"""
    url = os.getenv("TRADINGVIEW_MCP_URL", "http://tradingview-mcp:8060")
    async with MCPClient(url) as client:
        await wait_for_service(client, timeout=60)
        yield client

@pytest.fixture(scope="session")
async def memory_client() -> AsyncGenerator[MCPClient, None]:
    """Клиент для memory-mcp"""
    url = os.getenv("MEMORY_MCP_URL", "http://memory-mcp:8050")
    async with MCPClient(url) as client:
        await wait_for_service(client, timeout=60)
        yield client

@pytest.fixture(scope="session")
async def shell_client() -> AsyncGenerator[MCPClient, None]:
    """Клиент для shell-mcp"""
    url = os.getenv("SHELL_MCP_URL", "http://shell-mcp:8070")
    async with MCPClient(url) as client:
        await wait_for_service(client, timeout=60)
        yield client

@pytest.fixture(scope="session")
async def backtesting_client() -> AsyncGenerator[MCPClient, None]:
    """Клиент для backtesting-mcp"""
    url = os.getenv("BACKTESTING_MCP_URL", "http://backtesting-mcp:8082")
    async with MCPClient(url) as client:
        await wait_for_service(client, timeout=60)
        yield client

async def wait_for_service(client: MCPClient, timeout: float = 60):
    """Ожидание готовности сервиса"""
    start = asyncio.get_event_loop().time()
    logger.info(f"Waiting for service {client.base_url} to be ready...")
    
    while True:
        if await client.health_check():
            logger.info(f"Service {client.base_url} is ready!")
            return
        
        elapsed = asyncio.get_event_loop().time() - start
        if elapsed > timeout:
            raise TimeoutError(f"Service {client.base_url} not ready after {timeout}s")
        
        logger.debug(f"Service {client.base_url} not ready yet, waiting... ({elapsed:.1f}s)")
        await asyncio.sleep(2)

@pytest.fixture
def test_symbols():
    """Тестовые символы для торговли"""
    return ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

@pytest.fixture
def test_intervals():
    """Тестовые интервалы"""
    return ["1h", "4h", "1d"]

@pytest.fixture
def sample_klines():
    """Пример исторических данных"""
    return [
        {
            "symbol": "BTCUSDT",
            "openTime": 1698000000000,
            "open": "35000.00",
            "high": "36000.00",
            "low": "34000.00",
            "close": "35500.00",
            "volume": "100.5",
            "closeTime": 1698003599999,
            "quoteAssetVolume": "3550000.00",
            "numberOfTrades": 1500,
            "takerBuyBaseAssetVolume": "50.25",
            "takerBuyQuoteAssetVolume": "1775000.00",
            "ignore": "0"
        }
    ]
