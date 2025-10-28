"""Конфигурация pytest для тестов."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_binance_client():
    """Мок для Binance клиента."""
    client = MagicMock()
    client.get_account = AsyncMock()
    client.get_account_trades = AsyncMock()
    client.get_open_orders = AsyncMock()
    client.get_all_orders = AsyncMock()
    client.get_ticker = AsyncMock()
    client.get_klines = AsyncMock()
    client.get_order_book = AsyncMock()
    client.get_exchange_info = AsyncMock()
    client.get_server_time = AsyncMock()
    client.create_order = AsyncMock()
    return client


@pytest.fixture
def mock_config():
    """Мок для конфигурации."""
    return {
        "api_key": "test_api_key",
        "api_secret": "test_api_secret",
        "demo_trading": False,
        "host": "0.0.0.0",
        "port": 8000,
    }
