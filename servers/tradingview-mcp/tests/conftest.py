"""Test configuration and fixtures for TradingView MCP."""

import pytest
from unittest.mock import Mock, patch
from tradingview_mcp.config import Settings


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    return Settings(
        log_level="INFO",
        host="127.0.0.1",
        port=3000
    )


@pytest.fixture
def mock_tradingview_ta():
    """Mock TradingView TA module."""
    with patch('tradingview_mcp.server.TRADINGVIEW_TA_AVAILABLE', True):
        with patch('tradingview_mcp.server.TA_Handler') as mock_handler:
            mock_instance = Mock()
            mock_instance.get_analysis.return_value = {
                'RECOMMENDATION': 'BUY',
                'BUY': 5,
                'SELL': 2,
                'NEUTRAL': 3
            }
            mock_handler.return_value = mock_instance
            yield mock_handler


@pytest.fixture
def mock_tradingview_screener():
    """Mock TradingView Screener module."""
    with patch('tradingview_mcp.server.TRADINGVIEW_SCREENER_AVAILABLE', True):
        with patch('tradingview_mcp.server.Query') as mock_query:
            mock_instance = Mock()
            mock_instance.scan.return_value = [
                {'symbol': 'BTCUSDT', 'exchange': 'BINANCE', 'change': 5.2},
                {'symbol': 'ETHUSDT', 'exchange': 'BINANCE', 'change': 3.1}
            ]
            mock_query.return_value = mock_instance
            yield mock_query
