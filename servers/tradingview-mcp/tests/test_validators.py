"""Tests for validator functions."""

import pytest
from tradingview_mcp.core.utils.validators import (
    sanitize_exchange, 
    sanitize_timeframe,
    EXCHANGE_SCREENER,
    ALLOWED_TIMEFRAMES
)


class TestSanitizeExchange:
    """Test sanitize_exchange function."""
    
    def test_valid_exchanges(self):
        """Test sanitize_exchange with valid exchanges."""
        for exchange in EXCHANGE_SCREENER:
            result = sanitize_exchange(exchange)
            assert result == exchange.upper()
    
    def test_case_insensitive(self):
        """Test sanitize_exchange is case insensitive."""
        assert sanitize_exchange("binance") == "BINANCE"
        assert sanitize_exchange("Binance") == "BINANCE"
        assert sanitize_exchange("BINANCE") == "BINANCE"
    
    def test_invalid_exchange(self):
        """Test sanitize_exchange with invalid exchange."""
        with pytest.raises(ValueError, match="Invalid exchange"):
            sanitize_exchange("INVALID_EXCHANGE")
    
    def test_none_exchange(self):
        """Test sanitize_exchange with None."""
        with pytest.raises(ValueError, match="Invalid exchange"):
            sanitize_exchange(None)
    
    def test_empty_exchange(self):
        """Test sanitize_exchange with empty string."""
        with pytest.raises(ValueError, match="Invalid exchange"):
            sanitize_exchange("")
    
    def test_whitespace_exchange(self):
        """Test sanitize_exchange with whitespace."""
        with pytest.raises(ValueError, match="Invalid exchange"):
            sanitize_exchange("   ")


class TestSanitizeTimeframe:
    """Test sanitize_timeframe function."""
    
    def test_valid_timeframes(self):
        """Test sanitize_timeframe with valid timeframes."""
        for timeframe in ALLOWED_TIMEFRAMES:
            result = sanitize_timeframe(timeframe)
            assert result == timeframe
    
    def test_case_insensitive(self):
        """Test sanitize_timeframe is case insensitive."""
        assert sanitize_timeframe("1h") == "1h"
        assert sanitize_timeframe("1H") == "1h"
        assert sanitize_timeframe("1D") == "1d"
    
    def test_invalid_timeframe(self):
        """Test sanitize_timeframe with invalid timeframe."""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            sanitize_timeframe("invalid")
    
    def test_none_timeframe(self):
        """Test sanitize_timeframe with None."""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            sanitize_timeframe(None)
    
    def test_empty_timeframe(self):
        """Test sanitize_timeframe with empty string."""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            sanitize_timeframe("")
    
    def test_whitespace_timeframe(self):
        """Test sanitize_timeframe with whitespace."""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            sanitize_timeframe("   ")


class TestConstants:
    """Test validator constants."""
    
    def test_exchange_screener_not_empty(self):
        """Test EXCHANGE_SCREENER is not empty."""
        assert len(EXCHANGE_SCREENER) > 0
        assert isinstance(EXCHANGE_SCREENER, set)
    
    def test_allowed_timeframes_not_empty(self):
        """Test ALLOWED_TIMEFRAMES is not empty."""
        assert len(ALLOWED_TIMEFRAMES) > 0
        assert isinstance(ALLOWED_TIMEFRAMES, set)
    
    def test_exchange_screener_values(self):
        """Test EXCHANGE_SCREENER contains expected exchanges."""
        expected_exchanges = {"BINANCE", "KUCOIN", "BYBIT", "OKX", "GATEIO", "HUOBI"}
        for exchange in expected_exchanges:
            assert exchange in EXCHANGE_SCREENER
    
    def test_allowed_timeframes_values(self):
        """Test ALLOWED_TIMEFRAMES contains expected timeframes."""
        expected_timeframes = {"1m", "5m", "15m", "1h", "4h", "1d"}
        for timeframe in expected_timeframes:
            assert timeframe in ALLOWED_TIMEFRAMES
