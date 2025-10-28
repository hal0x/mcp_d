"""Tests for screener tools."""

import pytest
from unittest.mock import patch, Mock
from tradingview_mcp.server import top_gainers_batch, top_losers_batch, bollinger_scan_batch


class TestTopGainersBatch:
    """Test top_gainers_batch tool."""

    def test_top_gainers_batch_default_params(self, mock_tradingview_screener):
        """Test top_gainers_batch with default parameters."""
        requests = [{"exchange": "KUCOIN", "timeframe": "15m", "limit": 25}]
        result = top_gainers_batch(requests)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert "results" in result[0]

    def test_top_gainers_batch_custom_params(self, mock_tradingview_screener):
        """Test top_gainers_batch with custom parameters."""
        requests = [{"exchange": "BINANCE", "timeframe": "1h", "limit": 10}]
        result = top_gainers_batch(requests)
        
        assert isinstance(result, list)
        assert len(result) == 1

    def test_top_gainers_batch_invalid_exchange(self):
        """Test top_gainers_batch with invalid exchange."""
        requests = [{"exchange": "INVALID", "timeframe": "15m", "limit": 10}]
        result = top_gainers_batch(requests)
        
        assert isinstance(result, list)
        assert len(result) == 1

    def test_top_gainers_batch_invalid_timeframe(self):
        """Test top_gainers_batch with invalid timeframe."""
        requests = [{"exchange": "KUCOIN", "timeframe": "invalid", "limit": 10}]
        result = top_gainers_batch(requests)
        
        assert isinstance(result, list)
        assert len(result) == 1


class TestTopLosersBatch:
    """Test top_losers_batch tool."""

    def test_top_losers_batch_default_params(self, mock_tradingview_screener):
        """Test top_losers_batch with default parameters."""
        requests = [{"exchange": "KUCOIN", "timeframe": "15m", "limit": 25}]
        result = top_losers_batch(requests)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert "results" in result[0]

    def test_top_losers_batch_custom_params(self, mock_tradingview_screener):
        """Test top_losers_batch with custom parameters."""
        requests = [{"exchange": "KUCOIN", "timeframe": "4h", "limit": 15}]
        result = top_losers_batch(requests)
        
        assert isinstance(result, list)
        assert len(result) == 1

    def test_top_losers_batch_invalid_exchange(self):
        """Test top_losers_batch with invalid exchange."""
        requests = [{"exchange": "INVALID", "timeframe": "15m", "limit": 10}]
        result = top_losers_batch(requests)
        
        assert isinstance(result, list)
        assert len(result) == 1


class TestBollingerScanBatch:
    """Test bollinger_scan_batch tool."""

    def test_bollinger_scan_batch_default_params(self, mock_tradingview_ta):
        """Test bollinger_scan_batch with default parameters."""
        requests = [{"exchange": "KUCOIN", "timeframe": "4h", "bbw_threshold": 0.04, "limit": 50}]
        result = bollinger_scan_batch(requests)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert "results" in result[0]

    def test_bollinger_scan_batch_custom_params(self, mock_tradingview_ta):
        """Test bollinger_scan_batch with custom parameters."""
        requests = [{"exchange": "BINANCE", "timeframe": "1h", "bbw_threshold": 0.03, "limit": 20}]
        result = bollinger_scan_batch(requests)
        
        assert isinstance(result, list)
        assert len(result) == 1

    def test_bollinger_scan_batch_invalid_exchange(self):
        """Test bollinger_scan_batch with invalid exchange."""
        requests = [{"exchange": "INVALID", "timeframe": "4h", "bbw_threshold": 0.04, "limit": 50}]
        result = bollinger_scan_batch(requests)
        
        assert isinstance(result, list)
        assert len(result) == 1

    def test_bollinger_scan_batch_invalid_timeframe(self):
        """Test bollinger_scan_batch with invalid timeframe."""
        requests = [{"exchange": "KUCOIN", "timeframe": "invalid", "bbw_threshold": 0.04, "limit": 50}]
        result = bollinger_scan_batch(requests)
        
        assert isinstance(result, list)
        assert len(result) == 1


class TestScreenerIntegration:
    """Integration tests for screener tools."""

    def test_all_tools_return_lists(self):
        """Test that all screener tools return lists."""
        tools = [top_gainers_batch, top_losers_batch, bollinger_scan_batch]
        
        for tool in tools:
            requests = [{"exchange": "KUCOIN", "timeframe": "15m", "limit": 5}]
            result = tool(requests)
            assert isinstance(result, list)

    def test_tools_with_valid_exchanges(self):
        """Test tools with valid exchanges."""
        valid_exchanges = ["KUCOIN", "BINANCE", "BYBIT"]
        
        for exchange in valid_exchanges:
            requests = [{"exchange": exchange, "timeframe": "15m", "limit": 5}]
            result = top_gainers_batch(requests)
            assert isinstance(result, list)

    def test_tools_with_valid_timeframes(self):
        """Test tools with valid timeframes."""
        valid_timeframes = ["5m", "15m", "1h", "4h", "1D"]
        
        for timeframe in valid_timeframes:
            requests = [{"exchange": "KUCOIN", "timeframe": timeframe, "limit": 5}]
            result = top_gainers_batch(requests)
            assert isinstance(result, list)