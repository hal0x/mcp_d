"""Tests for health and version tools."""

import pytest
from datetime import datetime
from tradingview_mcp.server import health, version


def test_health_tool():
    """Test health tool returns correct structure."""
    result = health()
    
    assert isinstance(result, dict)
    assert "status" in result
    assert "timestamp" in result
    assert "services" in result
    
    assert result["status"] == "healthy"
    assert isinstance(result["timestamp"], str)
    assert isinstance(result["services"], dict)
    
    # Check timestamp is valid ISO format
    datetime.fromisoformat(result["timestamp"])


def test_health_tool_services():
    """Test health tool includes service status."""
    result = health()
    
    services = result["services"]
    assert "tradingview_ta" in services
    assert "tradingview_screener" in services
    
    # Values should be boolean
    assert isinstance(services["tradingview_ta"], bool)
    assert isinstance(services["tradingview_screener"], bool)


def test_version_tool():
    """Test version tool returns correct structure."""
    result = version()
    
    assert isinstance(result, dict)
    assert "version" in result
    assert "mode" in result
    assert "features" in result
    
    assert isinstance(result["version"], str)
    assert isinstance(result["mode"], str)
    assert isinstance(result["features"], list)
    
    # Check features list contains expected items
    expected_features = ["screener", "analysis", "patterns"]
    for feature in expected_features:
        assert feature in result["features"]


def test_version_tool_values():
    """Test version tool returns expected values."""
    result = version()
    
    assert result["version"] == "0.1.0"
    assert result["mode"] == "production"
