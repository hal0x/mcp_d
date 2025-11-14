"""Tests for the new MCP server using standard MCP Server."""

import os
import pytest
from datetime import datetime, timezone

from memory_mcp.mcp.adapters import MemoryServiceAdapter
from memory_mcp.mcp.schema import (
    FetchRequest,
    MemoryRecordPayload,
    SearchRequest,
    StoreTradingSignalRequest,
    SearchTradingPatternsRequest,
    GetSignalPerformanceRequest,
    ScrapedContentRequest,
)
from memory_mcp.mcp.server import get_health_payload, get_version_payload, call_tool


def test_health_tool():
    """Test the health tool."""
    # Временно устанавливаем путь к БД для теста
    original_db_path = os.environ.get("MEMORY_DB_PATH")
    try:
        os.environ["MEMORY_DB_PATH"] = ":memory:"  # Используем in-memory БД
        
        result = get_health_payload()
        assert result is not None
        assert "status" in result
        assert result["status"] in ["healthy", "degraded"]
        assert "services" in result
        assert "config" in result
    finally:
        if original_db_path:
            os.environ["MEMORY_DB_PATH"] = original_db_path
        elif "MEMORY_DB_PATH" in os.environ:
            del os.environ["MEMORY_DB_PATH"]


def test_version_tool():
    """Test the version tool."""
    result = get_version_payload()
    assert result is not None
    assert "name" in result
    assert "version" in result
    assert result["name"] == "memory-mcp"
    assert "features" in result


def test_ingest_records(mcp_server_adapter):
    """Test ingesting records."""
    records = [
        MemoryRecordPayload(
            record_id="test-1",
            source="test",
            content="Test message 1",
            timestamp=datetime.now(timezone.utc),
            tags=["test"],
            entities=[],
            attachments=[],
            metadata={},
        ),
        MemoryRecordPayload(
            record_id="test-2",
            source="test",
            content="Test message 2",
            timestamp=datetime.now(timezone.utc),
            tags=["test"],
            entities=[],
            attachments=[],
            metadata={},
        ),
    ]
    
    result = mcp_server_adapter.ingest(records)
    assert result is not None
    assert result.records_ingested == 2


def test_search_memory(mcp_server_adapter):
    """Test searching memory."""
    # Сначала инжестим данные
    records = [
        MemoryRecordPayload(
            record_id="search-test-1",
            source="test",
            content="Bitcoin достиг $120,000",
            timestamp=datetime.now(timezone.utc),
            tags=["crypto"],
            entities=[],
            attachments=[],
            metadata={},
        ),
    ]
    
    mcp_server_adapter.ingest(records)
    
    # Теперь ищем
    request = SearchRequest(query="Bitcoin", top_k=5)
    result = mcp_server_adapter.search(request)
    assert result is not None
    assert len(result.results) > 0


def test_fetch_record(mcp_server_adapter):
    """Test fetching a record by ID."""
    # Сначала инжестим запись
    record_id = "fetch-test-1"
    records = [
        MemoryRecordPayload(
            record_id=record_id,
            source="test",
            content="Test content for fetch",
            timestamp=datetime.now(timezone.utc),
            tags=[],
            entities=[],
            attachments=[],
            metadata={},
        ),
    ]
    
    mcp_server_adapter.ingest(records)
    
    # Теперь получаем запись
    request = FetchRequest(record_id=record_id)
    result = mcp_server_adapter.fetch(request)
    assert result is not None
    assert result.record is not None
    assert result.record.record_id == record_id


def test_store_trading_signal(mcp_server_adapter):
    """Test storing a trading signal."""
    signal = StoreTradingSignalRequest(
        symbol="BTCUSDT",
        signal_type="momentum",
        direction="long",
        entry=48250.5,
        confidence=78.5,
        context={"strategy": "momentum"},
        timestamp=datetime.now(timezone.utc),
    )
    
    result = mcp_server_adapter.store_trading_signal(signal)
    assert result is not None
    assert result.signal.symbol == "BTCUSDT"


def test_search_trading_patterns(mcp_server_adapter):
    """Test searching trading patterns."""
    # Сначала создаём сигнал
    signal = StoreTradingSignalRequest(
        symbol="BTCUSDT",
        signal_type="momentum",
        direction="long",
        entry=48250.5,
        confidence=78.5,
        context={"strategy": "momentum"},
        timestamp=datetime.now(timezone.utc),
    )
    
    mcp_server_adapter.store_trading_signal(signal)
    
    # Теперь ищем
    request = SearchTradingPatternsRequest(query="BTCUSDT", limit=10)
    result = mcp_server_adapter.search_trading_patterns(request)
    assert result is not None
    assert len(result.signals) > 0


def test_get_signal_performance(mcp_server_adapter):
    """Test getting signal performance."""
    # Сначала создаём сигнал
    signal = StoreTradingSignalRequest(
        symbol="BTCUSDT",
        signal_type="momentum",
        direction="long",
        entry=48250.5,
        confidence=78.5,
        context={"strategy": "momentum"},
        timestamp=datetime.now(timezone.utc),
    )
    
    store_result = mcp_server_adapter.store_trading_signal(signal)
    signal_id = store_result.signal.signal_id
    
    # Теперь получаем производительность
    request = GetSignalPerformanceRequest(signal_id=signal_id)
    result = mcp_server_adapter.get_signal_performance(request)
    assert result is not None
    assert result.signal.signal_id == signal_id


def test_ingest_scraped_content(mcp_server_adapter):
    """Test ingesting scraped content."""
    content = ScrapedContentRequest(
        url="https://example.com/article",
        title="Test Article",
        content="This is test content from a scraped article",
        metadata={},
        source="web",
        tags=["article"],
        entities=[],
    )
    
    result = mcp_server_adapter.ingest_scraped_content(content)
    assert result is not None
    assert result.record_id is not None


@pytest.mark.asyncio
async def test_call_tool_unknown_tool():
    """Test that unknown tool raises an error."""
    with pytest.raises(RuntimeError, match="Неизвестный инструмент"):
        await call_tool("unknown_tool", {})

