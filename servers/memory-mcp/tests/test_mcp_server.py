"""
🧪 Тестирование нового MCP сервера для унифицированной памяти

Тестирует все доступные инструменты нового MCP сервера (src/memory_mcp/mcp/server.py).
Этот файл заменяет старый test_mcp_server.py, который тестировал TelegramDumpMCP.
"""

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
    """Тест инструмента health."""
    result = get_health_payload()
    assert result is not None
    assert "status" in result
    assert result["status"] in ["healthy", "degraded"]
    assert "services" in result
    assert "config" in result


def test_version_tool():
    """Тест инструмента version."""
    result = get_version_payload()
    assert result is not None
    assert "name" in result
    assert "version" in result
    assert result["name"] == "memory-mcp"
    assert "features" in result
    assert isinstance(result["features"], list)
    assert len(result["features"]) > 0


def test_ingest_records(mcp_server_adapter):
    """Тест инжеста записей."""
    records = [
        MemoryRecordPayload(
            record_id="test-1",
            source="test",
            content="Bitcoin достиг $120,000",
            timestamp=datetime.now(timezone.utc),
            tags=["crypto", "bitcoin"],
            entities=[],
            attachments=[],
            metadata={},
        ),
        MemoryRecordPayload(
            record_id="test-2",
            source="test",
            content="капитализация 1.5 млрд долларов",
            timestamp=datetime.now(timezone.utc),
            tags=["crypto"],
            entities=[],
            attachments=[],
            metadata={},
        ),
    ]
    
    result = mcp_server_adapter.ingest(records)
    assert result is not None
    assert result.records_ingested == 2


def test_search_memory(mcp_server_adapter):
    """Тест поиска по памяти."""
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
        MemoryRecordPayload(
            record_id="search-test-2",
            source="test",
            content="рост на 15% за месяц",
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
    # Поиск может возвращать контент с HTML-разметкой для выделения найденных слов
    assert "Bitcoin" in result.results[0].content
    assert "$120,000" in result.results[0].content


def test_search_memory_with_filters(mcp_server_adapter):
    """Тест поиска с фильтрами."""
    # Инжестим данные с разными тегами
    records = [
        MemoryRecordPayload(
            record_id="filter-test-1",
            source="telegram",
            content="Сообщение из Telegram",
            timestamp=datetime.now(timezone.utc),
            tags=["telegram", "chat"],
            entities=[],
            attachments=[],
            metadata={},
        ),
        MemoryRecordPayload(
            record_id="filter-test-2",
            source="file",
            content="Сообщение из файла",
            timestamp=datetime.now(timezone.utc),
            tags=["file"],
            entities=[],
            attachments=[],
            metadata={},
        ),
    ]
    
    mcp_server_adapter.ingest(records)
    
    # Поиск с фильтром по источнику
    request = SearchRequest(query="Сообщение", top_k=5, source="telegram")
    result = mcp_server_adapter.search(request)
    assert result is not None
    assert len(result.results) > 0
    assert all(r.source == "telegram" for r in result.results)
    
    # Поиск с фильтром по тегам
    request = SearchRequest(query="Сообщение", top_k=5, tags=["file"])
    result = mcp_server_adapter.search(request)
    assert result is not None
    assert len(result.results) > 0
    assert "file" in result.results[0].metadata.get("tags", [])


def test_fetch_record(mcp_server_adapter):
    """Тест получения записи по ID."""
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
            metadata={"key": "value"},
        ),
    ]
    
    mcp_server_adapter.ingest(records)
    
    # Теперь получаем запись
    request = FetchRequest(record_id=record_id)
    result = mcp_server_adapter.fetch(request)
    assert result is not None
    assert result.record is not None
    assert result.record.record_id == record_id
    assert result.record.content == "Test content for fetch"
    assert result.record.metadata.get("key") == "value"


def test_store_trading_signal(mcp_server_adapter):
    """Тест сохранения торгового сигнала."""
    signal = StoreTradingSignalRequest(
        symbol="BTCUSDT",
        signal_type="momentum",
        direction="long",
        entry=48250.5,
        confidence=78.5,
        context={"strategy": "momentum", "timeframe": "1h"},
        timestamp=datetime.now(timezone.utc),
    )
    
    result = mcp_server_adapter.store_trading_signal(signal)
    assert result is not None
    assert result.signal.symbol == "BTCUSDT"
    assert result.signal.direction == "long"
    assert result.signal.entry == 48250.5


def test_search_trading_patterns(mcp_server_adapter):
    """Тест поиска торговых паттернов."""
    # Сначала создаём несколько сигналов
    signals = [
        StoreTradingSignalRequest(
            symbol="BTCUSDT",
            signal_type="momentum",
            direction="long",
            entry=48250.5,
            confidence=78.5,
            context={"strategy": "momentum"},
            timestamp=datetime.now(timezone.utc),
        ),
        StoreTradingSignalRequest(
            symbol="ETHUSDT",
            signal_type="breakout",
            direction="short",
            entry=2500.0,
            confidence=65.0,
            context={"strategy": "breakout"},
            timestamp=datetime.now(timezone.utc),
        ),
    ]
    
    for signal in signals:
        mcp_server_adapter.store_trading_signal(signal)
    
    # Теперь ищем по символу
    request = SearchTradingPatternsRequest(query="BTCUSDT", limit=10)
    result = mcp_server_adapter.search_trading_patterns(request)
    assert result is not None
    assert len(result.signals) > 0
    assert all(s.symbol == "BTCUSDT" for s in result.signals)


def test_get_signal_performance(mcp_server_adapter):
    """Тест получения производительности сигнала."""
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
    assert result.signal.symbol == "BTCUSDT"
    # Производительность может быть None, если сигнал ещё не закрыт
    assert result.performance is None or isinstance(result.performance.pnl, (int, float))


def test_ingest_scraped_content(mcp_server_adapter):
    """Тест инжеста скрапленного контента."""
    content = ScrapedContentRequest(
        url="https://example.com/article",
        title="Test Article",
        content="This is test content from a scraped article about Bitcoin",
        metadata={"author": "Test Author"},
        source="web",
        tags=["article", "bitcoin"],
        entities=["Bitcoin"],
    )
    
    result = mcp_server_adapter.ingest_scraped_content(content)
    assert result is not None
    assert result.record_id is not None
    
    # Проверяем, что контент можно найти через поиск
    request = SearchRequest(query="Bitcoin", top_k=5)
    search_result = mcp_server_adapter.search(request)
    assert search_result is not None
    assert len(search_result.results) > 0


@pytest.mark.asyncio
async def test_call_tool_unknown_tool():
    """Тест обработки неизвестного инструмента."""
    with pytest.raises(RuntimeError, match="Неизвестный инструмент"):
        await call_tool("unknown_tool", {})


def test_ingest_duplicates(mcp_server_adapter):
    """Тест обработки дубликатов при инжесте."""
    record = MemoryRecordPayload(
        record_id="duplicate-test",
        source="test",
        content="Test content",
        timestamp=datetime.now(timezone.utc),
        tags=[],
        entities=[],
        attachments=[],
        metadata={},
    )
    
    # Первый инжест
    result1 = mcp_server_adapter.ingest([record])
    assert result1.records_ingested == 1
    assert result1.duplicates_skipped == 0
    
    # Второй инжест того же record_id
    result2 = mcp_server_adapter.ingest([record])
    assert result2.records_ingested == 0
    assert result2.duplicates_skipped == 1

