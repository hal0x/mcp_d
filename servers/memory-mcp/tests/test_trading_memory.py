from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memory_mcp.mcp.adapters import MemoryServiceAdapter
from memory_mcp.mcp.schema import (
    GetSignalPerformanceRequest,
    SearchTradingPatternsRequest,
    StoreTradingSignalRequest,
)


def _build_request(timestamp: datetime | None = None) -> StoreTradingSignalRequest:
    ts = timestamp or datetime.now(timezone.utc)
    return StoreTradingSignalRequest(
        symbol="BTCUSDT",
        signal_type="momentum",
        direction="long",
        entry=48250.5,
        confidence=78.5,
        context={
            "strategy": "momentum",
            "telegram": {
                "chat_id": "12345",
                "chat_title": "HAL Signals",
                "chat_username": "hal_signals",
                "message_id": 678,
                "message_url": "https://t.me/hal_signals/678",
                "message_text": "Momentum long BTCUSDT @ 48.25k",
                "author": "HAL Bot",
            },
        },
        timestamp=ts,
    )


def test_store_trading_signal_creates_graph_links(temp_dir):
    db_path = temp_dir / "memory.db"
    adapter = MemoryServiceAdapter(db_path=str(db_path))
    try:
        request = _build_request()
        response = adapter.store_trading_signal(request)

        signal = response.signal
        assert signal.symbol == "BTCUSDT"
        assert signal.context["telegram"]["chat_title"] == "HAL Signals"

        graph = adapter.graph.graph
        signal_node = graph.nodes[signal.signal_id]
        assert signal_node["type"] == "TradingSignal"

        chat_node_id = "telegram:chat:12345"
        message_node_id = "telegram:msg:12345:678"
        assert chat_node_id in graph.nodes
        assert message_node_id in graph.nodes
        assert graph.nodes[chat_node_id]["properties"]["entity_type"] == "telegram_chat"
        assert graph.has_edge(signal.signal_id, chat_node_id)
        assert graph.has_edge(signal.signal_id, message_node_id)
        assert graph.has_edge(message_node_id, chat_node_id)
    finally:
        adapter.close()


def test_search_trading_patterns_and_performance(temp_dir):
    db_path = temp_dir / "trading.db"
    adapter = MemoryServiceAdapter(str(db_path))
    try:
        recent_request = _build_request()
        adapter.store_trading_signal(recent_request)

        older_ts = datetime.now(timezone.utc) - timedelta(days=40)
        older_request = _build_request(timestamp=older_ts)
        adapter.store_trading_signal(older_request)

        # recent timeframe should return only the fresh signal
        recent = adapter.search_trading_patterns(
            SearchTradingPatternsRequest(symbol="BTCUSDT", timeframe="recent", limit=5)
        )
        assert len(recent.signals) == 1

        # timeframe "all" should return both
        all_signals = adapter.search_trading_patterns(
            SearchTradingPatternsRequest(symbol="BTCUSDT", timeframe="all", limit=10)
        )
        assert len(all_signals.signals) == 2

        signal_id = recent.signals[0].signal_id
        performance = adapter.get_signal_performance(
            GetSignalPerformanceRequest(signal_id=signal_id)
        )
        assert performance.signal.signal_id == signal_id
        assert performance.performance is None
    finally:
        adapter.close()
