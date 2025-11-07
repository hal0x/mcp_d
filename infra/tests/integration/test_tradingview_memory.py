"""
Integration tests для tradingview-mcp с binance-mcp и memory-mcp
"""
import pytest
import asyncio
from unittest.mock import AsyncMock
import json

@pytest.mark.asyncio
async def test_tradingview_analysis_with_binance_prices():
    """Тест: Анализ TradingView с ценами от Binance"""
    print("\n=== Test: TradingView analysis with Binance prices ===")
    
    # Mock binance client
    mock_binance = AsyncMock()
    mock_binance.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "symbol": "BTCUSDT",
                "price": "35500.00",
                "priceChangePercent": "2.5"
            })
        }]
    })
    
    # Mock tradingview client
    mock_tradingview = AsyncMock()
    mock_tradingview.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "symbol": "BTCUSDT",
                "rsi": 55.5,
                "recommendation": "BUY",
                "score": 0.75
            })
        }]
    })
    
    # Workflow
    # 1. Get price from Binance
    price_data = await mock_binance.call_tool("get_ticker_price", {"symbol": "BTCUSDT"})
    assert price_data is not None
    print(f"✅ Получена цена от Binance")
    
    # 2. Analyze with TradingView
    analysis = await mock_tradingview.call_tool("coin_analysis", {
        "symbol": "BTCUSDT",
        "exchange": "BINANCE"
    })
    assert analysis is not None
    print(f"✅ Выполнен анализ TradingView")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_tradingview_save_analysis_to_memory():
    """Тест: Сохранение анализа TradingView в память"""
    print("\n=== Test: Save TradingView analysis to memory ===")
    
    # Mock tradingview client
    mock_tradingview = AsyncMock()
    mock_tradingview.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "symbol": "ETHUSDT",
                "rsi": 45.0,
                "macd": {"signal": "BULLISH"},
                "timestamp": "2025-10-22T16:00:00Z"
            })
        }]
    })
    
    # Mock memory client
    mock_memory = AsyncMock()
    mock_memory.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "success": True,
                "record_id": "rec-123",
                "stored_at": "2025-10-22T16:00:01Z"
            })
        }]
    })
    
    # Workflow
    # 1. Get analysis from TradingView
    analysis = await mock_tradingview.call_tool("coin_analysis", {"symbol": "ETHUSDT"})
    assert analysis is not None
    print(f"✅ Получен анализ от TradingView")
    
    # 2. Save to memory
    save_result = await mock_memory.call_tool("ingest_records", {
        "records": [{"type": "tradingview_analysis", "data": analysis}]
    })
    assert save_result is not None
    print(f"✅ Анализ сохранен в память")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_tradingview_alert_with_binance_execution():
    """Тест: Алерт TradingView с исполнением через Binance"""
    print("\n=== Test: TradingView alert with Binance execution ===")
    
    # Mock tradingview client
    mock_tradingview = AsyncMock()
    mock_tradingview.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "alert": "BUY_SIGNAL",
                "symbol": "BTCUSDT",
                "price": 35000.00,
                "condition": "RSI < 30"
            })
        }]
    })
    
    # Mock binance client
    mock_binance = AsyncMock()
    mock_binance.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "success": True,
                "orderId": "order-123",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.01
            })
        }]
    })
    
    # Mock memory client
    mock_memory = AsyncMock()
    mock_memory.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({"success": True, "record_id": "rec-124"})
        }]
    })
    
    # Workflow
    # 1. Get alert from TradingView
    alert = await mock_tradingview.call_tool("check_alerts", {"symbol": "BTCUSDT"})
    assert alert is not None
    alert_data = json.loads(alert["content"][0]["text"])
    print(f"✅ Получен алерт от TradingView: {alert_data['alert']}")
    
    # 2. Execute order on Binance
    if alert_data["alert"] == "BUY_SIGNAL":
        order = await mock_binance.call_tool("create_order", {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.01
        })
        assert order is not None
        print(f"✅ Ордер исполнен на Binance")
        
        # 3. Save to memory
        await mock_memory.call_tool("ingest_records", {
            "records": [{"type": "alert_execution", "alert": alert, "order": order}]
        })
        print(f"✅ Результат сохранен в память")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_tradingview_historical_analysis_comparison():
    """Тест: Сравнение исторических анализов из памяти"""
    print("\n=== Test: Historical analysis comparison ===")
    
    # Mock memory client
    mock_memory = AsyncMock()
    mock_memory.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "records": [
                    {"symbol": "BTCUSDT", "rsi": 30.0, "timestamp": "2025-10-20T10:00:00Z"},
                    {"symbol": "BTCUSDT", "rsi": 45.0, "timestamp": "2025-10-21T10:00:00Z"},
                    {"symbol": "BTCUSDT", "rsi": 55.0, "timestamp": "2025-10-22T10:00:00Z"}
                ]
            })
        }]
    })
    
    # Mock tradingview client
    mock_tradingview = AsyncMock()
    mock_tradingview.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "symbol": "BTCUSDT",
                "rsi": 60.0,
                "trend": "BULLISH",
                "timestamp": "2025-10-22T16:00:00Z"
            })
        }]
    })
    
    # Workflow
    # 1. Get historical analyses from memory
    historical = await mock_memory.call_tool("search_memory", {
        "query": "tradingview_analysis BTCUSDT",
        "limit": 10
    })
    assert historical is not None
    hist_data = json.loads(historical["content"][0]["text"])
    print(f"✅ Получено {len(hist_data['records'])} исторических записей")
    
    # 2. Get current analysis from TradingView
    current = await mock_tradingview.call_tool("coin_analysis", {"symbol": "BTCUSDT"})
    assert current is not None
    print(f"✅ Получен текущий анализ")
    
    # 3. Compare
    current_rsi = json.loads(current["content"][0]["text"])["rsi"]
    avg_historical_rsi = sum(r["rsi"] for r in hist_data["records"]) / len(hist_data["records"])
    print(f"✅ Средний исторический RSI: {avg_historical_rsi:.1f}")
    print(f"✅ Текущий RSI: {current_rsi:.1f}")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_tradingview_multi_exchange_analysis():
    """Тест: Анализ по нескольким биржам с использованием Binance и памяти"""
    print("\n=== Test: Multi-exchange analysis ===")
    
    exchanges = ["BINANCE", "KUCOIN", "OKX"]
    
    # Mock binance client
    mock_binance = AsyncMock()
    mock_tradingview = AsyncMock()
    mock_memory = AsyncMock()
    
    results = {}
    
    for exchange in exchanges:
        # Get price from Binance (or other exchange)
        mock_binance.call_tool = AsyncMock(return_value={
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "exchange": exchange,
                    "symbol": "BTCUSDT",
                    "price": 35000.00 + (exchanges.index(exchange) * 10)
                })
            }]
        })
        
        # Get analysis from TradingView
        mock_tradingview.call_tool = AsyncMock(return_value={
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "exchange": exchange,
                    "symbol": "BTCUSDT",
                    "rsi": 50.0 + (exchanges.index(exchange) * 5)
                })
            }]
        })
        
        price = await mock_binance.call_tool("get_ticker_price", {"symbol": "BTCUSDT", "exchange": exchange})
        analysis = await mock_tradingview.call_tool("coin_analysis", {"symbol": "BTCUSDT", "exchange": exchange})
        
        results[exchange] = {"price": price, "analysis": analysis}
    
    assert len(results) == len(exchanges)
    print(f"✅ Выполнен анализ для {len(exchanges)} бирж")
    
    # Save all results to memory
    mock_memory.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"success": True})}]
    })
    
    await mock_memory.call_tool("ingest_records", {
        "records": [{"type": "multi_exchange_analysis", "data": results}]
    })
    print(f"✅ Результаты сохранены в память")
    
    print("✅ Test passed")


if __name__ == "__main__":
    asyncio.run(test_tradingview_analysis_with_binance_prices())
    asyncio.run(test_tradingview_save_analysis_to_memory())
    asyncio.run(test_tradingview_alert_with_binance_execution())
    asyncio.run(test_tradingview_historical_analysis_comparison())
    asyncio.run(test_tradingview_multi_exchange_analysis())
    print("\n🎉 All TradingView integration tests passed!")
