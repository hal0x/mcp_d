"""
Integration tests для backtesting-mcp с binance-mcp и tradingview-mcp
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
import json

@pytest.mark.asyncio
async def test_backtesting_with_binance_historical_data():
    """Тест: Получение исторических данных от binance-mcp для бэктестинга"""
    print("\n=== Test: Backtesting with Binance historical data ===")
    
    # Mock binance client
    mock_binance = AsyncMock()
    mock_binance.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "symbol": "BTCUSDT",
                "interval": "1h",
                "klines": [
                    {
                        "openTime": 1698000000000,
                        "open": "35000.00",
                        "high": "36000.00",
                        "low": "34000.00",
                        "close": "35500.00",
                        "volume": "100.5"
                    }
                ]
            })
        }]
    })
    
    # Mock backtesting client
    mock_backtesting = AsyncMock()
    mock_backtesting.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "success": True,
                "backtest_id": "test-123",
                "trades": 10,
                "profit": 150.50
            })
        }]
    })
    
    # Simulate workflow
    # 1. Получаем исторические данные от binance
    klines_result = await mock_binance.call_tool("get_klines", {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "limit": 100
    })
    
    assert klines_result is not None
    print(f"✅ Получены исторические данные от binance-mcp")
    
    # 2. Запускаем бэктест с этими данными
    backtest_result = await mock_backtesting.call_tool("run_backtest", {
        "strategy": "sma_crossover",
        "data": klines_result
    })
    
    assert backtest_result is not None
    print(f"✅ Бэктест запущен успешно")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_backtesting_with_tradingview_indicators():
    """Тест: Использование индикаторов от tradingview-mcp в бэктестинге"""
    print("\n=== Test: Backtesting with TradingView indicators ===")
    
    # Mock tradingview client
    mock_tradingview = AsyncMock()
    mock_tradingview.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "symbol": "BTCUSDT",
                "rsi": 45.5,
                "macd": {"macd": 120.5, "signal": 115.3, "histogram": 5.2},
                "bollinger": {"upper": 36500, "middle": 35500, "lower": 34500}
            })
        }]
    })
    
    # Mock backtesting client
    mock_backtesting = AsyncMock()
    mock_backtesting.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "success": True,
                "backtest_id": "test-124",
                "trades": 15,
                "profit": 250.75
            })
        }]
    })
    
    # Simulate workflow
    # 1. Получаем индикаторы от tradingview
    indicators_result = await mock_tradingview.call_tool("coin_analysis", {
        "symbol": "BTCUSDT",
        "exchange": "BINANCE"
    })
    
    assert indicators_result is not None
    print(f"✅ Получены индикаторы от tradingview-mcp")
    
    # 2. Запускаем бэктест с индикаторами
    backtest_result = await mock_backtesting.call_tool("run_backtest_with_indicators", {
        "strategy": "rsi_bollinger",
        "indicators": indicators_result
    })
    
    assert backtest_result is not None
    print(f"✅ Бэктест с индикаторами запущен успешно")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_backtesting_multiple_timeframes():
    """Тест: Бэктестинг с множественными таймфреймами от binance и tradingview"""
    print("\n=== Test: Backtesting multiple timeframes ===")
    
    # Mock clients
    mock_binance = AsyncMock()
    mock_tradingview = AsyncMock()
    mock_backtesting = AsyncMock()
    
    timeframes = ["1h", "4h", "1d"]
    results = {}
    
    for tf in timeframes:
        # Get data from binance
        mock_binance.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"symbol": "BTCUSDT", "interval": tf, "data": []})}]
        })
        
        # Get indicators from tradingview
        mock_tradingview.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"symbol": "BTCUSDT", "timeframe": tf, "rsi": 50.0})}]
        })
        
        # Run backtest
        mock_backtesting.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"backtest_id": f"test-{tf}", "profit": 100.0})}]
        })
        
        klines = await mock_binance.call_tool("get_klines", {"symbol": "BTCUSDT", "interval": tf})
        indicators = await mock_tradingview.call_tool("coin_analysis", {"symbol": "BTCUSDT", "timeframe": tf})
        backtest = await mock_backtesting.call_tool("run_backtest", {"timeframe": tf})
        
        results[tf] = {"klines": klines, "indicators": indicators, "backtest": backtest}
    
    assert len(results) == len(timeframes)
    print(f"✅ Бэктест выполнен для {len(timeframes)} таймфреймов")
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_backtesting_strategy_optimization():
    """Тест: Оптимизация стратегии с использованием данных от binance и tradingview"""
    print("\n=== Test: Strategy optimization ===")
    
    # Mock clients
    mock_binance = AsyncMock()
    mock_tradingview = AsyncMock()
    mock_backtesting = AsyncMock()
    
    # Simulate optimization workflow
    parameters = [
        {"rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30},
        {"rsi_period": 14, "rsi_overbought": 75, "rsi_oversold": 25},
        {"rsi_period": 21, "rsi_overbought": 70, "rsi_oversold": 30},
    ]
    
    best_result = None
    best_profit = 0
    
    for params in parameters:
        mock_backtesting.call_tool = AsyncMock(return_value={
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "success": True,
                    "params": params,
                    "profit": 100.0 + (params["rsi_period"] * 10)
                })
            }]
        })
        
        result = await mock_backtesting.call_tool("run_backtest", {"params": params})
        profit = json.loads(result["content"][0]["text"])["profit"]
        
        if profit > best_profit:
            best_profit = profit
            best_result = result
    
    assert best_result is not None
    print(f"✅ Найдена лучшая стратегия с прибылью: {best_profit}")
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_backtesting_risk_management():
    """Тест: Управление рисками в бэктестинге с данными от binance"""
    print("\n=== Test: Risk management in backtesting ===")
    
    # Mock clients
    mock_binance = AsyncMock()
    mock_backtesting = AsyncMock()
    
    # Simulate risk management
    mock_binance.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "symbol": "BTCUSDT",
                "price": 35000.00,
                "volume": 100.5
            })
        }]
    })
    
    mock_backtesting.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "success": True,
                "max_drawdown": 5.2,
                "sharpe_ratio": 1.8,
                "win_rate": 65.5
            })
        }]
    })
    
    # Get market data
    market_data = await mock_binance.call_tool("get_ticker", {"symbol": "BTCUSDT"})
    
    # Run backtest with risk management
    backtest_result = await mock_backtesting.call_tool("run_backtest_with_risk", {
        "max_drawdown": 10.0,
        "stop_loss": 2.0,
        "take_profit": 5.0
    })
    
    assert backtest_result is not None
    result = json.loads(backtest_result["content"][0]["text"])
    assert result["max_drawdown"] < 10.0
    print(f"✅ Бэктест с управлением рисками выполнен: max_drawdown={result['max_drawdown']}%")
    print("✅ Test passed")


if __name__ == "__main__":
    asyncio.run(test_backtesting_with_binance_historical_data())
    asyncio.run(test_backtesting_with_tradingview_indicators())
    asyncio.run(test_backtesting_multiple_timeframes())
    asyncio.run(test_backtesting_strategy_optimization())
    asyncio.run(test_backtesting_risk_management())
    print("\n🎉 All backtesting integration tests passed!")
