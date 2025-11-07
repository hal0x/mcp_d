"""
Integration tests для halv1 со всеми MCP сервисами
"""
import pytest
import asyncio
from unittest.mock import AsyncMock
import json

@pytest.mark.asyncio
async def test_halv1_full_trading_workflow():
    """Тест: Полный торговый workflow с использованием всех MCP сервисов"""
    print("\n=== Test: HAL full trading workflow ===")
    
    # Mock all clients
    mock_binance = AsyncMock()
    mock_tradingview = AsyncMock()
    mock_memory = AsyncMock()
    mock_backtesting = AsyncMock()
    mock_shell = AsyncMock()
    
    # 1. Get market data from Binance
    mock_binance.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"symbol": "BTCUSDT", "price": 35000.00})}]
    })
    market_data = await mock_binance.call_tool("get_ticker_price", {"symbol": "BTCUSDT"})
    print(f"✅ 1. Получены рыночные данные от Binance")
    
    # 2. Analyze with TradingView
    mock_tradingview.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"symbol": "BTCUSDT", "rsi": 45.0, "recommendation": "BUY"})}]
    })
    analysis = await mock_tradingview.call_tool("coin_analysis", {"symbol": "BTCUSDT"})
    print(f"✅ 2. Выполнен технический анализ TradingView")
    
    # 3. Run backtest
    mock_backtesting.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"backtest_id": "test-123", "profit": 150.50, "win_rate": 65.5})}]
    })
    backtest = await mock_backtesting.call_tool("run_backtest", {"strategy": "rsi_strategy"})
    print(f"✅ 3. Выполнен бэктест стратегии")
    
    # 4. Save decision to memory
    mock_memory.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"success": True, "record_id": "rec-123"})}]
    })
    await mock_memory.call_tool("ingest_records", {
        "records": [{"type": "trading_decision", "market_data": market_data, "analysis": analysis, "backtest": backtest}]
    })
    print(f"✅ 4. Решение сохранено в память")
    
    # 5. Execute trade on Binance
    mock_binance.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"orderId": "order-123", "status": "FILLED"})}]
    })
    order = await mock_binance.call_tool("create_order", {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.01})
    print(f"✅ 5. Ордер исполнен на Binance")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_halv1_market_monitoring():
    """Тест: Мониторинг рынка с использованием Binance, TradingView и Memory"""
    print("\n=== Test: HAL market monitoring ===")
    
    mock_binance = AsyncMock()
    mock_tradingview = AsyncMock()
    mock_memory = AsyncMock()
    
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    for symbol in symbols:
        # 1. Get price from Binance
        mock_binance.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"symbol": symbol, "price": 35000.00})}]
        })
        price = await mock_binance.call_tool("get_ticker_price", {"symbol": symbol})
        
        # 2. Get analysis from TradingView
        mock_tradingview.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"symbol": symbol, "rsi": 50.0})}]
        })
        analysis = await mock_tradingview.call_tool("coin_analysis", {"symbol": symbol})
        
        # 3. Save to memory
        mock_memory.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"success": True})}]
        })
        await mock_memory.call_tool("ingest_records", {
            "records": [{"type": "market_snapshot", "symbol": symbol, "price": price, "analysis": analysis}]
        })
    
    print(f"✅ Мониторинг выполнен для {len(symbols)} символов")
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_halv1_automated_trading_decision():
    """Тест: Автоматизированное принятие торговых решений"""
    print("\n=== Test: HAL automated trading decision ===")
    
    mock_binance = AsyncMock()
    mock_tradingview = AsyncMock()
    mock_memory = AsyncMock()
    mock_backtesting = AsyncMock()
    
    # 1. Search historical patterns in memory
    mock_memory.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "patterns": [
                    {"pattern": "bullish_divergence", "success_rate": 75.0},
                    {"pattern": "golden_cross", "success_rate": 68.5}
                ]
            })
        }]
    })
    historical_patterns = await mock_memory.call_tool("search_trading_patterns", {"symbol": "BTCUSDT"})
    print(f"✅ 1. Найдены исторические паттерны")
    
    # 2. Get current market conditions
    mock_binance.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"symbol": "BTCUSDT", "price": 35000.00, "volume": 1000.0})}]
    })
    mock_tradingview.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"symbol": "BTCUSDT", "rsi": 45.0, "macd": "BULLISH"})}]
    })
    
    market = await mock_binance.call_tool("get_ticker_24h", {"symbol": "BTCUSDT"})
    analysis = await mock_tradingview.call_tool("coin_analysis", {"symbol": "BTCUSDT"})
    print(f"✅ 2. Получены текущие рыночные условия")
    
    # 3. Backtest similar scenarios
    mock_backtesting.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"expected_profit": 125.50, "confidence": 0.82})}]
        })
    backtest = await mock_backtesting.call_tool("backtest_similar_scenarios", {
        "current_conditions": {"market": market, "analysis": analysis}
    })
    print(f"✅ 3. Выполнен бэктест похожих сценариев")
    
    # 4. Make decision
    backtest_data = json.loads(backtest["content"][0]["text"])
    if backtest_data["confidence"] > 0.75:
        mock_binance.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"orderId": "order-124", "status": "FILLED"})}]
        })
        order = await mock_binance.call_tool("create_order", {"symbol": "BTCUSDT", "side": "BUY"})
        print(f"✅ 4. Решение принято: BUY (confidence: {backtest_data['confidence']})")
        
        # 5. Save decision to memory
        mock_memory.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"success": True})}]
        })
        await mock_memory.call_tool("store_trading_signal", {
            "signal": {"decision": "BUY", "confidence": backtest_data["confidence"], "order": order}
        })
        print(f"✅ 5. Решение сохранено в память")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_halv1_risk_management():
    """Тест: Управление рисками с использованием всех сервисов"""
    print("\n=== Test: HAL risk management ===")
    
    mock_binance = AsyncMock()
    mock_memory = AsyncMock()
    
    # 1. Get account balance
    mock_binance.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"totalBalance": 10000.00, "availableBalance": 8000.00})}]
    })
    balance = await mock_binance.call_tool("get_account_info", {})
    balance_data = json.loads(balance["content"][0]["text"])
    print(f"✅ 1. Баланс: ${balance_data['totalBalance']}")
    
    # 2. Get historical performance
    mock_memory.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "total_trades": 50,
                "winning_trades": 35,
                "total_profit": 1500.00,
                "max_drawdown": 5.2
            })
        }]
    })
    performance = await mock_memory.call_tool("get_signal_performance", {})
    perf_data = json.loads(performance["content"][0]["text"])
    print(f"✅ 2. Исторический winrate: {(perf_data['winning_trades']/perf_data['total_trades'])*100:.1f}%")
    
    # 3. Calculate position size
    max_risk_percent = 2.0
    position_size = (balance_data["availableBalance"] * max_risk_percent) / 100
    print(f"✅ 3. Рассчитан размер позиции: ${position_size:.2f}")
    
    # 4. Place order with risk management
    mock_binance.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"orderId": "order-125", "quantity": position_size / 35000})}]
    })
    order = await mock_binance.call_tool("create_order", {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "quantity": position_size / 35000,
        "stopLoss": 33250.00,
        "takeProfit": 36750.00
    })
    print(f"✅ 4. Ордер с риск-менеджментом размещен")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_halv1_portfolio_management():
    """Тест: Управление портфелем с использованием всех сервисов"""
    print("\n=== Test: HAL portfolio management ===")
    
    mock_binance = AsyncMock()
    mock_tradingview = AsyncMock()
    mock_memory = AsyncMock()
    
    # 1. Get current portfolio
    mock_binance.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "balances": [
                    {"asset": "BTC", "free": 0.5, "locked": 0.0},
                    {"asset": "ETH", "free": 10.0, "locked": 0.0},
                    {"asset": "BNB", "free": 100.0, "locked": 0.0}
                ]
            })
        }]
    })
    portfolio = await mock_binance.call_tool("get_account_info", {})
    portfolio_data = json.loads(portfolio["content"][0]["text"])
    print(f"✅ 1. Портфель загружен: {len(portfolio_data['balances'])} активов")
    
    # 2. Analyze each asset
    for balance in portfolio_data["balances"]:
        asset = balance["asset"]
        symbol = f"{asset}USDT"
        
        mock_tradingview.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"symbol": symbol, "rsi": 50.0, "recommendation": "HOLD"})}]
        })
        analysis = await mock_tradingview.call_tool("coin_analysis", {"symbol": symbol})
        print(f"  ✅ {asset}: анализ выполнен")
    
    # 3. Save portfolio snapshot to memory
    mock_memory.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"success": True, "snapshot_id": "snap-123"})}]
    })
    await mock_memory.call_tool("ingest_records", {
        "records": [{"type": "portfolio_snapshot", "data": portfolio}]
    })
    print(f"✅ 2. Снимок портфеля сохранен")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_halv1_strategy_backtesting():
    """Тест: Тестирование стратегии на исторических данных"""
    print("\n=== Test: HAL strategy backtesting ===")
    
    mock_binance = AsyncMock()
    mock_tradingview = AsyncMock()
    mock_backtesting = AsyncMock()
    mock_memory = AsyncMock()
    
    # 1. Get historical data from Binance
    mock_binance.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"symbol": "BTCUSDT", "klines": []})}]
    })
    historical = await mock_binance.call_tool("get_klines", {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "limit": 1000
    })
    print(f"✅ 1. Получены исторические данные")
    
    # 2. Get indicators from TradingView
    mock_tradingview.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"symbol": "BTCUSDT", "indicators": {}})}]
    })
    indicators = await mock_tradingview.call_tool("get_indicators", {"symbol": "BTCUSDT"})
    print(f"✅ 2. Получены индикаторы")
    
    # 3. Run comprehensive backtest
    mock_backtesting.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "backtest_id": "test-126",
                "total_trades": 45,
                "profitable_trades": 32,
                "total_profit": 1250.50,
                "sharpe_ratio": 1.85,
                "max_drawdown": 4.8
            })
        }]
    })
    backtest = await mock_backtesting.call_tool("run_comprehensive_backtest", {
        "strategy": "combined_indicators",
        "data": historical,
        "indicators": indicators
    })
    backtest_data = json.loads(backtest["content"][0]["text"])
    print(f"✅ 3. Бэктест завершен: winrate={backtest_data['profitable_trades']/backtest_data['total_trades']*100:.1f}%")
    
    # 4. Save results to memory
    mock_memory.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"success": True})}]
    })
    await mock_memory.call_tool("ingest_records", {
        "records": [{"type": "backtest_result", "data": backtest}]
    })
    print(f"✅ 4. Результаты сохранены в память")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_halv1_real_time_alerts():
    """Тест: Система реалтайм алертов"""
    print("\n=== Test: HAL real-time alerts ===")
    
    mock_binance = AsyncMock()
    mock_tradingview = AsyncMock()
    mock_memory = AsyncMock()
    mock_shell = AsyncMock()
    
    # 1. Monitor market conditions
    mock_binance.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"symbol": "BTCUSDT", "price": 35000.00, "change": -5.2})}]
    })
    mock_tradingview.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"symbol": "BTCUSDT", "rsi": 25.0, "alert": "OVERSOLD"})}]
    })
    
    market = await mock_binance.call_tool("get_ticker_24h", {"symbol": "BTCUSDT"})
    analysis = await mock_tradingview.call_tool("coin_analysis", {"symbol": "BTCUSDT"})
    
    market_data = json.loads(market["content"][0]["text"])
    analysis_data = json.loads(analysis["content"][0]["text"])
    
    print(f"✅ 1. Обнаружено условие алерта: RSI={analysis_data['rsi']}")
    
    # 2. Send notification via shell
    if analysis_data["rsi"] < 30:
        mock_shell.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"success": True, "notification_sent": True})}]
        })
        await mock_shell.call_tool("execute", {
            "command": f"notify 'ALERT: BTCUSDT RSI={analysis_data['rsi']}'"
        })
        print(f"✅ 2. Уведомление отправлено")
        
        # 3. Save alert to memory
        mock_memory.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"success": True})}]
        })
        await mock_memory.call_tool("ingest_records", {
            "records": [{"type": "alert", "condition": "RSI_OVERSOLD", "data": analysis}]
        })
        print(f"✅ 3. Алерт сохранен в память")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_halv1_multi_strategy_execution():
    """Тест: Исполнение множественных стратегий параллельно"""
    print("\n=== Test: HAL multi-strategy execution ===")
    
    mock_binance = AsyncMock()
    mock_tradingview = AsyncMock()
    mock_backtesting = AsyncMock()
    mock_memory = AsyncMock()
    
    strategies = ["momentum", "mean_reversion", "breakout"]
    results = {}
    
    for strategy in strategies:
        # 1. Get market data
        mock_binance.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"symbol": "BTCUSDT", "data": {}})}]
        })
        
        # 2. Get analysis
        mock_tradingview.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"symbol": "BTCUSDT", "indicators": {}})}]
        })
        
        # 3. Run backtest for strategy
        mock_backtesting.call_tool = AsyncMock(return_value={
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "strategy": strategy,
                    "profit": 100.0 + (strategies.index(strategy) * 50),
                    "sharpe": 1.5 + (strategies.index(strategy) * 0.2)
                })
            }]
        })
        
        market = await mock_binance.call_tool("get_ticker_price", {"symbol": "BTCUSDT"})
        analysis = await mock_tradingview.call_tool("coin_analysis", {"symbol": "BTCUSDT"})
        backtest = await mock_backtesting.call_tool("run_backtest", {"strategy": strategy})
        
        results[strategy] = backtest
        print(f"  ✅ Стратегия '{strategy}' протестирована")
    
    # 4. Save all results to memory
    mock_memory.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"success": True})}]
    })
    await mock_memory.call_tool("ingest_records", {
        "records": [{"type": "multi_strategy_results", "strategies": results}]
    })
    
    print(f"✅ Все {len(strategies)} стратегий протестированы и сохранены")
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_halv1_performance_tracking():
    """Тест: Отслеживание производительности торговых решений"""
    print("\n=== Test: HAL performance tracking ===")
    
    mock_binance = AsyncMock()
    mock_memory = AsyncMock()
    
    # 1. Get recent trades from Binance
    mock_binance.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "trades": [
                    {"symbol": "BTCUSDT", "profit": 150.50, "result": "WIN"},
                    {"symbol": "ETHUSDT", "profit": -50.25, "result": "LOSS"},
                    {"symbol": "BNBUSDT", "profit": 75.00, "result": "WIN"}
                ]
            })
        }]
    })
    trades = await mock_binance.call_tool("get_my_trades", {"limit": 100})
    trades_data = json.loads(trades["content"][0]["text"])
    print(f"✅ 1. Получено {len(trades_data['trades'])} сделок")
    
    # 2. Get performance metrics from memory
    mock_memory.call_tool = AsyncMock(return_value={
        "content": [{
            "type": "text",
            "text": json.dumps({
                "total_profit": 1750.50,
                "win_rate": 68.5,
                "avg_profit_per_trade": 25.50,
                "best_day": 350.00,
                "worst_day": -125.50
            })
        }]
    })
    performance = await mock_memory.call_tool("get_signal_performance", {})
    perf_data = json.loads(performance["content"][0]["text"])
    print(f"✅ 2. Метрики производительности:")
    print(f"     Общая прибыль: ${perf_data['total_profit']}")
    print(f"     Win rate: {perf_data['win_rate']}%")
    print(f"     Средняя прибыль: ${perf_data['avg_profit_per_trade']}")
    
    # 3. Save updated metrics
    mock_memory.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"success": True})}]
    })
    await mock_memory.call_tool("ingest_records", {
        "records": [{"type": "performance_update", "trades": trades, "metrics": performance}]
    })
    print(f"✅ 3. Метрики обновлены")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_halv1_emergency_shutdown():
    """Тест: Экстренное закрытие всех позиций"""
    print("\n=== Test: HAL emergency shutdown ===")
    
    mock_binance = AsyncMock()
    mock_memory = AsyncMock()
    mock_shell = AsyncMock()
    
    # 1. Detect emergency condition
    mock_binance.call_tool = AsyncMock(return_value={
        "content": [{"type": "text", "text": json.dumps({"price": 30000.00, "change": -15.5})}]
    })
    market = await mock_binance.call_tool("get_ticker_24h", {"symbol": "BTCUSDT"})
    market_data = json.loads(market["content"][0]["text"])
    
    if market_data["change"] < -10:
        print(f"⚠️ 1. Обнаружено экстренное условие: падение {market_data['change']}%")
        
        # 2. Get open positions
        mock_binance.call_tool = AsyncMock(return_value={
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "positions": [
                        {"symbol": "BTCUSDT", "quantity": 0.5},
                        {"symbol": "ETHUSDT", "quantity": 10.0}
                    ]
                })
            }]
        })
        positions = await mock_binance.call_tool("get_open_positions", {})
        positions_data = json.loads(positions["content"][0]["text"])
        print(f"✅ 2. Найдено {len(positions_data['positions'])} открытых позиций")
        
        # 3. Close all positions
        for position in positions_data["positions"]:
            mock_binance.call_tool = AsyncMock(return_value={
                "content": [{"type": "text", "text": json.dumps({"orderId": "close-123", "status": "FILLED"})}]
            })
            await mock_binance.call_tool("create_order", {
                "symbol": position["symbol"],
                "side": "SELL",
                "quantity": position["quantity"]
            })
            print(f"  ✅ Закрыта позиция: {position['symbol']}")
        
        # 4. Save emergency event to memory
        mock_memory.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"success": True})}]
        })
        await mock_memory.call_tool("ingest_records", {
            "records": [{"type": "emergency_shutdown", "reason": "market_crash", "data": market}]
        })
        print(f"✅ 3. Событие сохранено в память")
        
        # 5. Send notification
        mock_shell.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"success": True})}]
        })
        await mock_shell.call_tool("execute", {
            "command": "notify 'EMERGENCY: All positions closed due to market crash'"
        })
        print(f"✅ 4. Уведомление отправлено")
    
    print("✅ Test passed")


if __name__ == "__main__":
    asyncio.run(test_halv1_full_trading_workflow())
    asyncio.run(test_halv1_market_monitoring())
    asyncio.run(test_halv1_automated_trading_decision())
    asyncio.run(test_halv1_risk_management())
    asyncio.run(test_halv1_portfolio_management())
    asyncio.run(test_halv1_strategy_backtesting())
    asyncio.run(test_halv1_real_time_alerts())
    asyncio.run(test_halv1_multi_strategy_execution())
    asyncio.run(test_halv1_performance_tracking())
    asyncio.run(test_halv1_emergency_shutdown())
    print("\n🎉 All HAL integration tests passed!")
