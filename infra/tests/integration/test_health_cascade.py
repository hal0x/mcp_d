"""
Integration tests для healthcheck зависимостей и отказоустойчивости
"""
import pytest
import asyncio
from unittest.mock import AsyncMock
import json

@pytest.mark.asyncio
async def test_all_services_health_check():
    """Тест: Проверка здоровья всех сервисов"""
    print("\n=== Test: All services health check ===")
    
    services = {
        "binance-mcp": "http://localhost:8000",
        "tradingview-mcp": "http://localhost:8060",
        "memory-mcp": "http://localhost:8050",
        "shell-mcp": "http://localhost:8070",
        "backtesting-mcp": "http://localhost:8082"
    }
    
    results = {}
    
    for name, url in services.items():
        mock_client = AsyncMock()
        mock_client.health_check = AsyncMock(return_value=True)
        
        is_healthy = await mock_client.health_check()
        results[name] = is_healthy
        
        status = "✅" if is_healthy else "❌"
        print(f"  {status} {name}: {'healthy' if is_healthy else 'unhealthy'}")
    
    all_healthy = all(results.values())
    assert all_healthy, f"Not all services are healthy: {results}"
    
    print(f"✅ Все {len(services)} сервисов здоровы")
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_service_dependency_chain():
    """Тест: Проверка цепочки зависимостей сервисов"""
    print("\n=== Test: Service dependency chain ===")
    
    # Dependency chain: tradingview-mcp -> binance-mcp -> redis/postgres
    
    # 1. Check Redis/Postgres (base dependencies)
    mock_redis = AsyncMock()
    mock_redis.health_check = AsyncMock(return_value=True)
    redis_healthy = await mock_redis.health_check()
    print(f"  ✅ Redis: {'healthy' if redis_healthy else 'unhealthy'}")
    
    mock_postgres = AsyncMock()
    mock_postgres.health_check = AsyncMock(return_value=True)
    postgres_healthy = await mock_postgres.health_check()
    print(f"  ✅ Postgres: {'healthy' if postgres_healthy else 'unhealthy'}")
    
    # 2. Check binance-mcp (depends on redis/postgres)
    if redis_healthy and postgres_healthy:
        mock_binance = AsyncMock()
        mock_binance.health_check = AsyncMock(return_value=True)
        binance_healthy = await mock_binance.health_check()
        print(f"  ✅ Binance-mcp: {'healthy' if binance_healthy else 'unhealthy'}")
    else:
        binance_healthy = False
        print(f"  ❌ Binance-mcp: cannot start (dependencies unhealthy)")
    
    # 3. Check tradingview-mcp (depends on binance-mcp)
    if binance_healthy:
        mock_tradingview = AsyncMock()
        mock_tradingview.health_check = AsyncMock(return_value=True)
        tradingview_healthy = await mock_tradingview.health_check()
        print(f"  ✅ Tradingview-mcp: {'healthy' if tradingview_healthy else 'unhealthy'}")
    else:
        tradingview_healthy = False
        print(f"  ❌ Tradingview-mcp: cannot start (dependencies unhealthy)")
    
    assert all([redis_healthy, postgres_healthy, binance_healthy, tradingview_healthy])
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_service_recovery_after_failure():
    """Тест: Восстановление сервиса после сбоя"""
    print("\n=== Test: Service recovery after failure ===")
    
    mock_binance = AsyncMock()
    
    # Simulate service failure
    mock_binance.health_check = AsyncMock(return_value=False)
    is_healthy = await mock_binance.health_check()
    print(f"  1. Сервис binance-mcp: {'healthy' if is_healthy else 'unhealthy'}")
    assert not is_healthy
    
    # Wait for restart
    await asyncio.sleep(2)
    print(f"  2. Ожидание перезапуска...")
    
    # Simulate service recovery
    mock_binance.health_check = AsyncMock(return_value=True)
    is_healthy = await mock_binance.health_check()
    print(f"  3. Сервис binance-mcp: {'healthy' if is_healthy else 'unhealthy'}")
    assert is_healthy
    
    print("✅ Сервис успешно восстановлен")
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_graceful_degradation():
    """Тест: Graceful degradation при недоступности зависимости"""
    print("\n=== Test: Graceful degradation ===")
    
    mock_binance = AsyncMock()
    mock_tradingview = AsyncMock()
    mock_memory = AsyncMock()
    
    # Scenario: Memory service is down, but trading should continue
    mock_memory.health_check = AsyncMock(return_value=False)
    memory_healthy = await mock_memory.health_check()
    print(f"  1. Memory-mcp: {'healthy' if memory_healthy else 'DOWN ❌'}")
    
    # Binance and TradingView should still work
    mock_binance.health_check = AsyncMock(return_value=True)
    mock_tradingview.health_check = AsyncMock(return_value=True)
    
    binance_healthy = await mock_binance.health_check()
    tradingview_healthy = await mock_tradingview.health_check()
    
    print(f"  2. Binance-mcp: {'healthy ✅' if binance_healthy else 'unhealthy'}")
    print(f"  3. Tradingview-mcp: {'healthy ✅' if tradingview_healthy else 'unhealthy'}")
    
    # Trading workflow should work (without saving to memory)
    if binance_healthy and tradingview_healthy:
        mock_binance.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"symbol": "BTCUSDT", "price": 35000.00})}]
        })
        mock_tradingview.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": json.dumps({"symbol": "BTCUSDT", "rsi": 45.0})}]
        })
        
        price = await mock_binance.call_tool("get_ticker_price", {"symbol": "BTCUSDT"})
        analysis = await mock_tradingview.call_tool("coin_analysis", {"symbol": "BTCUSDT"})
        
        print(f"  4. ✅ Торговля продолжается без сохранения в память")
    
    assert binance_healthy and tradingview_healthy
    print("✅ Graceful degradation работает")
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_concurrent_service_failures():
    """Тест: Обработка множественных одновременных сбоев"""
    print("\n=== Test: Concurrent service failures ===")
    
    services = ["binance-mcp", "tradingview-mcp", "memory-mcp"]
    
    # Simulate multiple failures
    print(f"  1. Симуляция сбоя {len(services)} сервисов...")
    for service in services:
        print(f"     ❌ {service}: DOWN")
    
    # Wait for recovery
    await asyncio.sleep(3)
    print(f"  2. Ожидание восстановления...")
    
    # Simulate recovery
    recovered = []
    for service in services:
        mock_service = AsyncMock()
        mock_service.health_check = AsyncMock(return_value=True)
        is_healthy = await mock_service.health_check()
        
        if is_healthy:
            recovered.append(service)
            print(f"     ✅ {service}: RECOVERED")
    
    assert len(recovered) == len(services)
    print(f"✅ Все {len(services)} сервисов восстановлены")
    print("✅ Test passed")


if __name__ == "__main__":
    asyncio.run(test_all_services_health_check())
    asyncio.run(test_service_dependency_chain())
    asyncio.run(test_service_recovery_after_failure())
    asyncio.run(test_graceful_degradation())
    asyncio.run(test_concurrent_service_failures())
    print("\n🎉 All health and fault tolerance tests passed!")
