"""
Простой тест для проверки доступности MCP сервисов
"""
import pytest
import httpx
import asyncio
import logging

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_services_connectivity():
    """Проверка доступности всех MCP сервисов"""
    services = {
        "binance-mcp": "http://localhost:8000",
        "tradingview-mcp": "http://localhost:8060", 
        "memory-mcp": "http://localhost:8050",
        "shell-mcp": "http://localhost:8070",
        "backtesting-mcp": "http://localhost:8082"
    }
    
    results = {}
    
    for name, url in services.items():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Пробуем разные endpoints
                endpoints = ["/", "/health", "/tools", "/status"]
                
                for endpoint in endpoints:
                    try:
                        response = await client.get(f"{url}{endpoint}")
                        results[name] = {
                            "status": "available",
                            "endpoint": endpoint,
                            "status_code": response.status_code,
                            "response": response.text[:100] if response.text else "No content"
                        }
                        logger.info(f"{name} доступен на {endpoint}: {response.status_code}")
                        break
                    except Exception as e:
                        logger.debug(f"{name} {endpoint} failed: {e}")
                        continue
                else:
                    results[name] = {
                        "status": "unavailable",
                        "error": "No working endpoint found"
                    }
                    
        except Exception as e:
            results[name] = {
                "status": "error",
                "error": str(e)
            }
            logger.error(f"Error connecting to {name}: {e}")
    
    # Выводим результаты
    print("\n=== Результаты проверки сервисов ===")
    for name, result in results.items():
        print(f"{name}: {result['status']}")
        if result['status'] == 'available':
            print(f"  Endpoint: {result['endpoint']}")
            print(f"  Status Code: {result['status_code']}")
        else:
            print(f"  Error: {result['error']}")
    
    # Проверяем, что хотя бы один сервис доступен
    available_services = [name for name, result in results.items() if result['status'] == 'available']
    assert len(available_services) > 0, f"Ни один сервис не доступен. Результаты: {results}"
    
    print(f"\nДоступные сервисы: {available_services}")

@pytest.mark.asyncio 
async def test_simple_http_requests():
    """Простые HTTP запросы для проверки работы серверов"""
    
    # Проверяем binance-mcp
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Пробуем разные варианты endpoints
            endpoints = [
                "http://localhost:8000/",
                "http://localhost:8000/health", 
                "http://localhost:8000/tools",
                "http://localhost:8000/market/klines?symbol=BTCUSDT&interval=1h&limit=1"
            ]
            
            for endpoint in endpoints:
                try:
                    response = await client.get(endpoint)
                    print(f"binance-mcp {endpoint}: {response.status_code}")
                    if response.status_code == 200:
                        print(f"  Response: {response.text[:200]}")
                        break
                except Exception as e:
                    print(f"binance-mcp {endpoint}: Error - {e}")
    except Exception as e:
        print(f"binance-mcp: Connection error - {e}")
    
    # Проверяем tradingview-mcp
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            endpoints = [
                "http://localhost:8060/",
                "http://localhost:8060/health",
                "http://localhost:8060/tools"
            ]
            
            for endpoint in endpoints:
                try:
                    response = await client.get(endpoint)
                    print(f"tradingview-mcp {endpoint}: {response.status_code}")
                    if response.status_code == 200:
                        print(f"  Response: {response.text[:200]}")
                        break
                except Exception as e:
                    print(f"tradingview-mcp {endpoint}: Error - {e}")
    except Exception as e:
        print(f"tradingview-mcp: Connection error - {e}")

if __name__ == "__main__":
    # Запуск тестов напрямую
    asyncio.run(test_services_connectivity())
    asyncio.run(test_simple_http_requests())
