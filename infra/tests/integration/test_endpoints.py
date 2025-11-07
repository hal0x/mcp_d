"""
Тест для проверки реальных endpoints MCP сервисов
"""
import httpx
import asyncio
import json

async def test_binance_endpoints():
    """Тестирование endpoints binance-mcp"""
    print("=== Тестирование binance-mcp ===")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Пробуем разные варианты endpoints
        endpoints = [
            "/market/klines?symbol=BTCUSDT&interval=1h&limit=1",
            "/api/market/klines?symbol=BTCUSDT&interval=1h&limit=1", 
            "/v1/market/klines?symbol=BTCUSDT&interval=1h&limit=1",
            "/binance/market/klines?symbol=BTCUSDT&interval=1h&limit=1",
            "/tools",
            "/health",
            "/status",
            "/"
        ]
        
        for endpoint in endpoints:
            try:
                response = await client.get(f"http://localhost:8000{endpoint}")
                print(f"GET {endpoint}: {response.status_code}")
                if response.status_code == 200:
                    print(f"  Success! Response: {response.text[:200]}")
                    return endpoint
                elif response.status_code != 404:
                    print(f"  Response: {response.text[:100]}")
            except Exception as e:
                print(f"GET {endpoint}: Error - {e}")
        
        # Пробуем POST запросы
        post_endpoints = [
            "/call",
            "/tools/call", 
            "/mcp/call",
            "/api/call"
        ]
        
        for endpoint in post_endpoints:
            try:
                response = await client.post(
                    f"http://localhost:8000{endpoint}",
                    json={"tool": "get_ticker_price", "arguments": {"symbol": "BTCUSDT"}}
                )
                print(f"POST {endpoint}: {response.status_code}")
                if response.status_code == 200:
                    print(f"  Success! Response: {response.text[:200]}")
                    return endpoint
                elif response.status_code != 404:
                    print(f"  Response: {response.text[:100]}")
            except Exception as e:
                print(f"POST {endpoint}: Error - {e}")
    
    return None

async def test_tradingview_endpoints():
    """Тестирование endpoints tradingview-mcp"""
    print("\n=== Тестирование tradingview-mcp ===")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        endpoints = [
            "/tools",
            "/health", 
            "/status",
            "/call",
            "/tools/call",
            "/mcp/call",
            "/"
        ]
        
        for endpoint in endpoints:
            try:
                response = await client.get(f"http://localhost:8060{endpoint}")
                print(f"GET {endpoint}: {response.status_code}")
                if response.status_code == 200:
                    print(f"  Success! Response: {response.text[:200]}")
                    return endpoint
                elif response.status_code != 404:
                    print(f"  Response: {response.text[:100]}")
            except Exception as e:
                print(f"GET {endpoint}: Error - {e}")
    
    return None

async def test_all_services():
    """Тестирование всех сервисов"""
    services = [
        ("binance-mcp", "http://localhost:8000"),
        ("tradingview-mcp", "http://localhost:8060"),
        ("memory-mcp", "http://localhost:8050"),
        ("shell-mcp", "http://localhost:8070"),
        ("backtesting-mcp", "http://localhost:8082")
    ]
    
    print("=== Тестирование всех сервисов ===")
    
    for name, url in services:
        print(f"\n--- {name} ---")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Простая проверка доступности
                response = await client.get(f"{url}/")
                print(f"Status: {response.status_code}")
                print(f"Headers: {dict(response.headers)}")
                
                # Проверяем, есть ли какие-то полезные endpoints
                if "uvicorn" in response.headers.get("server", ""):
                    print("  Server: uvicorn (FastAPI/Starlette)")
                if "content-type" in response.headers:
                    print(f"  Content-Type: {response.headers['content-type']}")
                    
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_all_services())
    asyncio.run(test_binance_endpoints())
    asyncio.run(test_tradingview_endpoints())
