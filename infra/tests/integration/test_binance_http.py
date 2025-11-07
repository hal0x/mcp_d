"""
Тест для проверки работы binance-mcp в HTTP режиме
"""
import pytest
import httpx
import asyncio
import logging

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_binance_mcp_http_mode():
    """Проверка работы binance-mcp в HTTP режиме"""
    print("=== Тестирование binance-mcp в HTTP режиме ===")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Проверяем MCP endpoint
        try:
            response = await client.post(
                "http://localhost:8000/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list"
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
            )
            
            print(f"MCP tools/list: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {result}")
                
                if "result" in result:
                    tools = result["result"]["tools"]
                    print(f"Available tools: {len(tools)}")
                    for tool in tools[:5]:  # Показываем первые 5 инструментов
                        print(f"  - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')[:50]}...")
                    
                    return True
                elif "error" in result:
                    print(f"MCP Error: {result['error']}")
            else:
                print(f"HTTP Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    return False

@pytest.mark.asyncio
async def test_binance_mcp_tool_call():
    """Тестирование вызова инструмента binance-mcp"""
    print("\n=== Тестирование вызова инструмента ===")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Пробуем вызвать простой инструмент
            response = await client.post(
                "http://localhost:8000/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "get_server_time",
                        "arguments": {}
                    }
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
            )
            
            print(f"MCP tools/call get_server_time: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {result}")
                
                if "result" in result:
                    print("✅ Инструмент get_server_time работает!")
                    return True
                elif "error" in result:
                    print(f"MCP Error: {result['error']}")
            else:
                print(f"HTTP Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    return False

@pytest.mark.asyncio
async def test_binance_mcp_health():
    """Проверка health endpoint"""
    print("\n=== Проверка health endpoint ===")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Пробуем разные варианты health endpoints
        endpoints = [
            "/healthz",
            "/health", 
            "/status",
            "/ping"
        ]
        
        for endpoint in endpoints:
            try:
                response = await client.get(f"http://localhost:8000{endpoint}")
                print(f"GET {endpoint}: {response.status_code}")
                if response.status_code == 200:
                    print(f"  Success! Response: {response.text[:100]}")
                    return True
                elif response.status_code != 404:
                    print(f"  Response: {response.text[:100]}")
            except Exception as e:
                print(f"GET {endpoint}: Error - {e}")
    
    return False

if __name__ == "__main__":
    async def main():
        print("Запуск тестов binance-mcp...")
        
        # Тест 1: Проверка HTTP режима
        success1 = await test_binance_mcp_http_mode()
        
        # Тест 2: Вызов инструмента
        success2 = await test_binance_mcp_tool_call()
        
        # Тест 3: Health endpoint
        success3 = await test_binance_mcp_health()
        
        print(f"\n=== Результаты ===")
        print(f"HTTP режим: {'✅' if success1 else '❌'}")
        print(f"Вызов инструмента: {'✅' if success2 else '❌'}")
        print(f"Health endpoint: {'✅' if success3 else '❌'}")
        
        if success1 or success2:
            print("\n🎉 binance-mcp работает в HTTP режиме!")
        else:
            print("\n❌ binance-mcp не работает в HTTP режиме")
    
    asyncio.run(main())
