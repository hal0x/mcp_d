"""
Тест для проверки работы MCP серверов через правильный протокол
"""
import pytest
import asyncio
import httpx
import json
import logging

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_mcp_servers_respond():
    """Проверка, что MCP серверы отвечают на правильные запросы"""
    print("=== Тестирование MCP серверов ===")
    
    services = [
        ("binance-mcp", "http://localhost:8000"),
        ("tradingview-mcp", "http://localhost:8060"),
        ("memory-mcp", "http://localhost:8050"),
        ("shell-mcp", "http://localhost:8070"),
        ("backtesting-mcp", "http://localhost:8082")
    ]
    
    results = {}
    
    for service_name, base_url in services:
        print(f"\n--- Тестирование {service_name} ---")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Проверяем, что сервер отвечает на MCP endpoint
            try:
                response = await client.post(
                    f"{base_url}/mcp",
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
                        print(f"✅ {service_name}: {len(tools)} инструментов доступно")
                        results[service_name] = True
                    elif "error" in result:
                        print(f"❌ {service_name}: MCP Error - {result['error']}")
                        results[service_name] = False
                    else:
                        print(f"❓ {service_name}: Неожиданный ответ - {result}")
                        results[service_name] = False
                else:
                    print(f"❌ {service_name}: HTTP Error {response.status_code} - {response.text[:100]}")
                    results[service_name] = False
                    
            except Exception as e:
                print(f"❌ {service_name}: Exception - {e}")
                results[service_name] = False
    
    print(f"\n=== Результаты ===")
    working_services = []
    for service_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {service_name}")
        if success:
            working_services.append(service_name)
    
    print(f"\nРаботающие сервисы: {working_services}")
    return len(working_services) > 0

@pytest.mark.asyncio
async def test_mcp_tool_calls():
    """Тестирование вызова простых инструментов"""
    print("\n=== Тестирование вызова инструментов ===")
    
    # Простые инструменты для тестирования
    test_calls = [
        ("binance-mcp", "http://localhost:8000", "get_server_time", {}),
        ("tradingview-mcp", "http://localhost:8060", "health", {}),
        ("memory-mcp", "http://localhost:8050", "health", {}),
        ("shell-mcp", "http://localhost:8070", "health", {}),
        ("backtesting-mcp", "http://localhost:8082", "health", {})
    ]
    
    results = {}
    
    for service_name, base_url, tool_name, arguments in test_calls:
        print(f"\n--- Тестирование {service_name}.{tool_name} ---")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    f"{base_url}/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": tool_name,
                            "arguments": arguments
                        }
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream"
                    }
                )
                
                print(f"MCP tools/call {tool_name}: {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    print(f"Response: {result}")
                    
                    if "result" in result:
                        print(f"✅ {service_name}.{tool_name} работает!")
                        results[f"{service_name}.{tool_name}"] = True
                    elif "error" in result:
                        print(f"❌ {service_name}.{tool_name}: MCP Error - {result['error']}")
                        results[f"{service_name}.{tool_name}"] = False
                    else:
                        print(f"❓ {service_name}.{tool_name}: Неожиданный ответ - {result}")
                        results[f"{service_name}.{tool_name}"] = False
                else:
                    print(f"❌ {service_name}.{tool_name}: HTTP Error {response.status_code} - {response.text[:100]}")
                    results[f"{service_name}.{tool_name}"] = False
                    
            except Exception as e:
                print(f"❌ {service_name}.{tool_name}: Exception - {e}")
                results[f"{service_name}.{tool_name}"] = False
    
    print(f"\n=== Результаты вызовов инструментов ===")
    working_tools = []
    for tool_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {tool_name}")
        if success:
            working_tools.append(tool_name)
    
    print(f"\nРаботающие инструменты: {working_tools}")
    return len(working_tools) > 0

if __name__ == "__main__":
    async def main():
        print("Запуск тестов MCP серверов...")
        
        # Тест 1: Проверка ответов серверов
        success1 = await test_mcp_servers_respond()
        
        # Тест 2: Вызов инструментов
        success2 = await test_mcp_tool_calls()
        
        print(f"\n=== Итоговые результаты ===")
        print(f"Серверы отвечают: {'✅' if success1 else '❌'}")
        print(f"Инструменты работают: {'✅' if success2 else '❌'}")
        
        if success1 or success2:
            print("\n🎉 MCP серверы работают!")
        else:
            print("\n❌ MCP серверы не работают")
    
    asyncio.run(main())
