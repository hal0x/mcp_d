"""
Финальный тест для проверки работы MCP серверов
"""
import pytest
import asyncio
import sys
import os

# Добавляем путь к utils
sys.path.insert(0, os.path.dirname(__file__))

from utils.mcp_client import MCPClient

@pytest.mark.asyncio
async def test_all_mcp_servers():
    """Проверка всех MCP серверов"""
    print("\n=== Тестирование всех MCP серверов ===\n")
    
    services = [
        ("binance-mcp", "http://localhost:8000"),
        ("tradingview-mcp", "http://localhost:8060"),
        ("memory-mcp", "http://localhost:8050"),
        ("shell-mcp", "http://localhost:8070"),
        ("backtesting-mcp", "http://localhost:8082")
    ]
    
    results = {}
    
    for service_name, base_url in services:
        print(f"--- Тестирование {service_name} ({base_url}) ---")
        
        async with MCPClient(base_url, service_name) as client:
            # Проверка доступности
            is_available = await client.health_check()
            print(f"  Доступность: {'✅' if is_available else '❌'}")
            
            if not is_available:
                results[service_name] = {"available": False, "tools": 0}
                continue
            
            # Получение списка инструментов
            tools = await client.list_tools()
            print(f"  Инструменты: {len(tools)}")
            
            if tools:
                print(f"  Примеры инструментов:")
                for tool in tools[:3]:
                    print(f"    - {tool.get('name', 'Unknown')}")
            
            results[service_name] = {
                "available": is_available,
                "tools": len(tools),
                "has_tools": len(tools) > 0
            }
        
        print()
    
    print("=== Итоговые результаты ===")
    working_services = []
    for service_name, result in results.items():
        status = "✅" if result["available"] else "❌"
        tools_status = f"({result['tools']} tools)" if result.get("has_tools") else "(no tools)"
        print(f"{status} {service_name}: {tools_status}")
        
        if result["available"]:
            working_services.append(service_name)
    
    print(f"\nРаботающие сервисы: {len(working_services)}/{len(services)}")
    
    # Успех если хотя бы один сервис работает
    assert len(working_services) > 0, "Ни один MCP сервис не работает"
    
    return results


if __name__ == "__main__":
    asyncio.run(test_all_mcp_servers())

