"""
Улучшенные реальные интеграционные тесты
"""
import pytest
import asyncio
import sys
import os
import logging
import json

# Добавляем путь к utils
sys.path.insert(0, os.path.dirname(__file__))

from utils.hybrid_mcp_client import HybridMCPClient

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_all_services_real_connection():
    """Тест: Реальное подключение ко всем MCP сервисам"""
    print("\n=== Test: Real Connection to All MCP Services ===")
    
    services = [
        ("binance-mcp", "http://localhost:8000"),
        ("tradingview-mcp", "http://localhost:8060"),
        ("memory-mcp", "http://localhost:8050"),
        ("shell-mcp", "http://localhost:8070"),
        ("backtesting-mcp", "http://localhost:8082")
    ]
    
    results = {}
    
    for service_name, base_url in services:
        print(f"\n--- Тестирование {service_name} ({base_url}) ---")
        
        try:
            async with HybridMCPClient(base_url, service_name) as client:
                # Получаем информацию о клиенте
                client_info = client.get_client_info()
                print(f"  Клиент: {client_info['client_type']}")
                
                # Проверка доступности
                is_available = await client.health_check()
                print(f"  Доступность: {'✅' if is_available else '❌'}")
                
                # Получение списка инструментов
                tools = await client.list_tools()
                print(f"  Инструменты: {len(tools)}")
                
                if tools:
                    print(f"  Примеры инструментов:")
                    for tool in tools[:3]:
                        name = tool.get('name', 'Unknown')
                        desc = tool.get('description', 'No description')[:60]
                        print(f"    - {name}: {desc}...")
                
                results[service_name] = {
                    "available": is_available,
                    "client_type": client_info['client_type'],
                    "tools_count": len(tools),
                    "tools": [t.get('name') for t in tools[:5]]
                }
                
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            results[service_name] = {
                "available": False,
                "error": str(e),
                "client_type": None,
                "tools_count": 0
            }
    
    print(f"\n=== Итоговые результаты ===")
    working_services = []
    for service_name, result in results.items():
        status = "✅" if result["available"] else "❌"
        client_type = result.get("client_type", "FAILED")
        tools_count = result.get("tools_count", 0)
        print(f"{status} {service_name}: {client_type} ({tools_count} tools)")
        
        if result["available"]:
            working_services.append(service_name)
    
    print(f"\nРаботающие сервисы: {len(working_services)}/{len(services)}")
    
    # Сохраняем результаты для анализа
    with open("results/real_services_test.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    assert len(working_services) > 0, "Ни один сервис не работает"
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_real_binance_tools():
    """Тест: Реальные инструменты binance-mcp"""
    print("\n=== Test: Real Binance Tools ===")
    
    async with HybridMCPClient("http://localhost:8000", "binance-mcp") as client:
        client_info = client.get_client_info()
        print(f"  Используется клиент: {client_info['client_type']}")
        
        # Получаем список инструментов
        tools = await client.list_tools()
        print(f"  Найдено инструментов: {len(tools)}")
        
        if not tools:
            print("  ⚠️ Инструменты не найдены, пропускаем тест")
            pytest.skip("Инструменты binance-mcp недоступны")
        
        # Ищем основные инструменты
        tool_names = [tool.get('name') for tool in tools]
        expected_tools = ['get_server_time', 'get_ticker_price', 'get_account_info', 'get_klines']
        
        found_tools = []
        for expected in expected_tools:
            if expected in tool_names:
                found_tools.append(expected)
                print(f"    ✅ Найден: {expected}")
            else:
                print(f"    ❌ Не найден: {expected}")
        
        # Пробуем вызвать доступные инструменты
        for tool_name in found_tools[:2]:  # Тестируем первые 2
            print(f"\n  --- Тестирование {tool_name} ---")
            
            try:
                if tool_name == 'get_server_time':
                    result = await client.call_tool(tool_name, {})
                elif tool_name == 'get_ticker_price':
                    result = await client.call_tool(tool_name, {'symbol': 'BTCUSDT'})
                elif tool_name == 'get_klines':
                    result = await client.call_tool(tool_name, {
                        'symbol': 'BTCUSDT', 
                        'interval': '1h', 
                        'limit': 1
                    })
                else:
                    result = await client.call_tool(tool_name, {})
                
                if result:
                    print(f"    ✅ {tool_name} работает")
                    print(f"    Результат: {str(result)[:100]}...")
                    
                    # Пробуем извлечь полезную информацию
                    if 'content' in result:
                        content = result['content']
                        if isinstance(content, list) and content:
                            text_content = content[0].get('text', '')
                            if text_content:
                                try:
                                    data = json.loads(text_content)
                                    if tool_name == 'get_ticker_price' and 'price' in data:
                                        print(f"    💰 Цена BTCUSDT: ${float(data['price']):,.2f}")
                                    elif tool_name == 'get_server_time' and 'serverTime' in data:
                                        print(f"    🕐 Время сервера: {data['serverTime']}")
                                except json.JSONDecodeError:
                                    print(f"    📄 Текстовый ответ: {text_content[:50]}...")
                else:
                    print(f"    ❌ {tool_name} не работает")
                    
            except Exception as e:
                print(f"    ❌ Ошибка при вызове {tool_name}: {e}")
        
        assert len(found_tools) > 0, f"Не найдены ожидаемые инструменты: {expected_tools}"
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_real_tradingview_tools():
    """Тест: Реальные инструменты tradingview-mcp"""
    print("\n=== Test: Real TradingView Tools ===")
    
    async with HybridMCPClient("http://localhost:8060", "tradingview-mcp") as client:
        client_info = client.get_client_info()
        print(f"  Используется клиент: {client_info['client_type']}")
        
        # Получаем список инструментов
        tools = await client.list_tools()
        print(f"  Найдено инструментов: {len(tools)}")
        
        if not tools:
            print("  ⚠️ Инструменты не найдены, пропускаем тест")
            pytest.skip("Инструменты tradingview-mcp недоступны")
        
        # Ищем основные инструменты
        tool_names = [tool.get('name') for tool in tools]
        expected_tools = ['health', 'coin_analysis', 'top_gainers', 'exchanges_list']
        
        found_tools = []
        for expected in expected_tools:
            if expected in tool_names:
                found_tools.append(expected)
                print(f"    ✅ Найден: {expected}")
            else:
                print(f"    ❌ Не найден: {expected}")
        
        # Пробуем вызвать доступные инструменты
        for tool_name in found_tools[:2]:  # Тестируем первые 2
            print(f"\n  --- Тестирование {tool_name} ---")
            
            try:
                if tool_name == 'health':
                    result = await client.call_tool(tool_name, {})
                elif tool_name == 'coin_analysis':
                    result = await client.call_tool(tool_name, {
                        'symbol': 'BTCUSDT',
                        'exchange': 'BINANCE'
                    })
                elif tool_name == 'exchanges_list':
                    result = await client.call_tool(tool_name, {})
                else:
                    result = await client.call_tool(tool_name, {})
                
                if result:
                    print(f"    ✅ {tool_name} работает")
                    print(f"    Результат: {str(result)[:100]}...")
                else:
                    print(f"    ❌ {tool_name} не работает")
                    
            except Exception as e:
                print(f"    ❌ Ошибка при вызове {tool_name}: {e}")
        
        assert len(found_tools) > 0, f"Не найдены ожидаемые инструменты: {expected_tools}"
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_real_integration_workflow():
    """Тест: Реальный интеграционный workflow между сервисами"""
    print("\n=== Test: Real Integration Workflow ===")
    
    # Шаг 1: Получаем цену от Binance
    print("  1. Получение цены от Binance...")
    binance_result = None
    
    try:
        async with HybridMCPClient("http://localhost:8000", "binance-mcp") as binance_client:
            tools = await binance_client.list_tools()
            tool_names = [t.get('name') for t in tools]
            
            if 'get_ticker_price' in tool_names:
                binance_result = await binance_client.call_tool('get_ticker_price', {
                    'symbol': 'BTCUSDT'
                })
                
                if binance_result:
                    print("    ✅ Цена получена от Binance")
                else:
                    print("    ❌ Не удалось получить цену от Binance")
            else:
                print("    ⚠️ get_ticker_price недоступен в Binance")
    except Exception as e:
        print(f"    ❌ Ошибка Binance: {e}")
    
    # Шаг 2: Анализируем с TradingView
    print("  2. Анализ с TradingView...")
    tradingview_result = None
    
    try:
        async with HybridMCPClient("http://localhost:8060", "tradingview-mcp") as tv_client:
            tools = await tv_client.list_tools()
            tool_names = [t.get('name') for t in tools]
            
            if 'coin_analysis' in tool_names:
                tradingview_result = await tv_client.call_tool('coin_analysis', {
                    'symbol': 'BTCUSDT',
                    'exchange': 'BINANCE'
                })
                
                if tradingview_result:
                    print("    ✅ Анализ получен от TradingView")
                else:
                    print("    ❌ Не удалось получить анализ от TradingView")
            else:
                print("    ⚠️ coin_analysis недоступен в TradingView")
    except Exception as e:
        print(f"    ❌ Ошибка TradingView: {e}")
    
    # Шаг 3: Сохраняем в память (если доступно)
    print("  3. Сохранение в память...")
    memory_result = None
    
    try:
        async with HybridMCPClient("http://localhost:8050", "memory-mcp") as memory_client:
            tools = await memory_client.list_tools()
            tool_names = [t.get('name') for t in tools]
            
            if 'ingest_records' in tool_names and (binance_result or tradingview_result):
                records = []
                if binance_result:
                    records.append({
                        "type": "binance_price",
                        "symbol": "BTCUSDT",
                        "data": binance_result
                    })
                if tradingview_result:
                    records.append({
                        "type": "tradingview_analysis", 
                        "symbol": "BTCUSDT",
                        "data": tradingview_result
                    })
                
                memory_result = await memory_client.call_tool('ingest_records', {
                    'records': records
                })
                
                if memory_result:
                    print("    ✅ Данные сохранены в память")
                else:
                    print("    ❌ Не удалось сохранить в память")
            else:
                print("    ⚠️ ingest_records недоступен или нет данных для сохранения")
    except Exception as e:
        print(f"    ❌ Ошибка Memory: {e}")
    
    # Результат workflow
    workflow_success = any([binance_result, tradingview_result, memory_result])
    
    print(f"\n  === Результат workflow ===")
    print(f"  Binance: {'✅' if binance_result else '❌'}")
    print(f"  TradingView: {'✅' if tradingview_result else '❌'}")
    print(f"  Memory: {'✅' if memory_result else '❌'}")
    print(f"  Общий результат: {'✅' if workflow_success else '❌'}")
    
    # Сохраняем результаты workflow
    workflow_data = {
        "timestamp": asyncio.get_event_loop().time(),
        "binance_success": binance_result is not None,
        "tradingview_success": tradingview_result is not None,
        "memory_success": memory_result is not None,
        "overall_success": workflow_success
    }
    
    with open("results/real_workflow_test.json", "w") as f:
        json.dump(workflow_data, f, indent=2, ensure_ascii=False)
    
    assert workflow_success, "Ни один шаг workflow не выполнился успешно"
    print("✅ Test passed")


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создаем директорию для результатов
    os.makedirs("results", exist_ok=True)
    
    async def main():
        print("Запуск улучшенных реальных интеграционных тестов...")
        
        try:
            await test_all_services_real_connection()
            await test_real_binance_tools()
            await test_real_tradingview_tools()
            await test_real_integration_workflow()
            print("\n🎉 Все реальные интеграционные тесты завершены!")
        except Exception as e:
            print(f"\n❌ Ошибка в тестах: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(main())
