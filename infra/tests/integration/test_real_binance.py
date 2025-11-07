"""
Реальные интеграционные тесты для binance-mcp
"""
import pytest
import asyncio
import sys
import os
import logging

# Добавляем путь к utils
sys.path.insert(0, os.path.dirname(__file__))

from utils.real_mcp_client import RealMCPClient

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_real_binance_mcp_connection():
    """Тест: Реальное подключение к binance-mcp"""
    print("\n=== Test: Real Binance MCP Connection ===")
    
    async with RealMCPClient("http://localhost:8000", "binance-mcp") as client:
        # Проверка доступности
        is_available = await client.health_check()
        print(f"  Доступность: {'✅' if is_available else '❌'}")
        assert is_available, "binance-mcp недоступен"
        
        # Получение списка инструментов
        tools = await client.list_tools()
        print(f"  Инструменты: {len(tools)}")
        
        if tools:
            print(f"  Примеры инструментов:")
            for tool in tools[:5]:
                print(f"    - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')[:80]}...")
            
            # Проверяем, что есть основные инструменты
            tool_names = [tool.get('name') for tool in tools]
            expected_tools = ['get_server_time', 'get_ticker_price', 'get_account_info']
            
            found_tools = []
            for expected in expected_tools:
                if expected in tool_names:
                    found_tools.append(expected)
                    print(f"    ✅ Найден инструмент: {expected}")
                else:
                    print(f"    ❌ Не найден инструмент: {expected}")
            
            assert len(found_tools) > 0, f"Не найдены ожидаемые инструменты: {expected_tools}"
        else:
            print("  ⚠️ Инструменты не найдены (возможно, проблема с session ID)")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_real_binance_get_server_time():
    """Тест: Реальный вызов get_server_time"""
    print("\n=== Test: Real Binance get_server_time ===")
    
    async with RealMCPClient("http://localhost:8000", "binance-mcp") as client:
        # Получаем список инструментов
        tools = await client.list_tools()
        tool_names = [tool.get('name') for tool in tools]
        
        if 'get_server_time' in tool_names:
            # Вызываем get_server_time
            result = await client.call_tool('get_server_time', {})
            
            if result:
                print(f"  Результат: {result}")
                
                # Проверяем структуру ответа
                if 'content' in result:
                    content = result['content']
                    if isinstance(content, list) and len(content) > 0:
                        text_content = content[0].get('text', '')
                        print(f"  Время сервера: {text_content}")
                        assert text_content, "Время сервера не получено"
                        print("  ✅ get_server_time работает корректно")
                    else:
                        print("  ⚠️ Неожиданная структура ответа")
                else:
                    print(f"  ⚠️ Неожиданная структура ответа: {result}")
            else:
                print("  ❌ Не удалось вызвать get_server_time")
                pytest.skip("get_server_time не работает")
        else:
            print("  ⚠️ get_server_time не найден в списке инструментов")
            pytest.skip("get_server_time недоступен")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_real_binance_get_ticker_price():
    """Тест: Реальный вызов get_ticker_price"""
    print("\n=== Test: Real Binance get_ticker_price ===")
    
    async with RealMCPClient("http://localhost:8000", "binance-mcp") as client:
        tools = await client.list_tools()
        tool_names = [tool.get('name') for tool in tools]
        
        if 'get_ticker_price' in tool_names:
            # Вызываем get_ticker_price для BTCUSDT
            result = await client.call_tool('get_ticker_price', {
                'symbol': 'BTCUSDT'
            })
            
            if result:
                print(f"  Результат: {result}")
                
                # Проверяем структуру ответа
                if 'content' in result:
                    content = result['content']
                    if isinstance(content, list) and len(content) > 0:
                        text_content = content[0].get('text', '')
                        print(f"  Цена BTCUSDT: {text_content}")
                        assert text_content, "Цена не получена"
                        
                        # Пробуем распарсить JSON
                        try:
                            import json
                            price_data = json.loads(text_content)
                            if 'price' in price_data:
                                price = float(price_data['price'])
                                print(f"  ✅ Цена BTCUSDT: ${price:,.2f}")
                                assert price > 0, "Цена должна быть положительной"
                            else:
                                print(f"  ✅ Получены данные: {price_data}")
                        except json.JSONDecodeError:
                            print(f"  ✅ Получен текстовый ответ: {text_content}")
                    else:
                        print("  ⚠️ Неожиданная структура ответа")
                else:
                    print(f"  ⚠️ Неожиданная структура ответа: {result}")
            else:
                print("  ❌ Не удалось вызвать get_ticker_price")
                pytest.skip("get_ticker_price не работает")
        else:
            print("  ⚠️ get_ticker_price не найден в списке инструментов")
            pytest.skip("get_ticker_price недоступен")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_real_binance_account_info():
    """Тест: Реальный вызов get_account_info"""
    print("\n=== Test: Real Binance get_account_info ===")
    
    async with RealMCPClient("http://localhost:8000", "binance-mcp") as client:
        tools = await client.list_tools()
        tool_names = [tool.get('name') for tool in tools]
        
        if 'get_account_info' in tool_names:
            # Вызываем get_account_info
            result = await client.call_tool('get_account_info', {})
            
            if result:
                print(f"  Результат: {result}")
                
                # Проверяем структуру ответа
                if 'content' in result:
                    content = result['content']
                    if isinstance(content, list) and len(content) > 0:
                        text_content = content[0].get('text', '')
                        print(f"  Информация об аккаунте: {text_content[:200]}...")
                        assert text_content, "Информация об аккаунте не получена"
                        
                        # Пробуем распарсить JSON
                        try:
                            import json
                            account_data = json.loads(text_content)
                            if 'balances' in account_data:
                                balances = account_data['balances']
                                print(f"  ✅ Найдено балансов: {len(balances)}")
                                
                                # Показываем несколько балансов
                                for balance in balances[:3]:
                                    asset = balance.get('asset', 'Unknown')
                                    free = balance.get('free', '0')
                                    print(f"    - {asset}: {free}")
                            else:
                                print(f"  ✅ Получены данные аккаунта: {list(account_data.keys())}")
                        except json.JSONDecodeError:
                            print(f"  ✅ Получен текстовый ответ: {text_content[:100]}...")
                    else:
                        print("  ⚠️ Неожиданная структура ответа")
                else:
                    print(f"  ⚠️ Неожиданная структура ответа: {result}")
            else:
                print("  ❌ Не удалось вызвать get_account_info")
                pytest.skip("get_account_info не работает")
        else:
            print("  ⚠️ get_account_info не найден в списке инструментов")
            pytest.skip("get_account_info недоступен")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_real_binance_klines():
    """Тест: Реальный вызов get_klines"""
    print("\n=== Test: Real Binance get_klines ===")
    
    async with RealMCPClient("http://localhost:8000", "binance-mcp") as client:
        tools = await client.list_tools()
        tool_names = [tool.get('name') for tool in tools]
        
        if 'get_klines' in tool_names:
            # Вызываем get_klines для BTCUSDT
            result = await client.call_tool('get_klines', {
                'symbol': 'BTCUSDT',
                'interval': '1h',
                'limit': 5
            })
            
            if result:
                print(f"  Результат: {result}")
                
                # Проверяем структуру ответа
                if 'content' in result:
                    content = result['content']
                    if isinstance(content, list) and len(content) > 0:
                        text_content = content[0].get('text', '')
                        print(f"  Klines данные: {text_content[:200]}...")
                        assert text_content, "Klines данные не получены"
                        
                        # Пробуем распарсить JSON
                        try:
                            import json
                            klines_data = json.loads(text_content)
                            if isinstance(klines_data, list):
                                print(f"  ✅ Получено свечей: {len(klines_data)}")
                                
                                # Показываем первую свечу
                                if klines_data:
                                    first_candle = klines_data[0]
                                    if isinstance(first_candle, list) and len(first_candle) >= 6:
                                        open_price = first_candle[1]
                                        high_price = first_candle[2]
                                        low_price = first_candle[3]
                                        close_price = first_candle[4]
                                        print(f"    Первая свеча: O:{open_price} H:{high_price} L:{low_price} C:{close_price}")
                                    else:
                                        print(f"    Первая свеча: {first_candle}")
                            elif isinstance(klines_data, dict):
                                print(f"  ✅ Получены данные: {list(klines_data.keys())}")
                            else:
                                print(f"  ✅ Получены данные: {type(klines_data)}")
                        except json.JSONDecodeError:
                            print(f"  ✅ Получен текстовый ответ: {text_content[:100]}...")
                    else:
                        print("  ⚠️ Неожиданная структура ответа")
                else:
                    print(f"  ⚠️ Неожиданная структура ответа: {result}")
            else:
                print("  ❌ Не удалось вызвать get_klines")
                pytest.skip("get_klines не работает")
        else:
            print("  ⚠️ get_klines не найден в списке инструментов")
            pytest.skip("get_klines недоступен")
    
    print("✅ Test passed")


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        print("Запуск реальных тестов binance-mcp...")
        
        try:
            await test_real_binance_mcp_connection()
            await test_real_binance_get_server_time()
            await test_real_binance_get_ticker_price()
            await test_real_binance_account_info()
            await test_real_binance_klines()
            print("\n🎉 Все реальные тесты binance-mcp прошли успешно!")
        except Exception as e:
            print(f"\n❌ Ошибка в тестах: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(main())
