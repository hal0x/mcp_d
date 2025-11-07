"""
Тест для проверки работы MCP через WebSocket
"""
import pytest
import asyncio
import websockets
import json
import logging

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_mcp_websocket():
    """Проверка работы MCP через WebSocket"""
    print("=== Тестирование MCP через WebSocket ===")
    
    try:
        # Пробуем подключиться к WebSocket
        uri = "ws://localhost:8000/mcp"
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket подключение установлено")
            
            # Отправляем запрос на получение списка инструментов
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            }
            
            await websocket.send(json.dumps(request))
            print("📤 Запрос отправлен")
            
            # Ждем ответ
            response = await websocket.recv()
            print(f"📥 Ответ получен: {response[:200]}...")
            
            result = json.loads(response)
            if "result" in result:
                tools = result["result"]["tools"]
                print(f"✅ Доступно инструментов: {len(tools)}")
                for tool in tools[:3]:
                    print(f"  - {tool.get('name', 'Unknown')}")
                return True
            elif "error" in result:
                print(f"❌ MCP Error: {result['error']}")
            else:
                print(f"❓ Неожиданный ответ: {result}")
                
    except websockets.exceptions.InvalidURI:
        print("❌ WebSocket endpoint не найден")
    except websockets.exceptions.ConnectionClosed:
        print("❌ WebSocket соединение закрыто")
    except Exception as e:
        print(f"❌ Ошибка WebSocket: {e}")
    
    return False

@pytest.mark.asyncio
async def test_mcp_tool_call_websocket():
    """Тестирование вызова инструмента через WebSocket"""
    print("\n=== Тестирование вызова инструмента через WebSocket ===")
    
    try:
        uri = "ws://localhost:8000/mcp"
        async with websockets.connect(uri) as websocket:
            # Вызываем простой инструмент
            request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "get_server_time",
                    "arguments": {}
                }
            }
            
            await websocket.send(json.dumps(request))
            print("📤 Запрос на вызов инструмента отправлен")
            
            # Ждем ответ
            response = await websocket.recv()
            print(f"📥 Ответ получен: {response[:200]}...")
            
            result = json.loads(response)
            if "result" in result:
                print("✅ Инструмент get_server_time работает!")
                return True
            elif "error" in result:
                print(f"❌ MCP Error: {result['error']}")
            else:
                print(f"❓ Неожиданный ответ: {result}")
                
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    return False

if __name__ == "__main__":
    async def main():
        print("Запуск тестов MCP через WebSocket...")
        
        # Тест 1: WebSocket подключение
        success1 = await test_mcp_websocket()
        
        # Тест 2: Вызов инструмента через WebSocket
        success2 = await test_mcp_tool_call_websocket()
        
        print(f"\n=== Результаты ===")
        print(f"WebSocket подключение: {'✅' if success1 else '❌'}")
        print(f"Вызов инструмента: {'✅' if success2 else '❌'}")
        
        if success1 or success2:
            print("\n🎉 MCP работает через WebSocket!")
        else:
            print("\n❌ MCP не работает через WebSocket")
    
    asyncio.run(main())
