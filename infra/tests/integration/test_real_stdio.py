"""
Реальные интеграционные тесты через Docker exec (stdio)
"""
import pytest
import asyncio
import subprocess
import json
import sys
import os
import logging

logger = logging.getLogger(__name__)

class DockerMCPClient:
    """Клиент для работы с MCP через Docker exec (stdio)"""
    
    def __init__(self, container_name: str, service_name: str):
        self.container_name = container_name
        self.service_name = service_name
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        pass
    
    async def is_container_running(self) -> bool:
        """Проверка, что контейнер запущен"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return "Up" in result.stdout
        except Exception as e:
            logger.error(f"Error checking container {self.container_name}: {e}")
            return False
    
    async def call_mcp_stdio(self, request: dict) -> dict:
        """Вызов MCP через stdio"""
        try:
            # Создаем JSON-RPC запрос
            request_json = json.dumps(request)
            
            # Выполняем команду через docker exec
            cmd = [
                "docker", "exec", "-i", self.container_name,
                "python", "-c", f"""
import json
import sys
import asyncio
from mcp_server import create_server

async def main():
    server = create_server()
    
    # Читаем запрос
    request = {repr(request)}
    
    # Обрабатываем запрос
    if request['method'] == 'tools/list':
        tools = []
        for name, tool in server._tools.items():
            tools.append({{
                'name': name,
                'description': tool.description or '',
                'inputSchema': tool.inputSchema or {{}}
            }})
        response = {{
            'jsonrpc': '2.0',
            'id': request['id'],
            'result': {{'tools': tools}}
        }}
    elif request['method'] == 'tools/call':
        tool_name = request['params']['name']
        arguments = request['params']['arguments']
        
        if tool_name in server._tools:
            try:
                result = await server._tools[tool_name].func(**arguments)
                response = {{
                    'jsonrpc': '2.0',
                    'id': request['id'],
                    'result': result
                }}
            except Exception as e:
                response = {{
                    'jsonrpc': '2.0',
                    'id': request['id'],
                    'error': {{'code': -32000, 'message': str(e)}}
                }}
        else:
            response = {{
                'jsonrpc': '2.0',
                'id': request['id'],
                'error': {{'code': -32601, 'message': f'Tool {{tool_name}} not found'}}
            }}
    else:
        response = {{
            'jsonrpc': '2.0',
            'id': request['id'],
            'error': {{'code': -32601, 'message': f'Method {{request["method"]}} not supported'}}
        }}
    
    print(json.dumps(response))

if __name__ == '__main__':
    asyncio.run(main())
"""
            ]
            
            # Выполняем команду
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                try:
                    response = json.loads(result.stdout.strip())
                    return response
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {result.stdout}")
                    return {"error": f"JSON decode error: {e}"}
            else:
                logger.error(f"Command failed: {result.stderr}")
                return {"error": f"Command failed: {result.stderr}"}
                
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out for {self.container_name}")
            return {"error": "Command timed out"}
        except Exception as e:
            logger.error(f"Error calling MCP stdio for {self.container_name}: {e}")
            return {"error": str(e)}
    
    async def list_tools(self) -> list:
        """Получение списка инструментов"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        }
        
        response = await self.call_mcp_stdio(request)
        
        if "result" in response and "tools" in response["result"]:
            return response["result"]["tools"]
        elif "error" in response:
            logger.error(f"Error listing tools for {self.service_name}: {response['error']}")
        
        return []
    
    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Вызов инструмента"""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        response = await self.call_mcp_stdio(request)
        
        if "result" in response:
            return response["result"]
        elif "error" in response:
            logger.error(f"Error calling tool {tool_name} for {self.service_name}: {response['error']}")
            return None
        
        return None
    
    async def health_check(self) -> bool:
        """Проверка доступности контейнера"""
        return await self.is_container_running()


@pytest.mark.asyncio
async def test_docker_containers_running():
    """Тест: Проверка, что все Docker контейнеры запущены"""
    print("\n=== Test: Docker Containers Running ===")
    
    containers = [
        ("mcp-binance-mcp-1", "binance-mcp"),
        ("mcp-tradingview-mcp-1", "tradingview-mcp"),
        ("mcp-memory-mcp-1", "memory-mcp"),
        ("mcp-shell-mcp-1", "shell-mcp"),
        ("mcp-backtesting-mcp-1", "backtesting-mcp")
    ]
    
    results = {}
    
    for container_name, service_name in containers:
        client = DockerMCPClient(container_name, service_name)
        is_running = await client.health_check()
        
        results[service_name] = is_running
        status = "✅" if is_running else "❌"
        print(f"  {status} {service_name} ({container_name})")
    
    running_count = sum(results.values())
    print(f"\nЗапущено контейнеров: {running_count}/{len(containers)}")
    
    assert running_count > 0, "Ни один контейнер не запущен"
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_real_binance_stdio():
    """Тест: Реальный вызов binance-mcp через stdio"""
    print("\n=== Test: Real Binance MCP via stdio ===")
    
    async with DockerMCPClient("mcp-binance-mcp-1", "binance-mcp") as client:
        # Проверяем, что контейнер запущен
        is_running = await client.health_check()
        if not is_running:
            pytest.skip("binance-mcp контейнер не запущен")
        
        print("  Контейнер запущен: ✅")
        
        # Получаем список инструментов
        print("  Получение списка инструментов...")
        tools = await client.list_tools()
        print(f"  Найдено инструментов: {len(tools)}")
        
        if tools:
            print("  Примеры инструментов:")
            for tool in tools[:5]:
                name = tool.get('name', 'Unknown')
                desc = tool.get('description', 'No description')[:60]
                print(f"    - {name}: {desc}...")
            
            # Пробуем вызвать get_server_time
            if any(tool.get('name') == 'get_server_time' for tool in tools):
                print("\n  Вызов get_server_time...")
                result = await client.call_tool('get_server_time', {})
                
                if result:
                    print(f"  ✅ get_server_time работает: {str(result)[:100]}...")
                else:
                    print("  ❌ get_server_time не работает")
            
            # Пробуем вызвать get_ticker_price
            if any(tool.get('name') == 'get_ticker_price' for tool in tools):
                print("\n  Вызов get_ticker_price...")
                result = await client.call_tool('get_ticker_price', {'symbol': 'BTCUSDT'})
                
                if result:
                    print(f"  ✅ get_ticker_price работает: {str(result)[:100]}...")
                else:
                    print("  ❌ get_ticker_price не работает")
        else:
            print("  ⚠️ Инструменты не найдены")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_real_tradingview_stdio():
    """Тест: Реальный вызов tradingview-mcp через stdio"""
    print("\n=== Test: Real TradingView MCP via stdio ===")
    
    async with DockerMCPClient("mcp-tradingview-mcp-1", "tradingview-mcp") as client:
        # Проверяем, что контейнер запущен
        is_running = await client.health_check()
        if not is_running:
            pytest.skip("tradingview-mcp контейнер не запущен")
        
        print("  Контейнер запущен: ✅")
        
        # Получаем список инструментов
        print("  Получение списка инструментов...")
        tools = await client.list_tools()
        print(f"  Найдено инструментов: {len(tools)}")
        
        if tools:
            print("  Примеры инструментов:")
            for tool in tools[:5]:
                name = tool.get('name', 'Unknown')
                desc = tool.get('description', 'No description')[:60]
                print(f"    - {name}: {desc}...")
            
            # Пробуем вызвать health
            if any(tool.get('name') == 'health' for tool in tools):
                print("\n  Вызов health...")
                result = await client.call_tool('health', {})
                
                if result:
                    print(f"  ✅ health работает: {str(result)[:100]}...")
                else:
                    print("  ❌ health не работает")
            
            # Пробуем вызвать exchanges_list
            if any(tool.get('name') == 'exchanges_list' for tool in tools):
                print("\n  Вызов exchanges_list...")
                result = await client.call_tool('exchanges_list', {})
                
                if result:
                    print(f"  ✅ exchanges_list работает: {str(result)[:100]}...")
                else:
                    print("  ❌ exchanges_list не работает")
        else:
            print("  ⚠️ Инструменты не найдены")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_real_integration_workflow_stdio():
    """Тест: Реальный интеграционный workflow через stdio"""
    print("\n=== Test: Real Integration Workflow via stdio ===")
    
    workflow_results = {}
    
    # Шаг 1: Binance
    print("  1. Получение данных от Binance...")
    try:
        async with DockerMCPClient("mcp-binance-mcp-1", "binance-mcp") as binance_client:
            if await binance_client.health_check():
                tools = await binance_client.list_tools()
                if any(tool.get('name') == 'get_server_time' for tool in tools):
                    result = await binance_client.call_tool('get_server_time', {})
                    if result:
                        workflow_results['binance'] = result
                        print("    ✅ Данные получены от Binance")
                    else:
                        print("    ❌ Не удалось получить данные от Binance")
                else:
                    print("    ⚠️ get_server_time недоступен в Binance")
            else:
                print("    ❌ Binance контейнер не запущен")
    except Exception as e:
        print(f"    ❌ Ошибка Binance: {e}")
    
    # Шаг 2: TradingView
    print("  2. Получение данных от TradingView...")
    try:
        async with DockerMCPClient("mcp-tradingview-mcp-1", "tradingview-mcp") as tv_client:
            if await tv_client.health_check():
                tools = await tv_client.list_tools()
                if any(tool.get('name') == 'health' for tool in tools):
                    result = await tv_client.call_tool('health', {})
                    if result:
                        workflow_results['tradingview'] = result
                        print("    ✅ Данные получены от TradingView")
                    else:
                        print("    ❌ Не удалось получить данные от TradingView")
                else:
                    print("    ⚠️ health недоступен в TradingView")
            else:
                print("    ❌ TradingView контейнер не запущен")
    except Exception as e:
        print(f"    ❌ Ошибка TradingView: {e}")
    
    # Шаг 3: Memory (если доступно)
    print("  3. Сохранение в память...")
    try:
        async with DockerMCPClient("mcp-memory-mcp-1", "memory-mcp") as memory_client:
            if await memory_client.health_check():
                tools = await memory_client.list_tools()
                if any(tool.get('name') == 'ingest_records' for tool in tools) and workflow_results:
                    records = []
                    for service, data in workflow_results.items():
                        records.append({
                            "type": f"{service}_data",
                            "data": data
                        })
                    
                    result = await memory_client.call_tool('ingest_records', {
                        'records': records
                    })
                    
                    if result:
                        workflow_results['memory'] = result
                        print("    ✅ Данные сохранены в память")
                    else:
                        print("    ❌ Не удалось сохранить в память")
                else:
                    print("    ⚠️ ingest_records недоступен или нет данных")
            else:
                print("    ❌ Memory контейнер не запущен")
    except Exception as e:
        print(f"    ❌ Ошибка Memory: {e}")
    
    # Результат
    success_count = len(workflow_results)
    print(f"\n  === Результат workflow ===")
    print(f"  Успешных шагов: {success_count}/3")
    
    for service, result in workflow_results.items():
        print(f"  ✅ {service}: {str(result)[:50]}...")
    
    # Сохраняем результаты
    with open("results/real_stdio_workflow.json", "w") as f:
        json.dump(workflow_results, f, indent=2, ensure_ascii=False)
    
    assert success_count > 0, "Ни один шаг workflow не выполнился"
    print("✅ Test passed")


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создаем директорию для результатов
    os.makedirs("results", exist_ok=True)
    
    async def main():
        print("Запуск реальных тестов через Docker stdio...")
        
        try:
            await test_docker_containers_running()
            await test_real_binance_stdio()
            await test_real_tradingview_stdio()
            await test_real_integration_workflow_stdio()
            print("\n🎉 Все реальные stdio тесты завершены!")
        except Exception as e:
            print(f"\n❌ Ошибка в тестах: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(main())
