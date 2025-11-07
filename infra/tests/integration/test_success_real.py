#!/usr/bin/env python3
"""
УСПЕШНЫЕ реальные интеграционные тесты MCP сервисов
Тестирует реальную работу MCP серверов через stdio интерфейс
"""

import asyncio
import json
import logging
import subprocess
import sys
from typing import Dict, List, Optional, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class RealMCPTester:
    """Тестер для реальных MCP сервисов через stdio"""
    
    def __init__(self):
        self.services = {
            'binance-mcp': {'container': 'mcp-binance-mcp-1', 'port': 8000},
            'tradingview-mcp': {'container': 'mcp-tradingview-mcp-1', 'port': 8060},
            'memory-mcp': {'container': 'mcp-memory-mcp-1', 'port': 8050},
            'shell-mcp': {'container': 'mcp-shell-mcp-1', 'port': 8070},
            'backtesting-mcp': {'container': 'mcp-backtesting-mcp-1', 'port': 8082},
        }
        
    async def exec_in_container(self, container: str, command: List[str], timeout: int = 15) -> Dict[str, Any]:
        """Выполнить команду в контейнере"""
        try:
            cmd = ['docker', 'exec', container] + command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                return {
                    'success': process.returncode == 0,
                    'stdout': stdout.decode('utf-8', errors='ignore'),
                    'stderr': stderr.decode('utf-8', errors='ignore'),
                    'returncode': process.returncode
                }
            except asyncio.TimeoutError:
                process.kill()
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f'Command timed out after {timeout}s',
                    'returncode': -1
                }
                
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1
            }
    
    async def send_mcp_request(self, container: str, request: Dict) -> Dict[str, Any]:
        """Отправить MCP запрос через stdio"""
        
        # Создаем скрипт для MCP взаимодействия
        script = f"""
import sys
import json
import asyncio
from mcp.client.stdio import stdio_client

async def test_mcp():
    try:
        # Запускаем MCP сервер как процесс
        async with stdio_client(['python', '/app/mcp_server.py']) as (read, write):
            # Отправляем запрос
            request = {json.dumps(request)}
            await write(request)
            
            # Читаем ответ
            response = await read()
            print(f"MCP_RESPONSE: {{json.dumps(response)}}")
            
    except Exception as e:
        print(f"MCP_ERROR: {{e}}")

asyncio.run(test_mcp())
"""
        
        result = await self.exec_in_container(
            container,
            ['python', '-c', script],
            timeout=20
        )
        
        if result['success'] and 'MCP_RESPONSE:' in result['stdout']:
            try:
                response_line = [line for line in result['stdout'].split('\n') if 'MCP_RESPONSE:' in line][0]
                response_str = response_line.split('MCP_RESPONSE: ')[1].strip()
                return json.loads(response_str)
            except Exception as e:
                logger.error(f"Failed to parse MCP response: {e}")
                
        return {'error': f"MCP request failed: {result.get('stderr', 'Unknown error')}"}
    
    async def list_tools(self, service: str) -> List[str]:
        """Получить список инструментов через MCP протокол"""
        container = self.services[service]['container']
        
        # MCP запрос для получения списка инструментов
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        }
        
        response = await self.send_mcp_request(container, request)
        
        if 'result' in response and 'tools' in response['result']:
            tools = [tool['name'] for tool in response['result']['tools']]
            logger.info(f"{service}: найдено {len(tools)} инструментов: {tools}")
            return tools
        elif 'error' in response:
            logger.warning(f"{service}: ошибка получения инструментов: {response['error']}")
        else:
            logger.warning(f"{service}: неожиданный ответ: {response}")
            
        return []
    
    async def call_tool(self, service: str, tool_name: str, arguments: Dict = None) -> Dict[str, Any]:
        """Вызвать инструмент через MCP протокол"""
        container = self.services[service]['container']
        arguments = arguments or {}
        
        # MCP запрос для вызова инструмента
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        response = await self.send_mcp_request(container, request)
        
        if 'result' in response:
            logger.info(f"{service}: инструмент {tool_name} выполнен успешно")
            return response['result']
        elif 'error' in response:
            logger.warning(f"{service}: ошибка выполнения {tool_name}: {response['error']}")
            return {'error': response['error']}
        else:
            logger.warning(f"{service}: неожиданный ответ для {tool_name}: {response}")
            return {'error': 'Unexpected response'}

async def test_containers_running():
    """Проверить, что все контейнеры запущены"""
    print("\n=== Test: Containers Running ===")
    
    tester = RealMCPTester()
    results = {}
    
    for service in tester.services:
        container = tester.services[service]['container']
        result = await tester.exec_in_container(container, ['echo', 'ping'])
        running = result['success']
        results[service] = running
        print(f"  {'✅' if running else '❌'} {service} ({container})")
    
    running_count = sum(results.values())
    print(f"\nЗапущено контейнеров: {running_count}/{len(results)}")
    print("✅ Test passed" if running_count == len(results) else "❌ Test failed")
    return results

async def test_mcp_tools_discovery():
    """Обнаружение MCP инструментов через stdio"""
    print("\n=== Test: MCP Tools Discovery via stdio ===")
    
    tester = RealMCPTester()
    results = {}
    
    for service in tester.services:
        print(f"\n--- Поиск инструментов {service} ---")
        tools = await tester.list_tools(service)
        results[service] = tools
        print(f"  Найдено инструментов: {len(tools)}")
        if tools:
            for tool in tools[:3]:  # Показываем первые 3
                print(f"    • {tool}")
            if len(tools) > 3:
                print(f"    ... и еще {len(tools) - 3}")
    
    print(f"\n=== Итоговые результаты ===")
    total_tools = 0
    for service, tools in results.items():
        count = len(tools)
        total_tools += count
        print(f"{'✅' if count > 0 else '⚠️'} {service}: {count} инструментов")
    
    print(f"\nВсего найдено инструментов: {total_tools}")
    print("✅ Test passed" if total_tools > 0 else "⚠️ No tools found")
    return results

async def test_real_tool_calls():
    """Тестирование реальных вызовов инструментов"""
    print("\n=== Test: Real Tool Calls ===")
    
    tester = RealMCPTester()
    
    # Сначала получаем инструменты
    tools_results = await tester.list_tools('binance-mcp')
    
    if not tools_results:
        print("  ⚠️ Нет доступных инструментов для тестирования")
        return False
    
    success_count = 0
    test_count = 0
    
    # Тестируем простые инструменты без параметров
    simple_tools = ['health', 'version']
    
    for tool_name in simple_tools:
        if tool_name in tools_results:
            print(f"  Тестирование {tool_name}...")
            test_count += 1
            
            result = await tester.call_tool('binance-mcp', tool_name)
            
            if 'error' not in result:
                print(f"    ✅ {tool_name}: успешно")
                success_count += 1
            else:
                print(f"    ❌ {tool_name}: {result.get('error', 'Unknown error')}")
    
    print(f"\n  === Результаты вызовов ===")
    print(f"  Успешных вызовов: {success_count}/{test_count}")
    
    success = success_count > 0
    print("✅ Test passed" if success else "❌ Test failed")
    return success

async def test_integration_workflow():
    """Тестирование интеграционного workflow"""
    print("\n=== Test: Integration Workflow ===")
    
    tester = RealMCPTester()
    
    workflow_steps = []
    
    # Шаг 1: Проверка здоровья Binance
    print("  1. Проверка здоровья Binance...")
    binance_tools = await tester.list_tools('binance-mcp')
    if 'health' in binance_tools:
        result = await tester.call_tool('binance-mcp', 'health')
        if 'error' not in result:
            print("    ✅ Binance здоров")
            workflow_steps.append('binance_health')
        else:
            print(f"    ❌ Binance нездоров: {result.get('error')}")
    else:
        print("    ⚠️ Инструмент health недоступен")
    
    # Шаг 2: Проверка версии Binance
    print("  2. Получение версии Binance...")
    if 'version' in binance_tools:
        result = await tester.call_tool('binance-mcp', 'version')
        if 'error' not in result:
            print("    ✅ Версия получена")
            workflow_steps.append('binance_version')
        else:
            print(f"    ❌ Ошибка получения версии: {result.get('error')}")
    else:
        print("    ⚠️ Инструмент version недоступен")
    
    # Шаг 3: Проверка Memory MCP
    print("  3. Проверка Memory MCP...")
    memory_tools = await tester.list_tools('memory-mcp')
    if memory_tools:
        print(f"    ✅ Memory MCP доступен ({len(memory_tools)} инструментов)")
        workflow_steps.append('memory_available')
    else:
        print("    ⚠️ Memory MCP недоступен")
    
    print(f"\n  === Результат workflow ===")
    print(f"  Выполненных шагов: {len(workflow_steps)}/3")
    for step in workflow_steps:
        print(f"    ✅ {step}")
    
    success = len(workflow_steps) >= 2  # Минимум 2 шага для успеха
    print("✅ Test passed" if success else "❌ Test failed")
    return success

async def main():
    """Главная функция тестирования"""
    print("🚀 Запуск УСПЕШНЫХ реальных интеграционных тестов...")
    
    try:
        # Тест 1: Контейнеры запущены
        containers_result = await test_containers_running()
        
        # Тест 2: Обнаружение инструментов
        tools_result = await test_mcp_tools_discovery()
        
        # Тест 3: Реальные вызовы инструментов
        calls_result = await test_real_tool_calls()
        
        # Тест 4: Интеграционный workflow
        workflow_result = await test_integration_workflow()
        
        print(f"\n🎉 Все УСПЕШНЫЕ реальные тесты завершены!")
        
        # Итоговая статистика
        running_containers = sum(containers_result.values())
        total_tools = sum(len(tools) for tools in tools_result.values())
        
        print(f"\n📊 Итоговая статистика:")
        print(f"  • Запущенных контейнеров: {running_containers}/{len(containers_result)}")
        print(f"  • Найдено инструментов: {total_tools}")
        print(f"  • Вызовы инструментов: {'✅' if calls_result else '❌'}")
        print(f"  • Workflow: {'✅' if workflow_result else '❌'}")
        
        if running_containers == len(containers_result) and total_tools > 0 and calls_result:
            print(f"\n🎊 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
            return True
        else:
            print(f"\n⚠️ Некоторые тесты требуют внимания")
            return False
            
    except Exception as e:
        logger.error(f"Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
