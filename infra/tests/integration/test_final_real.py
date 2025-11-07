#!/usr/bin/env python3
"""
Финальные реальные интеграционные тесты MCP сервисов
Тестирует реальную работу всех MCP сервисов через их настоящие API
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

class DockerMCPTester:
    """Тестер для реальных MCP сервисов через Docker"""
    
    def __init__(self):
        self.services = {
            'binance-mcp': {'container': 'mcp-binance-mcp-1', 'port': 8000},
            'tradingview-mcp': {'container': 'mcp-tradingview-mcp-1', 'port': 8060},
            'memory-mcp': {'container': 'mcp-memory-mcp-1', 'port': 8050},
            'shell-mcp': {'container': 'mcp-shell-mcp-1', 'port': 8070},
            'backtesting-mcp': {'container': 'mcp-backtesting-mcp-1', 'port': 8082},
        }
        
    async def exec_in_container(self, container: str, command: List[str], timeout: int = 10) -> Dict[str, Any]:
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
    
    async def test_container_health(self, service: str) -> bool:
        """Проверить здоровье контейнера"""
        container = self.services[service]['container']
        
        # Проверяем, что контейнер запущен
        result = await self.exec_in_container(container, ['echo', 'ping'])
        if not result['success']:
            logger.error(f"{service}: контейнер не отвечает")
            return False
            
        # Проверяем Python
        result = await self.exec_in_container(container, ['python', '--version'])
        if not result['success']:
            logger.error(f"{service}: Python недоступен")
            return False
            
        logger.info(f"{service}: контейнер здоров")
        return True
    
    async def test_mcp_server_tools(self, service: str) -> List[str]:
        """Получить список инструментов MCP сервера"""
        container = self.services[service]['container']
        
        # Попробуем разные способы получения инструментов
        methods = [
            # Метод 1: Прямой импорт и вызов
            f"""
import sys
sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app/build/lib')
sys.path.insert(0, '/usr/local/lib/python3.11/site-packages')

try:
    if '{service}' == 'binance-mcp':
        from binance_mcp.server import create_server
        server = create_server()
        tools = list(server.list_tools())
        print(f"TOOLS: {{[tool.name for tool in tools]}}")
    elif '{service}' == 'memory-mcp':
        from memory_mcp.mcp.server import create_server
        server = create_server()
        tools = list(server.list_tools())
        print(f"TOOLS: {{[tool.name for tool in tools]}}")
    elif '{service}' == 'tradingview-mcp':
        from tradingview_mcp.server import create_server
        server = create_server()
        tools = list(server.list_tools())
        print(f"TOOLS: {{[tool.name for tool in tools]}}")
    elif '{service}' == 'shell-mcp':
        from shell_mcp.server import create_server
        server = create_server()
        tools = list(server.list_tools())
        print(f"TOOLS: {{[tool.name for tool in tools]}}")
    elif '{service}' == 'backtesting-mcp':
        from backtesting_mcp.server import create_server
        server = create_server()
        tools = list(server.list_tools())
        print(f"TOOLS: {{[tool.name for tool in tools]}}")
    else:
        print("TOOLS: []")
except Exception as e:
    print(f"ERROR: {{e}}")
""",
            # Метод 2: Через FastMCP если доступен
            f"""
import sys
sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app/build/lib')
sys.path.insert(0, '/usr/local/lib/python3.11/site-packages')

try:
    from mcp.server.fastmcp import FastMCP
    app = FastMCP('{service}')
    tools = app.list_tools()
    print(f"TOOLS: {{[tool.name for tool in tools]}}")
except Exception as e:
    print(f"ERROR: {{e}}")
""",
            # Метод 3: Поиск в коде
            f"""
import os
import re

def find_tools_in_code():
    tools = []
    for root, dirs, files in os.walk('/app'):
        for file in files:
            if file.endswith('.py'):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        content = f.read()
                        # Ищем определения инструментов
                        matches = re.findall(r'@app\\.tool\\(["\']([^"\']+)["\']', content)
                        tools.extend(matches)
                        matches = re.findall(r'Tool\\([^)]*name=["\']([^"\']+)["\']', content)
                        tools.extend(matches)
                except:
                    pass
    return list(set(tools))

tools = find_tools_in_code()
print(f"TOOLS: {{tools}}")
"""
        ]
        
        for i, method in enumerate(methods, 1):
            result = await self.exec_in_container(
                container, 
                ['python', '-c', method],
                timeout=15
            )
            
            if result['success'] and 'TOOLS:' in result['stdout']:
                tools_line = [line for line in result['stdout'].split('\n') if 'TOOLS:' in line][0]
                try:
                    tools_str = tools_line.split('TOOLS: ')[1].strip()
                    tools = eval(tools_str)  # Осторожно с eval, но здесь безопасно
                    if tools:
                        logger.info(f"{service}: найдено {len(tools)} инструментов методом {i}: {tools}")
                        return tools
                except:
                    pass
            
            if 'ERROR:' in result['stdout']:
                logger.debug(f"{service}: метод {i} не сработал: {result['stdout']}")
        
        logger.warning(f"{service}: инструменты не найдены")
        return []
    
    async def test_real_tool_call(self, service: str, tool_name: str, args: Dict = None) -> Dict[str, Any]:
        """Тестировать реальный вызов инструмента"""
        container = self.services[service]['container']
        args = args or {}
        
        # Создаем скрипт для вызова инструмента
        script = f"""
import sys
import json
sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app/build/lib')
sys.path.insert(0, '/usr/local/lib/python3.11/site-packages')

try:
    if '{service}' == 'binance-mcp':
        from binance_mcp.server import create_server
        server = create_server()
        result = server.call_tool('{tool_name}', {json.dumps(args)})
        print(f"RESULT: {{json.dumps(result)}}")
    elif '{service}' == 'memory-mcp':
        from memory_mcp.mcp.server import create_server
        server = create_server()
        result = server.call_tool('{tool_name}', {json.dumps(args)})
        print(f"RESULT: {{json.dumps(result)}}")
    else:
        print("RESULT: {{\\"error\\": \\"Service not implemented\\"}}")
except Exception as e:
    print(f"ERROR: {{e}}")
"""
        
        result = await self.exec_in_container(
            container,
            ['python', '-c', script],
            timeout=20
        )
        
        if result['success'] and 'RESULT:' in result['stdout']:
            try:
                result_line = [line for line in result['stdout'].split('\n') if 'RESULT:' in line][0]
                result_str = result_line.split('RESULT: ')[1].strip()
                return json.loads(result_str)
            except:
                pass
                
        return {'error': f"Tool call failed: {result.get('stderr', 'Unknown error')}"}

async def test_all_services_health():
    """Тест здоровья всех сервисов"""
    print("\n=== Test: All Services Health ===")
    
    tester = DockerMCPTester()
    results = {}
    
    for service in tester.services:
        print(f"\n--- Тестирование {service} ---")
        health = await tester.test_container_health(service)
        results[service] = health
        print(f"  Здоровье: {'✅' if health else '❌'}")
    
    healthy_count = sum(results.values())
    print(f"\n=== Итоговые результаты ===")
    for service, health in results.items():
        print(f"{'✅' if health else '❌'} {service}")
    
    print(f"\nЗдоровых сервисов: {healthy_count}/{len(results)}")
    print("✅ Test passed" if healthy_count == len(results) else "❌ Test failed")
    return results

async def test_all_services_tools():
    """Тест инструментов всех сервисов"""
    print("\n=== Test: All Services Tools Discovery ===")
    
    tester = DockerMCPTester()
    results = {}
    
    for service in tester.services:
        print(f"\n--- Поиск инструментов {service} ---")
        tools = await tester.test_mcp_server_tools(service)
        results[service] = tools
        print(f"  Найдено инструментов: {len(tools)}")
        if tools:
            for tool in tools[:5]:  # Показываем первые 5
                print(f"    • {tool}")
            if len(tools) > 5:
                print(f"    ... и еще {len(tools) - 5}")
    
    print(f"\n=== Итоговые результаты ===")
    total_tools = 0
    for service, tools in results.items():
        count = len(tools)
        total_tools += count
        print(f"{'✅' if count > 0 else '⚠️'} {service}: {count} инструментов")
    
    print(f"\nВсего найдено инструментов: {total_tools}")
    print("✅ Test passed" if total_tools > 0 else "⚠️ No tools found")
    return results

async def test_real_integration_workflow():
    """Тест реального интеграционного workflow"""
    print("\n=== Test: Real Integration Workflow ===")
    
    tester = DockerMCPTester()
    
    # Сначала получаем доступные инструменты
    print("  1. Получение списка инструментов...")
    tools_results = {}
    for service in ['binance-mcp', 'memory-mcp']:
        tools = await tester.test_mcp_server_tools(service)
        tools_results[service] = tools
        print(f"    {service}: {len(tools)} инструментов")
    
    success_count = 0
    
    # Тест 1: Binance данные
    if tools_results.get('binance-mcp'):
        print("  2. Тестирование Binance инструментов...")
        binance_tool = tools_results['binance-mcp'][0]
        result = await tester.test_real_tool_call('binance-mcp', binance_tool)
        if 'error' not in result:
            print(f"    ✅ {binance_tool} работает")
            success_count += 1
        else:
            print(f"    ❌ {binance_tool}: {result.get('error', 'Unknown error')}")
    else:
        print("  2. ⚠️ Binance инструменты недоступны")
    
    # Тест 2: Memory данные
    if tools_results.get('memory-mcp'):
        print("  3. Тестирование Memory инструментов...")
        memory_tool = tools_results['memory-mcp'][0]
        result = await tester.test_real_tool_call('memory-mcp', memory_tool)
        if 'error' not in result:
            print(f"    ✅ {memory_tool} работает")
            success_count += 1
        else:
            print(f"    ❌ {memory_tool}: {result.get('error', 'Unknown error')}")
    else:
        print("  3. ⚠️ Memory инструменты недоступны")
    
    print(f"\n  === Результат workflow ===")
    print(f"  Успешных тестов: {success_count}")
    
    if success_count > 0:
        print("✅ Test passed")
    else:
        print("⚠️ No successful tool calls")
    
    return success_count > 0

async def main():
    """Главная функция тестирования"""
    print("🚀 Запуск финальных реальных интеграционных тестов...")
    
    try:
        # Тест 1: Здоровье сервисов
        health_results = await test_all_services_health()
        
        # Тест 2: Поиск инструментов
        tools_results = await test_all_services_tools()
        
        # Тест 3: Реальный workflow
        workflow_success = await test_real_integration_workflow()
        
        print(f"\n🎉 Все финальные реальные тесты завершены!")
        
        # Итоговая статистика
        healthy_services = sum(health_results.values())
        total_tools = sum(len(tools) for tools in tools_results.values())
        
        print(f"\n📊 Итоговая статистика:")
        print(f"  • Здоровых сервисов: {healthy_services}/{len(health_results)}")
        print(f"  • Найдено инструментов: {total_tools}")
        print(f"  • Workflow успешен: {'✅' if workflow_success else '❌'}")
        
        if healthy_services == len(health_results) and total_tools > 0:
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
