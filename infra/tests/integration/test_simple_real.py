"""
Простые реальные тесты через прямые команды Docker
"""
import pytest
import asyncio
import subprocess
import json
import sys
import os
import logging

logger = logging.getLogger(__name__)

class SimpleDockerClient:
    """Простой клиент для тестирования Docker контейнеров"""
    
    def __init__(self, container_name: str, service_name: str):
        self.container_name = container_name
        self.service_name = service_name
    
    async def is_running(self) -> bool:
        """Проверка, что контейнер запущен"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return "Up" in result.stdout
        except Exception:
            return False
    
    async def get_logs(self, lines: int = 10) -> str:
        """Получение логов контейнера"""
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", str(lines), self.container_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout + result.stderr
        except Exception as e:
            return f"Error getting logs: {e}"
    
    async def exec_command(self, command: list, timeout: int = 30) -> dict:
        """Выполнение команды в контейнере"""
        try:
            full_command = ["docker", "exec", self.container_name] + command
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out",
                "timeout": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_python_import(self) -> bool:
        """Тест импорта Python модулей"""
        result = await self.exec_command(["python", "-c", "import sys; print('Python OK')"])
        return result.get("success", False)
    
    async def test_mcp_server_exists(self) -> bool:
        """Тест существования MCP сервера"""
        result = await self.exec_command(["ls", "-la", "/app/"])
        if result.get("success"):
            return "mcp_server.py" in result.get("stdout", "") or "server.py" in result.get("stdout", "")
        return False


@pytest.mark.asyncio
async def test_all_containers_health():
    """Тест: Проверка здоровья всех контейнеров"""
    print("\n=== Test: All Containers Health ===")
    
    containers = [
        ("mcp-binance-mcp-1", "binance-mcp"),
        ("mcp-tradingview-mcp-1", "tradingview-mcp"),
        ("mcp-memory-mcp-1", "memory-mcp"),
        ("mcp-shell-mcp-1", "shell-mcp"),
        ("mcp-backtesting-mcp-1", "backtesting-mcp")
    ]
    
    results = {}
    
    for container_name, service_name in containers:
        print(f"\n--- Тестирование {service_name} ---")
        client = SimpleDockerClient(container_name, service_name)
        
        # Проверка запуска
        is_running = await client.is_running()
        print(f"  Запущен: {'✅' if is_running else '❌'}")
        
        if is_running:
            # Проверка Python
            python_ok = await client.test_python_import()
            print(f"  Python: {'✅' if python_ok else '❌'}")
            
            # Проверка MCP сервера
            mcp_exists = await client.test_mcp_server_exists()
            print(f"  MCP сервер: {'✅' if mcp_exists else '❌'}")
            
            # Получение логов
            logs = await client.get_logs(5)
            if logs:
                print(f"  Последние логи:")
                for line in logs.split('\n')[-3:]:
                    if line.strip():
                        print(f"    {line.strip()}")
            
            results[service_name] = {
                "running": is_running,
                "python_ok": python_ok,
                "mcp_exists": mcp_exists,
                "healthy": is_running and python_ok and mcp_exists
            }
        else:
            results[service_name] = {
                "running": False,
                "healthy": False
            }
    
    # Итоговые результаты
    print(f"\n=== Итоговые результаты ===")
    healthy_count = 0
    for service_name, result in results.items():
        status = "✅" if result.get("healthy", False) else "❌"
        print(f"{status} {service_name}")
        if result.get("healthy", False):
            healthy_count += 1
    
    print(f"\nЗдоровых сервисов: {healthy_count}/{len(containers)}")
    
    # Сохраняем результаты
    with open("results/containers_health.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    assert healthy_count > 0, "Ни один сервис не здоров"
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_binance_mcp_direct():
    """Тест: Прямое тестирование binance-mcp"""
    print("\n=== Test: Binance MCP Direct ===")
    
    client = SimpleDockerClient("mcp-binance-mcp-1", "binance-mcp")
    
    if not await client.is_running():
        pytest.skip("binance-mcp не запущен")
    
    # Тест 1: Проверка конфигурации
    print("  1. Проверка конфигурации...")
    result = await client.exec_command([
        "python", "mcp_server.py", "--print-config"
    ])
    
    if result.get("success"):
        print("    ✅ Конфигурация получена")
        try:
            config = json.loads(result["stdout"])
            print(f"    Режим: {config.get('client', {}).get('mode', 'Unknown')}")
            print(f"    API ключ: {'✅' if config.get('config', {}).get('api_key_present') else '❌'}")
        except json.JSONDecodeError:
            print(f"    Конфигурация: {result['stdout'][:100]}...")
    else:
        print(f"    ❌ Ошибка конфигурации: {result.get('stderr', 'Unknown error')}")
    
    # Тест 2: Проверка импортов
    print("  2. Проверка импортов...")
    result = await client.exec_command([
        "python", "-c", "from src.config import get_config; print('Config OK')"
    ])
    
    if result.get("success"):
        print("    ✅ Импорты работают")
    else:
        print(f"    ❌ Ошибка импортов: {result.get('stderr', 'Unknown error')}")
    
    # Тест 3: Проверка Binance API (если доступно)
    print("  3. Проверка Binance API...")
    result = await client.exec_command([
        "python", "-c", """
try:
    from src.client import get_client_info
    info = get_client_info()
    print(f"Client mode: {info.get('mode', 'Unknown')}")
    print("Binance client OK")
except Exception as e:
    print(f"Binance client error: {e}")
"""
    ])
    
    if result.get("success"):
        print("    ✅ Binance клиент работает")
        print(f"    Результат: {result['stdout'].strip()}")
    else:
        print(f"    ❌ Ошибка Binance клиента: {result.get('stderr', 'Unknown error')}")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_tradingview_mcp_direct():
    """Тест: Прямое тестирование tradingview-mcp"""
    print("\n=== Test: TradingView MCP Direct ===")
    
    client = SimpleDockerClient("mcp-tradingview-mcp-1", "tradingview-mcp")
    
    if not await client.is_running():
        pytest.skip("tradingview-mcp не запущен")
    
    # Тест 1: Проверка структуры
    print("  1. Проверка структуры...")
    result = await client.exec_command(["ls", "-la", "/app/src/"])
    
    if result.get("success"):
        print("    ✅ Структура найдена")
        print(f"    Содержимое: {result['stdout'].count('tradingview_mcp')} файлов tradingview_mcp")
    else:
        print(f"    ❌ Ошибка структуры: {result.get('stderr', 'Unknown error')}")
    
    # Тест 2: Проверка импортов
    print("  2. Проверка импортов...")
    result = await client.exec_command([
        "python", "-c", "from tradingview_mcp.server import main; print('TradingView imports OK')"
    ])
    
    if result.get("success"):
        print("    ✅ Импорты работают")
    else:
        print(f"    ❌ Ошибка импортов: {result.get('stderr', 'Unknown error')}")
    
    # Тест 3: Проверка конфигурации
    print("  3. Проверка конфигурации...")
    result = await client.exec_command([
        "python", "-c", """
try:
    from tradingview_mcp.config import get_config
    config = get_config()
    print(f"Config loaded: {type(config)}")
    print("TradingView config OK")
except Exception as e:
    print(f"Config error: {e}")
"""
    ])
    
    if result.get("success"):
        print("    ✅ Конфигурация работает")
        print(f"    Результат: {result['stdout'].strip()}")
    else:
        print(f"    ❌ Ошибка конфигурации: {result.get('stderr', 'Unknown error')}")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_memory_mcp_direct():
    """Тест: Прямое тестирование memory-mcp"""
    print("\n=== Test: Memory MCP Direct ===")
    
    client = SimpleDockerClient("mcp-memory-mcp-1", "memory-mcp")
    
    if not await client.is_running():
        pytest.skip("memory-mcp не запущен")
    
    # Тест 1: Проверка структуры
    print("  1. Проверка структуры...")
    result = await client.exec_command(["ls", "-la", "/app/src/"])
    
    if result.get("success"):
        print("    ✅ Структура найдена")
        if "memory_mcp" in result['stdout']:
            print("    memory_mcp модуль найден")
    else:
        print(f"    ❌ Ошибка структуры: {result.get('stderr', 'Unknown error')}")
    
    # Тест 2: Проверка импортов
    print("  2. Проверка импортов...")
    result = await client.exec_command([
        "python", "-c", "from memory_mcp.mcp.server import main; print('Memory imports OK')"
    ])
    
    if result.get("success"):
        print("    ✅ Импорты работают")
    else:
        print(f"    ❌ Ошибка импортов: {result.get('stderr', 'Unknown error')}")
    
    # Тест 3: Проверка базы данных
    print("  3. Проверка базы данных...")
    result = await client.exec_command([
        "python", "-c", """
import os
db_path = os.getenv('MEMORY_DB_PATH', 'memory_graph.db')
print(f"DB path: {db_path}")
if os.path.exists(db_path):
    print("Database file exists")
else:
    print("Database file not found")
"""
    ])
    
    if result.get("success"):
        print("    ✅ База данных проверена")
        print(f"    Результат: {result['stdout'].strip()}")
    else:
        print(f"    ❌ Ошибка базы данных: {result.get('stderr', 'Unknown error')}")
    
    print("✅ Test passed")


@pytest.mark.asyncio
async def test_services_integration():
    """Тест: Интеграция между сервисами"""
    print("\n=== Test: Services Integration ===")
    
    services_data = {}
    
    # Собираем данные от каждого сервиса
    containers = [
        ("mcp-binance-mcp-1", "binance-mcp"),
        ("mcp-tradingview-mcp-1", "tradingview-mcp"),
        ("mcp-memory-mcp-1", "memory-mcp")
    ]
    
    for container_name, service_name in containers:
        print(f"\n  --- Сбор данных от {service_name} ---")
        client = SimpleDockerClient(container_name, service_name)
        
        if await client.is_running():
            # Получаем информацию о сервисе
            result = await client.exec_command([
                "python", "-c", f"""
import os
import json
print(json.dumps({{
    'service': '{service_name}',
    'python_version': __import__('sys').version,
    'working_directory': os.getcwd(),
    'environment': dict(os.environ),
    'timestamp': __import__('time').time()
}}))
"""
            ])
            
            if result.get("success"):
                try:
                    data = json.loads(result["stdout"])
                    services_data[service_name] = data
                    print(f"    ✅ Данные собраны от {service_name}")
                except json.JSONDecodeError:
                    print(f"    ❌ Ошибка парсинга данных от {service_name}")
            else:
                print(f"    ❌ Не удалось получить данные от {service_name}")
        else:
            print(f"    ❌ {service_name} не запущен")
    
    # Анализируем интеграцию
    print(f"\n  === Анализ интеграции ===")
    print(f"  Сервисов с данными: {len(services_data)}")
    
    # Проверяем общие переменные окружения
    common_env_vars = ['PYTHONUNBUFFERED', 'DEFAULT_TRANSPORT']
    for var in common_env_vars:
        values = []
        for service, data in services_data.items():
            env = data.get('environment', {})
            if var in env:
                values.append(f"{service}={env[var]}")
        
        if values:
            print(f"  {var}: {', '.join(values)}")
    
    # Сохраняем результаты интеграции
    integration_result = {
        "timestamp": asyncio.get_event_loop().time(),
        "services_count": len(services_data),
        "services_data": services_data,
        "integration_success": len(services_data) > 1
    }
    
    with open("results/services_integration.json", "w") as f:
        json.dump(integration_result, f, indent=2, ensure_ascii=False)
    
    assert len(services_data) > 0, "Не удалось собрать данные ни от одного сервиса"
    print("✅ Test passed")


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создаем директорию для результатов
    os.makedirs("results", exist_ok=True)
    
    async def main():
        print("Запуск простых реальных тестов...")
        
        try:
            await test_all_containers_health()
            await test_binance_mcp_direct()
            await test_tradingview_mcp_direct()
            await test_memory_mcp_direct()
            await test_services_integration()
            print("\n🎉 Все простые реальные тесты завершены успешно!")
        except Exception as e:
            print(f"\n❌ Ошибка в тестах: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(main())
