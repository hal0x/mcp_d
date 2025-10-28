#!/usr/bin/env python3
"""
Пример использования Shell MCP с установкой зависимостей.

Этот пример показывает, как установить пакеты через pip перед выполнением кода.
"""

# Пример с установкой зависимостей
code = """
import requests
import json
from datetime import datetime

print("Установленные пакеты:")
print(f"requests версия: {requests.__version__}")

# Простой HTTP запрос
try:
    response = requests.get("https://httpbin.org/json", timeout=10)
    data = response.json()
    print("\\nОтвет от httpbin.org:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Ошибка HTTP запроса: {e}")

# Работа с датами
now = datetime.now()
print(f"\\nТекущее время: {now.strftime('%Y-%m-%d %H:%M:%S')}")
"""

dependencies = [
    "requests==2.32.3",
    "urllib3==2.0.7"
]

print("Пример кода с зависимостями:")
print("=" * 50)
print(code)
print("=" * 50)

print("\nСписок зависимостей:")
for dep in dependencies:
    print(f"  - {dep}")

print("\nДля запуска через MCP используйте:")
print("mcp_shell-mcp_run_code_simple {")
print('  "code": """' + code.replace('"', '\\"') + '""",')
print('  "language": "python",')
print('  "dependencies": ' + str(dependencies).replace("'", '"') + ',')
print('  "timeout_seconds": 60')
print("}")
