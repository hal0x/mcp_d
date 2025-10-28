#!/usr/bin/env python3
"""
Базовый пример использования Shell MCP с Python.

Этот пример демонстрирует простой запуск Python кода через MCP сервер.
"""

# Пример простого Python скрипта
code = """
import sys
import datetime

print("Привет из Docker контейнера!")
print(f"Python версия: {sys.version}")
print(f"Текущее время: {datetime.datetime.now()}")

# Простые вычисления
numbers = [1, 2, 3, 4, 5]
sum_numbers = sum(numbers)
print(f"Сумма чисел {numbers} = {sum_numbers}")

# Работа с файлами
with open('/tmp/test.txt', 'w') as f:
    f.write("Тестовый файл создан через MCP!")

print("Файл /tmp/test.txt создан успешно")
"""

print("Пример кода для запуска через Shell MCP:")
print("=" * 50)
print(code)
print("=" * 50)

print("\nДля запуска через MCP используйте:")
print("mcp_shell-mcp_run_code_simple {")
print('  "code": """' + code.replace('"', '\\"') + '""",')
print('  "language": "python",')
print('  "timeout_seconds": 30')
print("}")
