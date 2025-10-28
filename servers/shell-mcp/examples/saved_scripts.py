#!/usr/bin/env python3
"""
Пример работы с сохранёнными скриптами в Shell MCP.

Этот пример показывает, как сохранить скрипт и затем запустить его.
"""

# Пример скрипта для сохранения
code = """
import sys
import datetime
import random

def generate_report():
    \"\"\"Генерирует простой отчёт.\"\"\"
    timestamp = datetime.datetime.now()
    
    # Генерируем случайные данные
    data = {
        "timestamp": timestamp.isoformat(),
        "random_number": random.randint(1, 100),
        "system_info": {
            "python_version": sys.version.split()[0],
            "platform": sys.platform
        },
        "calculations": {
            "fibonacci": [0, 1, 1, 2, 3, 5, 8, 13],
            "sum": sum(range(1, 11))
        }
    }
    
    return data

if __name__ == "__main__":
    print("Генерация отчёта...")
    report = generate_report()
    
    print("\\n=== ОТЧЁТ ===")
    print(f"Время: {report['timestamp']}")
    print(f"Случайное число: {report['random_number']}")
    print(f"Python версия: {report['system_info']['python_version']}")
    print(f"Платформа: {report['system_info']['platform']}")
    print(f"Сумма 1-10: {report['calculations']['sum']}")
    print(f"Фибоначчи: {report['calculations']['fibonacci']}")
    
    print("\\nОтчёт сгенерирован успешно!")
"""

print("Пример скрипта для сохранения:")
print("=" * 50)
print(code)
print("=" * 50)

print("\n1. Сохранение скрипта:")
print("mcp_shell-mcp_run_code_simple {")
print('  "code": """' + code.replace('"', '\\"') + '""",')
print('  "language": "python",')
print('  "save_name": "report-generator",')
print('  "timeout_seconds": 30')
print("}")

print("\n2. Просмотр сохранённых скриптов:")
print("mcp_shell-mcp_list_saved_scripts")

print("\n3. Запуск сохранённого скрипта:")
print("mcp_shell-mcp_run_saved_script {")
print('  "name": "report-generator",')
print('  "timeout_seconds": 30')
print("}")

print("\n4. Запуск с дополнительными параметрами:")
print("mcp_shell-mcp_run_saved_script {")
print('  "name": "report-generator",')
print('  "timeout_seconds": 60,')
print('  "out_artifacts_path": "./runs/saved-script-output"')
print("}")

print("\nПримечания:")
print("- Скрипт сохраняется автоматически при успешном выполнении")
print("- Можно запускать сохранённый скрипт многократно")
print("- Поддерживаются все параметры обычного run_code_simple")
