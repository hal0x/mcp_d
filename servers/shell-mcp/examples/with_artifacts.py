#!/usr/bin/env python3
"""
Пример использования Shell MCP с сохранением артефактов.

Этот пример показывает, как сохранить файлы из контейнера на хост.
"""

# Пример с созданием артефактов
code = """
import os
import json
import csv
from datetime import datetime

# Создаём папку для артефактов
os.makedirs('artifacts', exist_ok=True)

# Создаём JSON файл
data = {
    "timestamp": datetime.now().isoformat(),
    "message": "Привет из Docker контейнера!",
    "numbers": [1, 2, 3, 4, 5],
    "calculation": {
        "sum": sum([1, 2, 3, 4, 5]),
        "average": sum([1, 2, 3, 4, 5]) / 5
    }
}

with open('artifacts/result.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Создан файл artifacts/result.json")

# Создаём CSV файл
csv_data = [
    ["Name", "Age", "City"],
    ["Алексей", 25, "Москва"],
    ["Мария", 30, "Санкт-Петербург"],
    ["Дмитрий", 28, "Казань"]
]

with open('artifacts/people.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

print("Создан файл artifacts/people.csv")

# Создаём текстовый файл
with open('artifacts/readme.txt', 'w', encoding='utf-8') as f:
    f.write("Это пример артефактов из Shell MCP\\n")
    f.write("=" * 40 + "\\n")
    f.write("Создано: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\\n")
    f.write("Все файлы будут скопированы на хост\\n")

print("Создан файл artifacts/readme.txt")

print("\\nВсе артефакты созданы в папке artifacts/")
"""

print("Пример кода с созданием артефактов:")
print("=" * 50)
print(code)
print("=" * 50)

print("\nДля запуска через MCP используйте:")
print("mcp_shell-mcp_run_code_simple {")
print('  "code": """' + code.replace('"', '\\"') + '""",')
print('  "language": "python",')
print('  "out_artifacts_path": "./runs/artifacts-example",')
print('  "timeout_seconds": 30')
print("}")

print("\nАртефакты будут сохранены в:")
print("  - ./runs/artifacts-example/result.json")
print("  - ./runs/artifacts-example/people.csv") 
print("  - ./runs/artifacts-example/readme.txt")
