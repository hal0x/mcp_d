#!/usr/bin/env python3
"""
Скрипт-обертка для запуска интеграционных тестов.

Этот файл перенаправляет вызов на tests/integration/run_integration_tests.py
для удобства использования из корневой директории проекта.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Запускает интеграционные тесты."""
    # Путь к основному скрипту интеграционных тестов
    # Correct path: this file lives in tests/, so integration script is tests/integration/
    script_path = Path(__file__).parent / "integration" / "run_integration_tests.py"
    
    if not script_path.exists():
        print(f"❌ Файл интеграционных тестов не найден: {script_path}")
        sys.exit(1)
    
    # Устанавливаем переменную окружения для подавления предупреждений SWIG
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
    
    # Запускаем основной скрипт как модуль с переданными аргументами
    try:
        result = subprocess.run([sys.executable, "-m", "tests.integration.run_integration_tests"] + sys.argv[1:], 
                              check=False, capture_output=False, env=env)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"❌ Ошибка запуска интеграционных тестов: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
