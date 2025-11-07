#!/usr/bin/env python3
"""Скрипт для запуска тестов без предупреждений SWIG."""

import os
import subprocess
import sys

def main():
    """Запуск тестов с подавлением предупреждений SWIG."""
    # Устанавливаем переменную окружения для подавления предупреждений
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
    
    # Запускаем pytest с переданными аргументами
    cmd = [sys.executable, '-m', 'pytest'] + sys.argv[1:]
    
    try:
        result = subprocess.run(cmd, env=env, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nТесты прерваны пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка запуска тестов: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
