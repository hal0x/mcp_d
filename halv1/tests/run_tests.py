#!/usr/bin/env python3
"""
Скрипт для быстрого запуска тестов по группам
"""

import os
import subprocess
import sys
from pathlib import Path

TEST_GROUPS = {
    "agent": "🧠 Тесты компонентов агента",
    "executor": "⚡ Тесты исполнителей кода",
    "memory": "🧠 Тесты системы памяти",
    "search": "🔍 Тесты поиска и интернета",
    "index": "📚 Тесты индексации и векторов",
    "core": "🔧 Тесты основных компонентов",
    "security": "🛡️ Тесты безопасности",
    "integration": "🔗 Интеграционные тесты",
    "utils": "🛠️ Тесты утилит",
    "all": "🚀 Все тесты (кроме интеграционных)",
}


def run_tests(group, verbose=False):
    """Запуск тестов для указанной группы"""
    if group == "all":
        # Исключаем интеграционные тесты из команды "all"
        cmd = ["python", "-m", "pytest", "tests/", "--ignore=tests/integration/", "-v" if verbose else "-q"]
    else:
        # Специальная обработка для интеграционных тестов
        if group == "integration":
            cmd = [
                "python",
                "-m",
                "pytest",
                "tests/integration/",
                "-v" if verbose else "-q",
            ]
        else:
            cmd = [
                "python",
                "-m",
                "pytest",
                f"tests/test_{group}/",
                "-v" if verbose else "-q",
            ]

    print(f"🚀 Запуск: {TEST_GROUPS[group]}")
    print(f"📝 Команда: {' '.join(cmd)}")
    print("=" * 50)

    # Устанавливаем переменную окружения для подавления предупреждений SWIG
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, env=env)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⏹️ Тесты прерваны пользователем")
        return False


def show_help():
    """Показать справку"""
    print("🎯 Скрипт запуска тестов HALv1")
    print()
    print("📋 Доступные группы:")
    for group, description in TEST_GROUPS.items():
        print(f"  {group:<12} - {description}")
    print()
    print("🚀 Использование:")
    print(f"  python {sys.argv[0]} <группа> [--verbose]")
    print()
    print("📝 Примеры:")
    print(f"  python {sys.argv[0]} agent")
    print(f"  python {sys.argv[0]} memory --verbose")
    print(f"  python {sys.argv[0]} all")
    print()
    print("🔧 Опции:")
    print("  --verbose  - Подробный вывод")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help", "help"]:
        show_help()
        return

    group = sys.argv[1].lower()
    verbose = "--verbose" in sys.argv

    if group not in TEST_GROUPS:
        print(f"❌ Неизвестная группа: {group}")
        print("📋 Доступные группы:", ", ".join(TEST_GROUPS.keys()))
        return 1

    venv_path = Path(__file__).parent.parent / "venv" / "bin" / "activate"
    if venv_path.exists():
        print(f"🔧 Активация виртуального окружения: {venv_path}")

    success = run_tests(group, verbose)

    if success:
        print("✅ Тесты завершены успешно")
        return 0
    else:
        print("❌ Тесты завершены с ошибками")
        return 1


if __name__ == "__main__":
    sys.exit(main())
