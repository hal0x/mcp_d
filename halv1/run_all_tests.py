#!/usr/bin/env python3
"""
Скрипт для запуска всех тестов бота.

Использование:
    python run_all_tests.py [--debug] [--real] [--quick] [--all]
"""

import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)


def run_debug_test():
    """Запуск отладочного теста."""
    logger.info("🧪 Запуск отладочного теста...")
    try:
        result = subprocess.run([sys.executable, "test_bot_debug.py"], check=True)
        logger.info("✅ Отладочный тест завершен успешно")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Отладочный тест завершился с ошибкой: {e}")
        return False


def run_real_test():
    """Запуск реального теста."""
    logger.info("🧪 Запуск реального теста...")
    try:
        result = subprocess.run([sys.executable, "test_bot_real.py"], check=True)
        logger.info("✅ Реальный тест завершен успешно")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Реальный тест завершился с ошибкой: {e}")
        return False


def run_quick_test():
    """Запуск быстрого теста."""
    logger.info("🧪 Запуск быстрого теста...")
    try:
        result = subprocess.run([sys.executable, "test_bot_quick.py"], check=True)
        logger.info("✅ Быстрый тест завершен успешно")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Быстрый тест завершился с ошибкой: {e}")
        return False


def run_integration_tests():
    """Запуск интеграционных тестов."""
    logger.info("🧪 Запуск интеграционных тестов...")
    try:
        result = subprocess.run([sys.executable, "tests/run_bot_tests.py", "--verbose"], check=True)
        logger.info("✅ Интеграционные тесты завершены успешно")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Интеграционные тесты завершились с ошибкой: {e}")
        return False


def check_environment():
    """Проверка окружения."""
    logger.info("🔍 Проверяем окружение...")
    
    # Проверяем, что мы в правильной директории
    if not Path("main.py").exists():
        logger.error("❌ Запустите скрипт из корневой директории проекта")
        return False
    
    # Проверяем Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Ollama запущен и доступен")
        else:
            logger.warning("⚠️ Ollama недоступен")
    except Exception as e:
        logger.warning(f"⚠️ Не удалось проверить Ollama: {e}")
    
    # Проверяем Docker
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ Docker доступен")
        else:
            logger.warning("⚠️ Docker недоступен")
    except Exception as e:
        logger.warning(f"⚠️ Не удалось проверить Docker: {e}")
    
    return True


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description="Запуск всех тестов бота")
    parser.add_argument("--debug", action="store_true", help="Запустить отладочный тест")
    parser.add_argument("--real", action="store_true", help="Запустить реальный тест")
    parser.add_argument("--quick", action="store_true", help="Запустить быстрый тест")
    parser.add_argument("--integration", action="store_true", help="Запустить интеграционные тесты")
    parser.add_argument("--all", action="store_true", help="Запустить все тесты")
    
    args = parser.parse_args()
    
    # Проверяем окружение
    if not check_environment():
        sys.exit(1)
    
    success = True
    
    if args.debug or args.all:
        success &= run_debug_test()
    
    if args.real or args.all:
        success &= run_real_test()
    
    if args.quick or args.all:
        success &= run_quick_test()
    
    if args.integration or args.all:
        success &= run_integration_tests()
    
    if not any([args.debug, args.real, args.quick, args.integration, args.all]):
        # Если не указаны аргументы, запускаем отладочный тест
        success = run_debug_test()
    
    if success:
        logger.info("🎉 Все тесты прошли успешно!")
        sys.exit(0)
    else:
        logger.error("💥 Некоторые тесты завершились с ошибкой")
        sys.exit(1)


if __name__ == "__main__":
    main()
