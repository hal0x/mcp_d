#!/usr/bin/env python3
"""
Скрипт для запуска тестов бота.

Использование:
    python tests/run_bot_tests.py [--simple] [--full] [--verbose]
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


def run_simple_tests(verbose: bool = False):
    """Запуск простых тестов бота."""
    logger.info("🧪 Запуск простых тестов бота...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/integration/test_bot_simple.py",
        "-v" if verbose else "-q",
        "-s"  # Показывать print statements
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("✅ Простые тесты завершены успешно")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Простые тесты завершились с ошибкой: {e}")
        return False


def run_full_tests(verbose: bool = False):
    """Запуск полных тестов бота."""
    logger.info("🧪 Запуск полных тестов бота...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/integration/test_bot_queries.py",
        "-v" if verbose else "-q",
        "-s"  # Показывать print statements
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("✅ Полные тесты завершены успешно")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Полные тесты завершились с ошибкой: {e}")
        return False


def run_all_tests(verbose: bool = False):
    """Запуск всех тестов бота."""
    logger.info("🧪 Запуск всех тестов бота...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/integration/test_bot_simple.py",
        "tests/integration/test_bot_queries.py",
        "-v" if verbose else "-q",
        "-s"  # Показывать print statements
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("✅ Все тесты завершены успешно")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Тесты завершились с ошибкой: {e}")
        return False


def check_ollama_running():
    """Проверяет, запущен ли Ollama."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Ollama запущен и доступен")
            return True
    except Exception as e:
        logger.warning(f"⚠️ Ollama недоступен: {e}")
        return False


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description="Запуск тестов бота")
    parser.add_argument(
        "--simple", 
        action="store_true", 
        help="Запустить только простые тесты"
    )
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="Запустить только полные тесты"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Подробный вывод"
    )
    
    args = parser.parse_args()
    
    # Проверяем, что мы в правильной директории
    if not Path("main.py").exists():
        logger.error("❌ Запустите скрипт из корневой директории проекта")
        sys.exit(1)
    
    # Проверяем Ollama
    if not check_ollama_running():
        logger.warning("⚠️ Ollama не запущен. Некоторые тесты могут не работать.")
        response = input("Продолжить? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    success = True
    
    if args.simple:
        success = run_simple_tests(args.verbose)
    elif args.full:
        success = run_full_tests(args.verbose)
    else:
        success = run_all_tests(args.verbose)
    
    if success:
        logger.info("🎉 Все тесты прошли успешно!")
        sys.exit(0)
    else:
        logger.error("💥 Некоторые тесты завершились с ошибкой")
        sys.exit(1)


if __name__ == "__main__":
    main()
