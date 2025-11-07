#!/usr/bin/env python3
"""Integration test runner for HALv1.

This script:
1. Проверяет доступность Ollama
2. Запускает интеграционные тесты
3. Предоставляет диагностику для проблем
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


class IntegrationTestRunner:
    """Запускает интеграционные тесты с проверкой зависимостей."""

    def __init__(self):
        self.logger = logger
        # Point to repository root (tests/integration -> tests -> repo root)
        self.project_root = Path(__file__).resolve().parents[2]

    def check_ollama(self) -> bool:
        """Проверяет доступность Ollama."""
        try:
            self.logger.info("Проверяем доступность Ollama...")
            
            # Проверяем, что порт 11434 доступен
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', 11434))
            sock.close()
            
            if result == 0:
                self.logger.info("✅ Ollama доступен на порту 11434")
                return True
            else:
                self.logger.warning("⚠️  Ollama не отвечает на порту 11434")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка проверки Ollama: {e}")
            return False

    def check_docker(self) -> bool:
        """Проверяет доступность Docker."""
        try:
            self.logger.info("Проверяем доступность Docker...")
            result = subprocess.run(["docker", "info"], capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("✅ Docker доступен")
                return True
            else:
                self.logger.warning("⚠️  Docker не отвечает")
                return False
        except FileNotFoundError:
            self.logger.error("❌ Docker не установлен")
            return False

    def run_tests(self, mode: str, skip_llm_check: bool = False, skip_docker_check: bool = False, docker_only: bool = False) -> bool:
        """Запускает тесты в указанном режиме."""
        self.logger.info(f"Запуск интеграционных тестов в режиме: {mode}")

        # Проверяем Ollama (если не пропускаем)
        if not skip_llm_check:
            self.logger.info("Проверяем зависимости...")
            llm_available = self.check_ollama()

            if not llm_available:
                print("\n⚠️  Ollama недоступен.")
                print("1. Установите/настройте Ollama")
                print("2. Запустите Ollama и убедитесь, что он слушает порт 11434")
                print("3. Или используйте --skip-llm-check для пропуска проверки")
                return False

        # Проверяем Docker (если не пропускаем)
        if not skip_docker_check:
            docker_available = self.check_docker()
            if not docker_available:
                print("\n⚠️  Docker недоступен.")
                print("1. Установите Docker")
                print("2. Запустите Docker daemon")
                print("3. Или используйте --skip-docker-check для пропуска проверки")
                return False
        
        # Определяем тестовый файл в зависимости от режима
        if docker_only:
            # Запускаем только Docker-специфичные тесты
            test_path = self.project_root / "tests" / "integration" / "test_docker_integration.py"
            if not test_path.exists():
                self.logger.error(f"Тестовый файл не найден: {test_path}")
                return False
            cmd = [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"]
            self.logger.info(f"Выполняем команду: {' '.join(cmd)}")
            
            # Устанавливаем переменную окружения для подавления предупреждений SWIG
            env = os.environ.copy()
            env['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
            
            try:
                subprocess.run(cmd, cwd=self.project_root, check=True, env=env)
                self.logger.info("✅ Docker тесты завершены успешно")
                return True
            except subprocess.CalledProcessError as e:
                self.logger.error(f"❌ Docker тесты завершились с ошибкой: {e}")
                return False
        else:
            if mode == "check":
                test_file = "test_integration_check.py"
            elif mode == "fast":
                test_file = "test_integration_fast.py"
            elif mode == "full":
                test_file = "test_integration_full.py"
            else:
                self.logger.error(f"Неизвестный режим: {mode}")
                return False

        test_path = self.project_root / "tests" / "integration" / test_file

        # Fallback: if fast/check files are missing, try the full suite file
        if not test_path.exists():
            if mode in {"fast", "check"}:
                fallback = self.project_root / "tests" / "integration" / "test_integration_full.py"
                if fallback.exists():
                    self.logger.warning(
                        f"Тестовый файл не найден: {test_path}, используем fallback: {fallback}"
                    )
                    test_path = fallback
                else:
                    self.logger.error(f"Тестовый файл не найден: {test_path}")
                    return False
            else:
                self.logger.error(f"Тестовый файл не найден: {test_path}")
                return False
        
        # Запускаем тесты
        cmd = [
            sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"
        ]
        
        # Устанавливаем переменную окружения для подавления предупреждений SWIG
        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
        
        self.logger.info(f"Выполняем команду: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, check=True, env=env)
            self.logger.info("✅ Тесты завершены успешно")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Тесты завершились с ошибкой: {e}")
            return False

    def run_specific_test(self, test_name: str) -> bool:
        """Запускает конкретный тест."""
        self.logger.info(f"Запуск теста: {test_name}")
        
        cmd = [
            sys.executable, "-m", "pytest", test_name, "-v", "--tb=short"
        ]
        
        # Устанавливаем переменную окружения для подавления предупреждений SWIG
        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, check=True, env=env)
            self.logger.info("✅ Тест завершен успешно")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Тест завершился с ошибкой: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="HALv1 Integration Test Runner")
    parser.add_argument(
        "--mode",
        choices=["check", "fast", "full"],
        default="check",
        help="Режим запуска тестов (по умолчанию: check)"
    )
    parser.add_argument(
        "--test",
        help="Запустить конкретный тест (например: tests/integration/test_file.py::test_function)"
    )
    parser.add_argument(
        "--skip-llm-check",
        action="store_true",
        help="Пропустить проверку Ollama"
    )
    parser.add_argument(
        "--skip-docker-check",
        action="store_true",
        help="Пропустить проверку Docker"
    )
    parser.add_argument(
        "--docker-only",
        action="store_true",
        help="Запустить только Docker-специфичные тесты"
    )
    
    args = parser.parse_args()
    
    runner = IntegrationTestRunner()
    
    if args.test:
        success = runner.run_specific_test(args.test)
    else:
        success = runner.run_tests(
            args.mode,
            args.skip_llm_check,
            args.skip_docker_check,
            args.docker_only,
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
