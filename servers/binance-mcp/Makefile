.PHONY: help install install-dev test test-cov lint format clean run examples

help: ## Показать справку
	@echo "Доступные команды:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Установить зависимости
	pip install -e .

install-dev: ## Установить зависимости для разработки
	pip install -e ".[dev]"
	pre-commit install

test: ## Запустить тесты
	pytest

test-cov: ## Запустить тесты с покрытием
	pytest --cov=src --cov-report=html --cov-report=term

lint: ## Проверить код линтерами
	flake8 src tests
	black --check src tests
	isort --check-only src tests
	mypy src

format: ## Форматировать код
	black src tests
	isort src tests

clean: ## Очистить временные файлы
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run: ## Запустить MCP сервер
	python mcp_server.py

examples: ## Запустить примеры
	python examples/basic_usage.py
	python examples/trading_example.py

check-permissions: ## Проверить права API
	python scripts/check_permissions.py

utils: ## Запустить утилиты
	python scripts/utils.py

dev-setup: install-dev ## Настроить окружение для разработки
	@echo "Окружение для разработки настроено!"
	@echo "Используйте 'make run' для запуска сервера"
	@echo "Используйте 'make test' для запуска тестов"
	@echo "Используйте 'make format' для форматирования кода"
