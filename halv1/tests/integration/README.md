# 🔧 Интеграционные тесты HALv1

Эта папка содержит скрипты и тесты для проверки интеграции различных компонентов системы HALv1.

## 📁 Содержимое

- **`run_integration_tests.py`** - Основной скрипт для запуска интеграционных тестов
- **`__init__.py`** - Инициализация Python-пакета

## 🚀 Запуск тестов

### Из корневой директории проекта
```bash
# Активировать виртуальную среду
source venv/bin/activate

# Запустить через скрипт-обертку
python run_integration_tests.py

# Или напрямую как модуль
python -m tests.integration.run_integration_tests
```

### Из папки tests
```bash
# Активировать виртуальную среду
source ../venv/bin/activate

# Запустить как модуль
python -m tests.integration.run_integration_tests
```

## 🎯 Режимы работы

- **`--mode check`** - Проверка окружения и зависимостей
- **`--mode smoke`** - Быстрые тесты
- **`--mode fast`** - Ускоренные тесты
- **`--mode full`** - Полные интеграционные тесты
 - Дополнительно: `--docker-only` — запуск только Docker-специфичных тестов

## 📊 Примеры использования

```bash
# Проверка окружения
python -m tests.integration.run_integration_tests --mode check

# Быстрые тесты
python -m tests.integration.run_integration_tests --mode smoke

# Запуск конкретного теста
python -m tests.integration.run_integration_tests --test-file test_integration_full.py

# Подробный вывод
python -m tests.integration.run_integration_tests --verbose

# Только Docker-тесты
python -m tests.integration.run_integration_tests --docker-only --skip-llm-check

# Пропустить проверку Docker
python -m tests.integration.run_integration_tests --mode full --skip-docker-check

```

## 🐳 Требования Docker

Интеграционные тесты выполнения кода используют Docker. Убедитесь, что Docker установлен и запущен.

Проверка:

```
docker info
```

Быстрый тест сети Docker:

```
python -m pytest tests/integration/test_docker_integration.py::TestDockerIntegration::test_network_access -q
```

## 🔗 Связанные файлы

- [Основной README тестов](../README.md)
- [Документация проекта](../../docs/README.md)
- [Скрипт-обертка](../../run_integration_tests.py)
