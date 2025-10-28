# Docker-backed MCP Server

This package provides a Model Context Protocol (MCP) server that runs arbitrary code snippets inside short-lived Docker containers. It now follows the common workspace guidelines and exposes a console entrypoint `shell-mcp`.

## 🔌 Network Topology

| Service | Port | Transport | Dependencies |
|---------|------|-----------|--------------|
| `binance-mcp` | `8000` | `stdio` | `redis`, `postgres` |
| `tradingview-mcp` | `8060` | `streamable-http` | `redis`, `postgres`, `binance-mcp`, `memory-mcp` |
| `memory-mcp` | `8050` | `stdio` | `redis` |
| `shell-mcp` | `8070` | `stdio` | `redis` |
| `backtesting-mcp` | `8082` | `stdio` | `postgres`, `binance-mcp`, `tradingview-mcp` |

> Shell MCP listens on `0.0.0.0:8070`; other services should reference it as `http://shell-mcp:8070`.

## Prerequisites

- Python 3.10+
- Docker Engine installed and running
- (Optional) [uv](https://github.com/astral-sh/uv) or `pip` for dependency management

## Install dependencies

```bash
# using uv
uv sync

# or with pip (editable install)
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run the server (stdio)

```bash
uv run shell-mcp
```

Useful CLI options:

- `--image` – choose the default Docker image (defaults to `python:3.11` or `MCP_DOCKER_IMAGE`).
- `--network / --no-network` – enable or disable networking in containers by default.
- `--print-config` – print the effective configuration and exit.

Run with networking disabled:

```bash
uv run shell-mcp --no-network
```

## MCP integration

Point your MCP client to the server's executable command. Сервер предоставляет инструменты `run_code_simple`, `list_saved_scripts`, `run_saved_script`. Основной инструмент `run_code_simple` принимает:

- `code`: исходный текст (можно опустить, если задаёте `script_path`)
- `language`: one of `python`, `bash`, `sh`, `shell`, `node`
- `image`: optional Docker image override
- `command`: optional custom entrypoint (use `{script}` to reference the generated file)
- `network_enabled`: override the per-invocation networking choice (не может включить сеть, если глобально выключена)
- `timeout_seconds`: maximum runtime (default 120 seconds)
- `env`: optional list of `KEY=VALUE` pairs
- `memory`: ужесточающий лимит памяти (пример: `512m`), не превышает глобальный
- `cpus`: ужесточающий лимит CPU (пример: `0.5`), не превышает глобальный
- `readonly_fs`: запросить read-only FS (если глобально включено — отключить нельзя)
- `dependencies`: список пакетов, которые нужно установить через `pip` перед запуском (работает для Python-образов)
- `out_artifacts_path`: путь на хосте, куда скопировать файлы из папки `artifacts/` внутри контейнера
- `script_path`: путь к уже существующему файлу на хосте; содержимое файла будет выполнено
- `save_name`: имя, под которым сохранить код после успешного выполнения

### Упрощённый инструмент `run_code_simple`

Для интерфейсов без поддержки вложенных аргументов можно вызвать инструмент с плоскими параметрами:

```
mcp_shell-mcp_run_code_simple {
  "code": "print('hello')",
  "language": "python",
  "timeout_seconds": 60,
  "env": ["FOO=bar"],
  "memory": "256m",
  "cpus": "0.5",
  "readonly_fs": true,
  "dependencies": ["requests==2.32.3"],
  "out_artifacts_path": "./runs/run-001",
  "save_name": "hello-world"
}
```

Вместо `code` можно указать путь к существующему скрипту на хосте:

```
mcp_shell-mcp_run_code_simple {
  "script_path": "./examples/sample.py",
  "language": "python",
  "out_artifacts_path": "./runs/sample"
}
```

### Работа с сохранёнными скриптами

- Список сохранённых скриптов:

  ```
  mcp_shell-mcp_list_saved_scripts
  ```

  Ответ содержит `name`, `slug`, `language`, `path`, `updated_at`.

- Запуск сохранённого скрипта с дополнительными параметрами:

  ```
  mcp_shell-mcp_run_saved_script {
    "name": "hello-world",
    "timeout_seconds": 60,
    "out_artifacts_path": "./runs/hello-world"
  }
  ```

По умолчанию скрипты сохраняются в `/Users/hal/projects/mcp/scripts/`. Изменить каталог можно переменной окружения `SHELL_MCP_SAVED_SCRIPTS_DIR`.

Скрипт сохраняется автоматически, если вы передали `save_name` и выполнение завершилось без ошибок (exit code = 0).

Любые артефакты, созданные кодом в каталоге `./artifacts` внутри контейнера (например, `open('artifacts/output.txt', 'w')`), будут скопированы в указанную папку на хосте, а также перечислены в ответе инструмента.
```

The server returns stdout, stderr, exit status, whether the execution timed out, and metadata about the image/command that was used.

### HTTP transport (optional)

```
uv run shell-mcp --host 127.0.0.1 --port 3001
```

### Print effective config

```
uv run shell-mcp --print-config

## Concurrency

Each `run_code_simple` invocation runs in its own short‑lived container. Вы можете вызывать инструмент параллельно — для каждого запроса будет создан отдельный контейнер.

Ограничение одновременности на уровне сервиса:

- Переменная `SHELL_MCP_MAX_CONCURRENCY` задаёт максимальное число параллельных запусков `run_code_simple` (по умолчанию 3).
- Пример: ограничить до 3 одновременных контейнеров

```
export SHELL_MCP_MAX_CONCURRENCY=3
uv run shell-mcp
```

## Examples

В папке `examples/` находятся готовые примеры использования Shell MCP:

### [basic_python.py](examples/basic_python.py)
Простой пример запуска Python кода:
- Базовые вычисления
- Работа с системной информацией
- Создание файлов

### [with_dependencies.py](examples/with_dependencies.py)
Пример с установкой зависимостей:
- Установка пакетов через pip
- HTTP запросы с requests
- Работа с JSON данными

### [with_artifacts.py](examples/with_artifacts.py)
Пример с сохранением артефактов:
- Создание JSON файлов
- Генерация CSV данных
- Сохранение текстовых файлов
- Копирование результатов на хост

### [saved_scripts.py](examples/saved_scripts.py)
Пример работы с сохранёнными скриптами:
- Сохранение скрипта для повторного использования
- Просмотр списка сохранённых скриптов
- Запуск сохранённых скриптов с параметрами

### Запуск примеров

Для запуска любого примера используйте:

```bash
# Просмотр содержимого примера
cat examples/basic_python.py

# Запуск через MCP (скопируйте код из примера)
mcp_shell-mcp_run_code_simple {
  "code": "print('Hello from Docker!')",
  "language": "python"
}
```

Или используйте готовый файл:

```bash
mcp_shell-mcp_run_code_simple {
  "script_path": "./examples/basic_python.py",
  "language": "python"
}
```
