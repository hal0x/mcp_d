# Memory MCP Server — Архитектура и Руководство

## Обзор

**Memory MCP** — сервер Model Context Protocol для индексации и поиска по унифицированной памяти
(Telegram, файлы и другие источники), поддерживающий полнотекстовый и векторный поиск.

```
memory-mcp/
├── src/memory_mcp/
│   ├── cli/                 # CLI утилиты
│   ├── indexing/            # Индексаторы источников (Telegram и др.)
│   ├── memory/              # Граф памяти, FTS, векторное хранилище
│   ├── mcp/                 # MCP schema, adapters и сервер
│   └── core/, analysis/, …  # Унаследованные компоненты
├── docker-compose.memory.yml
├── Dockerfile
└── README.md
```

## Основные компоненты

### MCP сервер (`src/memory_mcp/mcp/server.py`)
- Поднимает FastMCP, регистрирует инструменты.
- Располагает `MemoryServiceAdapter`, который объединяет FTS и Qdrant-поиск.
- Читает настройки из переменных окружения (`MEMORY_DB_PATH`, `QDRANT_URL`, `EMBEDDINGS_URL`).

### Адаптер памяти (`src/memory_mcp/mcp/adapters.py`)
- Инжестирует записи (`MemoryRecordPayload`) в `TypedGraphMemory` и в Qdrant.
- Выполняет гибридный поиск: FTS5 + векторное ANN (при наличии Qdrant/эмбеддингов).
- Предоставляет `fetch` для чтения оригинальной записи с вложениями.

### Граф памяти (`src/memory_mcp/memory/typed_graph.py`)
- SQLite + NetworkX; FTS5 таблица `node_search`.
- Добавляет/обновляет узлы (`DocChunkNode`, `EventNode` и др.).
- Предоставляет метод `search_text` (BM25/snippet).

### Векторное хранилище (`src/memory_mcp/memory/vector_store.py`)
- Обёртка над Qdrant (создание коллекции, upsert, поиск).
- Конфигурируется через `QDRANT_URL`; gracefully выключается, если Qdrant недоступен.

### Embed-сервис (`src/memory_mcp/memory/embeddings.py`)
- HTTP клиент для `text-embeddings-inference` (или совместимых сервисов).
- Функция `build_embedding_service_from_env()` подхватывает `EMBEDDINGS_URL`.

### Индексаторы (`src/memory_mcp/indexing/*`)
- Unified интерфейс `BaseIndexer`.
- `TelegramIndexer` → нормализованные `MemoryRecord`.
- CLI команда `ingest-telegram` вызывает индексатор + `MemoryIngestor`.

## MCP инструменты

### Стандарт описания инструментов
- **Первая строка**: лаконичное действие на английском (≤90 символов), начинаем с глагола.
- **Где хранится**: docstring функций в `src/memory_mcp/mcp/tools.py`, FastMCP берёт оттуда `description`.
- **Детали**: разворачиваем во втором предложении/абзаце docstring, чтобы `list_tools` оставался компактным.
- **Единый подход**: новые инструменты должны следовать тем же правилам, чтобы клиенты видели консистентные подсказки.

### 1. `ingest_records`
- **Назначение**: Приём пачки `MemoryRecordPayload`.
- **Описание**: для каждого record создаётся узел в графе и (при наличии) вектор в Qdrant.
- **Ответ**: количество загруженных записей / вложений / пропущенных дубликатов.

### 2. `search_memory`
- **Назначение**: Поиск по памяти (FTS + векторный).
- **Параметры**: `query`, `top_k`, `source`, `tags`, `date_from`, `date_to`, `include_embeddings`.
- **Ответ**: список `SearchResultItem` (record_id, score, snippet, source, timestamp, metadata).

### 3. `fetch_record`
- **Назначение**: Получение полной записи по `record_id`.
- **Ответ**: `MemoryRecordPayload` либо `null`, если запись не найдена.

### Документация инструментов — best practices
- **Обязательные/взаимоисключающие параметры**: явно указывайте `code` XOR `script_path`, `steps` как массив и т.п.
- **Примеры вызова**: предоставляйте рабочие фрагменты (как сформировать `records`, как делать поиск).
- **Форматы полей**:
  - `env` — список строк `KEY=VALUE`.
  - `steps` — список строк.
  - `memory`/`cpus` — Docker-совместимые строки (`256m`, `1g`, `0.5`).
  - `tags`/`entities` — массивы строк.
- **Подсказки/use cases**: добавляйте bullet tips (как добавить зависимости, как выполнять multi-step сценарии, как фильтровать поиск).
- **Troubleshooting**: перечисляйте частые ошибки (пустой `steps`, неправильный формат `env`, отсутствие `records`).

## Конфигурация и запуск

### Переменные окружения
```bash
MEMORY_DB_PATH=/data/memory_graph.db
QDRANT_URL=http://qdrant:6333         # опционально, для векторного поиска
EMBEDDINGS_URL=http://embeddings:80  # опционально, для генерации эмбеддингов
MEMORY_LOG_LEVEL=INFO
TG_DUMP_PORT=8050                    # порт MCP memory сервера
```

### Docker Compose развертывание

#### Пересборка и перезапуск после изменений кода

**⚠️ ВАЖНО**: После внесения изменений в код необходимо пересобрать контейнер и перезапустить сервис:

```bash
# Пересборка контейнера с изменениями кода
docker-compose -f ../../infra/docker-compose.mcp.yml build memory-mcp

# Перезапуск сервиса с новым образом
docker-compose -f ../../infra/docker-compose.mcp.yml up -d memory-mcp

# Проверка статуса сервиса
docker-compose -f ../../infra/docker-compose.mcp.yml ps memory-mcp

# Просмотр логов для диагностики
docker-compose -f ../../infra/docker-compose.mcp.yml logs -f memory-mcp
```

#### Полная пересборка всех сервисов

```bash
# Пересборка всех MCP сервисов
docker-compose -f ../../infra/docker-compose.mcp.yml build

# Перезапуск всех сервисов
docker-compose -f ../../infra/docker-compose.mcp.yml up -d

# Проверка статуса всех сервисов
docker-compose -f ../../infra/docker-compose.mcp.yml ps
```

#### Отладка и диагностика

```bash
# Вход в контейнер для отладки
docker-compose -f ../../infra/docker-compose.mcp.yml exec memory-mcp bash

# Проверка переменных окружения в контейнере
docker-compose -f ../../infra/docker-compose.mcp.yml exec memory-mcp env

# Проверка доступности сервиса
curl http://localhost:8050/health
```

#### Очистка и пересборка с нуля

```bash
# Остановка и удаление контейнеров
docker-compose -f ../../infra/docker-compose.mcp.yml down

# Удаление образов (принудительная пересборка)
docker-compose -f ../../infra/docker-compose.mcp.yml build --no-cache memory-mcp

# Запуск с чистого листа
docker-compose -f ../../infra/docker-compose.mcp.yml up -d memory-mcp
```

### Docker Compose (`docker-compose.memory.yml`)
- **postgres**: хранение графа (или SQLite volume).
- **qdrant** *(опционально)*: векторное хранилище.
- **redis**: кэш/очередь (зарезервировано).
- **embeddings**: сервис `text-embeddings-inference`.
- **memory-core**: MCP сервер (entrypoint `python -m memory_mcp.mcp.server`).

### CLI
```bash
pip install -e .

# Инжест Telegram чатов
python -m memory_mcp.cli.main ingest-telegram --chats-dir chats --db-path memory_graph.db

# Запуск MCP сервера (stdio)
python -m memory_mcp.mcp.server
```

## Тестирование

- **Unit tests**: `pytest tests -q`
- **CLI тесты**: сценарии в `tests/test_cli.py`
- **Гибридный поиск**: `tests/test_mcp_server.py` проверяет FTS/ingest/search.

## Документация

- Обновляйте `README.md` и `CHANGELOG.md` при изменении API/CLI.
- В `AGENTS.md` фиксируйте архитектуру, правила оформления инструментов и конфигурацию.
- Для новых источников добавляйте секции в `indexing/*` + документацию по формату входных данных.
