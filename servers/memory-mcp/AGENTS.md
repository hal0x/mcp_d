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
│   ├── core/                # Основные компоненты (lmstudio_client, ollama_client)
│   ├── analysis/            # Анализ данных и саммаризация
│   ├── quality_analyzer/    # Анализ качества поиска
│   └── utils/               # Утилиты
├── tests/                   # Тесты
├── scripts/                 # Скрипты и утилиты
├── docs/                    # Документация
├── Dockerfile               # Docker образ
├── run_server.py            # HTTP сервер (FastAPI)
├── mcp_server.py            # Точка входа для HTTP режима
├── pyproject.toml           # Конфигурация проекта
├── README.md                # Пользовательская документация
├── AGENTS.md                # Архитектура и руководство (этот файл)
└── CHANGELOG.md             # История изменений
```

**Примечание:** Файл `docker-compose.mcp.yml` находится в родительском репозитории `infra/` по пути `../../infra/docker-compose.mcp.yml` относительно корня проекта.

## Основные компоненты

### MCP сервер (`src/memory_mcp/mcp/server.py`)
- Использует стандартный MCP Server (`mcp.server.Server`), регистрирует инструменты через `@server.list_tools()` и `@server.call_tool()`.
- Располагает `MemoryServiceAdapter`, который объединяет FTS и Qdrant-поиск.
- Читает настройки из переменных окружения (`MEMORY_DB_PATH`, `QDRANT_URL`, `EMBEDDINGS_URL`, `LMSTUDIO_HOST`, `LMSTUDIO_PORT`, `LMSTUDIO_MODEL`).

#### Почему используется стандартный MCP Server

Memory MCP использует низкоуровневый `mcp.server.Server` вместо FastMCP по следующим причинам:

1. **Ленивая инициализация адаптера памяти**: Адаптер памяти (`MemoryServiceAdapter`) создается только при первом вызове инструмента через функцию `_get_adapter()`. Это позволяет:
   - Отложить создание дорогих ресурсов (база данных, векторное хранилище) до момента реального использования
   - Ускорить запуск сервера, если инструменты не используются сразу
   - Обрабатывать ошибки инициализации более гибко

2. **Единое форматирование ответов**: Все инструменты используют функцию `_format_tool_response()` для консистентного форматирования ответов:
   - Преобразование Pydantic моделей в сериализуемый формат
   - Единый формат JSON с текстовым и структурированным содержимым
   - Обработка различных типов данных (dict, list, модели)

3. **Кастомная обработка ошибок**: Функция `format_error_message()` из модуля `quality_analyzer.utils.error_handler` обеспечивает единообразные сообщения об ошибках с префиксом "Ошибка:" для всех инструментов, что улучшает отладку и логирование.

**Пример использования:**
```python
@server.list_tools()
async def list_tools() -> List[Tool]:
    """Returns a list of available memory MCP tools."""
    return [
        Tool(name="ingest_records", ...),
        Tool(name="search_memory", ...),
        # ... другие инструменты
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> ToolResponse:
    """Execute a tool call and format the result."""
    adapter = _get_adapter()  # Ленивая инициализация
    
    if name == "ingest_records":
        records = [MemoryRecordPayload(**item) for item in arguments.get("records", [])]
        result = adapter.ingest(records)
        return _format_tool_response(result.model_dump())
    # ... другие инструменты
```

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
- **Где хранится**: описания находятся в `src/memory_mcp/mcp/server.py` в функции `list_tools()` через объекты `Tool()` с полем `description`.
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

### 4. `health`
- **Назначение**: Проверка состояния MCP сервера и конфигурации.
- **Параметры**: нет (пустой объект).
- **Ответ**: статус сервера, доступность сервисов (memory, vector_search, embeddings), конфигурация.

### 5. `version`
- **Назначение**: Получение информации о версии сервера и доступных возможностях.
- **Параметры**: нет (пустой объект).
- **Ответ**: название, версия, список доступных функций.

### 6. `store_trading_signal`
- **Назначение**: Сохранение торгового сигнала в хранилище памяти.
- **Параметры**: `signal` (объект с полями: `symbol`, `signal_type`, `direction`, `entry`, `confidence`, `context`, `timestamp`).
- **Ответ**: сохранённый сигнал (`TradingSignalRecord`).

### 7. `search_trading_patterns`
- **Назначение**: Поиск сохранённых торговых паттернов и сигналов.
- **Параметры**: `query` (обязательный), `symbol` (опционально), `limit` (по умолчанию 10).
- **Ответ**: список найденных сигналов (`TradingSignalRecord[]`).

### 8. `get_signal_performance`
- **Назначение**: Получение метрик производительности торгового сигнала.
- **Параметры**: `signal_id` (обязательный).
- **Ответ**: информация о сигнале и показатели эффективности (PnL, результат, время закрытия, заметки).

### 9. `ingest_scraped_content`
- **Назначение**: Инжест скрапленного веб-контента в память.
- **Параметры**: `content` (объект с полями: `url`, `title`, `content`, `metadata`, `source`, `tags`, `entities`).
- **Ответ**: `record_id`, статус инжеста, URL, сообщение.

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

#### Таблица переменных окружения

| Переменная | Обязательная | По умолчанию | Используется в | Описание |
|------------|--------------|--------------|----------------|----------|
| `MEMORY_DB_PATH` | Нет | `memory_graph.db` | MCP сервер | Путь к SQLite базе данных графа памяти |
| `QDRANT_URL` | Нет | - | MCP сервер | URL векторного хранилища Qdrant (опционально) |
| `EMBEDDINGS_URL` | Нет | - | MCP сервер | URL сервиса эмбеддингов (приоритет 1) |
| `LMSTUDIO_HOST` | Нет | `127.0.0.1` | MCP сервер, CLI | Хост LM Studio Server (приоритет 2) |
| `LMSTUDIO_PORT` | Нет | `1234` | MCP сервер, CLI | Порт LM Studio Server (приоритет 2) |
| `LMSTUDIO_MODEL` | Нет | - | MCP сервер, CLI | Модель для эмбеддингов в LM Studio |
| `MEMORY_LOG_LEVEL` | Нет | `INFO` | MCP сервер | Уровень логирования MCP сервера |
| `PORT` | Нет | `8050` | HTTP сервер | Порт HTTP сервера (приоритетный) |
| `TG_DUMP_PORT` | Нет | `8050` | HTTP сервер | Альтернативное имя для порта (для совместимости) |
| `HOST` | Нет | `0.0.0.0` | HTTP сервер | Хост для HTTP сервера |
| `LOG_LEVEL` | Нет | `INFO` | HTTP сервер | Уровень логирования HTTP сервера |

#### Приоритет конфигурации эмбеддингов

Для MCP сервера используется следующий приоритет:

1. **Приоритет 1**: `EMBEDDINGS_URL` — если установлен, используется напрямую
2. **Приоритет 2**: `LMSTUDIO_HOST` + `LMSTUDIO_PORT` + `LMSTUDIO_MODEL` — если `EMBEDDINGS_URL` не установлен, формируется URL из этих переменных

**Примеры конфигурации:**

```bash
# Вариант 1: Использование EMBEDDINGS_URL (text-embeddings-inference)
EMBEDDINGS_URL=http://embeddings:80

# Вариант 2: Использование LM Studio (автоматически формируется URL)
LMSTUDIO_HOST=127.0.0.1
LMSTUDIO_PORT=1234
LMSTUDIO_MODEL=text-embedding-qwen3-embedding-0.6b
```

#### Полный список переменных

```bash
# База данных
MEMORY_DB_PATH=/data/memory_graph.db

# Векторное хранилище (опционально)
QDRANT_URL=http://qdrant:6333

# Сервис эмбеддингов (приоритет 1)
EMBEDDINGS_URL=http://embeddings:80

# LM Studio Server (приоритет 2, используется если EMBEDDINGS_URL не установлен)
LMSTUDIO_HOST=127.0.0.1
LMSTUDIO_PORT=1234
LMSTUDIO_MODEL=text-embedding-qwen3-embedding-0.6b

# Логирование
MEMORY_LOG_LEVEL=INFO
LOG_LEVEL=INFO

# HTTP сервер
PORT=8050
TG_DUMP_PORT=8050  # альтернативное имя для совместимости
HOST=0.0.0.0
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

### Docker Compose

**⚠️ ВАЖНО:** Файл `docker-compose.mcp.yml` находится в родительском репозитории `infra/` по пути `../../infra/docker-compose.mcp.yml` относительно корня проекта.

**Для локальной разработки** можно создать собственный `docker-compose.yml` в корне проекта, но для продакшена используется общий файл в `infra/`.

Сервисы в docker-compose:
- **postgres**: хранение графа (или SQLite volume).
- **qdrant** *(опционально)*: векторное хранилище.
- **redis**: кэш/очередь (зарезервировано).
- **embeddings**: сервис `text-embeddings-inference` (опционально, можно использовать LM Studio).
- **memory-mcp**: MCP сервер (entrypoint `python run_server.py`).

### Режимы запуска MCP сервера

Memory MCP поддерживает два режима работы:

#### 1. HTTP режим (для Docker/продакшена)

Использует FastAPI с FastApiMCP для интеграции стандартного MCP Server с HTTP транспортом.

```bash
# Запуск HTTP сервера
python run_server.py

# Сервер будет доступен на http://localhost:8050
# Endpoints:
#   - GET  /          - информация о сервере
#   - GET  /health    - проверка здоровья
#   - GET  /healthz   - health check для Docker/K8s
#   - POST /mcp       - MCP endpoint для обработки запросов
```

**Особенности HTTP режима:**
- Использует FastAPI для HTTP транспорта
- Интеграция через `FastApiHttpSessionManager`
- Поддержка CORS middleware
- Автоматическое закрытие адаптера памяти при остановке

#### 2. Stdio режим (для локальной разработки и MCP клиентов)

Использует стандартный stdio транспорт MCP для интеграции с MCP клиентами (например, Cursor).

```bash
# Запуск через модуль (stdio режим)
python -m memory_mcp.mcp.server
```

**Особенности stdio режима:**
- Прямая интеграция с MCP клиентами через stdio
- Используется для локальной разработки
- Рекомендуется для настройки в MCP клиентах (например, `~/.cursor/mcp.json`)

### CLI команды

```bash
pip install -e .

# Инжест Telegram чатов
python -m memory_mcp.cli.main ingest-telegram --chats-dir chats --db-path memory_graph.db
```

## Тестирование

- **Unit tests**: `pytest tests -q`
- **CLI тесты**: сценарии в `tests/test_cli.py`
- **Гибридный поиск**: `tests/test_mcp_server.py` проверяет FTS/ingest/search.

## Документация

- Обновляйте `README.md` и `CHANGELOG.md` при изменении API/CLI.
- В `AGENTS.md` фиксируйте архитектуру, правила оформления инструментов и конфигурацию.
- Для новых источников добавляйте секции в `indexing/*` + документацию по формату входных данных.
