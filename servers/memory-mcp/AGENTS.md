# Memory MCP Server — Архитектура и Руководство

## Содержание

- [Обзор](#обзор)
- [Основные компоненты](#основные-компоненты)
- [MCP инструменты](#mcp-инструменты)
- [Конфигурация и запуск](#конфигурация-и-запуск)
- [Сценарии использования](#сценарии-использования)
- [Тестирование](#тестирование)

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

### Список всех инструментов

Memory MCP сервер предоставляет следующие инструменты:

1. **`health`** — Проверка состояния MCP сервера и конфигурации
2. **`version`** — Получение информации о версии сервера и доступных возможностях
3. **`ingest_records`** — Приём пачки записей памяти в хранилище
4. **`search_memory`** — Поиск по памяти (FTS + векторный)
5. **`fetch_record`** — Получение полной записи по идентификатору
6. **`store_trading_signal`** — Сохранение торгового сигнала в хранилище памяти
7. **`search_trading_patterns`** — Поиск сохранённых торговых паттернов и сигналов
8. **`get_signal_performance`** — Получение метрик производительности торгового сигнала
9. **`ingest_scraped_content`** — Инжест скрапленного веб-контента в память

### Недостающие инструменты (рекомендуемые для реализации)

Исходя из описанных сценариев использования, следующие инструменты могли бы дополнить функциональность:

#### Эмбеддинги
- **`generate_embedding`** — Генерация эмбеддинга для произвольного текста (сейчас доступно только через прямое использование `EmbeddingService`)

#### Управление записями
- **`update_record`** — Обновление существующей записи (метаданные, теги, контент)
- **`delete_record`** — Удаление записи из хранилища
- **`batch_update_records`** — Массовое обновление записей

#### Статистика и мониторинг
- **`get_indexing_progress`** — Получение прогресса инкрементальной индексации (эквивалент `memory_mcp indexing-progress`)
- **`get_statistics`** — Получение статистики системы (эквивалент `memory_mcp stats`)
  - Количество записей по источникам
  - Количество записей по тегам
  - Временная статистика (записи по периодам)
  - Размер базы данных

#### Работа с графом знаний
- **`get_graph_neighbors`** — Получение соседних узлов для записи
- **`find_graph_path`** — Поиск пути между двумя узлами в графе
- **`build_insight_graph`** — Построение графа знаний (эквивалент `memory_mcp insight-graph`)
- **`get_related_records`** — Получение связанных записей через граф

#### Саммаризация и отчёты
- **`update_summaries`** — Обновление markdown-отчётов без полной индексации (эквивалент `memory_mcp update-summaries`)
- **`review_summaries`** — Автоматическое ревью и исправление саммаризаций (эквивалент `memory_mcp review-summaries`)

#### Экспорт и импорт
- **`export_records`** — Экспорт записей в различных форматах (JSON, CSV, Markdown)
- **`import_records`** — Импорт записей из файлов

#### Расширенный поиск
- **`search_by_embedding`** — Поиск по эмбеддингу (без текстового запроса)
- **`similar_records`** — Поиск похожих записей по заданной записи
- **`search_explain`** — Объяснение результатов поиска с декомпозицией scores

#### Аналитика
- **`analyze_entities`** — Анализ сущностей в записях
- **`get_timeline`** — Получение временной линии записей
- **`get_tags_statistics`** — Статистика по тегам

### Стандарт описания инструментов
- **Первая строка**: лаконичное действие на английском (≤90 символов), начинаем с глагола.
- **Где хранится**: описания находятся в `src/memory_mcp/mcp/server.py` в функции `list_tools()` через объекты `Tool()` с полем `description`.
- **Детали**: разворачиваем во втором предложении/абзаце docstring, чтобы `list_tools` оставался компактным.
- **Единый подход**: новые инструменты должны следовать тем же правилам, чтобы клиенты видели консистентные подсказки.

### Диагностика проблем с загрузкой инструментов

Если инструменты не загружаются в MCP клиенте (например, Cursor показывает "Loading tools" без результата):

1. **Проверьте логи сервера**: При запуске сервера должны появиться сообщения:
   ```
   list_tools() вызвана, регистрируем инструменты...
   Зарегистрировано 9 инструментов: ['health', 'version', 'ingest_records', ...]
   ```

2. **Проверьте конфигурацию MCP клиента**: Убедитесь, что в `~/.cursor/mcp.json` правильно указан путь к серверу:
   ```json
   {
     "mcpServers": {
       "memory-mcp": {
         "command": "/path/to/venv/bin/python",
         "args": ["-m", "memory_mcp.mcp.server"],
         "cwd": "/path/to/memory-mcp"
       }
     }
   }
   ```

3. **Проверьте, что сервер запускается без ошибок**: Запустите сервер вручную:
   ```bash
   python -m memory_mcp.mcp.server
   ```
   Сервер должен запуститься и ждать ввода через stdio.

4. **Проверьте импорты**: Убедитесь, что все зависимости установлены:
   ```bash
   pip install -e .
   ```

5. **Проверьте логирование**: В коде сервера добавлено логирование, которое поможет диагностировать проблему. Проверьте логи при запуске сервера.

6. **Проверьте, что Python доступен**: Убедитесь, что путь к Python в конфигурации правильный:
   ```bash
   # Проверьте, что Python доступен по указанному пути
   /path/to/venv/bin/python --version
   
   # Проверьте, что модуль импортируется
   /path/to/venv/bin/python -m memory_mcp.mcp.server --help
   ```

7. **Проверьте логи Cursor**: В Cursor могут быть логи MCP сервера. Проверьте:
   - Откройте Developer Tools в Cursor (View → Developer → Toggle Developer Tools)
   - Проверьте консоль на наличие ошибок
   - Проверьте вкладку Network на наличие запросов к MCP серверу

8. **Важно**: `list_tools()` вызывается только когда клиент запрашивает список инструментов, а не при запуске сервера. Это означает, что логи "list_tools() вызвана..." появятся только после того, как Cursor запросит список инструментов.

9. **Проблема с базой данных**: Если вы видите ошибку "unable to open database file":
   - **Для локальной разработки**: БД создается автоматически в директории `data/` (`data/memory_graph.db`)
   - **Для Docker**: БД монтируется из локальной директории `data/` в контейнер (`/app/data/memory_graph.db`)
   - Убедитесь, что путь к БД правильный (можно задать через переменную окружения `MEMORY_DB_PATH`)
   - Если путь относительный, он будет разрешен относительно корня проекта (где находится `pyproject.toml`)
   - Убедитесь, что директория для БД существует и доступна для записи
   - В Docker контейнере директория `data/` автоматически создается и монтируется из хоста

### Краткая справка по инструментам

| Инструмент | Назначение | Ключевые параметры |
|------------|------------|-------------------|
| `ingest_records` | Приём записей в хранилище | `records[]` (MemoryRecordPayload) |
| `search_memory` | Гибридный поиск (FTS + векторный) | `query`, `top_k`, `source`, `tags`, `date_from`, `date_to` |
| `fetch_record` | Получение записи по ID | `record_id` |
| `health` | Проверка состояния сервера | - |
| `version` | Информация о версии | - |
| `store_trading_signal` | Сохранение торгового сигнала | `symbol`, `signal_type`, `direction`, `entry`, `confidence` |
| `search_trading_patterns` | Поиск торговых паттернов | `query`, `symbol`, `limit` |
| `get_signal_performance` | Метрики производительности сигнала | `signal_id` |
| `ingest_scraped_content` | Индексация веб-контента | `url`, `content`, `title`, `metadata` |

**Примечание**: Подробные описания параметров и примеры использования см. в `src/memory_mcp/mcp/server.py` и разделе [Сценарии использования](#сценарии-использования).

### Документация инструментов — best practices
- **Обязательные/взаимоисключающие параметры**: явно указывайте `code` XOR `script_path`, `steps` как массив и т.п.
- **Примеры вызова**: предоставляйте рабочие фрагменты (как сформировать `records`, как делать поиск).
- **Форматы полей**:
  - `env` — список строк `KEY=VALUE`.
  - `steps` — список строк.
  - `memory`/`cpus` — Docker-совместимые строки (`256m`, `1g`, `0.5`).
  - `tags`/`entities` — массивы строк.
  - `timestamp` — ISO 8601 строка (например, `"2024-01-15T14:30:00Z"` или `"2024-01-15T14:30:00+00:00"`).
- **Сериализация datetime**: Все datetime объекты автоматически сериализуются в ISO 8601 формат при возврате результатов.
- **Подсказки/use cases**: добавляйте bullet tips (как добавить зависимости, как выполнять multi-step сценарии, как фильтровать поиск).
- **Troubleshooting**: перечисляйте частые ошибки (пустой `steps`, неправильный формат `env`, отсутствие `records`).

## Конфигурация и запуск

### Переменные окружения

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `MEMORY_DB_PATH` | `data/memory_graph.db` | Путь к SQLite БД (относительный путь разрешается от корня проекта) |
| `QDRANT_URL` | - | URL векторного хранилища Qdrant (опционально) |
| `EMBEDDINGS_URL` | - | URL сервиса эмбеддингов (**приоритет 1**) |
| `LMSTUDIO_HOST` | `127.0.0.1` | Хост LM Studio Server (**приоритет 2**) |
| `LMSTUDIO_PORT` | `1234` | Порт LM Studio Server |
| `LMSTUDIO_MODEL` | - | Модель для эмбеддингов в LM Studio |
| `MEMORY_LOG_LEVEL` | `INFO` | Уровень логирования MCP сервера |
| `PORT` / `TG_DUMP_PORT` | `8050` | Порт HTTP сервера |
| `HOST` | `0.0.0.0` | Хост HTTP сервера |
| `LOG_LEVEL` | `INFO` | Уровень логирования HTTP сервера |

**Приоритет конфигурации эмбеддингов:**
1. `EMBEDDINGS_URL` (если установлен, используется напрямую)
2. `LMSTUDIO_HOST` + `LMSTUDIO_PORT` + `LMSTUDIO_MODEL` (формируется URL автоматически)

**Примеры:**
```bash
# Вариант 1: text-embeddings-inference
EMBEDDINGS_URL=http://embeddings:80

# Вариант 2: LM Studio
LMSTUDIO_HOST=127.0.0.1
LMSTUDIO_PORT=1234
LMSTUDIO_MODEL=text-embedding-qwen3-embedding-0.6b

# Для Docker
MEMORY_DB_PATH=/app/data/memory_graph.db
QDRANT_URL=http://qdrant:6333
```

### Docker Compose развертывание

**⚠️ ВАЖНО:** Файл `docker-compose.mcp.yml` находится в родительском репозитории `infra/` по пути `../../infra/docker-compose.mcp.yml` относительно корня проекта.

**Сервисы:**
- `postgres` — хранение графа (или SQLite volume)
- `qdrant` *(опционально)* — векторное хранилище
- `redis` — кэш/очередь (зарезервировано)
- `embeddings` — сервис `text-embeddings-inference` (опционально, можно использовать LM Studio)
- `memory-mcp` — MCP сервер (entrypoint `python run_server.py`)

**Основные команды:**

```bash
# Пересборка и перезапуск после изменений кода
docker-compose -f ../../infra/docker-compose.mcp.yml build memory-mcp
docker-compose -f ../../infra/docker-compose.mcp.yml up -d memory-mcp

# Полная пересборка всех сервисов
docker-compose -f ../../infra/docker-compose.mcp.yml build
docker-compose -f ../../infra/docker-compose.mcp.yml up -d

# Отладка
docker-compose -f ../../infra/docker-compose.mcp.yml exec memory-mcp bash
docker-compose -f ../../infra/docker-compose.mcp.yml logs -f memory-mcp
curl http://localhost:8050/health

# Очистка и пересборка с нуля
docker-compose -f ../../infra/docker-compose.mcp.yml down
docker-compose -f ../../infra/docker-compose.mcp.yml build --no-cache memory-mcp
docker-compose -f ../../infra/docker-compose.mcp.yml up -d memory-mcp
```

**Оптимизация сборки Docker образа:**

Dockerfile использует многоэтапную сборку для оптимизации кэширования:

1. **Этап `deps`**: Устанавливает зависимости из `pyproject.toml` и `uv.lock`
   - Этот слой кэшируется отдельно и пересобирается только при изменении зависимостей
   - Использует кэш Docker для ускорения установки пакетов (`--mount=type=cache`)

2. **Этап `final`**: Копирует исходный код приложения
   - Этот слой пересобирается при изменении кода
   - Зависимости уже установлены в предыдущем этапе

**Преимущества:**
- При изменении кода зависимости не переустанавливаются (используется кэш)
- Значительно ускоряется сборка при разработке
- Экономия времени и трафика при частых пересборках

**Пример:**
```bash
# Первая сборка - устанавливаются зависимости и копируется код
docker-compose -f ../../infra/docker-compose.mcp.yml build memory-mcp
# Время: ~5-10 минут (зависит от зависимостей)

# Изменение кода и пересборка - зависимости берутся из кэша
# (изменили файл в src/)
docker-compose -f ../../infra/docker-compose.mcp.yml build memory-mcp
# Время: ~10-30 секунд (только копирование кода)
```

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
python -m memory_mcp.cli.main ingest-telegram --chats-dir chats --db-path data/memory_graph.db
```

## Тестирование

- **Unit tests**: `pytest tests -q`
- **CLI тесты**: сценарии в `tests/test_cli.py`
- **Гибридный поиск**: `tests/test_mcp_server.py` проверяет FTS/ingest/search.

## Сценарии использования

### Индексация данных

**CLI (Telegram чаты):**
```bash
memory_mcp index --progress              # Первоначальная индексация
memory_mcp index                         # Инкрементальная (только новые)
memory_mcp index --force-full            # Полная переиндексация
```

**MCP API:**
```python
# Индексация записей
mcp_client.call_tool("ingest_records", {
    "records": [{
        "record_id": "msg-001",
        "source": "telegram",
        "content": "Текст сообщения",
        "timestamp": "2024-01-15T14:30:00Z",
        "tags": ["crypto", "news"]
    }]
})

# Индексация веб-контента
mcp_client.call_tool("ingest_scraped_content", {
    "url": "https://example.com/article",
    "content": "Основной текст...",
    "title": "Заголовок"
})
```

**Процесс:** Валидация → Создание узлов графа → Генерация эмбеддингов → Сохранение в векторное хранилище (Qdrant)

### Поиск по данным

**Гибридный поиск (FTS + векторный):**
```python
result = mcp_client.call_tool("search_memory", {
    "query": "криптовалюты и блокчейн",
    "top_k": 10,
    "source": "telegram",  # опционально
    "tags": ["crypto"],     # опционально
    "date_from": "2024-01-01T00:00:00Z",
    "include_embeddings": False
})
```

**CLI:**
```bash
memory_mcp search "криптовалюты"
memory_mcp search "TODO встреча" --collection tasks
```

**Алгоритм:** FTS (BM25, вес 0.4) + Vector Search (вес 0.6) → Объединение результатов

### Работа с эмбеддингами

**Конфигурация:**
```bash
# Приоритет 1: text-embeddings-inference
EMBEDDINGS_URL=http://embeddings:80

# Приоритет 2: LM Studio
LMSTUDIO_HOST=127.0.0.1
LMSTUDIO_PORT=1234
LMSTUDIO_MODEL=text-embedding-qwen3-embedding-0.6b
```

**Использование:**
- Автоматически при индексации (если настроен `EmbeddingService`)
- При поиске с `include_embeddings=True`
- Прямой вызов через `EmbeddingService.embed(text)`

### Специализированные сценарии

**Торговые сигналы:**
```python
mcp_client.call_tool("store_trading_signal", {
    "symbol": "BTCUSDT", "signal_type": "momentum", "direction": "long"
})
mcp_client.call_tool("search_trading_patterns", {"query": "momentum"})
mcp_client.call_tool("get_signal_performance", {"signal_id": "signal-001"})
```

**Мониторинг:**
```python
mcp_client.call_tool("health", {})    # Проверка состояния
mcp_client.call_tool("version", {})   # Информация о версии
```

### Схема потока данных

**Индексация:**
```
Telegram CLI / Web Scrape / MCP Ingest
    ↓
MemoryRecordPayload
    ↓
┌──────────────────┬──────────────────┐
│ TypedGraphMemory │ EmbeddingService │
│ (SQLite+NetworkX)│ → VectorStore    │
│                  │   (Qdrant)       │
└──────────────────┴──────────────────┘
```

**Поиск:**
```
Search Query
    ↓
┌──────────────┬──────────────┐
│ FTS (FTS5)   │ Vector Search│
│ (вес 0.4)    │ (вес 0.6)    │
└──────┬───────┴──────┬───────┘
       └──────┬───────┘
              ↓
      Hybrid Results
```

### Рекомендуемые workflow

**Первоначальная настройка:**
```bash
export EMBEDDINGS_URL=http://embeddings:80  # или LM Studio
memory_mcp check
memory_mcp index --progress
```

**Регулярное обновление:**
```bash
memory_mcp index                    # Инкрементальная индексация
memory_mcp update-summaries         # Обновление отчётов
memory_mcp search "новые темы"      # Поиск
```

**Программная работа:**
```python
mcp_client.call_tool("ingest_records", {"records": [...]})
mcp_client.call_tool("search_memory", {"query": "...", "top_k": 10})
mcp_client.call_tool("fetch_record", {"record_id": "..."})
```

## Документация

- Обновляйте `README.md` и `CHANGELOG.md` при изменении API/CLI.
- В `AGENTS.md` фиксируйте архитектуру, правила оформления инструментов и конфигурацию.
- Для новых источников добавляйте секции в `indexing/*` + документацию по формату входных данных.
