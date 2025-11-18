# Memory MCP Server — Архитектура и Руководство

## Содержание

- [Обзор](#обзор)
- [Основные компоненты](#основные-компоненты)
- [MCP инструменты](#mcp-инструменты)
- [Конфигурация и запуск](#конфигурация-и-запуск)
- [Сценарии использования](#сценарии-использования)
- [Стратегии группировки сообщений](#стратегии-группировки-сообщений)
- [Тестирование](#тестирование)

## Обзор

**Memory MCP** — сервер Model Context Protocol для индексации и поиска по унифицированной памяти (Telegram, файлы и другие источники), поддерживающий полнотекстовый и векторный поиск.

```
memory-mcp/
├── src/memory_mcp/
│   ├── cli/                 # CLI утилиты
│   ├── indexing/            # Индексаторы источников (Telegram и др.)
│   ├── memory/              # Граф памяти, FTS, векторное хранилище
│   ├── mcp/                 # MCP schema, adapters и сервер
│   ├── core/                # Основные компоненты (lmstudio_client, ollama_client, indexer)
│   ├── analysis/            # Анализ данных и саммаризация
│   ├── quality_analyzer/    # Анализ качества поиска
│   ├── search/              # Smart Search и гибридный поиск
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
- Читает настройки из `Settings` (`MEMORY_MCP_*`): пути БД/артефактов, LM Studio, embeddings и Qdrant.
- Поддерживает ленивую инициализацию компонентов для ускорения запуска.

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
- Поддерживает торговые сигналы и специализированные запросы.

### Граф памяти (`src/memory_mcp/memory/typed_graph.py`)

- SQLite + NetworkX; FTS5 таблица `node_search`.
- Добавляет/обновляет узлы (`DocChunkNode`, `EventNode` и др.).
- Предоставляет метод `search_text` (BM25/snippet).
- Поддерживает типизированные связи между узлами.

### Векторное хранилище (`src/memory_mcp/memory/vector_store.py`)

- Обёртка над Qdrant (создание коллекции, upsert, поиск).
- Конфигурируется через `MEMORY_MCP_QDRANT_URL`; gracefully выключается, если Qdrant недоступен.
- Поддерживает фильтрацию по метаданным.

### Embed-сервис (`src/memory_mcp/memory/embeddings.py`)

- HTTP клиент для `text-embeddings-inference` (или совместимых сервисов).
- Функция `build_embedding_service_from_env()` читает `MEMORY_MCP_EMBEDDINGS_URL` или параметры LM Studio.
- Поддерживает батчинг для эффективной обработки.

### Индексаторы (`src/memory_mcp/indexing/*`)

- Unified интерфейс `BaseIndexer`.
- `TelegramIndexer` → нормализованные `MemoryRecord`.
- CLI команда `ingest-telegram` вызывает индексатор + `MemoryIngestor`.

### Smart Search движок (`src/memory_mcp/search/*`)

- `SmartSearchEngine` объединяет адаптер памяти, `ArtifactsReader` и `SearchSessionStore` для интерактивного поиска.
- Сессии многократного поиска сохраняются в `data/search_sessions.db` (настраивается `SMART_SEARCH_SESSION_STORE_PATH`) и позволяют применять обратную связь (`feedback`).
- `hybrid_search.py` и `search_explainer.py` обеспечивают подсветку источников (граф, Qdrant, артефакты) и пояснения к результатам.
- Порог уверенности и стратегия уточняющих вопросов управляются через `SMART_SEARCH_MIN_CONFIDENCE` и связанные настройки `QualityAnalysis`.

### Менеджмент фоновой индексации

- `IndexingJobTracker` в `core/indexing_tracker.py` отслеживает жизненный цикл индексаций (`start_background_indexing`, `stop_background_indexing`, `get_background_indexing_status`).
- Стадии включают очистку старых данных, загрузку чатов, оптимизацию SQLite (VACUUM/ANALYZE) и проверку целостности, чтобы диагностировать узкие места.
- `MemoryServiceAdapter` и CLI используют общий менеджер для чтения/обновления статусов, что позволяет отображать прогресс и предотвращать параллельные full-rebuild для одного чата.
- Настройки фоновой индексации (`MEMORY_MCP_BACKGROUND_INDEXING_ENABLED`, `MEMORY_MCP_BACKGROUND_INDEXING_INTERVAL`) включают периодический обход `input/` и безопасное завершение при остановке сервера.

## MCP инструменты

### Категории и назначение

Все инструменты описаны и регистрируются в `src/memory_mcp/mcp/server.py::list_tools()`. Параметры и схемы запросов синхронизированы с обработчиками в `call_tool()`, поэтому при добавлении нового инструмента нужно обновлять оба места (и тесты `tests/test_mcp_server*.py`).

#### Системные

- `health` — проверка зависимостей, конфигурации и доступности базы данных/векторного стора.
- `version` — версия сервера, список включённых features; используется клиентами для capability detection.

#### Индексация и управление данными

- `ingest_records` — приём пачки `MemoryRecordPayload` в унифицированное хранилище.
- `ingest_scraped_content` — индексация веб-контента (скрапленные страницы).
- `index_chat` — индексация конкретного Telegram чата с отслеживанием прогресса (фоновый режим).
- `get_available_chats` — получение списка доступных чатов для индексации.
- `start_background_indexing` — запуск фоновой индексации (периодическая проверка `input/`).
- `stop_background_indexing` — остановка фоновой индексации.
- `get_background_indexing_status` — статус фоновой индексации.
- `get_indexing_progress` — прогресс индексации конкретного чата или всех чатов.
- `import_records` — массовый импорт записей из JSON/CSV.
- `export_records` — массовый экспорт записей в JSON/CSV/Markdown.
- `update_record` — обновление одной записи (контент, метаданные, теги).
- `batch_update_records` — пакетное обновление записей.
- `batch_delete_records` — пакетное удаление записей.
- `batch_fetch_records` — пакетное получение записей.
- `update_summaries` — пересборка markdown-отчётов без полной индексации.
- `review_summaries` — автоматическое ревью и исправление саммаризаций.

#### Поиск и аналитика

- `search_memory` — гибридный поиск (BM25 + векторный) по памяти.
- `smart_search` — интерактивный LLM-assisted поиск с сессиями и обратной связью.
- `search_by_embedding` — поиск по вектору эмбеддинга напрямую.
- `similar_records` — поиск похожих записей по эмбеддингам.
- `search_explain` — объяснение релевантности результата поиска с декомпозицией scores.
- `get_statistics` — статистика системы (граф, источники, теги, размер БД).
- `get_tags_statistics` — статистика использования тегов.
- `get_timeline` — временная шкала записей, отсортированная по timestamp.
- `analyze_entities` — извлечение и анализ сущностей из текста.

#### Граф знаний и навигация

- `get_graph_neighbors` — получение соседних узлов для узла графа.
- `find_graph_path` — поиск кратчайшего пути между двумя узлами.
- `get_related_records` — получение связанных записей через граф.
- `build_insight_graph` — построение графа инсайтов из markdown-саммаризаций.

#### Торговые сигналы

- `store_trading_signal` — сохранение торгового сигнала в память.
- `search_trading_patterns` — поиск по сохранённым торговым паттернам и сигналам.
- `get_signal_performance` — метрики производительности торгового сигнала.

#### Дополнительные сервисы

- `generate_embedding` — прямой вызов сервиса эмбеддингов; упрощает отладку и внешние интеграции.

> ⚠️ **Важно**: Перечисленные инструменты имеют строгие схемы (`schema.py`). Перед изменением аргументов синхронизируйте:
> 1. `Tool.inputSchema` в `list_tools()`
> 2. Pydantic модель запроса/ответа в `schema.py`
> 3. Обработчик в `call_tool()`
> 4. Документацию в этом файле и `README.md`
> 5. Смежные тесты (`tests/test_mcp_server*.py`, `tests/test_trading_memory.py`, `tests/test_smart_search.py`)

### Стандарт описания инструментов

- **Первая строка**: лаконичное действие на английском (≤90 символов), начинаем с глагола.
- **Где хранится**: описания находятся в `src/memory_mcp/mcp/server.py` в функции `list_tools()` через объекты `Tool()` с полем `description`.
- **Детали**: разворачиваем во втором предложении/абзаце docstring, чтобы `list_tools` оставался компактным.
- **Единый подход**: новые инструменты должны следовать тем же правилам, чтобы клиенты видели консистентные подсказки.

### Краткая справка по инструментам

| Инструмент | Назначение | Ключевые параметры |
|------------|------------|-------------------|
| `ingest_records` | Приём пачки `MemoryRecordPayload` | `records[]`, `upsert`, `include_embeddings` |
| `search_memory` | Гибридный поиск (BM25 + вектор) | `query`, `top_k`, фильтры `source/tags/date` |
| `smart_search` | Интерактивный поиск с обратной связью | `query`, `session_id`, `feedback[]`, `clarify` |
| `index_chat` | Индексация Telegram чата (фоновый режим) | `chat`, `aggregation_strategy`, фильтры дат, `enable_smart_aggregation` |
| `get_indexing_progress` | Прогресс индексации | `chat` (опционально) |
| `export_records` | Экспорт записей/вложений | `source`, `tags`, `date_from`, `include_embeddings`, `format` |
| `import_records` | Импорт JSONL/директорий с вложениями | `format`, `content`, `source` |
| `generate_embedding` | Получение векторов без записи в БД | `text`, `model` (опционально) |
| `store_trading_signal` | Сохранение торгового сигнала | `symbol`, `signal_type`, `direction`, `entry`, `confidence` |
| `search_trading_patterns` | Поиск по торговым сигналам | `query`, `symbol`, `limit` |
| `start_background_indexing` | Запуск фонового ingest по входной директории | (без параметров) |
| `get_background_indexing_status` | Статус фоновой индексации | (без параметров) |
| `get_available_chats` | Список доступных чатов | `include_stats` (опционально) |

**Примечание**: Подробные описания параметров и примеры использования см. в `src/memory_mcp/mcp/server.py` и разделе [Сценарии использования](#сценарии-использования).

### Диагностика проблем

**Инструменты не загружаются:**
1. Проверьте конфигурацию MCP клиента (`~/.cursor/mcp.json`):
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
2. Проверьте запуск сервера: `python -m memory_mcp.mcp.server`
3. Установите зависимости: `pip install -e .`
4. Проверьте логи в Cursor (Developer Tools → Console)

**Ошибка "unable to open database file":**
- Локально: БД создаётся в `data/memory_graph.db` (относительно корня проекта)
- Docker: БД монтируется из `data/` в `/app/data/memory_graph.db`
- Убедитесь, что директория существует и доступна для записи
- Проверьте `MEMORY_MCP_DB_PATH`; относительный путь резолвится от `pyproject.toml`.

**Примечание**: `list_tools()` вызывается только при запросе списка инструментов клиентом, не при запуске сервера.

## Конфигурация и запуск

### Переменные окружения

Основные настройки читаются через `pydantic.BaseSettings` с префиксом `MEMORY_MCP_`.

#### Основные настройки

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `MEMORY_MCP_DB_PATH` | `data/memory_graph.db` | Путь к SQLite БД. Относительный путь резолвится от `pyproject.toml`. |
| `MEMORY_MCP_CHROMA_PATH` | `./chroma_db` | Путь к базе ChromaDB. |
| `MEMORY_MCP_CHATS_PATH` | `./chats` | Путь к директории с чатами. |
| `MEMORY_MCP_ARTIFACTS_PATH` | `./artifacts` | Путь к артефактам (отчёты, саммаризации). |
| `MEMORY_MCP_INPUT_PATH` | `input` | Путь к директории input с новыми сообщениями. |

#### Настройки сервера

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `MEMORY_MCP_HOST` | `127.0.0.1` | HTTP-хост для FastAPI транспорта. |
| `MEMORY_MCP_PORT` | `8000` | HTTP-порт сервера / CLI RPC. |
| `MEMORY_MCP_LOG_LEVEL` | `INFO` | Уровень логирования MCP. |
| `MEMORY_MCP_TRANSPORT` | `stdio` | Транспорт запуска (`stdio`, `streamable-http`). |
| `MEMORY_MCP_DEBUG` | `False` | Включение расширенного логгирования. |

#### Настройки эмбеддингов и LLM

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `MEMORY_MCP_EMBEDDINGS_URL` | - | URL внешнего сервиса эмбеддингов (приоритетнее LM Studio). |
| `MEMORY_MCP_LMSTUDIO_HOST` | `127.0.0.1` | LM Studio (эмбеддинги/LLM) хост. |
| `MEMORY_MCP_LMSTUDIO_PORT` | `1234` | LM Studio порт. |
| `MEMORY_MCP_LMSTUDIO_MODEL` | `text-embedding-qwen3-embedding-0.6b` | Модель для `/v1/embeddings`. |
| `MEMORY_MCP_LMSTUDIO_LLM_MODEL` | `gpt-oss-20b` | LLM для `/v1/chat/completions`; если не задан, используется Ollama. |

#### Настройки векторного хранилища

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `MEMORY_MCP_QDRANT_URL` | - | URL Qdrant. Если пуст, векторный поиск не активен. |

#### Настройки фоновой индексации

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `MEMORY_MCP_BACKGROUND_INDEXING_ENABLED` | `False` | Авто-запуск фоновой индексации при старте. |
| `MEMORY_MCP_BACKGROUND_INDEXING_INTERVAL` | `60` | Интервал проверки `input/` (сек). |

#### Настройки Smart Search

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `SMART_SEARCH_SESSION_STORE_PATH` | `data/search_sessions.db` | Расположение БД для `SmartSearchEngine`. |
| `SMART_SEARCH_MIN_CONFIDENCE` | `0.5` | Порог уверенности для LLM-уточнений. |

#### Настройки больших контекстов

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `MEMORY_MCP_LARGE_CONTEXT_MAX_TOKENS` | `131072` | Максимальный контекст модели (токенов). |
| `MEMORY_MCP_LARGE_CONTEXT_PROMPT_RESERVE` | `5000` | Резерв токенов для промпта. |
| `MEMORY_MCP_LARGE_CONTEXT_HIERARCHICAL_THRESHOLD` | `100000` | Порог для включения иерархической обработки (токенов). |
| `MEMORY_MCP_LARGE_CONTEXT_ENABLE_HIERARCHICAL` | `True` | Включить иерархическую обработку. |
| `MEMORY_MCP_LARGE_CONTEXT_USE_SMART_SEARCH` | `True` | Использовать smart_search для анализа контекста. |
| `MEMORY_MCP_LARGE_CONTEXT_CACHE_SIZE` | `100` | Размер кэша для промежуточных результатов. |

#### Настройки анализа качества (QUALITY_ANALYSIS_*)

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `QUALITY_ANALYSIS_OLLAMA_MODEL` | `gpt-oss-20b:latest` | Модель Ollama для анализа качества (fallback, если нет LM Studio LLM). |
| `QUALITY_ANALYSIS_OLLAMA_BASE_URL` | `http://localhost:11434` | URL Ollama сервера. |

**Приоритет конфигурации эмбеддингов:**
1. `MEMORY_MCP_EMBEDDINGS_URL` — прямой HTTP endpoint.
2. `MEMORY_MCP_LMSTUDIO_HOST` + `MEMORY_MCP_LMSTUDIO_PORT` + `MEMORY_MCP_LMSTUDIO_MODEL` — формируется локальный URL.

**Важно - разделение моделей:**
- **Эмбеддинги**: `MEMORY_MCP_LMSTUDIO_MODEL` используется ТОЛЬКО для генерации эмбеддингов через endpoint `/v1/embeddings`.
- **Генерация текста (LLM)**: 
  - Если установлен `MEMORY_MCP_LMSTUDIO_LLM_MODEL`, используется для генерации текста через endpoint `/v1/chat/completions`.
  - Если `MEMORY_MCP_LMSTUDIO_LLM_MODEL` не установлен, используется Ollama с моделью `gpt-oss-20b:latest` (см. `QUALITY_ANALYSIS_OLLAMA_MODEL`).
- **НЕ используйте модель эмбеддингов для генерации текста** - это вызовет ошибку "Model is not llm"

**Примеры:**
```bash
# Вариант 1: text-embeddings-inference
MEMORY_MCP_EMBEDDINGS_URL=http://embeddings:80

# Вариант 2: LM Studio
MEMORY_MCP_LMSTUDIO_HOST=127.0.0.1
MEMORY_MCP_LMSTUDIO_PORT=1234
MEMORY_MCP_LMSTUDIO_MODEL=text-embedding-qwen3-embedding-0.6b  # для эмбеддингов
MEMORY_MCP_LMSTUDIO_LLM_MODEL=gpt-oss-20b                       # для LLM (опционально)

# Для Docker
MEMORY_MCP_DB_PATH=/app/data/memory_graph.db
MEMORY_MCP_QDRANT_URL=http://qdrant:6333
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
docker-compose -f ../../infra/docker-compose.mcp.yml build && docker-compose -f ../../infra/docker-compose.mcp.yml up -d

# Отладка
docker-compose -f ../../infra/docker-compose.mcp.yml exec memory-mcp bash
docker-compose -f ../../infra/docker-compose.mcp.yml logs -f memory-mcp
curl http://localhost:8050/health

# Очистка и пересборка с нуля
docker-compose -f ../../infra/docker-compose.mcp.yml down
docker-compose -f ../../infra/docker-compose.mcp.yml build --no-cache memory-mcp
docker-compose -f ../../infra/docker-compose.mcp.yml up -d memory-mcp
```

**Оптимизация сборки:**

Dockerfile использует многоэтапную сборку:
- **Этап `deps`**: Устанавливает зависимости (кэшируется отдельно)
- **Этап `final`**: Копирует код (пересобирается при изменениях)

**Преимущества**: При изменении кода зависимости не переустанавливаются (используется кэш). Первая сборка ~5-10 мин, последующие ~10-30 сек.

### Режимы запуска MCP сервера

**1. HTTP режим** (для Docker/продакшена):
```bash
python run_server.py
# Доступен на http://localhost:8050
# Endpoints: GET /, GET /health, GET /healthz, POST /mcp
```
- Использует FastAPI с FastApiMCP
- Поддержка CORS middleware
- Автоматическое закрытие адаптера при остановке

**2. Stdio режим** (для локальной разработки и MCP клиентов):
```bash
python -m memory_mcp.mcp.server
```
- Прямая интеграция с MCP клиентами через stdio
- Рекомендуется для настройки в `~/.cursor/mcp.json`

### CLI команды

```bash
pip install -e .

# Инжест Telegram чатов
python -m memory_mcp.cli.main ingest-telegram --chats-dir chats --db-path data/memory_graph.db

# Основные команды
memory_mcp check              # Проверка системы
memory_mcp index              # Инкрементальная индексация
memory_mcp index --force-full # Полная переиндексация
memory_mcp search "query"     # Поиск
memory_mcp stats              # Статистика
```

## Тестирование

- **Unit tests**: `pytest tests -q`
- **CLI тесты**: сценарии в `tests/test_cli.py`
- **Гибридный поиск**: `tests/test_mcp_server.py` проверяет FTS/ingest/search.
- **Smart Search**: `tests/test_smart_search.py` проверяет интерактивный поиск.

## Стратегии группировки сообщений

Параметр `aggregation_strategy` определяет, как сообщения группируются при индексации.

### `"smart"` — Умная стратегия (по умолчанию)

**Особенности:**
- Автоматически определяет тип чата (канал/группа/чат)
- Для чатов/групп: группировка по **сессиям** (связанные диалоги, разрыв до 4ч, макс. 24ч)
- Для каналов: группировка по **дням** (временная группировка)
- Создаёт NOW окно с контекстом для актуальных сообщений

**Временные окна:**
- **NOW** (0-1 день): сессии/дни, оригинальные сообщения, саммари с контекстом
- **FRESH** (1-14 дней): сессии/дни, оригинальные сообщения, без саммари
- **RECENT** (14-30 дней): недели, без оригиналов, саммари
- **OLD** (30+ дней): месяцы, без оригиналов, саммари

**Использование:**
```python
mcp_client.call_tool("index_chat", {
    "chat": "MyChat",
    "enable_smart_aggregation": True,
    "aggregation_strategy": "smart"
})
```

### `"channel"` — Стратегия для каналов

- Автоматически выбирается в `smart`, если чат определён как канал (<30% сообщений с автором)
- Группировка только по временным окнам (дни/недели/месяцы)
- Не использует `SessionSegmenter` для связанности

**Рекомендация**: Используйте `"smart"` (по умолчанию) — автоматически выберет оптимальную стратегию.

## Сценарии использования

### Индексация данных

**CLI:**
```bash
memory_mcp index              # Инкрементальная индексация
memory_mcp index --force-full # Полная переиндексация
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

# Индексация чата (фоновый режим)
mcp_client.call_tool("index_chat", {
    "chat": "MyChat",
    "force_full": False,
    "enable_smart_aggregation": True
})

# Проверка прогресса
progress = mcp_client.call_tool("get_indexing_progress", {
    "chat": "MyChat"  # опционально
})
```

**Процесс:** Валидация → Создание узлов графа → Генерация эмбеддингов → Сохранение в Qdrant

### Поиск по данным

**Гибридный поиск (FTS + векторный):**
```python
result = mcp_client.call_tool("search_memory", {
    "query": "криптовалюты и блокчейн",
    "top_k": 10,
    "source": "telegram",  # опционально
    "tags": ["crypto"],     # опционально
    "date_from": "2024-01-01T00:00:00Z"
})
```

**Smart Search (интерактивный):**
```python
# Первый запрос
result1 = mcp_client.call_tool("smart_search", {
    "query": "обсуждение криптовалют",
    "top_k": 10
})

# Уточнение с обратной связью
result2 = mcp_client.call_tool("smart_search", {
    "query": "обсуждение криптовалют",
    "session_id": result1["session_id"],
    "feedback": [
        {
            "record_id": "msg-001",
            "relevance": "relevant",
            "comment": "Именно то, что нужно"
        }
    ],
    "clarify": True
})
```

**CLI:** `memory_mcp search "криптовалюты"`

**Алгоритм:** FTS (BM25, вес 0.4) + Vector Search (вес 0.6) → Объединение результатов

### Эмбеддинги

**Конфигурация:**
```bash
# Приоритет 1: text-embeddings-inference
MEMORY_MCP_EMBEDDINGS_URL=http://embeddings:80

# Приоритет 2: LM Studio
MEMORY_MCP_LMSTUDIO_HOST=127.0.0.1
MEMORY_MCP_LMSTUDIO_PORT=1234
MEMORY_MCP_LMSTUDIO_MODEL=text-embedding-qwen3-embedding-0.6b
```

**Использование:** Автоматически при индексации, при поиске с `include_embeddings=True`, или через `generate_embedding` инструмент.

### Специализированные сценарии

**Торговые сигналы:**
```python
mcp_client.call_tool("store_trading_signal", {
    "symbol": "BTCUSDT",
    "signal_type": "momentum",
    "direction": "long",
    "entry": 50000.0,
    "confidence": 85
})
mcp_client.call_tool("search_trading_patterns", {
    "query": "momentum",
    "symbol": "BTCUSDT"
})
mcp_client.call_tool("get_signal_performance", {
    "signal_id": "signal-001"
})
```

**Мониторинг:** 
```python
health = mcp_client.call_tool("health", {})
version = mcp_client.call_tool("version", {})
stats = mcp_client.call_tool("get_statistics", {})
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
export MEMORY_MCP_EMBEDDINGS_URL=http://embeddings:80  # или LM Studio
memory_mcp check && memory_mcp index
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
mcp_client.call_tool("batch_fetch_records", {"record_ids": ["msg-001", "msg-002"]})
```

## Документация

- Обновляйте `README.md` и `CHANGELOG.md` при изменении API/CLI.
- В `AGENTS.md` фиксируйте архитектуру, правила оформления инструментов и конфигурацию.
- Для новых источников добавляйте секции в `indexing/*` + документацию по формату входных данных.
