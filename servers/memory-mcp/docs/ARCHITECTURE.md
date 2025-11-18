# Архитектура Memory MCP Server

Документ описывает ключевые слои системы, основные компоненты и потоки данных проекта Memory MCP.
Он предназначен для разработчиков, которые хотят быстрее разобраться в устройстве
индексатора, поискового движка и аналитического стека.

## Слойная модель

| Слой | Назначение | Ключевые директории/файлы |
|------|------------|---------------------------|
| Внешние интерфейсы | CLI, MCP‑сервер (stdio/HTTP), REST API | `src/memory_mcp/cli/`, `scripts/`, `src/memory_mcp/mcp/server.py`, `run_server.py` |
| Оркестрация и ядро | Управление индексированием, доступ к LM Studio/Ollama и хранилищам | `src/memory_mcp/core/indexer.py`, `src/memory_mcp/core/ollama_client.py`, `src/memory_mcp/core/lmstudio_client.py`, `src/memory_mcp/core/indexing_tracker.py` |
| Аналитика и агрегация | Сегментация, саммаризация, контроль качества, кластеры, отчёты | `src/memory_mcp/analysis/` |
| Поиск и память | Гибридный/векторный поиск, Smart Search, граф знаний, долговременная память | `src/memory_mcp/search/`, `src/memory_mcp/memory/` |
| Индексация источников | Индексаторы для различных источников данных | `src/memory_mcp/indexing/` |
| Утилиты и конфигурация | Общие вспомогательные функции и конфиги | `src/memory_mcp/utils/`, `config/`, `config.json` |

## Потоки данных

```text
входные дампы → предварительная фильтрация → сегментация сессий →
саммаризация (LM Studio/Ollama) → сохранение Markdown отчётов →
индексация сообщений/сессий в SQLite (TypedGraphMemory) + Qdrant/ChromaDB →
генерация эмбеддингов → (опц.) кластеризация + граф знаний →
гибридный поиск (BM25 + Vector) → Smart Search → результаты через MCP/CLI
```

1. **Загрузка данных**: JSON/JSONL файлы из `./chats/` читаются командами CLI или MCP инструментом `index_chat` и передаются в `TwoLevelIndexer`.
2. **Сегментация**: `SessionSegmenter` и `DayGroupingSegmenter` формируют сессии и дневные группы на основе временных интервалов. Поддерживается умная стратегия группировки (`smart`) с автоматическим определением типа чата.
3. **Саммаризация и контроль качества**: `SessionSummarizer` вызывает LM Studio/Ollama через соответствующие клиенты, оценивает результат через систему анализа качества и при необходимости повторно уточняет сводку. Поддерживается расширенный контекст для малых сессий через `IncrementalContextManager`.
4. **Агрегация и кластеры**: `SmartRollingAggregator`, `SessionClusterer` и `ClusterSummarizer` группируют активность и строят тематические кластеры (HDBSCAN).
5. **Индексирование**: 
   - Сообщения, сессии, задачи и кластеры пишутся в ChromaDB (коллекции `telegram_messages`, `chat_sessions`, `chat_tasks`, `session_clusters`)
   - Типизированный граф знаний сохраняется в SQLite через `TypedGraphMemory` (5 типов узлов, 8 типов рёбер)
   - Векторные эмбеддинги сохраняются в Qdrant (опционально) или ChromaDB
6. **Отчёты и память**: Markdown‑отчёты сохраняются в `artifacts/reports/`, граф знаний строится через `memory/typed_graph.py`, результаты доступны через MCP инструменты, CLI команды и REST API.

## Основные компоненты

### CLI и внешние интерфейсы

- `src/memory_mcp/cli/main.py` — точка входа для команд `index`, `search`, `insight-graph`, `stats`, `indexing-progress`, `update-summaries`, `review-summaries`.
- `src/memory_mcp/mcp/server.py` — MCP сервер для stdio режима (интеграции с MCP клиентами через стандартный ввод/вывод).
- `run_server.py` — HTTP сервер для Docker/продакшена (FastAPI + FastApiMCP, доступен на порту 8050).
- Директория `scripts/` содержит вспомогательные утилиты (проверка окружения, генерация конфигурации MCP и др.).

**MCP инструменты:**
- Универсальные: `search`, `batch_operations`, `graph_query`, `background_indexing`, `summaries`, `ingest`
- Специализированные: `index_chat`, `get_available_chats`, `generate_embedding`, `get_statistics`, `search_explain`
- Торговые сигналы: `store_trading_signal`, `search_trading_patterns`, `get_signal_performance`
- Системные: `health`, `version`

### Ядро индексатора

- `TwoLevelIndexer` управляет всем циклом: от подготовки коллекций ChromaDB до сохранения результатов в файловую систему. Поддерживает инкрементальную индексацию через отслеживание прогресса.
- `IndexingJobTracker` отслеживает жизненный цикл индексаций, прогресс по чатам и предотвращает параллельные full-rebuild для одного чата.
- `OllamaClient` инкапсулирует HTTP вызовы Ollama для LLM операций, поддерживает длинные контексты, тайм‑ауты и ретраи.
- `LMStudioClient` инкапсулирует HTTP вызовы LM Studio Server для эмбеддингов и LLM операций.
- `EmbeddingService` предоставляет единый интерфейс для работы с эмбеддингами (поддерживает text-embeddings-inference, LM Studio, Ollama).
- Менеджер инструкций `InstructionManager` позволяет централизованно обновлять промпты для LLM операций.

### Аналитические сервисы

- **Сегментация**: `SessionSegmenter`, `DayGroupingSegmenter`, `SmartRollingAggregator`
  (динамический выбор стратегии NOW/FRESH/RECENT для потоковых чатов).
- **Саммаризация**: `SessionSummarizer`, `ClusterSummarizer`, `MarkdownRenderer`.
- **Качество**: `analysis/quality_evaluator.py` и `IterativeRefiner` для итеративного
  повышения оценок саммари.
- **Тематика и сущности**: `EntityExtractor`, `SessionClusterer`, `ClusterSummarizer`.
- **Инкрементальные сценарии**: `incremental_context_manager.py` поддерживает добор
  свежих сообщений без пересоздания всего индекса.
- **Контекст и фильтрация**: `context_manager.py`, `message_filter.py`,
  `instruction_manager.py`.

### Поиск и долговременная память

- `search/hybrid_search.py` — гибридный поиск, комбинирующий BM25 (вес 0.4) и векторный поиск (вес 0.6) с Reciprocal Rank Fusion для улучшенной точности.
- `search/smart_search.py` — интерактивный LLM-assisted поиск с сессиями, обратной связью и автоматическим уточнением запросов.
- `search/search_explainer.py` — объяснение результатов поиска с декомпозицией scores (BM25, Vector, RRF) и трассировкой связей через граф.
- Модуль `memory/`:
  - `typed_graph.py` — типизированный граф знаний (SQLite + NetworkX) с 5 типами узлов (Entity, Event, DocChunk, Topic, ToolCall) и 8 типами рёбер
  - `vector_store.py` — обёртка над Qdrant для векторного поиска (gracefully выключается, если Qdrant недоступен)
  - `importance_scoring.py` — оценка важности сообщений (0.0-1.0) и интеллектуальная очистка памяти (Memory Pruning)
  - `embeddings.py` — сервис эмбеддингов с поддержкой батчинга и различных провайдеров

## Хранилища и артефакты

| Артефакт | Расположение | Что содержит |
|----------|--------------|--------------|
| Исходные дампы | `./chats/` | Экспортированные сообщения Telegram (JSON/JSONL) |
| Векторные коллекции | `./chroma_db/` | Сообщения, сессии, задачи, кластеры (ChromaDB) |
| Граф знаний | `data/memory_graph.db` | Типизированный граф (SQLite + NetworkX) |
| Векторное хранилище | Qdrant (опционально) | Векторные эмбеддинги для поиска |
| Markdown‑саммари | `artifacts/reports/` | Итоговые отчёты по чатам и кластерам |
| Контекст и состояние | `artifacts/smart_aggregation_state/`, `artifacts/chat_contexts/`, `artifacts/now_summaries/` | JSON состояния окон, контексты чатов, саммари NOW окна |
| Прогресс индексации | `data/indexing_jobs.json` | Отслеживание прогресса индексации по чатам |
| Сессии поиска | `data/search_sessions.db` | Сессии Smart Search для обратной связи |
| Конфигурация | `config.json`, `config/` | Значения по умолчанию для CLI и аналитики, словари сущностей |

## Внешние зависимости

- **LM Studio** (приоритет 1) или **Ollama** (fallback) — источник эмбеддингов и генеративных саммаризаций.
- **text-embeddings-inference** (опционально) — альтернативный сервис эмбеддингов через `MEMORY_MCP_EMBEDDINGS_URL`.
- **ChromaDB** — персистентное векторное хранилище коллекций (сообщения, сессии, задачи, кластеры).
- **Qdrant** (опционально) — векторное хранилище для масштабируемого поиска через `MEMORY_MCP_QDRANT_URL`.
- **SQLite** — хранение типизированного графа знаний (`TypedGraphMemory`).
- **PyTorch / ML‑библиотеки** — задействуются Ollama и локальными LLM‑моделями.

## Расширение системы

1. Добавьте новую аналитику в `src/memory_mcp/analysis/`, экспортируйте класс в
   `__init__.py`, подключите его в `TwoLevelIndexer` или CLI.
2. Новые CLI‑команды регистрируйте в `src/memory_mcp/cli/main.py`, переиспользуя существующие
   сервисы из `core` и `analysis`.
3. Для дополнительных коллекций определите схему в индексаторе и обновите раздел
   «Хранилища» в документации.

