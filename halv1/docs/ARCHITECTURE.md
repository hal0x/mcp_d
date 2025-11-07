# Архитектура HALv1

**HALv1** — это автономный AI агент для Telegram, являющийся частью экосистемы MCP (Model Context Protocol). Агент интегрируется с различными MCP серверами для выполнения сложных задач: анализа рынка, торговли, выполнения кода и управления памятью.

## Обзор архитектуры

HALv1 построен на модульной архитектуре с четким разделением ответственности:

```
HALv1 Core
├── Agent Core (планирование и выполнение задач)
├── Event Bus (координация компонентов)
├── Memory System (L0-L∞ память)
├── LLM Integration (Ollama, LM Studio)
├── Executor (Docker, MCP)
└── MCP Integration Layer (взаимодействие с MCP серверами)
    ├── Memory MCP
    ├── TradingView MCP
    ├── Binance MCP
    ├── Backtesting MCP
    ├── Shell MCP
    ├── Supervisor MCP
    ├── Learning MCP
    ├── Policy MCP
    └── Orchestrator MCP
```

## Интеграция с MCP экосистемой

HALv1 использует **MCP Integration Layer** для унифицированного взаимодействия с MCP серверами:

- **JSON Schema Registry** - нормализация request/response
- **Unified RPC** - retry, backoff, circuit-breaker
- **Health Aggregator** - мониторинг состояния всех MCP
- **Классы надежности** - critical/standard/optional

Подробнее: [memory/MCP_INTEGRATION.md](memory/MCP_INTEGRATION.md) и [MCP_ECOSYSTEM_GUIDELINES.md](../../MCP_ECOSYSTEM_GUIDELINES.md)

## Память L0–L∞

| Уровень | Компонент | Индексы | Связи |
|---------|-----------|---------|-------|
| **L0** | `EpisodicBuffer` — оперативная буферизация событий | нет | превышение TTL/размера → флеш в L1 |
| **L1** | `EpisodeGraph` + `WritePipeline` | SQLite c FTS5 и HNSW, таблицы `items`/`edges` | получает события из L0, узлы связаны рёбрами `time`, `entity`, `semantic` |
| **L2** | `MemoryStore` (`short_term`/`long_term`) + `schemas` | списки с эмбеддингами, JSON‑пersistence | кластеризация и сводки через `schemas`, суммаризация `short_term` → `long_term`, поиск по фильтрации и косинусному сходству; удалённые записи → L∞ |
| **L∞** | `MemoryArchive` | JSONL для текста, FAISS OPQ+PQ для эмбеддингов | архивирует записи из L2 и может восстановить их обратно |

Краткосрочная память (часть L2) хранит последние события в оперативной памяти.
Долгосрочная память сохраняет суммаризованные записи на диск.

При превышении `short_term_limit` содержимое краткосрочной памяти
суммаризируется и переносится в долговременную. Функции `recall`,
`search` и `semantic_search` принимают флаг `long_term` для выбора
уровня.

Вызов `MemoryService.consolidate()` очищает буфер L0 и передаёт его содержимое в `schemas.consolidate_graph`, который группирует недавние эпизоды и формирует для них сводки.

```python
from memory.service import MemoryService

svc = MemoryService("memory.db")
svc.write_event("buy milk")
svc.consolidate()
```


## Система промтов

Модули промтов управляются через файл конфигурации
[`config/prompts.yaml`](../config/prompts.yaml).

- **PromptManager** загружает и шаблонизирует промты.
- **ModuleCoordinator** координирует модули `events`, `themes`,
  `emotions` и прокидывает их контекст в агента.
- Инструменты (`search`, `code`, `planning`) используют общие
  шаблоны из [`llm/prompts.py`](../llm/prompts.py).

Такая схема позволяет централизованно обновлять промты и подключать
новые модули без изменения ядра.

## Исполнение кода

- **Фабрика исполнителей** [`executor/factory.py`](../executor/factory.py) выбирает исполнителя:
  - `DockerExecutor` (по умолчанию) - изолированное выполнение в Docker контейнерах
  - `MCPCodeExecutor` - выполнение через MCP сервер (`@pydantic/mcp-run-python`)
  - `SubprocessCodeExecutor` - локальное выполнение (для тестов)
- **MCP Executor** [`executor/mcp_executor.py`](../executor/mcp_executor.py) подключает Model Context Protocol сервер через stdio или SSE и транслирует результаты в формат `ExecutionResult`.
- Конфигурация управляется блоком `executor.mcp` в [`config/settings.yaml`](../config/settings.yaml); можно задать команду `deno`, URL SSE и дополнительные аргументы для tools.

## Event Bus и координация

**AsyncEventBus** [`services/event_bus.py`](../services/event_bus.py) обеспечивает:
- Централизованную систему событий для координации компонентов
- Асинхронную публикацию и подписку на события
- Изоляцию компонентов через событийную архитектуру
- Поддержку типизированных событий (`MessageReceived`, `ReplyReady`, и др.)

## Планирование задач

**LLMTaskPlanner** [`planner/task_planner.py`](../planner/task_planner.py) и **DAGExecutor** [`planner/dag_executor.py`](../planner/dag_executor.py):
- Разбивают сложные задачи на шаги
- Анализируют зависимости между шагами
- Выполняют шаги в правильном порядке
- Поддерживают параллельное выполнение независимых шагов

## Интеграция с MCP серверами

### Memory MCP
- Замена локального модуля памяти через `MCPMemoryAdapter`
- Семантический поиск и управление памятью
- Интеграция с графовой памятью HALv1

### TradingView MCP
- Анализ криптовалютного рынка
- Получение сигналов и индикаторов
- Интеграция с Decision MCP для принятия решений

### Binance MCP
- Исполнение торговых операций
- Управление портфелем
- Интеграция с Policy MCP для контроля рисков

### Supervisor MCP
- Единый реестр всех MCP серверов
- Мониторинг здоровья и метрик
- Агрегация фактов для обучения

### Orchestrator MCP
- Координация задач между MCP серверами
- FSM для управления жизненным циклом задач
- Интеграция с Telegram для подтверждений

Подробнее о плане развития: [../../impl_plan/hal_ai_agent_plan.md](../../impl_plan/hal_ai_agent_plan.md)
