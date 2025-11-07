# Архитектура

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

- Фабрика [`executor/factory.py`](../executor/factory.py) выбирает исполнителя: Docker (по умолчанию), локальный `SubprocessCodeExecutor` и новый `MCPCodeExecutor`.
- [`executor/mcp_executor.py`](../executor/mcp_executor.py) подключает Model Context Protocol сервер (`@pydantic/mcp-run-python`) через stdio или SSE и транслирует результаты в формат `ExecutionResult`.
- Конфигурация управляется блоком `executor.mcp` в [`config/settings.yaml`](../config/settings.yaml); можно задать команду `deno`, URL SSE и дополнительные аргументы для tools.
