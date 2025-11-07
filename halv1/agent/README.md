# AgentCore

## Обработка событий
`AgentCore` подписывается на три канала шины: входящие сообщения, планы и результаты выполнения. Это позволяет реагировать на новые сообщения, запускать выполнение плана и сохранять полученные результаты.

При получении `MessageReceived` агент извлекает релевантный контекст из памяти и генерирует план действий. План публикуется обратно в шину в виде события `PlanGenerated`.

## Планирование и выполнение
Событие `PlanGenerated` запускает выполнение шага плана. Шаги исполняются через `run_plan`, который вызывает зарегистрированные инструменты и возвращает результаты для каждого шага.

После каждого шага агент публикует `ExecutionCompleted`, чтобы другие компоненты могли реагировать на результаты.

## Взаимодействие с executor, memory и retriever
`AgentCore` регистрирует набор встроенных инструментов, включая обработку кода, поиск, файловый ввод-вывод и HTTP-запросы. Для выполнения шагов используется `executor`, который запускает сгенерированный код и возвращает вывод программы.

Перед планированием агент запрашивает `MemoryStore`, выполняя семантический поиск по предыдущим результатам. После выполнения новые факты сохраняются в память, избегая дубликатов.
Сравнение выполняется без учета регистра: записи, отличающиеся только регистром, считаются дубликатами и не добавляются повторно.

Для доступа к внешнему знанию может быть подключён модуль `Retriever`, выполняющий семантический поиск по векторному индексу и возвращающий наиболее релевантные фрагменты текста.

## Структура памяти

Отдельного хранилища диалоговой истории нет: после перезапуска бот начинает новую сессию, а сохранившиеся факты извлекаются только из `MemoryStore`.
`MemoryStore` разделяет знания агента на два уровня:

- **Краткосрочная память** хранится только в оперативной памяти и очищается либо после завершения диалога, либо при превышении лимита. После переполнения она автоматически суммаризируется и переносится в долговременное хранилище.
- **Долговременная память** сохраняется на диск в виде JSON-файла и переживает перезапуски. Каждая запись представляет собой объект с полями `text` и `embedding`:

```json
{
  "short_term": [{"text": "привет", "embedding": [0.1, 0.2, ...]}],
  "long_term": [{"text": "имя пользователя — Алексей", "embedding": [0.3, 0.4, ...]}]
}
```

При поиске контекста агент сначала обращается к `MemoryStore`, а затем использует `Retriever` и `VectorIndex`, которые позволяют искать по более широкому индексу сообщений и документов.

Пустой запрос в `search` возвращает все записи, тогда как `semantic_search` с пустой строкой возвращает пустой список. Это сделано намеренно, чтобы пустой семантический запрос не возвращал случайные данные. Если требуется получить все элементы, используйте `search("")`.

### Пример использования
```python
from memory import MemoryStore
from index.vector_index import VectorIndex

memory = MemoryStore(long_term_path="mem.json")
index = VectorIndex()

# добавление записей
memory.remember("поздоровался с пользователем")
memory.remember("имя пользователя — Алексей", long_term=True)

# индексация долговременных воспоминаний
for i, text in enumerate(memory.recall(long_term=True)):
    await index.add(f"mem-{i}", text, metadata={})

# извлечение
recent = memory.recall()
archive = memory.recall(long_term=True)
similar = memory.semantic_search("как зовут", long_term=True)
no_matches = memory.semantic_search("")  # пустой запрос ничего не возвращает
all_items = memory.search("")  # так можно получить все записи
```

## Пример конфигурации
```python
from services.event_bus import AsyncEventBus
from agent.core import AgentCore
from planner.task_planner import SimpleTaskPlanner
from executor import create_executor
from internet.search_client import SearchClient
from memory import MemoryStore
from index.vector_index import VectorIndex
from index.cluster_manager import ClusterManager
from retriever.retriever import Retriever

bus = AsyncEventBus()
planner = SimpleTaskPlanner()
executor = create_executor("docker")
search = SearchClient()
memory = MemoryStore()
retriever = Retriever(VectorIndex(), ClusterManager())
core = AgentCore(bus, planner, executor, search, memory, code_generator=lambda x: x)
```

## Типичный сценарий диалога
1. Пользователь отправляет запрос в бот.
2. `AgentCore` извлекает контекст из `memory` и при необходимости обращается к `retriever` за дополнительными данными.
3. Планировщик формирует последовательность шагов.
4. `executor` исполняет каждый шаг и возвращает результаты.
5. Результаты сохраняются в `memory`, а пользователю отправляется ответ.

Обновлено: 2025-08-22.
