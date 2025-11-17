# Комплексный анализ недостающих функций Memory MCP

**Дата анализа:** 2025-01-XX  
**Версия системы:** 2.0.0  
**Статус:** Завершён

## Резюме

Проведён комплексный анализ кодовой базы Memory MCP сервера для выявления пробелов в функциональности. Обнаружено:

- **34 реализованных MCP инструмента** (все из первоначального плана)
- **15+ неэкспонированных компонентов** (реализованы в коде, но не доступны через MCP API)
- **30+ отсутствующих функций** (необходимы для production-ready решения)

**Ключевые выводы:**
1. Система оценки важности (Importance Scoring) полностью реализована, но не экспонирована
2. Множество CLI команд не имеют MCP-эквивалентов
3. Отсутствуют критичные функции управления данными (backup, restore, optimize)
4. Нет расширенных операций с графом знаний
5. Отсутствуют функции мониторинга и аналитики

---

## 1. Неэкспонированные компоненты

### 1.1. Система оценки важности (Importance Scoring)

**Файлы:** `src/memory_mcp/memory/importance_scoring.py`

**Статус:** Полностью реализовано, но не экспонировано через MCP

#### Компоненты:

1. **`ImportanceScorer`** (строки 19-152)
   - Метод `compute_importance()` - вычисление важности записи (0.0-1.0)
   - Факторы: сущности, задачи, длина, частота поиска, специальные паттерны
   - Настраиваемые веса для разных факторов

2. **`EvictionScorer`** (строки 154-323)
   - Метод `compute_eviction_score()` - приоритет удаления (выше = быстрее удалить)
   - Формула: `(1 - importance) * time_decay * (1 - usage_freq)`
   - Метод `get_eviction_candidates()` - получение кандидатов на удаление

3. **`MemoryPruner`** (строки 326-428)
   - Метод `should_prune()` - проверка необходимости очистки
   - Метод `prune_messages()` - автоматическая очистка неважных записей
   - Настраиваемые пороги и лимиты

#### Рекомендуемые MCP инструменты:

**1. `calculate_importance`**
```python
# Схема запроса
class CalculateImportanceRequest(BaseModel):
    record_id: str
    # Опционально: переопределить веса
    entity_weight: Optional[float] = None
    task_weight: Optional[float] = None
    length_weight: Optional[float] = None
    search_hits_weight: Optional[float] = None

# Схема ответа
class CalculateImportanceResponse(BaseModel):
    record_id: str
    importance_score: float  # 0.0-1.0
    factors: Dict[str, float]  # Декомпозиция по факторам
```

**2. `get_low_importance_records`**
```python
class GetLowImportanceRecordsRequest(BaseModel):
    threshold: float = 0.3  # Порог важности
    limit: int = 100
    source: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None

class GetLowImportanceRecordsResponse(BaseModel):
    records: List[Dict[str, Any]]  # record_id, importance_score, content_preview
    total_found: int
```

**3. `prune_memory`**
```python
class PruneMemoryRequest(BaseModel):
    max_records: int = 100000  # Максимальное количество записей
    eviction_threshold: float = 0.7  # Порог для удаления
    dry_run: bool = False  # Только анализ, без удаления
    source: Optional[str] = None  # Фильтр по источнику

class PruneMemoryResponse(BaseModel):
    pruned: bool
    removed_count: int
    removed_ids: List[str]
    remaining_count: int
    candidates: List[Dict[str, Any]]  # Для архивации
```

**4. `update_importance_scores`**
```python
class UpdateImportanceScoresRequest(BaseModel):
    source: Optional[str] = None  # Обновить только для источника
    batch_size: int = 1000  # Размер батча для обработки

class UpdateImportanceScoresResponse(BaseModel):
    records_updated: int
    average_importance: float
    min_importance: float
    max_importance: float
```

**Оценка сложности:** Средняя (2-3 дня)  
**Приоритет:** Высокий

---

### 1.2. Построитель графа знаний (Graph Builder)

**Файлы:** `src/memory_mcp/memory/graph_builder.py`

**Статус:** Реализовано, частично используется в индексации, но не экспонировано

#### Компоненты:

1. **`GraphBuilder.build_from_sessions()`** (строки 42-76)
   - Построение графа из списка сессий
   - Создание узлов: Event, Entity, DocChunk, Topic
   - Создание рёбер: MENTIONS, PART_OF, HAS_TOPIC

2. **`GraphBuilder.add_temporal_edges()`** (строки 278-334)
   - Добавление временных связей между событиями
   - Связывание событий в временном окне

3. **`GraphBuilder.add_semantic_edges()`** (строки 336-381)
   - Добавление семантических связей на основе эмбеддингов
   - Использует косинусное сходство

#### Рекомендуемые MCP инструменты:

**1. `build_graph_from_sessions`**
```python
class BuildGraphFromSessionsRequest(BaseModel):
    sessions: List[Dict[str, Any]]  # Данные сессий
    create_temporal_edges: bool = True
    create_semantic_edges: bool = False
    time_window_minutes: int = 60
    similarity_threshold: float = 0.8

class BuildGraphFromSessionsResponse(BaseModel):
    stats: Dict[str, int]  # sessions_processed, entities_created, etc.
    nodes_created: int
    edges_created: int
```

**2. `extract_entities_from_text`**
```python
class ExtractEntitiesFromTextRequest(BaseModel):
    text: str
    entity_types: Optional[List[str]] = None  # Фильтр по типам

class ExtractEntitiesFromTextResponse(BaseModel):
    entities: List[Dict[str, Any]]  # name, type, confidence
    total_entities: int
```

**3. `create_graph_edges`**
```python
class CreateGraphEdgesRequest(BaseModel):
    edges: List[Dict[str, Any]]  # source_id, target_id, type, weight, properties

class CreateGraphEdgesResponse(BaseModel):
    edges_created: int
    edges_failed: int
    errors: List[str]
```

**Оценка сложности:** Средняя (2-3 дня)  
**Приоритет:** Средний

---

### 1.3. Анализатор качества (Quality Analyzer)

**Файлы:** `src/memory_mcp/quality_analyzer/quality_analyzer.py`

**Статус:** Полностью реализовано, используется в CLI, но не экспонировано

#### Компоненты:

1. **`QualityAnalyzer.analyze_chat_quality()`** (строки 124-236)
   - Анализ качества индексации для чата
   - Генерация тестовых запросов
   - Расчёт метрик качества (precision, recall, MRR, etc.)

2. **`QualityAnalyzer.analyze_multiple_chats()`** (строка 271)
   - Анализ качества для нескольких чатов

3. **Компоненты:**
   - `QueryGenerator` - генерация тестовых запросов
   - `RelevanceAnalyzer` - анализ релевантности через LLM
   - `MetricsCalculator` - расчёт метрик
   - `ReportGenerator` - создание отчётов

#### Рекомендуемые MCP инструменты:

**1. `analyze_indexing_quality`**
```python
class AnalyzeIndexingQualityRequest(BaseModel):
    chat: str
    test_queries: Optional[List[Dict[str, Any]]] = None
    batch_size: Optional[int] = None
    max_queries: Optional[int] = 20
    custom_queries: Optional[List[str]] = None

class AnalyzeIndexingQualityResponse(BaseModel):
    chat: str
    metrics: Dict[str, float]  # precision, recall, mrr, etc.
    total_queries: int
    results: List[Dict[str, Any]]  # Детальные результаты
    recommendations: List[str]  # Рекомендации по улучшению
```

**2. `generate_test_queries`**
```python
class GenerateTestQueriesRequest(BaseModel):
    chat: str
    count: int = 20
    query_types: Optional[List[str]] = None  # factual, exploratory, etc.

class GenerateTestQueriesResponse(BaseModel):
    queries: List[Dict[str, Any]]  # query, type, expected_results
    total_generated: int
```

**3. `get_quality_metrics`**
```python
class GetQualityMetricsRequest(BaseModel):
    chat: Optional[str] = None  # Если None - для всех чатов
    date_from: Optional[datetime] = None

class GetQualityMetricsResponse(BaseModel):
    metrics: Dict[str, Any]  # Агрегированные метрики
    chat_metrics: Dict[str, Dict[str, float]]  # По чатам
    trends: Dict[str, List[float]]  # Тренды во времени
```

**Оценка сложности:** Высокая (4-5 дней)  
**Приоритет:** Средний

---

### 1.4. Другие неэкспонированные компоненты

#### MessageExtractor
**Файлы:** `src/memory_mcp/utils/message_extractor.py`  
**CLI команда:** `extract-messages`

**Функциональность:**
- Извлечение новых сообщений из input в chats
- Фильтрация по дате
- Дедупликация

**Рекомендуемый MCP инструмент:** `extract_messages`

#### MessageDeduplicator
**Файлы:** `src/memory_mcp/cli/main.py` (строки 39-151)  
**CLI команда:** `deduplicate`

**Функциональность:**
- Удаление дубликатов сообщений по ID
- Дедупликация по хешу контента

**Рекомендуемый MCP инструмент:** `deduplicate_records`

#### InstructionManager
**Файлы:** `src/memory_mcp/analysis/instruction_manager.py`  
**CLI команды:** `set-instruction`, `list-instructions`

**Функциональность:**
- Управление инструкциями саммаризации
- Индивидуальные инструкции для чатов
- Общие инструкции для типов чатов

**Рекомендуемые MCP инструменты:**
- `set_summarization_instruction`
- `get_summarization_instructions`
- `clear_summarization_instruction`

#### TimeProcessor
**Файлы:** `src/memory_mcp/analysis/time_processor.py`

**Функциональность:**
- Анализ временных паттернов активности
- Метод `analyze_activity_patterns()`

**Рекомендуемый MCP инструмент:** `analyze_activity_patterns`

---

## 2. Отсутствующие базовые функции

### 2.1. Массовые операции

**Текущее состояние:** Есть `batch_update_records`, отсутствуют другие batch операции

#### 2.1.1. `batch_delete_records`

**Описание:** Массовое удаление записей с транзакционной безопасностью

**Схема:**
```python
class BatchDeleteRecordsRequest(BaseModel):
    record_ids: List[str]
    confirm: bool = False  # Подтверждение для больших батчей

class BatchDeleteResult(BaseModel):
    record_id: str
    deleted: bool
    message: Optional[str] = None

class BatchDeleteRecordsResponse(BaseModel):
    results: List[BatchDeleteResult]
    total_deleted: int
    total_failed: int
```

**Оценка сложности:** Низкая (1 день)  
**Приоритет:** Высокий

#### 2.1.2. `batch_fetch_records`

**Описание:** Получение множества записей за один запрос (оптимизация)

**Схема:**
```python
class BatchFetchRecordsRequest(BaseModel):
    record_ids: List[str]
    include_embeddings: bool = False

class BatchFetchRecordsResponse(BaseModel):
    records: List[MemoryRecordPayload]
    not_found: List[str]  # ID записей, которые не найдены
    total_found: int
```

**Оценка сложности:** Низкая (1 день)  
**Приоритет:** Средний

---

### 2.2. Управление данными

#### 2.2.1. `backup_database`

**Описание:** Создание резервной копии SQLite БД и ChromaDB

**Схема:**
```python
class BackupDatabaseRequest(BaseModel):
    backup_path: Optional[str] = None  # Путь для сохранения (по умолчанию: backups/)
    include_chromadb: bool = True
    include_reports: bool = False
    compress: bool = True  # Создать .tar.gz архив

class BackupDatabaseResponse(BaseModel):
    backup_path: str
    backup_size_bytes: int
    backup_created_at: datetime
    includes: List[str]  # Что включено в backup
```

**Реализация:**
- Копирование SQLite файла
- Копирование ChromaDB директории (если включено)
- Создание архива (если compress=True)
- Валидация backup

**Оценка сложности:** Средняя (2 дня)  
**Приоритет:** Высокий

#### 2.2.2. `restore_database`

**Описание:** Восстановление из резервной копии

**Схема:**
```python
class RestoreDatabaseRequest(BaseModel):
    backup_path: str
    confirm: bool = False  # Подтверждение (удалит текущие данные)
    restore_chromadb: bool = True
    restore_reports: bool = False

class RestoreDatabaseResponse(BaseModel):
    restored: bool
    backup_date: datetime
    records_restored: int
    message: str
```

**Оценка сложности:** Средняя (2 дня)  
**Приоритет:** Высокий

#### 2.2.3. `optimize_database`

**Описание:** Оптимизация SQLite БД (VACUUM, ANALYZE, REINDEX)

**Схема:**
```python
class OptimizeDatabaseRequest(BaseModel):
    vacuum: bool = True  # VACUUM для освобождения места
    analyze: bool = True  # ANALYZE для обновления статистики
    reindex: bool = False  # REINDEX для пересоздания индексов
    optimize_fts: bool = True  # Оптимизация FTS5 индекса

class OptimizeDatabaseResponse(BaseModel):
    optimized: bool
    operations_performed: List[str]
    size_before_bytes: int
    size_after_bytes: int
    space_freed_bytes: int
    duration_seconds: float
```

**Оценка сложности:** Низкая (1 день)  
**Приоритет:** Высокий

#### 2.2.4. `validate_database`

**Описание:** Проверка целостности данных

**Схема:**
```python
class ValidateDatabaseRequest(BaseModel):
    check_integrity: bool = True  # PRAGMA integrity_check
    check_foreign_keys: bool = True  # PRAGMA foreign_key_check
    check_orphaned_nodes: bool = True  # Узлы без связей
    check_orphaned_edges: bool = True  # Рёбра с несуществующими узлами

class ValidationIssue(BaseModel):
    type: str  # integrity, foreign_key, orphaned_node, etc.
    severity: str  # error, warning
    message: str
    details: Dict[str, Any]

class ValidateDatabaseResponse(BaseModel):
    valid: bool
    issues: List[ValidationIssue]
    total_issues: int
    checks_performed: List[str]
```

**Оценка сложности:** Средняя (2 дня)  
**Приоритет:** Средний

#### 2.2.5. `migrate_schema`

**Описание:** Миграция схемы БД при обновлениях

**Схема:**
```python
class MigrateSchemaRequest(BaseModel):
    target_version: Optional[str] = None  # Версия схемы (auto-detect если None)
    dry_run: bool = False

class MigrationStep(BaseModel):
    step: str
    description: str
    status: str  # pending, completed, failed
    error: Optional[str] = None

class MigrateSchemaResponse(BaseModel):
    migrated: bool
    current_version: str
    target_version: str
    steps: List[MigrationStep]
    duration_seconds: float
```

**Оценка сложности:** Высокая (5-7 дней)  
**Приоритет:** Низкий (для будущих версий)

---

### 2.3. Работа с графом

**Текущее состояние:** Есть базовые операции (`get_graph_neighbors`, `find_graph_path`), нет расширенных

#### 2.3.1. `add_graph_edge`

**Описание:** Добавление ребра в граф

**Схема:**
```python
class AddGraphEdgeRequest(BaseModel):
    source_id: str
    target_id: str
    edge_type: str  # EdgeType enum
    weight: float = 1.0
    properties: Dict[str, Any] = {}
    bidirectional: bool = False

class AddGraphEdgeResponse(BaseModel):
    edge_id: str
    added: bool
    message: str
```

**Оценка сложности:** Низкая (0.5 дня)  
**Приоритет:** Средний

#### 2.3.2. `delete_graph_edge`

**Описание:** Удаление ребра из графа

**Схема:**
```python
class DeleteGraphEdgeRequest(BaseModel):
    source_id: str
    target_id: str
    # Или по edge_id, если будет добавлен

class DeleteGraphEdgeResponse(BaseModel):
    deleted: bool
    message: str
```

**Оценка сложности:** Низкая (0.5 дня)  
**Приоритет:** Средний

#### 2.3.3. `get_graph_components`

**Описание:** Получение компонент связности графа

**Схема:**
```python
class GetGraphComponentsRequest(BaseModel):
    min_size: int = 2  # Минимальный размер компоненты
    node_type: Optional[str] = None  # Фильтр по типу узлов

class GraphComponent(BaseModel):
    component_id: int
    nodes: List[str]
    size: int
    node_types: Dict[str, int]

class GetGraphComponentsResponse(BaseModel):
    components: List[GraphComponent]
    total_components: int
    largest_component_size: int
    isolated_nodes: int  # Узлы без связей
```

**Оценка сложности:** Средняя (1-2 дня)  
**Приоритет:** Низкий

#### 2.3.4. `find_cycles`

**Описание:** Поиск циклов в графе

**Схема:**
```python
class FindCyclesRequest(BaseModel):
    max_length: int = 10
    node_type: Optional[str] = None

class Cycle(BaseModel):
    nodes: List[str]
    length: int
    edge_types: List[str]

class FindCyclesResponse(BaseModel):
    cycles: List[Cycle]
    total_cycles: int
```

**Оценка сложности:** Средняя (1-2 дня)  
**Приоритет:** Низкий

#### 2.3.5. `get_node_centrality`

**Описание:** Вычисление центральности узла (degree, betweenness, closeness)

**Схема:**
```python
class GetNodeCentralityRequest(BaseModel):
    node_id: str
    centrality_type: str = "degree"  # degree, betweenness, closeness

class GetNodeCentralityResponse(BaseModel):
    node_id: str
    centrality_type: str
    centrality_score: float
    rank: Optional[int] = None  # Ранг среди всех узлов
    percentile: Optional[float] = None
```

**Оценка сложности:** Средняя (2 дня)  
**Приоритет:** Низкий

#### 2.3.6. `merge_nodes`

**Описание:** Объединение узлов (например, при дедупликации сущностей)

**Схема:**
```python
class MergeNodesRequest(BaseModel):
    source_node_id: str  # Узел, который будет удалён
    target_node_id: str  # Узел, в который будет объединён
    merge_properties: bool = True  # Объединить properties
    merge_edges: bool = True  # Перенаправить рёбра

class MergeNodesResponse(BaseModel):
    merged: bool
    target_node_id: str
    edges_redirected: int
    properties_merged: bool
    message: str
```

**Оценка сложности:** Высокая (3-4 дня)  
**Приоритет:** Низкий

---

## 3. Отсутствующие продвинутые функции

### 3.1. Производительность и кэширование

#### 3.1.1. `get_performance_metrics`

**Описание:** Метрики производительности инструментов

**Схема:**
```python
class GetPerformanceMetricsRequest(BaseModel):
    tool_name: Optional[str] = None  # Если None - для всех инструментов
    time_window_hours: int = 24

class ToolMetrics(BaseModel):
    tool_name: str
    call_count: int
    avg_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    error_rate: float
    success_rate: float

class GetPerformanceMetricsResponse(BaseModel):
    metrics: List[ToolMetrics]
    total_calls: int
    time_window_hours: int
```

**Оценка сложности:** Средняя (2-3 дня) - требует добавления метрик  
**Приоритет:** Средний

#### 3.1.2. `get_query_statistics`

**Описание:** Статистика по поисковым запросам

**Схема:**
```python
class GetQueryStatisticsRequest(BaseModel):
    time_window_hours: int = 24
    top_n: int = 20

class QueryStat(BaseModel):
    query: str
    count: int
    avg_results: float
    avg_duration_ms: float
    success_rate: float

class GetQueryStatisticsResponse(BaseModel):
    top_queries: List[QueryStat]
    total_queries: int
    unique_queries: int
    avg_results_per_query: float
```

**Оценка сложности:** Средняя (2 дня) - требует логирования запросов  
**Приоритет:** Низкий

#### 3.1.3. `optimize_search_index`

**Описание:** Оптимизация поисковых индексов (FTS5, векторные)

**Схема:**
```python
class OptimizeSearchIndexRequest(BaseModel):
    optimize_fts: bool = True
    optimize_vector: bool = True
    rebuild_fts: bool = False  # Полная пересборка FTS индекса

class OptimizeSearchIndexResponse(BaseModel):
    optimized: bool
    fts_optimized: bool
    vector_optimized: bool
    duration_seconds: float
    index_size_before: int
    index_size_after: int
```

**Оценка сложности:** Низкая (1 день)  
**Приоритет:** Средний

---

### 3.2. Мониторинг и алерты

#### 3.2.1. `get_system_health`

**Описание:** Расширенная проверка здоровья системы

**Схема:**
```python
class SystemHealthCheck(BaseModel):
    component: str  # database, vector_store, embeddings, etc.
    status: str  # healthy, degraded, down
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    details: Dict[str, Any] = {}

class GetSystemHealthResponse(BaseModel):
    overall_status: str
    checks: List[SystemHealthCheck]
    timestamp: datetime
    uptime_seconds: float
```

**Оценка сложности:** Средняя (2 дня)  
**Приоритет:** Средний

#### 3.2.2. `get_resource_usage`

**Описание:** Использование ресурсов (память, диск, CPU)

**Схема:**
```python
class ResourceUsage(BaseModel):
    memory_mb: float
    disk_usage_mb: float
    cpu_percent: Optional[float] = None
    database_size_mb: float
    vector_store_size_mb: float

class GetResourceUsageResponse(BaseModel):
    usage: ResourceUsage
    limits: Optional[Dict[str, float]] = None  # Если заданы лимиты
    warnings: List[str] = []  # Предупреждения о превышении
```

**Оценка сложности:** Низкая (1 день)  
**Приоритет:** Низкий

---

### 3.3. Интеграции

#### 3.3.1. `register_webhook`

**Описание:** Регистрация webhook для событий

**Схема:**
```python
class RegisterWebhookRequest(BaseModel):
    url: str
    events: List[str]  # indexing_complete, record_ingested, etc.
    secret: Optional[str] = None  # Для подписи запросов
    enabled: bool = True

class RegisterWebhookResponse(BaseModel):
    webhook_id: str
    registered: bool
    url: str
    events: List[str]
```

**Оценка сложности:** Высокая (4-5 дней)  
**Приоритет:** Низкий

#### 3.3.2. `export_to_format`

**Описание:** Экспорт в дополнительные форматы

**Схема:**
```python
class ExportToFormatRequest(BaseModel):
    format: str  # xml, yaml, parquet, excel
    source: Optional[str] = None
    tags: List[str] = []
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = 1000

class ExportToFormatResponse(BaseModel):
    format: str
    content: str  # Или base64 для бинарных форматов
    records_count: int
    file_size_bytes: int
```

**Оценка сложности:** Средняя (2-3 дня)  
**Приоритет:** Низкий

---

## 4. Безопасность и валидация

### 4.1. Валидация данных

#### 4.1.1. `validate_record`

**Описание:** Валидация записи перед инжестом

**Схема:**
```python
class ValidateRecordRequest(BaseModel):
    record: MemoryRecordPayload

class ValidationError(BaseModel):
    field: str
    error: str
    severity: str  # error, warning

class ValidateRecordResponse(BaseModel):
    valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    sanitized_record: Optional[MemoryRecordPayload] = None
```

**Оценка сложности:** Средняя (2 дня)  
**Приоритет:** Средний

#### 4.1.2. `sanitize_content`

**Описание:** Очистка контента от вредоносных данных

**Схема:**
```python
class SanitizeContentRequest(BaseModel):
    content: str
    remove_html: bool = True
    remove_scripts: bool = True
    max_length: Optional[int] = None

class SanitizeContentResponse(BaseModel):
    sanitized_content: str
    removed_elements: List[str]  # Что было удалено
    original_length: int
    sanitized_length: int
```

**Оценка сложности:** Низкая (1 день)  
**Приоритет:** Низкий

---

## 5. Удобство использования

### 5.1. Поиск и фильтрация

#### 5.1.1. `search_by_author`

**Описание:** Поиск по автору

**Схема:**
```python
class SearchByAuthorRequest(BaseModel):
    author: str
    query: Optional[str] = None  # Дополнительный текстовый запрос
    top_k: int = 10
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None

class SearchByAuthorResponse(BaseModel):
    results: List[SearchResultItem]
    total_matches: int
    author: str
```

**Оценка сложности:** Низкая (0.5 дня)  
**Приоритет:** Низкий

#### 5.1.2. `fuzzy_search`

**Описание:** Нечёткий поиск (для опечаток)

**Схема:**
```python
class FuzzySearchRequest(BaseModel):
    query: str
    threshold: float = 0.7  # Порог схожести
    top_k: int = 10

class FuzzySearchResponse(BaseModel):
    results: List[SearchResultItem]
    suggestions: List[str]  # Предложения для исправления опечаток
```

**Оценка сложности:** Средняя (2 дня) - требует библиотеки fuzzy matching  
**Приоритет:** Низкий

#### 5.1.3. `search_suggestions`

**Описание:** Предложения для поиска (autocomplete)

**Схема:**
```python
class SearchSuggestionsRequest(BaseModel):
    prefix: str
    limit: int = 10

class SearchSuggestionsResponse(BaseModel):
    suggestions: List[str]
    total_found: int
```

**Оценка сложности:** Средняя (2 дня) - требует индексации популярных запросов  
**Приоритет:** Низкий

---

### 5.2. Аналитика

#### 5.2.1. `get_usage_statistics`

**Описание:** Статистика использования инструментов

**Схема:**
```python
class GetUsageStatisticsRequest(BaseModel):
    time_window_hours: int = 24

class ToolUsageStat(BaseModel):
    tool_name: str
    call_count: int
    unique_users: int  # Если будет добавлена аутентификация
    avg_duration_ms: float

class GetUsageStatisticsResponse(BaseModel):
    tools: List[ToolUsageStat]
    total_calls: int
    most_used_tool: str
    least_used_tool: str
```

**Оценка сложности:** Средняя (2 дня) - требует логирования  
**Приоритет:** Низкий

#### 5.2.2. `get_data_growth`

**Описание:** Рост объёма данных во времени

**Схема:**
```python
class GetDataGrowthRequest(BaseModel):
    time_window_days: int = 30
    granularity: str = "day"  # hour, day, week, month

class DataPoint(BaseModel):
    timestamp: datetime
    records_count: int
    database_size_mb: float
    vector_store_size_mb: float

class GetDataGrowthResponse(BaseModel):
    data_points: List[DataPoint]
    growth_rate_percent: float  # Средний рост в день
    projection_30_days: Dict[str, float]  # Прогноз на 30 дней
```

**Оценка сложности:** Высокая (3-4 дня) - требует исторических данных  
**Приоритет:** Низкий

#### 5.2.3. `get_top_searches`

**Описание:** Топ поисковых запросов

**Схема:**
```python
class GetTopSearchesRequest(BaseModel):
    limit: int = 20
    time_window_hours: int = 24

class TopSearch(BaseModel):
    query: str
    count: int
    avg_results: float
    success_rate: float

class GetTopSearchesResponse(BaseModel):
    searches: List[TopSearch]
    total_unique_queries: int
```

**Оценка сложности:** Средняя (2 дня) - требует логирования  
**Приоритет:** Низкий

#### 5.2.4. `get_entity_cooccurrence`

**Описание:** Совместное появление сущностей

**Схема:**
```python
class GetEntityCooccurrenceRequest(BaseModel):
    entity: Optional[str] = None  # Если None - все пары
    min_cooccurrence: int = 2
    limit: int = 50

class EntityPair(BaseModel):
    entity1: str
    entity2: str
    cooccurrence_count: int
    cooccurrence_rate: float  # Относительно общего количества

class GetEntityCooccurrenceResponse(BaseModel):
    pairs: List[EntityPair]
    total_pairs: int
```

**Оценка сложности:** Средняя (2-3 дня)  
**Приоритет:** Низкий

---

## 6. Приоритизация функций

### Критичность функций

#### Высокий приоритет (реализовать в первую очередь)

1. **`batch_delete_records`** - Массовое удаление записей
   - **Сложность:** Низкая (1 день)
   - **Польза:** Критична для управления данными
   - **Зависимости:** Нет

2. **`backup_database` / `restore_database`** - Резервное копирование
   - **Сложность:** Средняя (2 дня каждый)
   - **Польза:** Критична для production
   - **Зависимости:** Нет

3. **`optimize_database`** - Оптимизация БД
   - **Сложность:** Низкая (1 день)
   - **Польза:** Улучшение производительности
   - **Зависимости:** Нет

4. **`calculate_importance`** - Оценка важности записей
   - **Сложность:** Средняя (1 день)
   - **Польза:** Основа для управления памятью
   - **Зависимости:** Нет (компонент уже реализован)

5. **`prune_memory`** - Автоматическая очистка памяти
   - **Сложность:** Средняя (2 дня)
   - **Польза:** Управление размером БД
   - **Зависимости:** `calculate_importance`

#### Средний приоритет (реализовать во вторую очередь)

1. **`analyze_indexing_quality`** - Анализ качества
   - **Сложность:** Высокая (4-5 дней)
   - **Польза:** Улучшение качества индексации
   - **Зависимости:** Нет (компонент уже реализован)

2. **`get_performance_metrics`** - Метрики производительности
   - **Сложность:** Средняя (2-3 дня)
   - **Польза:** Мониторинг производительности
   - **Зависимости:** Требует добавления метрик

3. **`add_graph_edge` / `delete_graph_edge`** - Управление графом
   - **Сложность:** Низкая (0.5 дня каждый)
   - **Польза:** Расширенная работа с графом
   - **Зависимости:** Нет

4. **`validate_record`** - Валидация данных
   - **Сложность:** Средняя (2 дня)
   - **Польза:** Предотвращение ошибок
   - **Зависимости:** Нет

5. **`get_system_health`** - Расширенный health check
   - **Сложность:** Средняя (2 дня)
   - **Польза:** Мониторинг системы
   - **Зависимости:** Нет

6. **`get_low_importance_records`** - Записи с низкой важностью
   - **Сложность:** Средняя (1 день)
   - **Польза:** Поддержка `prune_memory`
   - **Зависимости:** `calculate_importance`

7. **`update_importance_scores`** - Пересчёт важности
   - **Сложность:** Средняя (2 дня)
   - **Польза:** Обновление оценок важности
   - **Зависимости:** `calculate_importance`

8. **`extract_messages`** - Извлечение сообщений
   - **Сложность:** Средняя (2 дня)
   - **Польза:** Интеграция с CLI функциональностью
   - **Зависимости:** Нет (компонент уже реализован)

9. **`deduplicate_records`** - Дедупликация записей
   - **Сложность:** Средняя (2 дня)
   - **Польза:** Очистка дубликатов
   - **Зависимости:** Нет (компонент уже реализован)

10. **`optimize_search_index`** - Оптимизация индексов
    - **Сложность:** Низкая (1 день)
    - **Польза:** Улучшение производительности поиска
    - **Зависимости:** Нет

#### Низкий приоритет (реализовать в последнюю очередь)

1. Webhooks и уведомления (`register_webhook`, `trigger_webhook`)
2. Расширенная аналитика (`get_data_growth`, `get_top_searches`, `get_entity_cooccurrence`)
3. Дополнительные форматы экспорта (`export_to_format`)
4. Расширенные операции с графом (`get_graph_components`, `find_cycles`, `get_node_centrality`, `merge_nodes`)
5. Rate limiting и audit logging
6. Fuzzy search и search suggestions
7. Миграция схемы БД

---

## 7. Оценка сложности реализации

### Методология оценки

- **Низкая (0.5-1 день):** Простые операции, использующие существующие методы
- **Средняя (2-3 дня):** Требуют новой логики, но используют существующие компоненты
- **Высокая (4-7 дней):** Требуют значительной новой логики или интеграций

### Сводная таблица

| Функция | Сложность | Приоритет | Оценка времени |
|---------|-----------|-----------|----------------|
| `batch_delete_records` | Низкая | Высокий | 1 день |
| `backup_database` | Средняя | Высокий | 2 дня |
| `restore_database` | Средняя | Высокий | 2 дня |
| `optimize_database` | Низкая | Высокий | 1 день |
| `calculate_importance` | Средняя | Высокий | 1 день |
| `prune_memory` | Средняя | Высокий | 2 дня |
| `analyze_indexing_quality` | Высокая | Средний | 4-5 дней |
| `get_performance_metrics` | Средняя | Средний | 2-3 дня |
| `add_graph_edge` | Низкая | Средний | 0.5 дня |
| `delete_graph_edge` | Низкая | Средний | 0.5 дня |
| `validate_record` | Средняя | Средний | 2 дня |
| `get_system_health` | Средняя | Средний | 2 дня |
| `get_low_importance_records` | Средняя | Средний | 1 день |
| `update_importance_scores` | Средняя | Средний | 2 дня |
| `extract_messages` | Средняя | Средний | 2 дня |
| `deduplicate_records` | Средняя | Средний | 2 дня |
| `optimize_search_index` | Низкая | Средний | 1 день |

**Итого для высокого приоритета:** ~9 дней  
**Итого для среднего приоритета:** ~20 дней  
**Итого для низкого приоритета:** ~30+ дней

---

## 8. Примеры API для новых инструментов

### 8.1. Пример: `calculate_importance`

```python
# Request
{
  "record_id": "msg-001",
  "entity_weight": 0.1,
  "task_weight": 0.3
}

# Response
{
  "record_id": "msg-001",
  "importance_score": 0.75,
  "factors": {
    "entities": 0.2,
    "task": 0.3,
    "length": 0.15,
    "search_hits": 0.1,
    "special_patterns": 0.0
  }
}
```

### 8.2. Пример: `backup_database`

```python
# Request
{
  "backup_path": "/backups/memory_backup_20250115",
  "include_chromadb": true,
  "compress": true
}

# Response
{
  "backup_path": "/backups/memory_backup_20250115.tar.gz",
  "backup_size_bytes": 524288000,
  "backup_created_at": "2025-01-15T10:30:00Z",
  "includes": [
    "sqlite_database",
    "chromadb",
    "reports"
  ]
}
```

### 8.3. Пример: `prune_memory`

```python
# Request
{
  "max_records": 50000,
  "eviction_threshold": 0.7,
  "dry_run": true
}

# Response
{
  "pruned": false,
  "removed_count": 0,
  "remaining_count": 75000,
  "candidates": [
    {
      "record_id": "msg-123",
      "eviction_score": 0.85,
      "importance_score": 0.1,
      "age_days": 365
    }
  ]
}
```

---

## 9. Рекомендации по архитектурным улучшениям

### 9.1. Система метрик

**Проблема:** Нет централизованной системы сбора метрик производительности

**Рекомендация:**
- Добавить декоратор для автоматического логирования вызовов инструментов
- Хранить метрики в отдельной таблице SQLite или временном хранилище
- Реализовать `get_performance_metrics` на основе этих данных

**Пример реализации:**
```python
@metrics_tracker
async def call_tool(name: str, arguments: Dict[str, Any]) -> ToolResponse:
    start_time = time.time()
    try:
        result = await _execute_tool(name, arguments)
        _record_metric(name, "success", time.time() - start_time)
        return result
    except Exception as e:
        _record_metric(name, "error", time.time() - start_time)
        raise
```

### 9.2. Система логирования запросов

**Проблема:** Нет истории поисковых запросов для аналитики

**Рекомендация:**
- Добавить таблицу `search_history` в SQLite
- Логировать все поисковые запросы с результатами
- Использовать для `get_top_searches`, `get_query_statistics`

### 9.3. Кэширование результатов

**Проблема:** Нет кэширования для часто запрашиваемых данных

**Рекомендация:**
- Добавить простой in-memory кэш для `get_statistics`, `get_tags_statistics`
- TTL для кэша (например, 5 минут)
- Инвалидация при изменении данных

### 9.4. Транзакционная безопасность

**Проблема:** Batch операции не всегда транзакционны

**Рекомендация:**
- Использовать SQLite транзакции для batch операций
- Rollback при ошибках
- Поддержка частичного успеха с детальными результатами

---

## 10. Итоговые выводы

### Статистика

- **Всего проанализировано компонентов:** 50+
- **Неэкспонированных компонентов:** 15+
- **Отсутствующих функций:** 30+
- **Рекомендуемых новых инструментов:** 25-30

### Ключевые пробелы

1. **Управление данными:** Отсутствуют критичные функции backup/restore/optimize
2. **Система важности:** Полностью реализована, но не доступна через MCP
3. **Массовые операции:** Неполный набор batch операций
4. **Мониторинг:** Нет метрик производительности и расширенного health check
5. **Граф знаний:** Базовые операции есть, но нет расширенных

### Рекомендуемый план реализации

**Фаза 1 (Высокий приоритет, ~2 недели):**
1. `batch_delete_records`
2. `backup_database` / `restore_database`
3. `optimize_database`
4. `calculate_importance`
5. `prune_memory`

**Фаза 2 (Средний приоритет, ~4 недели):**
1. `analyze_indexing_quality`
2. `get_performance_metrics`
3. `add_graph_edge` / `delete_graph_edge`
4. `validate_record`
5. `get_system_health`
6. `get_low_importance_records`
7. `update_importance_scores`
8. `extract_messages`
9. `deduplicate_records`
10. `optimize_search_index`

**Фаза 3 (Низкий приоритет, по необходимости):**
- Остальные функции из списка

---

## Приложение A: Сравнение CLI и MCP

### CLI команды без MCP эквивалентов

| CLI команда | MCP инструмент | Статус |
|-------------|----------------|--------|
| `extract-messages` | `extract_messages` | ❌ Отсутствует |
| `deduplicate` | `deduplicate_records` | ❌ Отсутствует |
| `set-instruction` | `set_summarization_instruction` | ❌ Отсутствует |
| `list-instructions` | `get_summarization_instructions` | ❌ Отсутствует |
| `rebuild-vector-db` | `rebuild_vector_database` | ❌ Отсутствует |
| `stop-indexing` | `stop_background_indexing` | ✅ Есть (частично) |

---

## Приложение B: Список всех рекомендуемых инструментов

### Высокий приоритет (5 инструментов)
1. `batch_delete_records`
2. `backup_database`
3. `restore_database`
4. `optimize_database`
5. `calculate_importance`
6. `prune_memory`

### Средний приоритет (15 инструментов)
1. `analyze_indexing_quality`
2. `generate_test_queries`
3. `get_quality_metrics`
4. `get_performance_metrics`
5. `add_graph_edge`
6. `delete_graph_edge`
7. `validate_record`
8. `get_system_health`
9. `get_low_importance_records`
10. `update_importance_scores`
11. `extract_messages`
12. `deduplicate_records`
13. `optimize_search_index`
14. `set_summarization_instruction`
15. `get_summarization_instructions`
16. `build_graph_from_sessions`
17. `batch_fetch_records`

### Низкий приоритет (10+ инструментов)
1. `get_graph_components`
2. `find_cycles`
3. `get_node_centrality`
4. `merge_nodes`
5. `get_resource_usage`
6. `register_webhook`
7. `export_to_format`
8. `sanitize_content`
9. `search_by_author`
10. `fuzzy_search`
11. `search_suggestions`
12. `get_usage_statistics`
13. `get_data_growth`
14. `get_top_searches`
15. `get_entity_cooccurrence`
16. `validate_database`
17. `migrate_schema`

---

**Конец отчёта**

