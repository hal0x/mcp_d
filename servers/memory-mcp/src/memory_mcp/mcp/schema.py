"""Pydantic schemas describing the MCP tools for memory ingestion and search."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class AttachmentPayload(BaseModel):
    """Attachment supplied with a memory record."""

    type: str = Field(..., description="Тип вложения (image/file/link/...)")
    uri: Optional[str] = Field(
        None, description="URI файла или ресурса (file://, http:// и т.д.)"
    )
    text: Optional[str] = Field(
        None, description="Извлечённый текст или подпись, если доступно"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Дополнительные метаданные вложения"
    )


class MemoryRecordPayload(BaseModel):
    """Normalized memory event used by the MCP ingest tool."""

    record_id: str = Field(..., description="Уникальный идентификатор записи")
    source: str = Field(..., description="Источник данных (telegram/files/... )")
    content: str = Field(..., description="Основной текст события")
    timestamp: datetime = Field(..., description="Временная метка в UTC")
    author: Optional[str] = Field(None, description="Автор/отправитель записи")
    tags: list[str] = Field(default_factory=list, description="Теги записи")
    entities: list[str] = Field(
        default_factory=list, description="Извлечённые сущности"
    )
    attachments: list[AttachmentPayload] = Field(
        default_factory=list, description="Список вложений"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Произвольные дополнительные данные"
    )
    embedding: Optional[list[float]] = Field(
        None, description="Эмбеддинг записи, если доступен"
    )


class IngestRequest(BaseModel):
    """Request body for the ingest tool."""

    records: list[MemoryRecordPayload] = Field(
        ..., description="Пакет записей для загрузки в память"
    )


class IngestResponse(BaseModel):
    """Response payload returned by the ingest tool."""

    records_ingested: int = Field(
        ..., description="Количество успешно загруженных записей"
    )
    attachments_ingested: int = Field(
        ..., description="Количество обработанных вложений"
    )
    duplicates_skipped: int = Field(
        ..., description="Сколько записей пропущено из-за дубликатов"
    )


class SearchRequest(BaseModel):
    """Query parameters for semantic search."""

    query: str = Field(..., description="Поисковый запрос (естественный язык)")
    top_k: int = Field(5, ge=1, le=50, description="Макс. количество результатов")
    source: Optional[str] = Field(
        None, description="Фильтр по источнику (например, telegram)"
    )
    tags: list[str] = Field(default_factory=list, description="Фильтр по тегам")
    date_from: Optional[datetime] = Field(
        None, description="Фильтр: дата не раньше указанной"
    )
    date_to: Optional[datetime] = Field(
        None, description="Фильтр: дата не позже указанной"
    )
    include_embeddings: bool = Field(
        False, description="Возвращать ли эмбеддинги найденных элементов"
    )


class SearchResultItem(BaseModel):
    """Single search result entry."""

    record_id: str = Field(..., description="Идентификатор найденной записи")
    score: float = Field(..., description="Схожесть (чем выше, тем лучше)")
    content: str = Field(..., description="Фрагмент или полный текст результата")
    source: str = Field(..., description="Источник записи")
    timestamp: datetime = Field(..., description="Временная метка записи")
    author: Optional[str] = Field(None, description="Автор записи")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Дополнительные данные результата"
    )
    embedding: Optional[list[float]] = Field(
        None, description="Эмбеддинг записи, если был запрошен"
    )


class SearchResponse(BaseModel):
    """Response returned by the search tool."""

    results: list[SearchResultItem] = Field(
        default_factory=list, description="Список результатов поиска"
    )
    total_matches: int = Field(
        ..., description="Общее количество совпадений до применения top_k"
    )


class FetchRequest(BaseModel):
    """Fetch a memory record by its identifier."""

    record_id: str = Field(..., description="Идентификатор записи для получения")


class FetchResponse(BaseModel):
    """Full record retrieved from storage."""

    record: Optional[MemoryRecordPayload] = Field(
        None, description="Полные данные записи, если найдены"
    )


# --------------------------- Trading signal schema ---------------------------


class TradingSignalRecord(BaseModel):
    """Stored trading signal representation."""

    signal_id: str = Field(..., description="Идентификатор торгового сигнала")
    timestamp: datetime = Field(..., description="Время генерации сигнала (UTC)")
    symbol: str = Field(..., description="Торговый инструмент (например BTCUSDT)")
    signal_type: str = Field(..., description="Тип сигнала (momentum/breakout/...)")
    direction: Optional[str] = Field(None, description="Направление (long/short)")
    entry: Optional[float] = Field(None, description="Цена входа")
    confidence: Optional[float] = Field(None, description="Уверенность (0-100)")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Дополнительные данные"
    )


class StoreTradingSignalRequest(BaseModel):
    """Request for storing trading signal in memory."""

    symbol: str = Field(..., description="Инструмент (BTCUSDT и т.д.)")
    signal_type: str = Field(..., description="Тип сигнала")
    direction: Optional[str] = Field(None, description="Направление торговли")
    entry: Optional[float] = Field(None, description="Цена входа")
    confidence: Optional[float] = Field(None, description="Уверенность (0-100)")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Произвольный контекст"
    )
    timestamp: Optional[datetime] = Field(
        None, description="Метка времени сигнала (по умолчанию сейчас)"
    )


class StoreTradingSignalResponse(BaseModel):
    """Response after storing trading signal."""

    signal: TradingSignalRecord = Field(..., description="Сохранённый сигнал")


class SearchTradingPatternsRequest(BaseModel):
    """Search trading signals by metadata/timeframe."""

    query: Optional[str] = Field(
        None, description="Поисковый запрос (тип сигнала, контекст)"
    )
    symbol: Optional[str] = Field(None, description="Фильтр по инструменту")
    timeframe: str = Field("recent", description="Период: recent/24h/week/month/all")
    limit: int = Field(20, ge=1, le=100, description="Максимум результатов")


class SearchTradingPatternsResponse(BaseModel):
    """Result of trading pattern search."""

    signals: list[TradingSignalRecord] = Field(
        default_factory=list, description="Найденные сигналы"
    )


class SignalPerformance(BaseModel):
    """Performance stats for a trading signal."""

    pnl: Optional[float] = Field(None, description="Финальный PnL")
    result: Optional[str] = Field(None, description="Outcome (win/loss/active)")
    closed_at: Optional[datetime] = Field(None, description="Время закрытия сделки")
    notes: Optional[str] = Field(None, description="Комментарии/заметки")


class GetSignalPerformanceRequest(BaseModel):
    """Request for signal performance."""

    signal_id: str = Field(..., description="Идентификатор сигнала")


class GetSignalPerformanceResponse(BaseModel):
    """Response with performance details."""

    signal: TradingSignalRecord = Field(..., description="Информация о сигнале")
    performance: Optional[SignalPerformance] = Field(
        None, description="Показатели эффективности, если доступны"
    )


# Web scraping schemas
class ScrapedContentRequest(BaseModel):
    """Request for ingesting scraped content."""

    url: str = Field(..., description="URL of the scraped page")
    title: Optional[str] = Field(None, description="Page title")
    content: str = Field(..., description="Main content text")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    source: str = Field(default="bright_data", description="Scraping source")
    tags: list[str] = Field(default_factory=list, description="Content tags")
    entities: list[str] = Field(default_factory=list, description="Extracted entities")


class ScrapedContentResponse(BaseModel):
    """Response for scraped content ingestion."""

    record_id: str = Field(..., description="Generated record ID")
    status: str = Field(..., description="Ingestion status")
    url: str = Field(..., description="Original URL")
    message: Optional[str] = Field(None, description="Status message")


# --------------------------- Embedding schemas ---------------------------


class GenerateEmbeddingRequest(BaseModel):
    """Request for generating embedding for text."""

    text: str = Field(..., description="Текст для генерации эмбеддинга")
    model: Optional[str] = Field(
        None, description="Модель для эмбеддингов (опционально, используется из конфигурации)"
    )


class GenerateEmbeddingResponse(BaseModel):
    """Response with generated embedding."""

    embedding: list[float] = Field(..., description="Векторное представление текста")
    dimension: int = Field(..., description="Размерность эмбеддинга")
    model: Optional[str] = Field(None, description="Использованная модель")


# --------------------------- Record management schemas ---------------------------


class UpdateRecordRequest(BaseModel):
    """Request for updating a memory record."""

    record_id: str = Field(..., description="Идентификатор записи для обновления")
    content: Optional[str] = Field(None, description="Новый контент записи")
    source: Optional[str] = Field(None, description="Новый источник")
    tags: Optional[list[str]] = Field(None, description="Новые теги")
    entities: Optional[list[str]] = Field(None, description="Новые сущности")
    metadata: Optional[dict[str, Any]] = Field(
        None, description="Дополнительные метаданные (объединяются с существующими)"
    )


class UpdateRecordResponse(BaseModel):
    """Response after updating a record."""

    record_id: str = Field(..., description="Идентификатор обновлённой записи")
    updated: bool = Field(..., description="Успешно ли обновлена запись")
    message: Optional[str] = Field(None, description="Сообщение о результате")


class DeleteRecordRequest(BaseModel):
    """Request for deleting a memory record."""

    record_id: str = Field(..., description="Идентификатор записи для удаления")


class DeleteRecordResponse(BaseModel):
    """Response after deleting a record."""

    record_id: str = Field(..., description="Идентификатор удалённой записи")
    deleted: bool = Field(..., description="Успешно ли удалена запись")
    message: Optional[str] = Field(None, description="Сообщение о результате")


# --------------------------- Statistics schemas ---------------------------


class GetStatisticsResponse(BaseModel):
    """Response with system statistics."""

    graph_stats: dict[str, Any] = Field(..., description="Статистика графа знаний")
    sources_count: dict[str, int] = Field(
        default_factory=dict, description="Количество записей по источникам"
    )
    tags_count: dict[str, int] = Field(
        default_factory=dict, description="Количество записей по тегам"
    )
    database_size_bytes: Optional[int] = Field(
        None, description="Размер базы данных в байтах"
    )


class GetIndexingProgressRequest(BaseModel):
    """Request for indexing progress."""

    chat: Optional[str] = Field(None, description="Фильтр по конкретному чату")


class IndexingProgressItem(BaseModel):
    """Single indexing progress entry."""

    chat_name: str = Field(..., description="Название чата")
    last_indexed_date: Optional[str] = Field(None, description="Дата последнего проиндексированного сообщения")
    last_indexing_time: Optional[str] = Field(None, description="Время последней индексации")
    total_messages: int = Field(0, description="Всего проиндексированных сообщений")
    total_sessions: int = Field(0, description="Всего созданных сессий")
    job_id: Optional[str] = Field(None, description="Идентификатор активной задачи индексации")
    status: Optional[str] = Field(None, description="Статус индексации: 'running', 'completed', 'failed'")
    started_at: Optional[str] = Field(None, description="Время начала индексации (ISO format)")
    current_stage: Optional[str] = Field(None, description="Текущий этап индексации")


class GetIndexingProgressResponse(BaseModel):
    """Response with indexing progress."""

    progress: list[IndexingProgressItem] = Field(
        default_factory=list, description="Список прогресса индексации"
    )
    message: Optional[str] = Field(None, description="Сообщение о статусе")


# --------------------------- Graph operations schemas ---------------------------


class GetGraphNeighborsRequest(BaseModel):
    """Request for getting graph neighbors."""

    node_id: str = Field(..., description="ID узла")
    edge_type: Optional[str] = Field(None, description="Фильтр по типу рёбер")
    direction: str = Field(
        default="both", description="Направление: 'out', 'in', 'both'"
    )


class GraphNeighborItem(BaseModel):
    """Single neighbor node entry."""

    node_id: str = Field(..., description="ID соседнего узла")
    edge_type: Optional[str] = Field(None, description="Тип ребра")
    edge_data: dict[str, Any] = Field(
        default_factory=dict, description="Данные ребра"
    )


class GetGraphNeighborsResponse(BaseModel):
    """Response with graph neighbors."""

    neighbors: list[GraphNeighborItem] = Field(
        default_factory=list, description="Список соседних узлов"
    )


class FindGraphPathRequest(BaseModel):
    """Request for finding graph path."""

    source_id: str = Field(..., description="ID исходного узла")
    target_id: str = Field(..., description="ID целевого узла")
    max_length: int = Field(default=5, ge=1, le=20, description="Максимальная длина пути")


class FindGraphPathResponse(BaseModel):
    """Response with graph path."""

    path: Optional[list[str]] = Field(None, description="Путь как список ID узлов")
    total_weight: Optional[float] = Field(None, description="Суммарный вес пути")
    found: bool = Field(..., description="Найден ли путь")


class GetRelatedRecordsRequest(BaseModel):
    """Request for getting related records."""

    record_id: str = Field(..., description="ID записи")
    max_depth: int = Field(default=1, ge=1, le=3, description="Максимальная глубина поиска")
    limit: int = Field(default=10, ge=1, le=50, description="Максимальное количество результатов")


class GetRelatedRecordsResponse(BaseModel):
    """Response with related records."""

    records: list[MemoryRecordPayload] = Field(
        default_factory=list, description="Список связанных записей"
    )


# --------------------------- Advanced search schemas ---------------------------


class SearchByEmbeddingRequest(BaseModel):
    """Request for searching by embedding vector."""

    embedding: list[float] = Field(..., description="Вектор эмбеддинга для поиска")
    top_k: int = Field(default=5, ge=1, le=50, description="Максимальное количество результатов")
    source: Optional[str] = Field(None, description="Фильтр по источнику")
    tags: list[str] = Field(default_factory=list, description="Фильтр по тегам")
    date_from: Optional[datetime] = Field(None, description="Фильтр: дата не раньше")
    date_to: Optional[datetime] = Field(None, description="Фильтр: дата не позже")


class SearchByEmbeddingResponse(BaseModel):
    """Response with search results by embedding."""

    results: list[SearchResultItem] = Field(
        default_factory=list, description="Список результатов поиска"
    )
    total_matches: int = Field(..., description="Общее количество совпадений")


class SimilarRecordsRequest(BaseModel):
    """Request for finding similar records."""

    record_id: str = Field(..., description="ID записи для поиска похожих")
    top_k: int = Field(default=5, ge=1, le=50, description="Максимальное количество результатов")


class SimilarRecordsResponse(BaseModel):
    """Response with similar records."""

    results: list[SearchResultItem] = Field(
        default_factory=list, description="Список похожих записей"
    )


class SearchExplainRequest(BaseModel):
    """Request for explaining search results."""

    query: str = Field(..., description="Поисковый запрос")
    record_id: str = Field(..., description="ID записи для объяснения")
    rank: int = Field(default=0, description="Позиция в результатах (0-based)")


class SearchExplainResponse(BaseModel):
    """Response with search explanation."""

    explanation: dict[str, Any] = Field(..., description="Объяснение релевантности")
    score_breakdown: dict[str, Any] = Field(..., description="Декомпозиция score")
    connection_paths: list[dict[str, Any]] = Field(
        default_factory=list, description="Пути связей через граф"
    )
    explanation_text: str = Field(..., description="Текстовое объяснение")


# --------------------------- Analytics schemas ---------------------------


class GetTagsStatisticsResponse(BaseModel):
    """Response with tags statistics."""

    tags_count: dict[str, int] = Field(
        default_factory=dict, description="Количество записей по тегам"
    )
    total_tags: int = Field(..., description="Всего уникальных тегов")
    total_tagged_records: int = Field(..., description="Всего записей с тегами")


class GetTimelineRequest(BaseModel):
    """Request for timeline."""

    source: Optional[str] = Field(None, description="Фильтр по источнику")
    date_from: Optional[datetime] = Field(None, description="Начало периода")
    date_to: Optional[datetime] = Field(None, description="Конец периода")
    limit: int = Field(default=50, ge=1, le=500, description="Максимальное количество записей")


class TimelineItem(BaseModel):
    """Single timeline entry."""

    record_id: str = Field(..., description="ID записи")
    timestamp: datetime = Field(..., description="Временная метка")
    source: str = Field(..., description="Источник")
    content_preview: str = Field(..., description="Превью контента")


class GetTimelineResponse(BaseModel):
    """Response with timeline."""

    items: list[TimelineItem] = Field(
        default_factory=list, description="Список записей временной линии"
    )
    total: int = Field(..., description="Всего записей в периоде")


class AnalyzeEntitiesRequest(BaseModel):
    """Request for entity analysis."""

    text: str = Field(..., description="Текст для анализа сущностей")
    entity_types: Optional[list[str]] = Field(
        None, description="Типы сущностей для извлечения (опционально)"
    )


class EntityItem(BaseModel):
    """Single entity entry."""

    value: str = Field(..., description="Значение сущности")
    entity_type: str = Field(..., description="Тип сущности")
    count: int = Field(..., description="Количество упоминаний")


class AnalyzeEntitiesResponse(BaseModel):
    """Response with entity analysis."""

    entities: list[EntityItem] = Field(
        default_factory=list, description="Список найденных сущностей"
    )
    total_entities: int = Field(..., description="Всего найдено сущностей")


# --------------------------- Batch operations schemas ---------------------------


class BatchUpdateRecordItem(BaseModel):
    """Single record update item."""

    record_id: str = Field(..., description="ID записи для обновления")
    content: Optional[str] = Field(None, description="Новый контент")
    source: Optional[str] = Field(None, description="Новый источник")
    tags: Optional[list[str]] = Field(None, description="Новые теги")
    entities: Optional[list[str]] = Field(None, description="Новые сущности")
    metadata: Optional[dict[str, Any]] = Field(None, description="Новые метаданные")


class BatchUpdateRecordsRequest(BaseModel):
    """Request for batch updating records."""

    updates: list[BatchUpdateRecordItem] = Field(
        ..., description="Список обновлений записей"
    )


class BatchUpdateResult(BaseModel):
    """Single batch update result."""

    record_id: str = Field(..., description="ID записи")
    updated: bool = Field(..., description="Успешно ли обновлена")
    message: Optional[str] = Field(None, description="Сообщение о результате")


class BatchUpdateRecordsResponse(BaseModel):
    """Response after batch update."""

    results: list[BatchUpdateResult] = Field(
        default_factory=list, description="Результаты обновления"
    )
    total_updated: int = Field(..., description="Всего успешно обновлено")
    total_failed: int = Field(..., description="Всего неудачных обновлений")


class BatchDeleteRecordsRequest(BaseModel):
    """Request for batch deleting records."""

    record_ids: list[str] = Field(..., description="Список идентификаторов записей для удаления")


class BatchDeleteResult(BaseModel):
    """Single batch delete result."""

    record_id: str = Field(..., description="ID записи")
    deleted: bool = Field(..., description="Успешно ли удалена")
    message: Optional[str] = Field(None, description="Сообщение о результате")


class BatchDeleteRecordsResponse(BaseModel):
    """Response after batch delete."""

    results: list[BatchDeleteResult] = Field(
        default_factory=list, description="Результаты удаления"
    )
    total_deleted: int = Field(..., description="Всего успешно удалено")
    total_failed: int = Field(..., description="Всего неудачных удалений")


class BatchFetchRecordsRequest(BaseModel):
    """Request for batch fetching records."""

    record_ids: list[str] = Field(..., description="Список идентификаторов записей для получения")


class BatchFetchResult(BaseModel):
    """Single batch fetch result."""

    record_id: str = Field(..., description="ID записи")
    record: Optional[MemoryRecordPayload] = Field(
        None, description="Данные записи, если найдена"
    )
    found: bool = Field(..., description="Найдена ли запись")
    message: Optional[str] = Field(None, description="Сообщение о результате")


class BatchFetchRecordsResponse(BaseModel):
    """Response after batch fetch."""

    results: list[BatchFetchResult] = Field(
        default_factory=list, description="Результаты получения"
    )
    total_found: int = Field(..., description="Всего найдено записей")
    total_not_found: int = Field(..., description="Всего не найдено записей")


# --------------------------- Export/Import schemas ---------------------------


class ExportRecordsRequest(BaseModel):
    """Request for exporting records."""

    format: str = Field(default="json", description="Формат экспорта: json, csv, markdown")
    source: Optional[str] = Field(None, description="Фильтр по источнику")
    tags: list[str] = Field(default_factory=list, description="Фильтр по тегам")
    date_from: Optional[datetime] = Field(None, description="Начало периода")
    date_to: Optional[datetime] = Field(None, description="Конец периода")
    limit: int = Field(default=100, ge=1, le=10000, description="Максимальное количество записей")


class ExportRecordsResponse(BaseModel):
    """Response with exported records."""

    format: str = Field(..., description="Формат экспорта")
    content: str = Field(..., description="Экспортированные данные")
    records_count: int = Field(..., description="Количество экспортированных записей")


class ImportRecordsRequest(BaseModel):
    """Request for importing records."""

    format: str = Field(..., description="Формат импорта: json, csv")
    content: str = Field(..., description="Содержимое для импорта")
    source: Optional[str] = Field(None, description="Источник для импортируемых записей")


class ImportRecordsResponse(BaseModel):
    """Response after importing records."""

    records_imported: int = Field(..., description="Количество импортированных записей")
    records_failed: int = Field(..., description="Количество неудачных импортов")
    message: Optional[str] = Field(None, description="Сообщение о результате")


# --------------------------- Summaries schemas ---------------------------


class UpdateSummariesRequest(BaseModel):
    """Request for updating markdown summaries."""

    chat: Optional[str] = Field(None, description="Обновить отчеты только для конкретного чата")
    force: bool = Field(default=False, description="Принудительно пересоздать существующие артефакты")


class UpdateSummariesResponse(BaseModel):
    """Response after updating summaries."""

    chats_updated: int = Field(..., description="Количество обновленных чатов")
    message: str = Field(..., description="Сообщение о результате")


class ReviewSummariesRequest(BaseModel):
    """Request for reviewing summaries."""

    dry_run: bool = Field(default=False, description="Только анализ, без изменения файлов")
    chat: Optional[str] = Field(None, description="Обработать только конкретный чат")
    limit: Optional[int] = Field(None, description="Максимальное количество файлов для обработки")


class ReviewSummariesResponse(BaseModel):
    """Response after reviewing summaries."""

    files_processed: int = Field(..., description="Количество обработанных файлов")
    files_fixed: int = Field(..., description="Количество исправленных файлов")
    message: str = Field(..., description="Сообщение о результате")


# --------------------------- Insight Graph schemas ---------------------------


class BuildInsightGraphRequest(BaseModel):
    """Request for building insight graph."""

    summaries_dir: Optional[str] = Field(None, description="Директория с саммаризациями")
    chroma_path: Optional[str] = Field(None, description="Путь к ChromaDB")
    similarity_threshold: float = Field(default=0.76, description="Порог схожести")
    max_similar_results: int = Field(default=8, description="Максимальное количество похожих результатов")


class InsightItem(BaseModel):
    """Single insight item."""

    title: str = Field(..., description="Заголовок инсайта")
    description: str = Field(..., description="Описание инсайта")
    confidence: float = Field(..., description="Уверенность в инсайте")


class BuildInsightGraphResponse(BaseModel):
    """Response with insight graph."""

    nodes_count: int = Field(..., description="Количество узлов в графе")
    edges_count: int = Field(..., description="Количество связей в графе")
    insights: list[InsightItem] = Field(default_factory=list, description="Список инсайтов")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Метрики графа")
    message: str = Field(..., description="Сообщение о результате")


# --------------------------- Indexing schemas ---------------------------


class IndexChatRequest(BaseModel):
    """Request for indexing a specific chat."""

    chat: str = Field(..., description="Название чата для индексации")
    force_full: bool = Field(default=False, description="Полная пересборка индекса")
    recent_days: int = Field(default=7, description="Пересаммаризировать последние N дней")
    
    # Параметры качества и улучшения
    enable_quality_check: Optional[bool] = Field(default=True, description="Включить проверку качества саммаризации")
    enable_iterative_refinement: Optional[bool] = Field(default=True, description="Включить автоматическое улучшение саммаризаций")
    min_quality_score: Optional[float] = Field(default=80.0, description="Минимальный приемлемый балл качества")
    
    # Параметры кластеризации
    enable_clustering: Optional[bool] = Field(default=True, description="Включить автоматическую кластеризацию сессий")
    clustering_threshold: Optional[float] = Field(default=0.8, description="Порог сходства для кластеризации")
    min_cluster_size: Optional[int] = Field(default=2, description="Минимальный размер кластера")
    
    # Параметры группировки
    max_messages_per_group: Optional[int] = Field(default=100, description="Максимальное количество сообщений в группе")
    max_session_hours: Optional[int] = Field(default=6, description="Максимальная длительность сессии в часах")
    gap_minutes: Optional[int] = Field(default=60, description="Максимальный разрыв между сообщениями в минутах")
    
    # Параметры умной группировки
    enable_smart_aggregation: Optional[bool] = Field(default=True, description="Включить умную группировку с скользящими окнами")
    aggregation_strategy: Optional[str] = Field(default="smart", description="Стратегия группировки: 'smart', 'channel', 'legacy'")
    now_window_hours: Optional[int] = Field(default=24, description="Размер NOW окна в часах")
    fresh_window_days: Optional[int] = Field(default=14, description="Размер FRESH окна в днях")
    recent_window_days: Optional[int] = Field(default=30, description="Размер RECENT окна в днях")
    strategy_threshold: Optional[int] = Field(default=1000, description="Порог количества сообщений для перехода между стратегиями")
    
    # Дополнительные параметры
    enable_entity_learning: Optional[bool] = Field(default=True, description="Включить автоматическое обучение словарей сущностей")
    enable_time_analysis: Optional[bool] = Field(default=True, description="Включить анализ временных паттернов")


class IndexChatResponse(BaseModel):
    """Response after indexing a chat."""

    job_id: str = Field(..., description="Идентификатор задачи индексации")
    status: str = Field(..., description="Статус: 'started', 'running', 'completed', 'failed'")
    chat: str = Field(..., description="Название чата")
    message: str = Field(..., description="Сообщение о результате")


class ChatInfo(BaseModel):
    """Information about an available chat."""

    name: str = Field(..., description="Название чата")
    path: str = Field(..., description="Путь к директории чата")
    message_count: int = Field(default=0, description="Количество сообщений (если доступно)")
    last_modified: Optional[str] = Field(None, description="Дата последнего изменения (ISO format)")


class GetAvailableChatsRequest(BaseModel):
    """Request for getting available chats."""

    include_stats: bool = Field(default=False, description="Включить статистику (количество сообщений, дата изменения)")


class GetAvailableChatsResponse(BaseModel):
    """Response with available chats."""

    chats: list[ChatInfo] = Field(default_factory=list, description="Список доступных чатов")
    total_count: int = Field(default=0, description="Общее количество чатов")
    message: str = Field(..., description="Сообщение о результате")


# --------------------------- Background indexing schemas ---------------------------


class StartBackgroundIndexingRequest(BaseModel):
    """Request to start background indexing."""


class StartBackgroundIndexingResponse(BaseModel):
    """Response after starting background indexing."""

    success: bool = Field(..., description="Успешность запуска")
    message: str = Field(..., description="Сообщение о результате")


class StopBackgroundIndexingRequest(BaseModel):
    """Request to stop background indexing."""


class StopBackgroundIndexingResponse(BaseModel):
    """Response after stopping background indexing."""

    success: bool = Field(..., description="Успешность остановки")
    message: str = Field(..., description="Сообщение о результате")


class GetBackgroundIndexingStatusRequest(BaseModel):
    """Request for background indexing status."""


class GetBackgroundIndexingStatusResponse(BaseModel):
    """Response with background indexing status."""

    running: bool = Field(..., description="Запущен ли фоновый процесс")
    check_interval: int = Field(..., description="Интервал проверки в секундах")
    last_check_time: Optional[str] = Field(None, description="Время последней проверки (ISO format)")
    input_path: str = Field(..., description="Путь к input директории")
    chats_path: str = Field(..., description="Путь к chats директории")
    message: str = Field(..., description="Сообщение о статусе")


# --------------------------- Smart search schemas ---------------------------


class SearchFeedback(BaseModel):
    """Модель обратной связи по результату поиска."""

    record_id: str = Field(..., description="Идентификатор результата")
    artifact_path: Optional[str] = Field(
        None, description="Путь к артифакту (если результат из артифакта)"
    )
    relevance: Literal["relevant", "irrelevant", "partially_relevant"] = Field(
        ..., description="Релевантность результата"
    )
    comment: Optional[str] = Field(None, description="Комментарий пользователя")


class SmartSearchRequest(BaseModel):
    """Запрос для интерактивного смарт-поиска."""

    query: str = Field(..., description="Поисковый запрос (естественный язык)")
    session_id: Optional[str] = Field(
        None, description="ID сессии для многошагового диалога"
    )
    feedback: Optional[list[SearchFeedback]] = Field(
        None, description="Обратная связь по предыдущим результатам"
    )
    clarify: Optional[bool] = Field(
        False, description="Запросить уточняющие вопросы, если результаты неоднозначны"
    )
    top_k: int = Field(10, ge=1, le=50, description="Максимальное количество результатов")
    source: Optional[str] = Field(
        None, description="Фильтр по источнику (например, telegram)"
    )
    tags: list[str] = Field(
        default_factory=list, description="Фильтр по тегам"
    )
    date_from: Optional[datetime] = Field(
        None, description="Фильтр: дата не раньше указанной"
    )
    date_to: Optional[datetime] = Field(
        None, description="Фильтр: дата не позже указанной"
    )
    artifact_types: Optional[list[str]] = Field(
        None,
        description="Фильтр по типам артифактов (chat_context, now_summary, report, aggregation_state)",
    )


class SmartSearchResponse(BaseModel):
    """Ответ интерактивного смарт-поиска."""

    results: list[SearchResultItem] = Field(
        default_factory=list, description="Список результатов поиска"
    )
    clarifying_questions: Optional[list[str]] = Field(
        None, description="Уточняющие вопросы для улучшения поиска"
    )
    suggested_refinements: Optional[list[str]] = Field(
        None, description="Предложенные уточнения запроса"
    )
    session_id: str = Field(..., description="ID сессии поиска")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Уверенность в релевантности результатов"
    )
    artifacts_found: int = Field(
        0, description="Количество найденных артифактов"
    )
    db_records_found: int = Field(
        0, description="Количество найденных записей из БД"
    )
    total_matches: int = Field(
        ..., description="Общее количество совпадений до применения top_k"
    )
