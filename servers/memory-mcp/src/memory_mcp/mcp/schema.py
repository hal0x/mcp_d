"""Pydantic schemas describing the MCP tools for memory ingestion and search."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

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
