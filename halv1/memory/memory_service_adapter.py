"""Адаптер для MemoryService, обеспечивающий совместимость с MemoryStore."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from llm.base_client import EmbeddingsClient, LLMClient
from memory.service import MemoryService
from memory.memory_store import MemoryEntry

logger = logging.getLogger(__name__)


class MemoryServiceAdapter:
    """Адаптер MemoryService для совместимости с MemoryStore API."""

    def __init__(
        self,
        path: str | Path,
        embeddings_client: EmbeddingsClient | None = None,
        *,
        buffer_ttl_days: int = 7,
        buffer_max_size: int = 1000,
        short_term_limit: int = 100,
        llm_client: LLMClient | None = None,
    ) -> None:
        """Инициализация адаптера.
        
        Args:
            path: Путь к базе данных памяти
            embeddings_client: Клиент для эмбеддингов
            buffer_ttl_days: TTL буфера в днях
            buffer_max_size: Максимальный размер буфера
            short_term_limit: Лимит краткосрочной памяти
            llm_client: LLM клиент для обратной совместимости
        """
        self.memory_service = MemoryService(
            path=path,
            embeddings_client=embeddings_client,
            buffer_ttl_days=buffer_ttl_days,
            buffer_max_size=buffer_max_size,
        )
        self.short_term_limit = short_term_limit
        self.llm_client = llm_client
        
        # Для обратной совместимости с MemoryStore API
        self.short_term: List[MemoryEntry] = []
        self.long_term: List[MemoryEntry] = []
        self.long_term_path = Path(path) if isinstance(path, str) else path
        self.embeddings_client = embeddings_client
        self.archive = None

    def _embed(self, text: str) -> List[float]:
        """Создание эмбеддинга для текста."""
        if self.embeddings_client:
            try:
                return self.embeddings_client.embed(text)
            except Exception as exc:
                logger.warning(f"Ошибка создания эмбеддинга: {exc}")
                return [0.0] * 384  # Fallback размер
        return [0.0] * 384

    def _load_entry(self, item: dict[str, Any]) -> MemoryEntry:
        """Загрузка записи из словаря."""
        return MemoryEntry(
            text=item.get("text", ""),
            embedding=item.get("embedding", []),
            timestamp=item.get("timestamp", datetime.now(timezone.utc).timestamp()),
        )

    def remember(self, item: str, long_term: bool = False) -> None:
        """Сохранение элемента в память.
        
        Args:
            item: Текст для сохранения
            long_term: Сохранить в долгосрочную память
        """
        # Используем MemoryService для сохранения
        if long_term:
            self.memory_service.remember(item, frozen=True)
        else:
            # Для краткосрочной памяти используем буфер
            self.memory_service.write_event(item)
            
        # Также обновляем локальные списки для совместимости
        entry = MemoryEntry(
            text=item,
            embedding=self._embed(item),
            timestamp=datetime.now(timezone.utc).timestamp(),
        )
        
        if long_term:
            self.long_term.append(entry)
        else:
            self.short_term.append(entry)
            # Проверяем лимит краткосрочной памяти
            if len(self.short_term) > self.short_term_limit:
                self._summarize_to_long_term()

    def recall(self, long_term: bool = False) -> List[str]:
        """Получение записей из памяти.
        
        Args:
            long_term: Получить из долгосрочной памяти
            
        Returns:
            Список текстовых записей
        """
        if long_term:
            return self.memory_service.recall(long_term=True)
        else:
            return self.memory_service.recall(long_term=False)

    def search(self, query: str, long_term: bool | None = None) -> List[str]:
        """Поиск в памяти.
        
        Args:
            query: Поисковый запрос
            long_term: Искать в долгосрочной памяти
            
        Returns:
            Список найденных записей
        """
        return self.memory_service.search(query, long_term=long_term)

    def semantic_search(
        self, query: str, long_term: bool | None = None, top_k: int = 5
    ) -> List[str]:
        """Семантический поиск в памяти.
        
        Args:
            query: Поисковый запрос
            long_term: Искать в долгосрочной памяти
            top_k: Количество результатов
            
        Returns:
            Список найденных записей
        """
        return self.memory_service.semantic_search(query, long_term=long_term, top_k=top_k)

    def consolidate(self) -> None:
        """Консолидация памяти."""
        self.memory_service.consolidate()

    def save(self) -> None:
        """Сохранение памяти (заглушка для совместимости)."""
        # MemoryService автоматически сохраняет данные
        pass

    def prune_long_term(self, max_items: int = 1000) -> None:
        """Обрезка долгосрочной памяти для совместимости с MemoryStore API.
        
        Args:
            max_items: Максимальное количество элементов в долгосрочной памяти
        """
        # MemoryService автоматически управляет размером памяти
        # Этот метод добавлен для совместимости с тестами
        pass

    def _summarize_to_long_term(self) -> None:
        """Суммаризация краткосрочной памяти в долгосрочную."""
        if not self.short_term:
            return
            
        # Простая суммаризация - берем первые несколько записей
        summary_text = " | ".join([entry.text for entry in self.short_term[:5]])
        if len(self.short_term) > 5:
            summary_text += f" ... и еще {len(self.short_term) - 5} записей"
            
        # Сохраняем суммаризацию в долгосрочную память
        self.memory_service.remember(summary_text, frozen=True)
        
        # Очищаем краткосрочную память
        self.short_term.clear()

    def forget(self, item: str, *, long_term: bool = False) -> bool:
        """Удаление элемента из памяти.
        
        Args:
            item: Элемент для удаления
            long_term: Удалить из долгосрочной памяти
            
        Returns:
            True если элемент был удален
        """
        return self.memory_service.forget(item, long_term=long_term)

    def read_events(self) -> List[str]:
        """Чтение событий из буфера."""
        return self.memory_service.read_events()

    def write_event(
        self,
        content: str,
        *,
        timestamp: Optional[datetime] = None,
        entities: Optional[List[str]] = None,
        frozen: bool = False,
    ) -> None:
        """Запись события в буфер."""
        self.memory_service.write_event(
            content=content,
            timestamp=timestamp,
            entities=entities,
            frozen=frozen,
        )

    def read_schemas(self) -> List[tuple[str, List[str]]]:
        """Чтение схем из памяти."""
        return self.memory_service.read_schemas()

    def explain(self, item_id: int) -> List[tuple[str, Any]]:
        """Объяснение связей для элемента памяти."""
        return self.memory_service.explain(item_id)
