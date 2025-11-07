"""Адаптер MCP сервера для интеграции с системой памяти HALv1.

Этот модуль предоставляет адаптер, который реализует интерфейс MemoryServiceAdapter
и использует MCP сервер для поиска и управления данными Telegram чатов.
"""

from __future__ import annotations

import logging
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Dict, Union
import json

from llm.base_client import EmbeddingsClient, LLMClient
from memory.memory_store import MemoryEntry
from memory.service import MemoryService
from utils.vector_math import cosine_similarity

logger = logging.getLogger(__name__)


class MCPMemoryAdapter:
    """Адаптер MCP сервера для интеграции с системой памяти HALv1.
    
    Реализует интерфейс MemoryServiceAdapter, используя MCP сервер
    для поиска и управления данными Telegram чатов.
    """
    
    def __init__(
        self,
        mcp_server_path: str | Path | None = None,
        embeddings_client: EmbeddingsClient | None = None,
        *,
        long_term_path: str | Path | None = None,
        buffer_ttl_days: int = 7,
        buffer_max_size: int = 1000,
        short_term_limit: int = 100,
        llm_client: LLMClient | None = None,
        archive: "MemoryArchive | None" = None,
        episode_graph_path: str | Path | None = None,
    ) -> None:
        """Инициализация адаптера MCP сервера.
        
        Args:
            mcp_server_path: Путь к MCP серверу
            embeddings_client: Клиент для эмбеддингов
            long_term_path: Путь к долгосрочной памяти (для совместимости)
            buffer_ttl_days: TTL буфера в днях
            buffer_max_size: Максимальный размер буфера
            short_term_limit: Лимит краткосрочной памяти
            llm_client: LLM клиент для совместимости
            archive: Архив памяти
            episode_graph_path: Путь к графу эпизодов
        """
        self.mcp_server_path = Path(mcp_server_path) if mcp_server_path else None
        self.embeddings_client = embeddings_client
        self.llm_client = llm_client
        self.archive = archive
        
        # Параметры для совместимости
        self.short_term_limit = short_term_limit
        self.buffer_ttl_days = buffer_ttl_days
        self.buffer_max_size = buffer_max_size
        
        # Определяем путь к базе данных
        db_path_input = long_term_path or episode_graph_path
        if db_path_input is None:
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="halv1_mcp_"))
            db_path = temp_dir / "mcp_memory.db"
            db_path_str = str(db_path)
        else:
            db_path_str = str(db_path_input)
        
        # Инициализируем MemoryService для локального буферирования
        self.memory_service = MemoryService(
            path=db_path_str,
            embeddings_client=embeddings_client,
            buffer_ttl_days=buffer_ttl_days,
            buffer_max_size=buffer_max_size,
        )
        
        # Локальные кэши для совместимости
        self._short_term_cache: List[MemoryEntry] = []
        self._long_term_cache: List[MemoryEntry] = []
        self._cache_dirty = True
        
        # Путь для сохранения
        self.path = Path(db_path_str) if db_path_str != ":memory:" else ":memory:"
        
        logger.info(f"MCP Memory Adapter инициализирован: {self.path}")
    
    # ======================================================================
    # Основные методы MemoryService (L0-L∞ архитектура)
    # ======================================================================
    
    def remember(self, text: str, *, frozen: bool = False, long_term: bool = False) -> None:
        """Сохранение текста в память.
        
        Args:
            text: Текст для сохранения
            frozen: Заморозить запись
            long_term: Сохранить в долгосрочную память
        """
        # Проверяем дедупликацию
        text_lc = text.lower()
        
        if long_term:
            # Сохраняем в долгосрочную память через MemoryService
            try:
                result = self.memory_service.remember(text, frozen=frozen)
                logger.debug(f"Сохранено в долгосрочную память: {text[:50]}...")
            except Exception as exc:
                logger.warning(f"Ошибка при сохранении в MemoryService: {exc}")
            
            # Добавляем в кэш долгосрочной памяти
            entry = MemoryEntry(
                text=text,
                embedding=self._embed(text),
                timestamp=datetime.now(timezone.utc).timestamp(),
                frozen=frozen,
            )
            self._long_term_cache.append(entry)
        else:
            # Сохраняем в буфер L0
            self.memory_service.write_event(text, frozen=frozen)
            
            # Добавляем в кэш краткосрочной памяти
            entry = MemoryEntry(
                text=text,
                embedding=self._embed(text),
                timestamp=datetime.now(timezone.utc).timestamp(),
                frozen=frozen,
            )
            self._short_term_cache.append(entry)
            
            # Проверяем лимит краткосрочной памяти
            if len(self._short_term_cache) > self.short_term_limit:
                self._summarize_to_long_term()
        
        self._cache_dirty = True
        logger.debug(f"Сохранено в память: {text[:50]}...")
    
    def write_event(
        self,
        content: str,
        *,
        timestamp: Optional[datetime] = None,
        entities: Optional[List[str]] = None,
        frozen: bool = False,
    ) -> None:
        """Запись события в буфер L0.
        
        Args:
            content: Содержимое события
            timestamp: Временная метка
            entities: Связанные сущности
            frozen: Заморозить событие
        """
        self.memory_service.write_event(
            content=content,
            timestamp=timestamp,
            entities=entities,
            frozen=frozen,
        )
        
        # Также добавляем в локальный кэш для совместимости
        entry = MemoryEntry(
            text=content,
            embedding=self._embed(content),
            timestamp=timestamp.timestamp() if timestamp else datetime.now(timezone.utc).timestamp(),
            frozen=frozen,
        )
        self._short_term_cache.append(entry)
        
        self._cache_dirty = True
        logger.debug(f"Событие записано в буфер: {content[:50]}...")
    
    def read_events(self) -> List[str]:
        """Чтение событий из буфера L0."""
        # Возвращаем события из локального кэша для совместимости
        return [entry.text for entry in self._short_term_cache]
    
    def consolidate(self) -> None:
        """Консолидация памяти (L0 → L1 → L2)."""
        self.memory_service.consolidate()
        self._cache_dirty = True
        logger.info("Память консолидирована")
    
    def read_schemas(self) -> List[tuple[str, List[str]]]:
        """Чтение схем из памяти."""
        return self.memory_service.read_schemas()
    
    def explain(self, item_id: int) -> List[tuple[str, Any]]:
        """Объяснение связей для элемента памяти."""
        return self.memory_service.explain(item_id)
    
    # ======================================================================
    # Методы для обратной совместимости с MemoryStore API
    # ======================================================================
    
    def recall(self, long_term: bool = False) -> List[str]:
        """Получение записей из памяти.
        
        Args:
            long_term: Получить из долгосрочной памяти
            
        Returns:
            Список текстовых записей
        """
        # Не перезагружаем кэш автоматически, чтобы сохранить локальные изменения
        # if self._cache_dirty:
        #     self._ensure_cache_loaded()
        
        if long_term:
            return [entry.text for entry in self._long_term_cache]
        else:
            return [entry.text for entry in self._short_term_cache]
    
    def search(self, query: str, long_term: bool | None = None) -> List[str]:
        """Поиск в памяти.
        
        Args:
            query: Поисковый запрос
            long_term: Искать в долгосрочной памяти
            
        Returns:
            Список найденных записей
        """
        if not query:
            return self.recall(long_term=long_term)
        
        # Используем MCP сервер для поиска в Telegram данных
        try:
            # Синхронный вызов MCP поиска
            import subprocess
            import json
            
            if self.mcp_server_path:
                cmd = [
                    "python", str(self.mcp_server_path),
                    "--tool", "search",
                    "--query", query
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    if data.get("success") and "results" in data:
                        return data["results"]
        except Exception as exc:
            logger.warning(f"Ошибка при поиске через MCP: {exc}")
        
        # Fallback к локальному поиску
        q = query.lower()
        results = []
        
        if long_term is False or long_term is None:
            for content in self.recall(long_term=False):
                if q in content.lower():
                    results.append(content)
        
        if long_term is True or long_term is None:
            for content in self.recall(long_term=True):
                if q in content.lower():
                    results.append(content)
        
        return results
    
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
        if not query or not query.strip():
            return []
        
        # Используем MCP сервер для семантического поиска
        try:
            # Синхронный вызов MCP семантического поиска
            import subprocess
            import json
            
            if self.mcp_server_path:
                cmd = [
                    "python", str(self.mcp_server_path),
                    "--tool", "semantic_search",
                    "--query", query,
                    "--top_k", str(top_k)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    if data.get("success") and "results" in data:
                        return data["results"]
        except Exception as exc:
            logger.warning(f"Ошибка при семантическом поиске через MCP: {exc}")
        
        # Fallback к локальному семантическому поиску
        if self._cache_dirty:
            self._ensure_cache_loaded()
        
        if long_term is True:
            entries = self._long_term_cache
        elif long_term is False:
            entries = self._short_term_cache
        else:
            entries = self._short_term_cache + self._long_term_cache
        
        if not entries:
            return []
        
        query_embedding = self._embed(query)
        query_tokens = {tok for tok in query.lower().split() if tok}

        scored_entries = []
        for entry in entries:
            similarity = self._cosine_similarity(query_embedding, entry.embedding)
            entry_tokens = {tok for tok in entry.text.lower().split() if tok}
            has_token_overlap = bool(query_tokens & entry_tokens)
            scored_entries.append((similarity, has_token_overlap, entry.text))

        scored_entries.sort(key=lambda x: x[0], reverse=True)

        filtered = []
        for similarity, has_overlap, text in scored_entries:
            if similarity > 0:
                filtered.append((similarity, text))
            elif has_overlap:
                filtered.append((similarity, text))
            if len(filtered) >= top_k:
                break

        return [text for _, text in filtered[:top_k]] if filtered else []
    
    def forget(self, item: str, *, long_term: bool = False, archive: "MemoryArchive | None" = None) -> bool:
        """Удаление элемента из памяти.
        
        Args:
            item: Элемент для удаления
            long_term: Удалить из долгосрочной памяти
            archive: Архив для архивирования элемента
            
        Returns:
            True если элемент был найден и удален
        """
        if self._cache_dirty:
            self._ensure_cache_loaded()
        
        target_cache = self._long_term_cache if long_term else self._short_term_cache
        entry_to_archive = None
        
        logger.debug(f"Ищем элемент '{item}' в {'долгосрочной' if long_term else 'краткосрочной'} памяти")
        logger.debug(f"Количество записей в кэше: {len(target_cache)}")
        
        # Ищем точное совпадение или частичное совпадение
        for i, entry in enumerate(target_cache):
            logger.debug(f"Запись {i}: '{entry.text[:50]}...'")
            if entry.text == item or item in entry.text:
                entry_to_archive = entry
                logger.debug(f"Найден элемент для удаления: '{entry.text[:50]}...'")
                break
        
        if not entry_to_archive:
            logger.debug(f"Элемент '{item}' не найден в кэше")
            return False
        
        if archive:
            archive.archive([entry_to_archive])
        
        target_cache.remove(entry_to_archive)
        logger.debug(f"Элемент удален. Осталось записей: {len(target_cache)}")
        
        if long_term:
            try:
                self.memory_service.forget(item, long_term=True)
            except Exception as exc:
                logger.warning(f"Ошибка при удалении из MemoryService: {exc}")
        
        self._cache_dirty = True
        return True
    
    # ======================================================================
    # Свойства для обратной совместимости с MemoryStore
    # ======================================================================
    
    @property
    def short_term(self) -> List[MemoryEntry]:
        """Краткосрочная память."""
        self._ensure_cache_loaded()
        return self._short_term_cache
    
    @short_term.setter
    def short_term(self, value: List[MemoryEntry]) -> None:
        """Установка краткосрочной памяти."""
        self._short_term_cache = value
        self._cache_dirty = False

    @property
    def long_term(self) -> List[MemoryEntry]:
        """Долгосрочная память."""
        self._ensure_cache_loaded()
        return self._long_term_cache
    
    @long_term.setter
    def long_term(self, value: List[MemoryEntry]) -> None:
        """Установка долгосрочной памяти."""
        self._long_term_cache = value
        self._cache_dirty = False
    
    def save(self) -> None:
        """Сохранение памяти."""
        try:
            self.memory_service.consolidate()
        except Exception as exc:
            logger.warning(f"Ошибка при сохранении памяти: {exc}")
        
        if self.path and self.path != ":memory:":
            try:
                data = {
                    "short_term": [e.__dict__ for e in self._short_term_cache],
                    "long_term": [e.__dict__ for e in self._long_term_cache],
                }
                
                path = Path(self.path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                logger.debug(f"Память сохранена в {path}")
            except Exception as exc:
                logger.warning(f"Ошибка при сохранении JSON: {exc}")
    
    def prune_long_term(self, max_items: int = 1000) -> None:
        """Обрезка долгосрочной памяти."""
        current_items = self.recall(long_term=True)
        if len(current_items) <= max_items:
            return
        
        if max_items <= 0:
            if not self._long_term_cache:
                return
            
            all_items = [entry.text for entry in self._long_term_cache]
            summary = self._summarize_texts(all_items)
            
            if self.archive:
                self.archive.archive(self._long_term_cache)
            
            self._long_term_cache.clear()
            
            if summary:
                self.remember(summary, long_term=True)
        else:
            # Сортируем по важности и оставляем только max_items
            sorted_entries = sorted(
                self._long_term_cache,
                key=lambda x: x.timestamp or 0,
                reverse=True
            )
            
            to_remove = sorted_entries[max_items:]
            if self.archive:
                self.archive.archive(to_remove)
            
            self._long_term_cache = sorted_entries[:max_items]
        
        self._cache_dirty = True
    
    # ======================================================================
    # Вспомогательные методы
    # ======================================================================
    
    def _ensure_cache_loaded(self) -> None:
        """Загружает кэш из MemoryService если он устарел."""
        if not self._cache_dirty:
            return
        
        self._short_term_cache = []
        try:
            for content in self.memory_service.read_events():
                entry = MemoryEntry(
                    text=content,
                    embedding=self._embed(content),
                    timestamp=datetime.now(timezone.utc).timestamp(),
                )
                self._short_term_cache.append(entry)
        except Exception as exc:
            logger.warning(f"Ошибка при загрузке краткосрочной памяти: {exc}")
        
        self._long_term_cache = []
        try:
            for content in self.memory_service.recall(long_term=True):
                entry = MemoryEntry(
                    text=content,
                    embedding=self._embed(content),
                    timestamp=datetime.now(timezone.utc).timestamp(),
                )
                self._long_term_cache.append(entry)
        except Exception as exc:
            logger.warning(f"Ошибка при загрузке долгосрочной памяти: {exc}")
        
        self._cache_dirty = False
    
    def _embed(self, text: str) -> List[float]:
        """Создание эмбеддинга для текста."""
        if self.embeddings_client:
            try:
                return self._deterministic_embed(text)
            except Exception as exc:
                logger.warning(f"Ошибка создания эмбеддинга: {exc}")
                return self._deterministic_embed(text)
        return self._deterministic_embed(text)
    
    def _deterministic_embed(self, text: str) -> List[float]:
        """Детерминированное создание эмбеддинга."""
        import hashlib
        import numpy as np
        from numpy.typing import NDArray
        
        tokens = text.lower().split()
        dim = 64
        vec: NDArray[np.float64] = np.zeros(dim, dtype=float)
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return [float(x) for x in vec]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Вычисление косинусного сходства между двумя векторами."""
        return float(cosine_similarity(vec1, vec2))
    
    def _summarize_to_long_term(self) -> None:
        """Суммаризация краткосрочной памяти в долгосрочную."""
        if not self._short_term_cache:
            return
        
        texts = [entry.text for entry in self._short_term_cache]
        summary = self._summarize_texts(texts)
        
        if summary:
            self.remember(summary, long_term=True)
        
        self._short_term_cache.clear()
        self._cache_dirty = True
    
    def _summarize_texts(self, texts: List[str]) -> str:
        """Суммаризация списка текстов."""
        if not texts:
            return ""
        
        if len(texts) == 1:
            return texts[0]
        
        # Простая суммаризация - объединяем тексты
        return " | ".join(texts[:5])  # Ограничиваем количество для читаемости
    
    # ======================================================================
    # MCP сервер интеграция (синхронные методы)
    # ======================================================================
    
    # ======================================================================
    # Свойства для совместимости
    # ======================================================================
    
    @property
    def graph(self):
        """Expose underlying EpisodeGraph for legacy tests."""
        return self.memory_service.graph
    
    def _legacy_save(self) -> None:
        """Legacy no-op: kept for API compatibility with MemoryStore."""
        return None
