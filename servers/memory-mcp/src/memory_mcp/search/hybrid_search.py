#!/usr/bin/env python3
"""
Гибридный поиск: BM25 + Векторный поиск с Reciprocal Rank Fusion

Вдохновлено архитектурой памяти HALv1:
- Векторный поиск (FAISS/HNSW) для семантического сходства
- BM25 для точного текстового поиска
- RRF (Reciprocal Rank Fusion) для объединения результатов
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from rank_bm25 import BM25Okapi

from ..utils.russian_tokenizer import get_tokenizer

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Гибридный поисковый движок с BM25 + Vector search"""

    def __init__(
        self,
        chroma_collection: chromadb.Collection,
        alpha: float = 0.6,
        k: int = 60,
    ):
        """
        Инициализация гибридного поиска

        Args:
            chroma_collection: Коллекция для векторного поиска
            alpha: Вес векторного поиска (0-1), остальное - BM25
            k: Параметр для Reciprocal Rank Fusion (по умолчанию 60)
        """
        self.collection = chroma_collection
        self.alpha = alpha  # Вес векторного поиска
        self.k = k  # RRF параметр

        # BM25 индекс
        self.bm25_index: Optional[BM25Okapi] = None
        self.documents_map: Dict[str, Any] = {}  # id -> полный документ
        self.tokenized_corpus: List[List[str]] = []

        logger.info(
            f"Инициализирован HybridSearchEngine (alpha={alpha}, k={k}) "
            f"для коллекции {chroma_collection.name}"
        )

    def build_bm25_index(self, force_rebuild: bool = False):
        """
        Построение BM25 индекса из коллекции

        Args:
            force_rebuild: Принудительная пересборка индекса
        """
        if self.bm25_index is not None and not force_rebuild:
            logger.info("BM25 индекс уже построен")
            return

        logger.info("Построение BM25 индекса...")

        try:
            # Получаем все документы из коллекции
            result = self.collection.get(include=["documents", "metadatas"])

            if not result["documents"]:
                logger.warning("Нет документов для индексации")
                return

            # Токенизируем документы для BM25
            self.tokenized_corpus = []
            self.documents_map = {}

            for doc_id, doc_text, metadata in zip(
                result["ids"], result["documents"], result["metadatas"]
            ):
                # Простая токенизация (можно улучшить для русского языка)
                tokens = self._tokenize(doc_text)
                self.tokenized_corpus.append(tokens)

                # Сохраняем документ с метаданными
                self.documents_map[doc_id] = {
                    "id": doc_id,
                    "text": doc_text,
                    "metadata": metadata,
                }

            # Создаём BM25 индекс
            self.bm25_index = BM25Okapi(self.tokenized_corpus)

            logger.info(
                f"BM25 индекс построен: {len(self.tokenized_corpus)} документов"
            )

        except Exception as e:
            logger.error(f"Ошибка при построении BM25 индекса: {e}")
            raise

    def _tokenize(self, text: str) -> List[str]:
        """
        Токенизация текста с использованием russian_tokenizer для улучшенной обработки русского языка.

        Использует морфологическую обработку для нормализации слов.
        Fallback на простую токенизацию, если russian_tokenizer недоступен.

        Args:
            text: Текст для токенизации

        Returns:
            Список токенов
        """
        if not text:
            return []
        
        try:
            # Используем russian_tokenizer для улучшенной токенизации
            tokenizer = get_tokenizer()
            tokens = tokenizer.tokenize(text)
            return tokens
        except Exception as e:
            # Fallback на простую токенизацию
            logger.debug(f"Ошибка при токенизации через russian_tokenizer, используем fallback: {e}")
            import re
            text = text.lower()
            tokens = re.findall(r"\b\w+\b", text)
            return tokens

    def _bm25_search(self, query: str, top_k: int = 200) -> List[Tuple[str, float]]:
        """
        BM25 поиск

        Args:
            query: Поисковый запрос
            top_k: Количество результатов

        Returns:
            Список (doc_id, score)
        """
        if self.bm25_index is None:
            logger.warning("BM25 индекс не построен, выполняется построение...")
            self.build_bm25_index()

        if self.bm25_index is None:
            return []

        # Токенизируем запрос
        query_tokens = self._tokenize(query)

        # Получаем BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)

        # Сортируем по убыванию
        doc_ids = list(self.documents_map.keys())
        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)

        return ranked[:top_k]

    def _vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 200,
        where: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        """
        Векторный поиск

        Args:
            query_embedding: Эмбеддинг запроса
            top_k: Количество результатов
            where: Фильтр метаданных

        Returns:
            Список (doc_id, distance)
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            # Конвертируем distance в similarity (меньше distance = больше similarity)
            # Конвертируем L2 distance в similarity score
            ranked = []
            for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
                # Конвертация: similarity = 1 / (1 + distance)
                similarity = 1.0 / (1.0 + distance)
                ranked.append((doc_id, similarity))

            return ranked

        except Exception as e:
            logger.error(f"Ошибка векторного поиска: {e}")
            return []

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion (RRF)

        Формула: score = alpha * (1/(k + rank_vec)) + (1-alpha) * (1/(k + rank_bm25))

        Args:
            vector_results: Результаты векторного поиска [(doc_id, score), ...]
            bm25_results: Результаты BM25 поиска [(doc_id, score), ...]

        Returns:
            Объединённые результаты [(doc_id, fused_score), ...]
        """
        fused_scores = defaultdict(float)

        # Добавляем scores из векторного поиска
        for rank, (doc_id, _score) in enumerate(vector_results):
            fused_scores[doc_id] += self.alpha / (self.k + rank + 1)

        # Добавляем scores из BM25
        for rank, (doc_id, _score) in enumerate(bm25_results):
            fused_scores[doc_id] += (1.0 - self.alpha) / (self.k + rank + 1)

        # Сортируем по финальному score
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        return ranked

    def search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict] = None,
        use_hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Гибридный поиск

        Args:
            query: Текстовый запрос (для BM25)
            query_embedding: Эмбеддинг запроса (для векторного поиска)
            top_k: Количество результатов
            where: Фильтр метаданных
            use_hybrid: Использовать гибридный поиск (True) или только векторный (False)

        Returns:
            Список результатов с метаданными
        """
        if not use_hybrid or self.bm25_index is None:
            # Только векторный поиск
            logger.info("Используется только векторный поиск")
            vector_results = self._vector_search(
                query_embedding, top_k=top_k, where=where
            )
            final_results = vector_results
        else:
            # Гибридный поиск
            logger.info("Используется гибридный поиск (BM25 + Vector)")

            # 1. Векторный поиск (top 200)
            vector_results = self._vector_search(
                query_embedding, top_k=200, where=where
            )

            # 2. BM25 поиск (top 200)
            bm25_results = self._bm25_search(query, top_k=200)

            # Фильтруем BM25 результаты по where условию если задано
            if where:
                bm25_results = self._filter_bm25_by_where(bm25_results, where)

            # 3. Reciprocal Rank Fusion
            final_results = self._reciprocal_rank_fusion(vector_results, bm25_results)

            logger.info(
                f"Гибридный поиск: {len(vector_results)} vector + "
                f"{len(bm25_results)} BM25 → {len(final_results)} fused"
            )

        # Формируем финальный результат с полными данными
        results = []
        for doc_id, score in final_results[:top_k]:
            if doc_id in self.documents_map:
                doc = self.documents_map[doc_id].copy()
                doc["score"] = score
                results.append(doc)

        return results

    def _filter_bm25_by_where(
        self, bm25_results: List[Tuple[str, float]], where: Dict
    ) -> List[Tuple[str, float]]:
        """
        Фильтрация BM25 результатов по условию where

        Args:
            bm25_results: Результаты BM25
            where: Условие фильтрации (например, {"chat": "chatname"})

        Returns:
            Отфильтрованные результаты
        """
        filtered = []
        for doc_id, score in bm25_results:
            if doc_id in self.documents_map:
                metadata = self.documents_map[doc_id]["metadata"]
                # Проверяем соответствие условию
                match = True
                for key, value in where.items():
                    if metadata.get(key) != value:
                        match = False
                        break
                if match:
                    filtered.append((doc_id, score))

        return filtered

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Статистика индексов

        Returns:
            Словарь со статистикой
        """
        return {
            "collection_name": self.collection.name,
            "total_documents": len(self.documents_map),
            "bm25_indexed": self.bm25_index is not None,
            "alpha": self.alpha,
            "k": self.k,
        }


class HybridSearchManager:
    """Менеджер для управления несколькими гибридными поисковыми движками"""

    def __init__(self, chroma_client: chromadb.Client, alpha: float = 0.6):
        """
        Инициализация менеджера

        Args:
            chroma_client: Клиент векторного хранилища
            alpha: Вес векторного поиска (для всех коллекций)
        """
        self.chroma_client = chroma_client
        self.alpha = alpha
        self.engines: Dict[str, HybridSearchEngine] = {}

        logger.info(f"Инициализирован HybridSearchManager (alpha={alpha})")

    def get_engine(
        self, collection_name: str, build_index: bool = True
    ) -> HybridSearchEngine:
        """
        Получить или создать поисковый движок для коллекции

        Args:
            collection_name: Имя коллекции
            build_index: Построить BM25 индекс сразу

        Returns:
            Поисковый движок
        """
        if collection_name not in self.engines:
            try:
                collection = self.chroma_client.get_collection(collection_name)
                engine = HybridSearchEngine(collection, alpha=self.alpha)

                if build_index:
                    engine.build_bm25_index()

                self.engines[collection_name] = engine
                logger.info(f"Создан движок для коллекции {collection_name}")

            except Exception as e:
                logger.error(f"Ошибка создания движка для {collection_name}: {e}")
                raise

        return self.engines[collection_name]

    def rebuild_all_indexes(self):
        """Пересборка всех BM25 индексов"""
        logger.info("Пересборка всех BM25 индексов...")
        for collection_name, engine in self.engines.items():
            logger.info(f"Пересборка индекса для {collection_name}")
            engine.build_bm25_index(force_rebuild=True)

    def get_stats(self) -> Dict[str, Any]:
        """
        Статистика всех движков

        Returns:
            Словарь со статистикой
        """
        return {
            "total_engines": len(self.engines),
            "alpha": self.alpha,
            "engines": {
                name: engine.get_index_stats() for name, engine in self.engines.items()
            },
        }
