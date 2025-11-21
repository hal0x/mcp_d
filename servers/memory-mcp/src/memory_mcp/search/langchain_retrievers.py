"""LangChain Retrievers для гибридного поиска."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    from langchain.retrievers import BM25Retriever, EnsembleRetriever
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError:
    BM25Retriever = None  # type: ignore
    EnsembleRetriever = None  # type: ignore
    ContextualCompressionRetriever = None  # type: ignore
    LLMChainExtractor = None  # type: ignore
    Document = None  # type: ignore
    BaseRetriever = None  # type: ignore

logger = logging.getLogger(__name__)


class HybridMemoryRetriever:
    """Гибридный ретривер на базе LangChain для поиска в памяти.
    
    Объединяет BM25 и векторный поиск через EnsembleRetriever.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: BaseRetriever,
        weights: List[float] = [0.4, 0.6],
        use_compression: bool = False,
        llm=None,  # Для ContextualCompressionRetriever
    ):
        """Инициализация гибридного ретривера.
        
        Args:
            bm25_retriever: BM25 ретривер для текстового поиска
            vector_retriever: Векторный ретривер (Qdrant)
            weights: Веса для ретриверов [bm25_weight, vector_weight]
            use_compression: Использовать contextual compression
            llm: LLM для compression (если use_compression=True)
        """
        if EnsembleRetriever is None:
            raise ImportError(
                "LangChain не установлен. Установите: pip install langchain"
            )
        
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.weights = weights
        self.use_compression = use_compression
        
        # Создаем ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=weights,
        )
        
        # Опционально добавляем compression
        if use_compression and llm is not None and ContextualCompressionRetriever is not None:
            compressor = LLMChainExtractor.from_llm(llm)
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.ensemble_retriever,
            )
        else:
            self.retriever = self.ensemble_retriever

    def get_relevant_documents(self, query: str, *, top_k: int = 10) -> List[Document]:
        """Получение релевантных документов для запроса.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            
        Returns:
            Список релевантных документов
        """
        try:
            # LangChain retrievers используют get_relevant_documents
            results = self.retriever.get_relevant_documents(query)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Ошибка при поиске через LangChain retriever: {e}")
            return []

    async def aget_relevant_documents(self, query: str, *, top_k: int = 10) -> List[Document]:
        """Асинхронное получение релевантных документов.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            
        Returns:
            Список релевантных документов
        """
        try:
            if hasattr(self.retriever, "aget_relevant_documents"):
                results = await self.retriever.aget_relevant_documents(query)
                return results[:top_k]
            else:
                # Fallback на синхронный метод
                return self.get_relevant_documents(query, top_k=top_k)
        except Exception as e:
            logger.error(f"Ошибка при асинхронном поиске через LangChain retriever: {e}")
            return []


def create_bm25_retriever_from_documents(
    documents: List[Document],
    k: int = 10,
) -> Optional[BM25Retriever]:
    """Создание BM25 ретривера из документов.
    
    Args:
        documents: Список LangChain Document объектов
        k: Количество результатов по умолчанию
        
    Returns:
        BM25Retriever или None
    """
    if BM25Retriever is None:
        logger.warning("BM25Retriever не доступен")
        return None
    
    try:
        retriever = BM25Retriever.from_documents(documents)
        retriever.k = k
        return retriever
    except Exception as e:
        logger.error(f"Ошибка при создании BM25 ретривера: {e}")
        return None


def create_hybrid_retriever(
    bm25_retriever: BM25Retriever,
    vector_retriever: BaseRetriever,
    weights: List[float] = [0.4, 0.6],
    use_compression: bool = False,
    llm=None,
) -> Optional[HybridMemoryRetriever]:
    """Создание гибридного ретривера.
    
    Args:
        bm25_retriever: BM25 ретривер
        vector_retriever: Векторный ретривер
        weights: Веса для ретриверов
        use_compression: Использовать compression
        llm: LLM для compression
        
    Returns:
        HybridMemoryRetriever или None
    """
    if EnsembleRetriever is None:
        logger.warning("EnsembleRetriever не доступен")
        return None
    
    try:
        return HybridMemoryRetriever(
            bm25_retriever=bm25_retriever,
            vector_retriever=vector_retriever,
            weights=weights,
            use_compression=use_compression,
            llm=llm,
        )
    except Exception as e:
        logger.error(f"Ошибка при создании гибридного ретривера: {e}")
        return None

