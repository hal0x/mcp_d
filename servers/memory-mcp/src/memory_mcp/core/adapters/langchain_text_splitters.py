"""LangChain Text Splitters для разбивки текста на чанки."""

from __future__ import annotations

import logging
from typing import List, Optional

try:
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        TokenTextSplitter,
        CharacterTextSplitter,
    )
    # SemanticChunker может быть недоступен в старых версиях
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        SemanticChunker = None  # type: ignore
except ImportError:
    RecursiveCharacterTextSplitter = None  # type: ignore
    TokenTextSplitter = None  # type: ignore
    CharacterTextSplitter = None  # type: ignore
    SemanticChunker = None  # type: ignore

logger = logging.getLogger(__name__)


class LangChainTextSplitter:
    """Обертка над LangChain Text Splitters для разбивки текста."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        use_semantic: bool = False,
        embeddings=None,  # Для SemanticChunker
    ):
        """Инициализация text splitter.
        
        Args:
            chunk_size: Максимальный размер чанка (символов или токенов)
            chunk_overlap: Перекрытие между чанками
            separators: Список разделителей для рекурсивной разбивки
            use_semantic: Использовать семантическую разбивку (требует embeddings)
            embeddings: Сервис эмбеддингов для SemanticChunker
        """
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "LangChain не установлен. Установите: pip install langchain"
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic = use_semantic and SemanticChunker is not None and embeddings is not None
        
        if self.use_semantic:
            self.splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
            )
        else:
            # Используем рекурсивную разбивку по умолчанию
            if separators is None:
                separators = ["\n\n", "\n", ". ", " ", ""]
            
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
            )

    def split_text(self, text: str) -> List[str]:
        """Разбивка текста на чанки.
        
        Args:
            text: Текст для разбивки
            
        Returns:
            Список чанков текста
        """
        if not text:
            return []
        
        try:
            chunks = self.splitter.split_text(text)
            logger.debug(f"Текст разбит на {len(chunks)} чанков")
            return chunks
        except Exception as e:
            logger.warning(f"Ошибка при разбивке текста: {e}, возвращаем весь текст")
            return [text]

    def split_documents(self, documents: List[str]) -> List[List[str]]:
        """Разбивка нескольких документов на чанки.
        
        Args:
            documents: Список документов для разбивки
            
        Returns:
            Список списков чанков (по одному списку на документ)
        """
        return [self.split_text(doc) for doc in documents]


def create_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_langchain: bool = True,
    embeddings=None,
) -> LangChainTextSplitter | Any:
    """Фабричная функция для создания text splitter.
    
    Args:
        chunk_size: Максимальный размер чанка
        chunk_overlap: Перекрытие между чанками
        use_langchain: Использовать LangChain splitter (True) или простую реализацию (False)
        embeddings: Сервис эмбеддингов для семантической разбивки
        
    Returns:
        LangChainTextSplitter или простая реализация
    """
    if use_langchain and RecursiveCharacterTextSplitter is not None:
        try:
            return LangChainTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                use_semantic=embeddings is not None,
                embeddings=embeddings,
            )
        except Exception as e:
            logger.warning(f"Не удалось создать LangChain text splitter: {e}, используем простую реализацию")
    
    # Простая реализация как fallback
    class SimpleTextSplitter:
        def __init__(self, chunk_size: int, chunk_overlap: int):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        
        def split_text(self, text: str) -> List[str]:
            if not text or len(text) <= self.chunk_size:
                return [text]
            
            chunks = []
            start = 0
            
            while start < len(text):
                end = start + self.chunk_size
                if end >= len(text):
                    chunks.append(text[start:])
                    break
                
                # Ищем последний пробел
                last_space = text.rfind(" ", start, end)
                if last_space > start + self.chunk_size * 0.7:
                    chunks.append(text[start:last_space])
                    start = last_space + 1 - self.chunk_overlap
                else:
                    chunks.append(text[start:end])
                    start = end - self.chunk_overlap
                
                if start < 0:
                    start = 0
            
            return chunks
    
    return SimpleTextSplitter(chunk_size, chunk_overlap)

