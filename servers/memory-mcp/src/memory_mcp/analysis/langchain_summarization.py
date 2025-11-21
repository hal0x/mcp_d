"""LangChain Summarization Chains для саммаризации."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    from langchain.chains.summarize import load_summarize_chain
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_core.documents import Document
    from langchain_core.language_models.chat_models import BaseChatModel
except ImportError:
    load_summarize_chain = None  # type: ignore
    LLMChain = None  # type: ignore
    PromptTemplate = None  # type: ignore
    Document = None  # type: ignore
    BaseChatModel = None  # type: ignore

logger = logging.getLogger(__name__)


class LangChainSummarizationChain:
    """Обертка над LangChain summarize chains для саммаризации."""

    def __init__(
        self,
        llm: BaseChatModel,
        chain_type: str = "map_reduce",
        verbose: bool = False,
        custom_prompt: Optional[str] = None,
    ):
        """Инициализация summarization chain.
        
        Args:
            llm: LangChain LLM для генерации
            chain_type: Тип цепочки ("stuff", "map_reduce", "refine")
            verbose: Включить подробное логирование
            custom_prompt: Кастомный промпт для саммаризации
        """
        if load_summarize_chain is None:
            raise ImportError(
                "LangChain не установлен. Установите: pip install langchain"
            )
        
        self.llm = llm
        self.chain_type = chain_type
        self.verbose = verbose
        
        # Создаем промпты
        if custom_prompt:
            prompt = PromptTemplate.from_template(custom_prompt)
        else:
            prompt = None
        
        # Загружаем цепочку
        try:
            self.chain = load_summarize_chain(
                llm=llm,
                chain_type=chain_type,
                verbose=verbose,
                prompt=prompt,
            )
        except Exception as e:
            logger.error(f"Ошибка при создании summarization chain: {e}")
            raise

    def summarize(
        self,
        text: str,
        max_chunk_size: int = 10000,
    ) -> str:
        """Саммаризация текста.
        
        Args:
            text: Текст для саммаризации
            max_chunk_size: Максимальный размер чанка (для map_reduce и refine)
            
        Returns:
            Саммаризация текста
        """
        if not text:
            return ""
        
        try:
            # Создаем Document из текста
            if Document is None:
                logger.error("LangChain Document не доступен")
                return text
            
            # Для "stuff" просто передаем весь текст
            if self.chain_type == "stuff":
                docs = [Document(page_content=text)]
            else:
                # Для map_reduce и refine разбиваем на чанки
                from ..core.langchain_text_splitters import create_text_splitter
                splitter = create_text_splitter(
                    chunk_size=max_chunk_size,
                    chunk_overlap=200,
                    use_langchain=True,
                )
                chunks = splitter.split_text(text)
                docs = [Document(page_content=chunk) for chunk in chunks]
            
            # Выполняем саммаризацию
            result = self.chain.run(docs)
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            logger.error(f"Ошибка при саммаризации через LangChain: {e}")
            return text

    async def asummarize(
        self,
        text: str,
        max_chunk_size: int = 10000,
    ) -> str:
        """Асинхронная саммаризация текста.
        
        Args:
            text: Текст для саммаризации
            max_chunk_size: Максимальный размер чанка
            
        Returns:
            Саммаризация текста
        """
        if not text:
            return ""
        
        try:
            if Document is None:
                logger.error("LangChain Document не доступен")
                return text
            
            # Создаем Document из текста
            if self.chain_type == "stuff":
                docs = [Document(page_content=text)]
            else:
                from ..core.langchain_text_splitters import create_text_splitter
                splitter = create_text_splitter(
                    chunk_size=max_chunk_size,
                    chunk_overlap=200,
                    use_langchain=True,
                )
                chunks = splitter.split_text(text)
                docs = [Document(page_content=chunk) for chunk in chunks]
            
            # Асинхронное выполнение
            if hasattr(self.chain, "arun"):
                result = await self.chain.arun(docs)
            elif hasattr(self.chain, "ainvoke"):
                result = await self.chain.ainvoke({"input_documents": docs})
            else:
                # Fallback на синхронный метод
                result = self.chain.run(docs)
            
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            logger.error(f"Ошибка при асинхронной саммаризации через LangChain: {e}")
            return text


def create_summarization_chain(
    llm: BaseChatModel,
    chain_type: str = "map_reduce",
    verbose: bool = False,
    custom_prompt: Optional[str] = None,
) -> Optional[LangChainSummarizationChain]:
    """Фабричная функция для создания summarization chain.
    
    Args:
        llm: LangChain LLM
        chain_type: Тип цепочки ("stuff", "map_reduce", "refine")
        verbose: Включить логирование
        custom_prompt: Кастомный промпт
        
    Returns:
        LangChainSummarizationChain или None
    """
    if load_summarize_chain is None:
        logger.warning("LangChain summarization chains не доступны")
        return None
    
    try:
        return LangChainSummarizationChain(
            llm=llm,
            chain_type=chain_type,
            verbose=verbose,
            custom_prompt=custom_prompt,
        )
    except Exception as e:
        logger.error(f"Ошибка при создании summarization chain: {e}")
        return None

