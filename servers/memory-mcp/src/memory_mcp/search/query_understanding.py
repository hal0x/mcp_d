#!/usr/bin/env python3
"""
Движок понимания запросов (Query Understanding Engine).

Использует LLM для глубокого понимания запросов:
- Декомпозиция сложных запросов на подзапросы
- Извлечение неявных требований
- Генерация альтернативных формулировок
- Определение ключевых концепций и их связей
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..core.langchain_adapters import LangChainLLMAdapter, get_llm_client_factory
from ..core.lmql_adapter import LMQLAdapter, build_lmql_adapter_from_env

logger = logging.getLogger(__name__)


@dataclass
class QueryUnderstanding:
    """Результат понимания запроса."""

    original_query: str
    sub_queries: List[str]
    implicit_requirements: List[str]
    alternative_formulations: List[str]
    key_concepts: List[str]
    concept_relationships: Dict[str, List[str]]
    enhanced_query: str

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь."""
        return {
            "original_query": self.original_query,
            "sub_queries": self.sub_queries,
            "implicit_requirements": self.implicit_requirements,
            "alternative_formulations": self.alternative_formulations,
            "key_concepts": self.key_concepts,
            "concept_relationships": self.concept_relationships,
            "enhanced_query": self.enhanced_query,
        }


class QueryUnderstandingEngine:
    """Движок для глубокого понимания запросов."""

    def __init__(self, lmql_adapter: Optional[LMQLAdapter] = None):
        """Инициализация движка понимания запросов.
        
        Args:
            lmql_adapter: Опциональный LMQL адаптер для структурированной генерации.
                         Если не указан, создается из настроек окружения.
        """
        self._llm_client: Optional[LangChainLLMAdapter] = None
        try:
            self.lmql_adapter = lmql_adapter or build_lmql_adapter_from_env()
        except RuntimeError:
            self.lmql_adapter = None

    def _get_llm_client(self) -> Optional[LangChainLLMAdapter]:
        """Получение или создание LLM клиента."""
        if self._llm_client is not None:
            return self._llm_client

        try:
            settings = get_settings()

            if settings.lmstudio_llm_model:
                self._llm_client = LangChainLLMAdapter(
                    model_name=settings.lmstudio_model,
                    llm_model_name=settings.lmstudio_llm_model,
                    base_url=f"http://{settings.lmstudio_host}:{settings.lmstudio_port}",
                )
                logger.debug("Используется LM Studio для понимания запросов")
            else:
                logger.error("LM Studio не настроен, понимание запросов невозможно")
                raise ValueError("Для понимания запросов требуется настроенный LM Studio (MEMORY_MCP_LMSTUDIO_LLM_MODEL)")
        except Exception as e:
            logger.error(f"Не удалось инициализировать LLM клиент для понимания запросов: {e}")
            raise

        return self._llm_client

    async def understand_query(self, query: str) -> QueryUnderstanding:
        """Глубокое понимание запроса с декомпозицией и улучшениями."""
        if not query or not query.strip():
            return QueryUnderstanding(
                original_query=query,
                sub_queries=[],
                implicit_requirements=[],
                alternative_formulations=[],
                key_concepts=[],
                concept_relationships={},
                enhanced_query=query,
            )

        # Используем LMQL для понимания запроса
        if self.lmql_adapter:
            logger.debug("Используется LMQL для понимания запроса")
            return await self._understand_query_with_lmql(query)

        # Если LMQL недоступен, выбрасываем ошибку
        raise RuntimeError(
            "LMQL не настроен. Установите MEMORY_MCP_USE_LMQL=true и настройте модель"
        )

    async def _understand_query_with_lmql(self, query: str) -> QueryUnderstanding:
        """Понимание запроса с использованием LMQL.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            QueryUnderstanding
            
        Raises:
            RuntimeError: Если произошла ошибка при выполнении запроса
        """

        try:
            # Создаем промпт для LMQL
            prompt = f"""Проанализируй поисковый запрос и выполни глубокий анализ.

Запрос: "{query}"

Выполни следующие задачи:
1. Декомпозиция: разбей сложный запрос на простые подзапросы (если запрос простой, верни его как единственный подзапрос)
2. Неявные требования: определи, что пользователь может иметь в виду, но не сказал явно
3. Альтернативные формулировки: предложи 2-3 альтернативных способа сформулировать этот запрос
4. Ключевые концепции: выдели основные концепции и термины из запроса
5. Связи концепций: определи, как концепции связаны между собой"""

            # Создаем JSON схему с переменными
            json_schema = """{{
    "sub_queries": [SUB_QUERIES],
    "implicit_requirements": [IMPLICIT_REQ],
    "alternative_formulations": [ALT_FORM],
    "key_concepts": [KEY_CONCEPTS],
    "concept_relationships": {{CONCEPT_REL}},
    "enhanced_query": "[ENHANCED_QUERY]"
}}"""

            # Ограничения для переменных
            constraints = """
    len(SUB_QUERIES) <= 5 and
    len(IMPLICIT_REQ) <= 5 and
    len(ALT_FORM) <= 3 and
    len(KEY_CONCEPTS) <= 10
"""

            # Выполняем LMQL запрос
            result = await self.lmql_adapter.execute_json_query(
                prompt=prompt,
                json_schema=json_schema,
                constraints=constraints,
                temperature=0.3,
                max_tokens=2048,
            )

            if not result:
                return None

            # Преобразуем результат в QueryUnderstanding
            sub_queries = result.get("sub_queries", [])
            implicit_requirements = result.get("implicit_requirements", [])
            alternative_formulations = result.get("alternative_formulations", [])
            key_concepts = result.get("key_concepts", [])
            concept_relationships = result.get("concept_relationships", {})
            enhanced_query = result.get("enhanced_query", query)

            # Валидация и нормализация
            if not isinstance(sub_queries, list):
                sub_queries = []
            if not isinstance(implicit_requirements, list):
                implicit_requirements = []
            if not isinstance(alternative_formulations, list):
                alternative_formulations = []
            if not isinstance(key_concepts, list):
                key_concepts = []
            if not isinstance(concept_relationships, dict):
                concept_relationships = {}
            if not isinstance(enhanced_query, str):
                enhanced_query = query

            return QueryUnderstanding(
                original_query=query,
                sub_queries=sub_queries[:5],
                implicit_requirements=implicit_requirements[:5],
                alternative_formulations=alternative_formulations[:3],
                key_concepts=key_concepts[:10],
                concept_relationships=concept_relationships,
                enhanced_query=enhanced_query,
            )

        except Exception as e:
            logger.error(f"Ошибка при использовании LMQL для понимания запроса: {e}")
            raise RuntimeError(f"Ошибка понимания запроса через LMQL: {e}") from e


