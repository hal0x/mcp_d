#!/usr/bin/env python3
"""
Движок понимания запросов (Query Understanding Engine).

Использует LLM для глубокого понимания запросов:
- Декомпозиция сложных запросов на подзапросы
- Извлечение неявных требований
- Генерация альтернативных формулировок
- Определение ключевых концепций и их связей
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..core.langchain_adapters import LangChainLLMAdapter, get_llm_client_factory

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

    def __init__(self):
        """Инициализация движка понимания запросов."""
        self._llm_client: Optional[LangChainLLMAdapter] = None

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

        llm_client = self._get_llm_client()
        if not llm_client:
            logger.error("LLM клиент не настроен, понимание запросов невозможно")
            raise ValueError("Для понимания запросов требуется настроенный LLM клиент")

        # Используем LLM для глубокого понимания
        prompt = self._create_understanding_prompt(query)
        
        # Используем максимальное значение модели (131072 для gpt-oss-20b)
        settings = get_settings()
        max_tokens = settings.large_context_max_tokens  # 131072
        
        async with llm_client:
            response = await llm_client.generate_summary(
                prompt=prompt,
                temperature=0.3,
                max_tokens=max_tokens,  # Максимальное значение модели
                top_p=0.9,
            )

        # Парсим ответ LLM
        understanding = self._parse_llm_response(response, query)
        if not understanding:
            logger.error("Не удалось распарсить ответ LLM для понимания запроса")
            raise ValueError("LLM вернул некорректный ответ для понимания запроса")
        
        return understanding

    def _create_understanding_prompt(self, query: str) -> str:
        """Создание промпта для понимания запроса"""
        return f"""Проанализируй поисковый запрос и выполни глубокий анализ.

Запрос: "{query}"

Выполни следующие задачи:
1. Декомпозиция: разбей сложный запрос на простые подзапросы (если запрос простой, верни его как единственный подзапрос)
2. Неявные требования: определи, что пользователь может иметь в виду, но не сказал явно
3. Альтернативные формулировки: предложи 2-3 альтернативных способа сформулировать этот запрос
4. Ключевые концепции: выдели основные концепции и термины из запроса
5. Связи концепций: определи, как концепции связаны между собой

Верни ответ в формате JSON:
{{
    "sub_queries": ["подзапрос 1", "подзапрос 2"],
    "implicit_requirements": ["требование 1", "требование 2"],
    "alternative_formulations": ["вариант 1", "вариант 2"],
    "key_concepts": ["концепция 1", "концепция 2"],
    "concept_relationships": {{
        "концепция 1": ["связанная концепция 1", "связанная концепция 2"]
    }},
    "enhanced_query": "улучшенная формулировка запроса"
}}

Только JSON, без дополнительного текста."""

    def _parse_llm_response(self, response: str, query: str) -> Optional[QueryUnderstanding]:
        """Парсинг ответа LLM"""
        try:
            response = response.strip()
            # Убираем markdown code blocks, если есть
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()

            data = json.loads(response)
            
            sub_queries = data.get("sub_queries", [])
            implicit_requirements = data.get("implicit_requirements", [])
            alternative_formulations = data.get("alternative_formulations", [])
            key_concepts = data.get("key_concepts", [])
            concept_relationships = data.get("concept_relationships", {})
            enhanced_query = data.get("enhanced_query", query)

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
                sub_queries=sub_queries[:5],  # Максимум 5 подзапросов
                implicit_requirements=implicit_requirements[:5],  # Максимум 5 требований
                alternative_formulations=alternative_formulations[:3],  # Максимум 3 варианта
                key_concepts=key_concepts[:10],  # Максимум 10 концепций
                concept_relationships=concept_relationships,
                enhanced_query=enhanced_query,
            )

        except Exception as e:
            logger.debug(f"Ошибка парсинга ответа LLM для понимания запроса: {e}")
            return None


