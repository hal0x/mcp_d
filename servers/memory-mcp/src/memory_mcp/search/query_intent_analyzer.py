#!/usr/bin/env python3
"""
Анализатор намерения запроса (Query Intent Classification).

Классифицирует поисковые запросы на типы намерений и адаптирует стратегию поиска.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..config import get_settings
from ..core.adapters.langchain_adapters import LangChainLLMAdapter, get_llm_client_factory
from ..core.adapters.lmql_adapter import LMQLAdapter, build_lmql_adapter_from_env

logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """Результат анализа намерения запроса."""

    intent_type: str
    confidence: float
    recommended_db_weight: float
    recommended_artifact_weight: float
    recommended_top_k: Optional[int] = None
    recommended_filters: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь."""
        return {
            "intent_type": self.intent_type,
            "confidence": self.confidence,
            "recommended_db_weight": self.recommended_db_weight,
            "recommended_artifact_weight": self.recommended_artifact_weight,
            "recommended_top_k": self.recommended_top_k,
            "recommended_filters": self.recommended_filters or {},
        }


class QueryIntentAnalyzer:
    """Анализатор намерения запроса для адаптации стратегии поиска."""

    def __init__(self, lmql_adapter: Optional[LMQLAdapter] = None):
        """Инициализация анализатора намерений.
        
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
                logger.debug("Используется LM Studio для анализа намерения")
            else:
                logger.error("LM Studio не настроен, анализ намерения невозможен")
                raise ValueError("Для анализа намерения требуется настроенный LM Studio (MEMORY_MCP_LMSTUDIO_LLM_MODEL)")
        except Exception as e:
            logger.error(f"Не удалось инициализировать LLM клиент для анализа намерения: {e}")
            raise

        return self._llm_client

    async def analyze_intent(self, query: str) -> QueryIntent:
        """
        Анализ намерения запроса

        Args:
            query: Поисковый запрос

        Returns:
            QueryIntent с типом намерения и рекомендациями
        """
        if not query or not query.strip():
            # По умолчанию информационный поиск
            return QueryIntent(
                intent_type="informational",
                confidence=0.5,
                recommended_db_weight=0.6,
                recommended_artifact_weight=0.4,
            )

        # Используем LMQL для анализа намерения
        if self.lmql_adapter:
            logger.debug("Используется LMQL для анализа намерения запроса")
            return await self._analyze_intent_with_lmql(query)

        # Если LMQL недоступен, выбрасываем ошибку
        raise RuntimeError(
            "LMQL не настроен. Установите MEMORY_MCP_USE_LMQL=true и настройте модель"
        )

    async def _analyze_intent_with_lmql(self, query: str) -> QueryIntent:
        """Анализ намерения запроса с использованием LMQL.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            QueryIntent с типом намерения и рекомендациями
            
        Raises:
            RuntimeError: Если произошла ошибка при выполнении запроса
        """
        try:
            lmql_query_str = self._create_lmql_intent_query(query)
            response_data = await self.lmql_adapter.execute_json_query(
                prompt=f"""Проанализируй поисковый запрос и определи его намерение.

Запрос: "{query}"

Типы намерений:
1. informational - информационный поиск (что это?, расскажи о, информация о)
2. transactional - транзакционный поиск (как сделать, инструкция, шаги)
3. navigational - навигационный поиск (где найти, найти конкретную вещь)
4. analytical - аналитический поиск (почему, как работает, анализ, сравнение)""",
                json_schema='{"intent_type": "[INTENT_TYPE]", "confidence": [CONFIDENCE], "reasoning": "[REASONING]"}',
                constraints="""
                    INTENT_TYPE in ["informational", "transactional", "navigational", "analytical"] and
                    0.0 <= CONFIDENCE <= 1.0
                """,
                temperature=0.2,
                max_tokens=512,
            )

            return self._parse_lmql_response(response_data, query)

        except Exception as e:
            logger.error(f"Ошибка при использовании LMQL для анализа намерения: {e}")
            raise RuntimeError(f"Ошибка анализа намерения через LMQL: {e}") from e

    def _create_lmql_intent_query(self, query: str) -> str:
        """Создание LMQL запроса для анализа намерения.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Строка с LMQL запросом (для логирования/отладки)
        """
        # Этот метод возвращает строку для отладки, фактический запрос выполняется в execute_json_query
        return f"LMQL query for intent analysis: {query}"

    def _parse_lmql_response(self, data: Dict[str, Any], query: str) -> QueryIntent:
        """Парсинг ответа LMQL в QueryIntent.
        
        Args:
            data: Данные из LMQL ответа
            query: Оригинальный запрос
            
        Returns:
            QueryIntent объект
        """
        intent_type = data.get("intent_type", "informational")
        confidence = float(data.get("confidence", 0.7))

        # Генерируем рекомендации на основе типа намерения
        recommendations = self._generate_recommendations(intent_type, query)

        return QueryIntent(
            intent_type=intent_type,
            confidence=confidence,
            recommended_db_weight=recommendations["db_weight"],
            recommended_artifact_weight=recommendations["artifact_weight"],
            recommended_top_k=recommendations.get("top_k"),
            recommended_filters=recommendations.get("filters"),
        )

    def _generate_recommendations(
        self, intent_type: str, query: str
    ) -> Dict[str, Any]:
        """
        Генерация рекомендаций по параметрам поиска на основе типа намерения

        Args:
            intent_type: Тип намерения
            query: Поисковый запрос

        Returns:
            Словарь с рекомендациями
        """
        recommendations = {
            "db_weight": 0.6,
            "artifact_weight": 0.4,
            "top_k": None,
            "filters": {},
        }

        if intent_type == "informational":
            # Информационный поиск: больше артифактов (саммаризации, отчёты)
            recommendations["db_weight"] = 0.5
            recommendations["artifact_weight"] = 0.5
            recommendations["top_k"] = 15  # Больше результатов для информационного поиска

        elif intent_type == "transactional":
            # Транзакционный поиск: больше БД (конкретные сообщения с инструкциями)
            recommendations["db_weight"] = 0.7
            recommendations["artifact_weight"] = 0.3
            recommendations["top_k"] = 10

        elif intent_type == "navigational":
            # Навигационный поиск: больше БД (конкретные записи)
            recommendations["db_weight"] = 0.8
            recommendations["artifact_weight"] = 0.2
            recommendations["top_k"] = 5  # Меньше результатов для навигационного поиска

        elif intent_type == "analytical":
            # Аналитический поиск: больше артифактов (анализ, саммаризации)
            recommendations["db_weight"] = 0.4
            recommendations["artifact_weight"] = 0.6
            recommendations["top_k"] = 20  # Больше результатов для анализа

        return recommendations


