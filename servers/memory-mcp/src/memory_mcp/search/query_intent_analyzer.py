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
from ..core.langchain_adapters import LangChainLLMAdapter, get_llm_client_factory

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

    def __init__(self):
        """Инициализация анализатора намерений."""
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

        llm_client = self._get_llm_client()
        if not llm_client:
            logger.error("LLM клиент не настроен, анализ намерения невозможен")
            raise ValueError("Для анализа намерения требуется настроенный LLM клиент")

        # Используем LLM для классификации
        prompt = self._create_intent_classification_prompt(query)
        
        # Используем максимальное значение модели (131072 для gpt-oss-20b)
        from ..config import get_settings
        settings = get_settings()
        max_tokens = settings.large_context_max_tokens  # 131072
        
        async with llm_client:
            response = await llm_client.generate_summary(
                prompt=prompt,
                temperature=0.2,
                max_tokens=max_tokens,  # Максимальное значение модели
                top_p=0.9,
            )

        # Парсим ответ LLM
        intent = self._parse_llm_response(response, query)
        if not intent:
            logger.error("Не удалось распарсить ответ LLM для анализа намерения")
            raise ValueError("LLM вернул некорректный ответ для анализа намерения")
        
        return intent

    def _create_intent_classification_prompt(self, query: str) -> str:
        """Создание промпта для классификации намерения"""
        return f"""Проанализируй поисковый запрос и определи его намерение.

Запрос: "{query}"

Типы намерений:
1. informational - информационный поиск (что это?, расскажи о, информация о)
2. transactional - транзакционный поиск (как сделать, инструкция, шаги)
3. navigational - навигационный поиск (где найти, найти конкретную вещь)
4. analytical - аналитический поиск (почему, как работает, анализ, сравнение)

Верни ответ в формате JSON:
{{
    "intent_type": "informational|transactional|navigational|analytical",
    "confidence": 0.0-1.0,
    "reasoning": "краткое объяснение"
}}

Только JSON, без дополнительного текста."""

    def _parse_llm_response(self, response: str, query: str) -> Optional[QueryIntent]:
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
            intent_type = data.get("intent_type", "informational")
            confidence = float(data.get("confidence", 0.7))
            
            # Валидация типа намерения
            valid_types = ["informational", "transactional", "navigational", "analytical"]
            if intent_type not in valid_types:
                intent_type = "informational"
                confidence = 0.5

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

        except Exception as e:
            logger.debug(f"Ошибка парсинга ответа LLM для анализа намерения: {e}")
            return None

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


