#!/usr/bin/env python3
"""
Анализатор релевантности результатов поиска с поддержкой адаптивных промптов.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ...core.langchain_adapters import LangChainLLMAdapter
from ...core.lmql_adapter import LMQLAdapter, build_lmql_adapter_from_env
from .templates import DEFAULT_PROMPTS_DIR, PromptTemplateManager

logger = logging.getLogger(__name__)


class RelevanceAnalyzer:
    """Анализатор релевантности результатов поиска."""

    def __init__(
        self,
        model_name: str = "gemma3n:e4b-it-q8_0",
        base_url: str = "http://localhost:1234",  # URL LM Studio сервера
        max_context_tokens: int = 131072,  # Для gpt-oss-20b
        temperature: float = 0.1,
        max_response_tokens: int = 131072,  # Для gpt-oss-20b (максимальный лимит)
        prompts_dir: Path | None = None,
        thinking_level: str | None = None,
        lmql_adapter: LMQLAdapter | None = None,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature
        self.max_response_tokens = max_response_tokens
        self.thinking_level = thinking_level

        # Используем LangChainLLMAdapter для работы с LLM
        self.embedding_client = LangChainLLMAdapter(
            model_name=model_name,
            base_url=base_url,
        )

        self.prompt_manager = PromptTemplateManager(prompts_dir or DEFAULT_PROMPTS_DIR)
        
        # LMQL адаптер для структурированной генерации
        try:
            self.lmql_adapter = lmql_adapter or build_lmql_adapter_from_env()
        except RuntimeError:
            self.lmql_adapter = None

        logger.info("Инициализирован RelevanceAnalyzer (модель: %s)", model_name)

    async def analyze_relevance(
        self,
        query_data: dict[str, Any],
        search_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        logger.debug(
            "Анализ релевантности для запроса %s", query_data.get("query", "<empty>")
        )

        if not search_results:
            return self._analyze_empty_results(query_data)

        # Используем LMQL для анализа релевантности
        if self.lmql_adapter:
            logger.debug("Используется LMQL для анализа релевантности")
            return await self._analyze_relevance_with_lmql(query_data, search_results)

        # Если LMQL недоступен, выбрасываем ошибку
        raise RuntimeError(
            "LMQL не настроен. Установите MEMORY_MCP_USE_LMQL=true и настройте модель"
        )

    def _generate_adaptive_prompt(
        self,
        query_data: dict[str, Any],
        search_results: list[dict[str, Any]],
    ) -> str:
        query_type = (query_data.get("type") or "unknown").lower()
        query_text = query_data.get("query", "")

        instructions_map = {
            "factual": "- Оцени точность фактов\n- Укажи, хватает ли ответа для пользователя",
            "contextual": "- Проверь соответствие временного интервала\n- Оцени полноту контекста",
            "analytical": "- Оцени глубину и структурированность ответа\n- Проверь охват ключевых тем",
            "custom": "- Используй описание задачи в запросе\n- Укажи возможные ограничения",
        }
        instructions = instructions_map.get(
            query_type, "- Оцени соответствие результатов запросу"
        )

        formatted_results = self._format_results_for_prompt(search_results)

        return self.prompt_manager.format(
            "relevance_analysis_base",
            query_text=query_text,
            query_type=query_type,
            formatted_results=formatted_results,
            instructions=instructions,
        )

    async def _analyze_relevance_with_lmql(
        self,
        query_data: dict[str, Any],
        search_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Анализ релевантности с использованием LMQL.
        
        Args:
            query_data: Данные запроса
            search_results: Результаты поиска
            
        Returns:
            Словарь с анализом
            
        Raises:
            RuntimeError: Если произошла ошибка при выполнении запроса
        """

        try:
            query_type = (query_data.get("type") or "unknown").lower()
            query_text = query_data.get("query", "")

            instructions_map = {
                "factual": "- Оцени точность фактов\n- Укажи, хватает ли ответа для пользователя",
                "contextual": "- Проверь соответствие временного интервала\n- Оцени полноту контекста",
                "analytical": "- Оцени глубину и структурированность ответа\n- Проверь охват ключевых тем",
                "custom": "- Используй описание задачи в запросе\n- Укажи возможные ограничения",
            }
            instructions = instructions_map.get(
                query_type, "- Оцени соответствие результатов запросу"
            )

            formatted_results = self._format_results_for_prompt(search_results)

            # Создаем промпт для LMQL
            prompt = f"""Ты - эксперт по анализу качества поиска в чатах.

ЗАДАЧА: Оценить релевантность результатов поиска к запросу пользователя.

Запрос: "{query_text}"
Тип запроса: {query_type}

Результаты поиска:
{formatted_results}

Инструкции:
{instructions}

Типы проблем:
- indexing: проблемы с индексацией
- search: проблемы поиска
- context: проблемы контекста

Верни анализ в формате JSON."""

            # Создаем JSON схему с переменными
            json_schema = """{{
    "overall_score": [SCORE],
    "individual_scores": [INDIVIDUAL_SCORES],
    "problems": {{
        "indexing": [INDEXING_PROBLEMS],
        "search": [SEARCH_PROBLEMS],
        "context": [CONTEXT_PROBLEMS]
    }},
    "explanation": "[EXPLANATION]",
    "recommendations": [RECOMMENDATIONS]
}}"""

            # Ограничения для переменных
            results_count = len(search_results)
            constraints = f"""
    0.0 <= SCORE <= 10.0 and
    len(INDIVIDUAL_SCORES) == {results_count} and
    all(0.0 <= score <= 10.0 for score in INDIVIDUAL_SCORES)
"""

            # Выполняем LMQL запрос
            result = await self.lmql_adapter.execute_json_query(
                prompt=prompt,
                json_schema=json_schema,
                constraints=constraints,
                temperature=self.temperature,
                max_tokens=self.max_response_tokens,
            )

            if not result:
                return None

            # Валидируем структуру результата
            if not self._validate_analysis_structure(result):
                raise RuntimeError("LMQL вернул невалидную структуру анализа")

            return result

        except Exception as e:
            logger.error(f"Ошибка при использовании LMQL для анализа релевантности: {e}")
            raise RuntimeError(f"Ошибка анализа релевантности через LMQL: {e}") from e

    def _format_results_for_prompt(self, search_results: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for idx, result in enumerate(search_results[:10], start=1):
            snippet = (result.get("text") or result.get("content", ""))[:600]
            score = result.get("score", 0)
            chat = result.get("metadata", {}).get("chat")
            timestamp = result.get("metadata", {}).get("date") or ""
            lines.append(
                f"{idx}. (score={score:.3f}, chat={chat or 'unknown'}, date={timestamp})\n{snippet}"
            )
        return "\n\n".join(lines)

    def _validate_analysis_structure(self, analysis: dict[str, Any]) -> bool:
        required_fields = [
            "overall_score",
            "individual_scores",
            "problems",
            "explanation",
        ]

        for field in required_fields:
            if field not in analysis:
                return False

        if not isinstance(analysis["overall_score"], (int, float)):
            return False
        if not isinstance(analysis["individual_scores"], list):
            return False
        if not isinstance(analysis["problems"], dict):
            return False

        return True

    def _analyze_empty_results(self, query_data: dict[str, Any]) -> dict[str, Any]:
        return {
            "overall_score": 0.0,
            "individual_scores": [],
            "problems": {
                "indexing": 1,
                "search": 1,
                "context": 0,
            },
            "explanation": "Результаты поиска не найдены - возможны проблемы с индексацией или поиском",
            "recommendations": [
                "Проверить индексацию данных",
                "Проверить настройки поиска",
                "Убедиться в корректности запроса",
            ],
        }

    def _create_error_analysis(self, error_message: str) -> dict[str, Any]:
        return {
            "overall_score": 0.0,
            "individual_scores": [],
            "problems": {
                "indexing": 0,
                "search": 0,
                "context": 0,
            },
            "explanation": f"Ошибка анализа: {error_message}",
            "recommendations": [
                "Проверить подключение к LLM серверу",
                "Проверить корректность модели",
                "Повторить анализ",
            ],
        }

    async def batch_analyze_relevance(
        self,
        queries_and_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        logger.info(
            "Пакетный анализ релевантности для %d запросов",
            len(queries_and_results),
        )

        analyses = []

        for item in queries_and_results:
            query_data = item.get("query_data", {})
            search_results = item.get("search_results", [])

            try:
                analysis = await self.analyze_relevance(query_data, search_results)
                analyses.append(analysis)
            except Exception as exc:  # pragma: no cover - только логируем
                logger.error("Ошибка анализа запроса: %s", exc)
                analyses.append(self._create_error_analysis(str(exc)))

            await asyncio.sleep(0.5)

        logger.info("Завершен пакетный анализ для %d запросов", len(analyses))
        return analyses
