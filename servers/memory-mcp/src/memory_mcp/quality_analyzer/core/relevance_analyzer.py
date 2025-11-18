#!/usr/bin/env python3
"""
Анализатор релевантности результатов поиска с поддержкой адаптивных промптов.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ...core.lmstudio_client import LMStudioEmbeddingClient
from .templates import DEFAULT_PROMPTS_DIR, PromptTemplateManager

logger = logging.getLogger(__name__)


class RelevanceAnalyzer:
    """Анализатор релевантности результатов поиска."""

    def __init__(
        self,
        model_name: str = "gemma3n:e4b-it-q8_0",
        base_url: str = "http://localhost:11434",
        max_context_tokens: int = 131072,  # Для gpt-oss-20b
        temperature: float = 0.1,
        max_response_tokens: int = 131072,  # Для gpt-oss-20b (максимальный лимит)
        prompts_dir: Path | None = None,
        thinking_level: str | None = None,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature
        self.max_response_tokens = max_response_tokens
        self.thinking_level = thinking_level

        self.embedding_client = LMStudioEmbeddingClient(
            model_name=model_name,
            base_url=base_url,
        )

        self.prompt_manager = PromptTemplateManager(prompts_dir or DEFAULT_PROMPTS_DIR)

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

        prompt = self._generate_adaptive_prompt(query_data, search_results)

        try:
            async with self.embedding_client:
                response = await self.embedding_client.generate_summary(
                    prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_response_tokens,
                )

            return self._parse_ollama_response(response, query_data, search_results)
        except Exception as exc:  # pragma: no cover - логируем и возвращаем fallback
            logger.error("Ошибка анализа релевантности: %s", exc)
            return self._create_error_analysis(str(exc))

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

    # --- существующие методы ниже остаются без изменений ---

    def _parse_ollama_response(
        self,
        response: str,
        query_data: dict[str, Any],
        search_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            import json
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)

                if self._validate_analysis_structure(parsed):
                    return parsed

            return self._create_text_based_analysis(
                response, query_data, search_results
            )

        except Exception as exc:
            logger.error("Ошибка парсинга ответа Ollama: %s", exc)
            response_preview = response[:500] if 'response' in locals() and response else 'N/A'
            raise RuntimeError(
                f"Ошибка парсинга ответа LLM для анализа релевантности: {exc}. "
                f"Ответ: {response_preview}. "
                f"Проверьте конфигурацию LLM клиента."
            ) from exc

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

    def _create_text_based_analysis(
        self,
        response: str,
        query_data: dict[str, Any],
        search_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        import re

        score_match = re.search(r"(\d+(?:\.\d+)?)", response)
        overall_score = float(score_match.group(1)) if score_match else 5.0
        overall_score = max(1.0, min(10.0, overall_score))

        individual_scores = []
        for i in range(len(search_results)):
            score = overall_score * (1.0 - i * 0.1)
            individual_scores.append(max(1.0, score))

        return {
            "overall_score": overall_score,
            "individual_scores": individual_scores,
            "problems": {
                "indexing": 0,
                "search": 0,
                "context": 0,
            },
            "explanation": response[:500],
            "recommendations": [],
        }

    def _create_fallback_analysis(
        self,
        query_data: dict[str, Any],
        search_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if len(search_results) == 0:
            overall_score = 1.0
            problems = {"indexing": 1, "search": 0, "context": 0}
        elif len(search_results) < 3:
            overall_score = 4.0
            problems = {"indexing": 0, "search": 1, "context": 0}
        else:
            overall_score = 6.0
            problems = {"indexing": 0, "search": 0, "context": 0}

        individual_scores = [overall_score] * len(search_results)

        return {
            "overall_score": overall_score,
            "individual_scores": individual_scores,
            "problems": problems,
            "explanation": "Анализ выполнен с использованием резервной логики",
            "recommendations": [],
        }

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
                "Проверить подключение к Ollama",
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
