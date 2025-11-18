"""Интерактивный поисковый движок с поддержкой LLM для уточнений и рефайнинга."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import get_settings, get_quality_analysis_settings
from ..core.lmstudio_client import LMStudioEmbeddingClient
from ..core.ollama_client import OllamaEmbeddingClient
from ..memory.artifacts_reader import ArtifactsReader, ArtifactSearchResult
from ..mcp.adapters import MemoryServiceAdapter
from ..mcp.schema import (
    SearchRequest,
    SearchResultItem,
    SmartSearchRequest,
    SmartSearchResponse,
)
from .search_session_store import SearchSessionStore

logger = logging.getLogger(__name__)


class SmartSearchEngine:
    """Интерактивный поисковый движок с поддержкой LLM."""

    def __init__(
        self,
        adapter: MemoryServiceAdapter,
        artifacts_reader: ArtifactsReader,
        session_store: SearchSessionStore,
        min_confidence: float = 0.5,
    ):
        """
        Инициализация поискового движка.

        Args:
            adapter: Адаптер памяти для поиска по БД
            artifacts_reader: Читатель артифактов
            session_store: Хранилище сессий поиска
            min_confidence: Минимальный порог уверенности для запроса уточнений
        """
        self.adapter = adapter
        self.artifacts_reader = artifacts_reader
        self.session_store = session_store
        self.min_confidence = min_confidence
        self._llm_client: Optional[LMStudioEmbeddingClient | OllamaEmbeddingClient] = None

    def _get_llm_client(self) -> Optional[LMStudioEmbeddingClient | OllamaEmbeddingClient]:
        """Получение или создание LLM клиента."""
        if self._llm_client is not None:
            return self._llm_client

        try:
            settings = get_settings()

            # Пытаемся использовать LM Studio, если указана LLM модель
            if settings.lmstudio_llm_model:
                self._llm_client = LMStudioEmbeddingClient(
                    model_name=settings.lmstudio_model,  # Для эмбеддингов (не используется здесь)
                    llm_model_name=settings.lmstudio_llm_model,  # Для генерации текста
                    base_url=f"http://{settings.lmstudio_host}:{settings.lmstudio_port}",
                )
                logger.debug("Используется LM Studio для интерактивного поиска")
            else:
                # Используем Ollama как fallback
                qa_settings = get_quality_analysis_settings()
                self._llm_client = OllamaEmbeddingClient(
                    llm_model_name=qa_settings.ollama_model,
                    base_url=qa_settings.ollama_base_url,
                )
                logger.debug("Используется Ollama для интерактивного поиска")
        except Exception as e:
            logger.warning(f"Не удалось инициализировать LLM клиент: {e}")
            return None

        return self._llm_client

    async def search(self, request: SmartSearchRequest) -> SmartSearchResponse:
        """
        Выполнение интерактивного поиска.

        Args:
            request: Запрос на поиск

        Returns:
            Результаты поиска с интерактивными элементами
        """
        # Получаем или создаем сессию
        session_id = request.session_id
        if not session_id:
            session_id = self.session_store.create_session(request.query)
        else:
            # Сохраняем уточненный запрос, если есть feedback
            if request.feedback:
                refined_query = await self._refine_query(request.query, request.feedback)
                if refined_query and refined_query != request.query:
                    self.session_store.add_refined_query(session_id, refined_query)
                    request.query = refined_query

        # Выполняем параллельный поиск
        db_results, artifact_results = self._parallel_search(request)

        # Объединяем результаты
        combined_results = self._combine_results(
            db_results, artifact_results, request.top_k
        )

        # Вычисляем confidence score
        confidence_score = self._calculate_confidence(combined_results)

        # Сохраняем результаты в сессию
        query_id = None
        if request.feedback:
            # Получаем последний query_id из сессии
            session = self.session_store.get_session(session_id)
            if session and session.get("queries"):
                query_id = session["queries"][-1].get("id")

        self.session_store.save_results(
            session_id,
            query_id,
            [
                {
                    "record_id": r.record_id,
                    "artifact_path": r.metadata.get("artifact_path"),
                    "score": r.score,
                    "content": r.content,
                }
                for r in combined_results
            ],
        )

        # Обрабатываем feedback, если есть
        if request.feedback:
            for feedback in request.feedback:
                self.session_store.add_feedback(
                    session_id,
                    feedback.record_id,
                    feedback.relevance,
                    feedback.artifact_path,
                    feedback.comment,
                )

        # Генерируем уточняющие вопросы, если нужно
        clarifying_questions = None
        suggested_refinements = None
        if request.clarify or confidence_score < self.min_confidence:
            clarifying_questions = await self._generate_clarifying_questions(
                request.query, combined_results
            )
            suggested_refinements = await self._suggest_refinements(
                request.query, combined_results
            )

        return SmartSearchResponse(
            results=combined_results,
            clarifying_questions=clarifying_questions,
            suggested_refinements=suggested_refinements,
            session_id=session_id,
            confidence_score=confidence_score,
            artifacts_found=len(artifact_results),
            db_records_found=len(db_results),
            total_matches=len(db_results) + len(artifact_results),
        )

    def _parallel_search(
        self, request: SmartSearchRequest
    ) -> tuple[List[SearchResultItem], List[SearchResultItem]]:
        """Параллельный поиск по БД и артифактам."""
        # Поиск по БД
        db_search_request = SearchRequest(
            query=request.query,
            top_k=request.top_k * 2,  # Берем больше для объединения
            source=request.source,
            tags=request.tags,
            date_from=request.date_from,
            date_to=request.date_to,
        )
        db_response = self.adapter.search(db_search_request)
        db_results = db_response.results

        # Поиск по артифактам
        artifact_results_raw = self.artifacts_reader.search_artifacts(
            query=request.query,
            artifact_types=request.artifact_types,
            limit=request.top_k * 2,
        )

        # Конвертируем результаты артифактов в SearchResultItem
        artifact_results = []
        for artifact_result in artifact_results_raw:
            # Извлекаем timestamp из метаданных
            timestamp = datetime.now(timezone.utc)
            if "modified_time" in artifact_result.metadata:
                try:
                    timestamp = datetime.fromisoformat(
                        artifact_result.metadata["modified_time"]
                    )
                except Exception:
                    pass

            artifact_results.append(
                SearchResultItem(
                    record_id=f"artifact:{artifact_result.artifact_path}",
                    score=artifact_result.score,
                    content=artifact_result.content,
                    source="artifact",
                    timestamp=timestamp,
                    author=None,
                    metadata={
                        **artifact_result.metadata,
                        "artifact_path": artifact_result.artifact_path,
                        "artifact_type": artifact_result.artifact_type,
                        "snippet": artifact_result.snippet,
                        "line_number": artifact_result.line_number,
                    },
                )
            )

        return db_results, artifact_results

    def _combine_results(
        self,
        db_results: List[SearchResultItem],
        artifact_results: List[SearchResultItem],
        top_k: int,
    ) -> List[SearchResultItem]:
        """
        Объединение результатов с весами (БД: 0.6, артифакты: 0.4).

        Args:
            db_results: Результаты из БД
            artifact_results: Результаты из артифактов
            top_k: Максимальное количество результатов

        Returns:
            Объединенный список результатов
        """
        # Применяем веса
        db_weight = 0.6
        artifact_weight = 0.4

        # Нормализуем scores и применяем веса
        all_results: Dict[str, SearchResultItem] = {}

        # Обрабатываем результаты БД
        if db_results:
            max_db_score = max(r.score for r in db_results) if db_results else 1.0
            for result in db_results:
                normalized_score = result.score / max_db_score if max_db_score > 0 else 0.0
                weighted_score = normalized_score * db_weight
                result.score = weighted_score
                all_results[result.record_id] = result

        # Обрабатываем результаты артифактов
        if artifact_results:
            max_artifact_score = (
                max(r.score for r in artifact_results) if artifact_results else 1.0
            )
            for result in artifact_results:
                normalized_score = (
                    result.score / max_artifact_score if max_artifact_score > 0 else 0.0
                )
                weighted_score = normalized_score * artifact_weight

                # Если результат уже есть, объединяем scores
                if result.record_id in all_results:
                    all_results[result.record_id].score += weighted_score
                else:
                    result.score = weighted_score
                    all_results[result.record_id] = result

        # Сортируем по score и возвращаем top_k
        sorted_results = sorted(
            all_results.values(), key=lambda x: x.score, reverse=True
        )
        return sorted_results[:top_k]

    def _calculate_confidence(self, results: List[SearchResultItem]) -> float:
        """
        Вычисление уверенности в релевантности результатов.

        Args:
            results: Список результатов

        Returns:
            Confidence score от 0.0 до 1.0
        """
        if not results:
            return 0.0

        # Используем средний score и количество результатов
        avg_score = sum(r.score for r in results) / len(results)
        count_factor = min(len(results) / 10.0, 1.0)  # Нормализуем к 1.0 при 10+ результатах

        # Комбинируем факторы
        confidence = (avg_score * 0.7 + count_factor * 0.3)
        return min(confidence, 1.0)

    async def _refine_query(
        self, original_query: str, feedback: List[Any]
    ) -> Optional[str]:
        """
        Рефайнинг запроса на основе обратной связи.

        Args:
            original_query: Исходный запрос
            feedback: Список обратной связи

        Returns:
            Уточненный запрос или None
        """
        llm_client = self._get_llm_client()
        if not llm_client:
            return None

        # Формируем промпт
        relevant_items = [f for f in feedback if f.relevance == "relevant"]
        irrelevant_items = [f for f in feedback if f.relevance == "irrelevant"]

        prompt = f"""Ты помощник для улучшения поисковых запросов. Пользователь искал: "{original_query}"

Обратная связь:
- Релевантные результаты: {len(relevant_items)}
- Нерелевантные результаты: {len(irrelevant_items)}

Помоги улучшить поисковый запрос, чтобы найти больше релевантных результатов и меньше нерелевантных.

Верни только улучшенный запрос, без дополнительных объяснений. Если запрос уже хорош, верни его без изменений."""

        try:
            async with llm_client:
                if isinstance(llm_client, LMStudioEmbeddingClient):
                    refined = await llm_client.generate_summary(
                        prompt=prompt,
                        temperature=0.3,
                        max_tokens=200,
                        top_p=0.9,
                        presence_penalty=0.1,
                    )
                else:  # OllamaEmbeddingClient
                    refined = await llm_client.generate_summary(
                        prompt=prompt,
                        temperature=0.3,
                        max_tokens=200,
                        top_p=0.9,
                        presence_penalty=0.1,
                    )

                # Очищаем ответ от лишних символов
                refined = refined.strip().strip('"').strip("'")
                return refined if refined and refined != original_query else None
        except Exception as e:
            logger.warning(f"Ошибка при рефайнинге запроса: {e}", exc_info=True)
            return None

    async def _generate_clarifying_questions(
        self, query: str, results: List[SearchResultItem]
    ) -> Optional[List[str]]:
        """
        Генерация уточняющих вопросов через LLM.

        Args:
            query: Поисковый запрос
            results: Текущие результаты поиска

        Returns:
            Список уточняющих вопросов или None
        """
        llm_client = self._get_llm_client()
        if not llm_client:
            return None

        # Формируем промпт
        results_preview = "\n".join(
            [f"- {r.content[:100]}..." for r in results[:5]]
        )

        prompt = f"""Ты помощник для улучшения поиска. Пользователь искал: "{query}"

Текущие результаты поиска:
{results_preview}

Сгенерируй 2-3 уточняющих вопроса, которые помогут пользователю лучше сформулировать запрос.

Верни вопросы в формате JSON массива строк, например: ["Вопрос 1?", "Вопрос 2?", "Вопрос 3?"]
Только JSON, без дополнительного текста."""

        try:
            async with llm_client:
                if isinstance(llm_client, LMStudioEmbeddingClient):
                    response = await llm_client.generate_summary(
                        prompt=prompt,
                        temperature=0.5,
                        max_tokens=300,
                        top_p=0.9,
                        presence_penalty=0.1,
                    )
                else:  # OllamaEmbeddingClient
                    response = await llm_client.generate_summary(
                        prompt=prompt,
                        temperature=0.5,
                        max_tokens=300,
                        top_p=0.9,
                        presence_penalty=0.1,
                    )

                # Парсим JSON
                import json
                response = response.strip()
                # Убираем markdown code blocks, если есть
                if response.startswith("```"):
                    response = response.split("```")[1]
                    if response.startswith("json"):
                        response = response[4:]
                response = response.strip()

                questions = json.loads(response)
                if isinstance(questions, list) and all(
                    isinstance(q, str) for q in questions
                ):
                    return questions[:3]  # Максимум 3 вопроса
        except Exception as e:
            logger.warning(
                f"Ошибка при генерации уточняющих вопросов: {e}", exc_info=True
            )

        return None

    async def _suggest_refinements(
        self, query: str, results: List[SearchResultItem]
    ) -> Optional[List[str]]:
        """
        Генерация предложений по уточнению запроса.

        Args:
            query: Поисковый запрос
            results: Текущие результаты поиска

        Returns:
            Список предложенных уточнений или None
        """
        llm_client = self._get_llm_client()
        if not llm_client:
            return None

        # Формируем промпт
        prompt = f"""Ты помощник для улучшения поиска. Пользователь искал: "{query}"

Предложи 2-3 альтернативных формулировки этого запроса, которые могут дать лучшие результаты.

Верни предложения в формате JSON массива строк, например: ["Вариант 1", "Вариант 2", "Вариант 3"]
Только JSON, без дополнительного текста."""

        try:
            async with llm_client:
                if isinstance(llm_client, LMStudioEmbeddingClient):
                    response = await llm_client.generate_summary(
                        prompt=prompt,
                        temperature=0.5,
                        max_tokens=300,
                        top_p=0.9,
                        presence_penalty=0.1,
                    )
                else:  # OllamaEmbeddingClient
                    response = await llm_client.generate_summary(
                        prompt=prompt,
                        temperature=0.5,
                        max_tokens=300,
                        top_p=0.9,
                        presence_penalty=0.1,
                    )

                # Парсим JSON
                import json
                response = response.strip()
                # Убираем markdown code blocks, если есть
                if response.startswith("```"):
                    response = response.split("```")[1]
                    if response.startswith("json"):
                        response = response[4:]
                response = response.strip()

                refinements = json.loads(response)
                if isinstance(refinements, list) and all(
                    isinstance(r, str) for r in refinements
                ):
                    return refinements[:3]  # Максимум 3 предложения
        except Exception as e:
            logger.warning(
                f"Ошибка при генерации предложений: {e}", exc_info=True
            )

        return None

