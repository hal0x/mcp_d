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
        adapter: "MemoryServiceAdapter",  # Используем строковую аннотацию для избежания циклического импорта
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
        
        # Инициализируем обогатитель контекста сущностей
        from .entity_context_enricher import EntityContextEnricher
        from ..analysis.entity_dictionary import get_entity_dictionary
        from ..memory.embeddings import build_embedding_service_from_env
        from ..memory.vector_store import build_entity_vector_store_from_env
        
        entity_dict = get_entity_dictionary(graph=adapter.graph if hasattr(adapter, 'graph') else None)
        embedding_service = build_embedding_service_from_env()
        entity_vector_store = build_entity_vector_store_from_env()
        
        self.entity_enricher = EntityContextEnricher(
            entity_dictionary=entity_dict,
            graph=adapter.graph if hasattr(adapter, 'graph') else None,
            entity_vector_store=entity_vector_store,
            embedding_service=embedding_service,
        )
        
        # Инициализируем ConnectionGraphBuilder для улучшения результатов через граф связей
        from .search_explainer import ConnectionGraphBuilder
        self.connection_builder = ConnectionGraphBuilder(
            typed_graph_memory=adapter.graph if hasattr(adapter, 'graph') else None
        )
        
        # Инициализируем анализатор намерений запроса
        from .query_intent_analyzer import QueryIntentAnalyzer
        self.intent_analyzer = QueryIntentAnalyzer()
        
        # Инициализируем движок понимания запросов
        from .query_understanding import QueryUnderstandingEngine
        self.query_understanding = QueryUnderstandingEngine()

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
        # Глубокое понимание запроса через LLM
        understanding = await self.query_understanding.understand_query(request.query)
        
        # Используем улучшенный запрос, если он был сгенерирован
        query_to_use = understanding.enhanced_query if understanding.enhanced_query != request.query else request.query
        
        # Если есть подзапросы, используем первый как основной (или объединяем)
        if understanding.sub_queries and len(understanding.sub_queries) > 1:
            # Для сложных запросов используем первый подзапрос как основной
            query_to_use = understanding.sub_queries[0]
            logger.debug(
                f"Запрос декомпозирован на {len(understanding.sub_queries)} подзапросов, "
                f"используется: {query_to_use}"
            )
        
        # Анализируем намерение запроса для адаптации стратегии поиска
        intent = await self.intent_analyzer.analyze_intent(query_to_use)
        
        # Получаем персонализированные веса на основе истории сессий
        personalization = self._get_personalization_context()
        
        # Адаптируем параметры поиска на основе намерения и персонализации
        adapted_top_k = intent.recommended_top_k or request.top_k
        adapted_db_weight = intent.recommended_db_weight
        adapted_artifact_weight = intent.recommended_artifact_weight
        
        # Применяем персонализацию (смешиваем рекомендации намерения с историей)
        if personalization:
            # Смешиваем веса: 70% намерение, 30% персонализация
            adapted_db_weight = (
                adapted_db_weight * 0.7 + personalization.get("db_weight", adapted_db_weight) * 0.3
            )
            adapted_artifact_weight = (
                adapted_artifact_weight * 0.7 + personalization.get("artifact_weight", adapted_artifact_weight) * 0.3
            )
            logger.debug(
                f"Применена персонализация: БД={personalization.get('db_weight', 0.6):.2f}, "
                f"артифакты={personalization.get('artifact_weight', 0.4):.2f}"
            )
        
        logger.debug(
            f"Намерение запроса: {intent.intent_type} (уверенность: {intent.confidence:.2f}), "
            f"адаптированные веса: БД={adapted_db_weight:.2f}, артифакты={adapted_artifact_weight:.2f}"
        )
        
        # Обогащаем запрос контекстом сущностей (полный профиль + семантически похожие)
        # Используем улучшенный запрос из understanding
        enriched_query = self.entity_enricher.enrich_query_with_entity_context(query_to_use)
        
        # Добавляем неявные требования и ключевые концепции в контекст
        if understanding.implicit_requirements:
            implicit_context = " [неявно: " + ", ".join(understanding.implicit_requirements[:3]) + "]"
            enriched_query = enriched_query + implicit_context
        
        if understanding.key_concepts:
            concepts_context = " [концепции: " + ", ".join(understanding.key_concepts[:5]) + "]"
            enriched_query = enriched_query + concepts_context
        
        # Расширяем запрос связанными сущностями через граф
        enriched_query = self.entity_enricher.expand_query_with_related_entities(
            enriched_query, 
            entities=None  # Сущности будут извлечены автоматически
        )
        
        # Получаем или создаем сессию
        session_id = request.session_id
        if not session_id:
            session_id = self.session_store.create_session(enriched_query)
        else:
            # Сохраняем уточненный запрос, если есть feedback
            if request.feedback:
                refined_query = await self._refine_query(enriched_query, request.feedback)
                if refined_query and refined_query != enriched_query:
                    self.session_store.add_refined_query(session_id, refined_query)
                    enriched_query = refined_query
        
        # Обновляем запрос в request для использования в поиске
        original_query = request.query
        request.query = enriched_query

        # Выполняем параллельный поиск с адаптированным top_k
        original_top_k = request.top_k
        request.top_k = adapted_top_k
        db_results, artifact_results = self._parallel_search(request)
        request.top_k = original_top_k
        
        # Восстанавливаем оригинальный запрос для ответа
        request.query = original_query

        # Объединяем результаты с адаптированными весами
        combined_results = self._combine_results(
            db_results, artifact_results, request.top_k,
            db_weight=adapted_db_weight,
            artifact_weight=adapted_artifact_weight,
        )

        # Улучшаем результаты через граф связей
        combined_results = self._boost_results_with_graph_connections(
            combined_results, original_query
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
        db_weight: float = 0.6,
        artifact_weight: float = 0.4,
    ) -> List[SearchResultItem]:
        """
        Объединение результатов с настраиваемыми весами.

        Args:
            db_results: Результаты из БД
            artifact_results: Результаты из артифактов
            top_k: Максимальное количество результатов
            db_weight: Вес для результатов БД (по умолчанию 0.6)
            artifact_weight: Вес для результатов артифактов (по умолчанию 0.4)

        Returns:
            Объединенный список результатов
        """
        # Нормализуем веса, чтобы они в сумме давали 1.0
        total_weight = db_weight + artifact_weight
        if total_weight > 0:
            db_weight = db_weight / total_weight
            artifact_weight = artifact_weight / total_weight

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

    def _boost_results_with_graph_connections(
        self, results: List[SearchResultItem], query: str
    ) -> List[SearchResultItem]:
        """
        Повышение релевантности результатов на основе графа связей.

        Args:
            results: Список результатов поиска
            query: Исходный запрос

        Returns:
            Список результатов с обновленными scores
        """
        if not self.connection_builder or not self.connection_builder.graph:
            return results

        # Извлекаем сущности из запроса
        query_entities = self.entity_enricher.extract_entities_from_query(query)
        if not query_entities:
            return results

        # Собираем названия сущностей для поиска связей
        entity_names = [e.get("value", "") for e in query_entities if e.get("value")]

        # Для каждого результата ищем пути к сущностям запроса
        boosted_results = []
        for result in results:
            result_id = result.record_id
            
            # Ищем пути между сущностями запроса и результатом
            connections = self.connection_builder.find_connections(
                query_entities=entity_names,
                result_id=result_id,
                max_paths=3,
                max_depth=3,
            )

            # Применяем буст на основе найденных путей
            boost_factor = 1.0
            if connections:
                # Находим самый короткий путь
                min_path_length = min(c.path_length for c in connections)
                
                # Буст зависит от длины пути:
                # - путь длиной 1: +0.3
                # - путь длиной 2: +0.2
                # - путь длиной 3: +0.1
                if min_path_length == 1:
                    boost_factor = 1.3
                elif min_path_length == 2:
                    boost_factor = 1.2
                elif min_path_length == 3:
                    boost_factor = 1.1
                
                # Дополнительный буст на основе силы пути
                max_strength = max(c.path_strength for c in connections)
                boost_factor += max_strength * 0.1  # До +0.1 за силу связи
                
                logger.debug(
                    f"Буст для результата {result_id}: "
                    f"путь={min_path_length}, сила={max_strength:.2f}, "
                    f"фактор={boost_factor:.2f}"
                )

            # Применяем буст к score
            result.score *= boost_factor
            boosted_results.append(result)

        # Пересортировываем результаты по обновленным scores
        boosted_results.sort(key=lambda x: x.score, reverse=True)
        
        return boosted_results

    def _get_personalization_context(self) -> Optional[Dict[str, Any]]:
        """
        Получение контекста персонализации на основе общей истории сессий.

        Returns:
            Словарь с персонализированными весами или None
        """
        try:
            # Получаем общую историю обратной связи
            feedback_history = self.session_store.get_feedback_for_learning(limit=100)
            
            if not feedback_history:
                return None

            # Анализируем успешные запросы
            relevant_count = 0
            irrelevant_count = 0
            db_preference = 0.0
            artifact_preference = 0.0
            total_feedback = 0

            for feedback in feedback_history:
                relevance = feedback.get("relevance", "")
                result_id = feedback.get("result_id", "")
                artifact_path = feedback.get("artifact_path")
                
                # Определяем тип результата по result_id или artifact_path
                result_type = ""
                if artifact_path or (result_id and result_id.startswith("artifact:")):
                    result_type = "artifact"
                elif result_id:
                    result_type = "db_record"
                
                if relevance == "relevant":
                    relevant_count += 1
                    # Учитываем тип результата для предпочтений
                    if result_type == "db_record":
                        db_preference += 1.0
                    elif result_type == "artifact":
                        artifact_preference += 1.0
                elif relevance == "irrelevant":
                    irrelevant_count += 1
                
                total_feedback += 1

            if total_feedback == 0:
                return None

            # Вычисляем адаптированные веса на основе предпочтений
            if db_preference + artifact_preference > 0:
                db_ratio = db_preference / (db_preference + artifact_preference)
                artifact_ratio = artifact_preference / (db_preference + artifact_preference)
                
                # Нормализуем к диапазону 0.4-0.8 для каждого веса
                adapted_db_weight = 0.4 + (db_ratio * 0.4)
                adapted_artifact_weight = 0.4 + (artifact_ratio * 0.4)
            else:
                # Если нет предпочтений, используем значения по умолчанию
                adapted_db_weight = 0.6
                adapted_artifact_weight = 0.4

            logger.debug(
                f"Персонализация: {relevant_count} релевантных, {irrelevant_count} нерелевантных, "
                f"предпочтения: БД={db_preference:.1f}, артифакты={artifact_preference:.1f}"
            )

            return {
                "db_weight": adapted_db_weight,
                "artifact_weight": adapted_artifact_weight,
                "relevant_count": relevant_count,
                "irrelevant_count": irrelevant_count,
                "total_feedback": total_feedback,
            }

        except Exception as e:
            logger.warning(f"Ошибка при получении контекста персонализации: {e}")
            return None

