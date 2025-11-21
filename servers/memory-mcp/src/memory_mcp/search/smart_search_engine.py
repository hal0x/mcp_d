"""Интерактивный поисковый движок с поддержкой LLM для уточнений и рефайнинга запросов."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..core.langchain_adapters import LangChainLLMAdapter, get_llm_client_factory
from ..core.lmql_adapter import LMQLAdapter, build_lmql_adapter_from_env
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
        adapter: "MemoryServiceAdapter",
        artifacts_reader: ArtifactsReader,
        session_store: SearchSessionStore,
        min_confidence: float = 0.5,
        lmql_adapter: Optional[LMQLAdapter] = None,
    ):
        """Инициализация интерактивного поискового движка."""
        self.adapter = adapter
        self.artifacts_reader = artifacts_reader
        self.session_store = session_store
        self.min_confidence = min_confidence
        self._llm_client: Optional[LangChainLLMAdapter] = None
        try:
            self.lmql_adapter = lmql_adapter or build_lmql_adapter_from_env()
        except RuntimeError:
            self.lmql_adapter = None
        
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
        
        from .search_explainer import ConnectionGraphBuilder
        self.connection_builder = ConnectionGraphBuilder(
            typed_graph_memory=adapter.graph if hasattr(adapter, 'graph') else None
        )
        
        from .query_intent_analyzer import QueryIntentAnalyzer
        self.intent_analyzer = QueryIntentAnalyzer()
        
        from .query_understanding import QueryUnderstandingEngine
        self.query_understanding = QueryUnderstandingEngine()

    def _get_llm_client(self) -> Optional[LangChainLLMAdapter]:
        """Получение или создание LangChain LLM клиента."""
        if self._llm_client is not None:
            return self._llm_client

        try:
            from ..core.langchain_adapters import get_llm_client_factory
            
            self._llm_client = get_llm_client_factory()
            if self._llm_client is None:
                logger.error("Не удалось инициализировать LLM клиент")
                raise ValueError("Для интерактивного поиска требуется настроенный LLM (LM Studio)")
            
            logger.debug("LLM клиент инициализирован")
        except Exception as e:
            logger.error(f"Не удалось инициализировать LLM клиент: {e}")
            raise

        return self._llm_client

    def _select_query(
        self, understanding: Any, original_query: str
    ) -> str:
        """
        Выбирает оптимальный запрос из понимания запроса.
        
        Приоритет:
        1. sub_queries[0] если есть несколько подзапросов
        2. enhanced_query если он отличается от оригинального
        3. original_query в остальных случаях
        
        Args:
            understanding: Результат понимания запроса
            original_query: Оригинальный запрос пользователя
            
        Returns:
            Выбранный запрос для поиска
        """
        if understanding.sub_queries and len(understanding.sub_queries) > 1:
            query_to_use = understanding.sub_queries[0]
            logger.debug(
                f"Запрос декомпозирован на {len(understanding.sub_queries)} подзапросов, "
                f"используется: {query_to_use}"
            )
            return query_to_use
        
        if understanding.enhanced_query and understanding.enhanced_query != original_query:
            return understanding.enhanced_query
        
        return original_query

    async def search(self, request: SmartSearchRequest) -> SmartSearchResponse:
        """Выполнение интерактивного поиска с пониманием запроса."""
        understanding = await self.query_understanding.understand_query(request.query)
        
        # Выбор оптимального запроса для поиска
        query_to_use = self._select_query(understanding, request.query)
        
        intent = await self.intent_analyzer.analyze_intent(query_to_use)
        personalization = self._get_personalization_context()
        
        adapted_top_k = intent.recommended_top_k or request.top_k
        adapted_db_weight = intent.recommended_db_weight
        adapted_artifact_weight = intent.recommended_artifact_weight
        
        if personalization:
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
        
        enriched_query = self.entity_enricher.enrich_query_with_entity_context(query_to_use)
        
        if understanding.implicit_requirements:
            implicit_context = " [неявно: " + ", ".join(understanding.implicit_requirements[:3]) + "]"
            enriched_query = enriched_query + implicit_context
        
        if understanding.key_concepts:
            concepts_context = " [концепции: " + ", ".join(understanding.key_concepts[:5]) + "]"
            enriched_query = enriched_query + concepts_context
        
        enriched_query = self.entity_enricher.expand_query_with_related_entities(
            enriched_query, 
            entities=None
        )
        
        session_id = request.session_id
        if not session_id:
            session_id = self.session_store.create_session(enriched_query)
        else:
            if request.feedback:
                refined_query = await self._refine_query(enriched_query, request.feedback)
                if refined_query and refined_query != enriched_query:
                    self.session_store.add_refined_query(session_id, refined_query)
                    enriched_query = refined_query
        
        original_query = request.query
        request.query = enriched_query

        original_top_k = request.top_k
        request.top_k = adapted_top_k
        db_results, artifact_results = self._parallel_search(request)
        request.top_k = original_top_k
        
        request.query = original_query

        combined_results = self._combine_results(
            db_results, artifact_results, request.top_k,
            db_weight=adapted_db_weight,
            artifact_weight=adapted_artifact_weight,
        )

        combined_results = self._boost_results_with_graph_connections(
            combined_results, original_query
        )

        confidence_score = self._calculate_confidence(combined_results)

        query_id = None
        if request.feedback:
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

        if request.feedback:
            for feedback in request.feedback:
                self.session_store.add_feedback(
                    session_id,
                    feedback.record_id,
                    feedback.relevance,
                    feedback.artifact_path,
                    feedback.comment,
                )

        clarifying_questions = None
        suggested_refinements = None
        if request.clarify or confidence_score < self.min_confidence:
            try:
                clarifying_questions = await self._generate_clarifying_questions(
                    request.query, combined_results
                )
            except RuntimeError as e:
                logger.warning(f"Не удалось сгенерировать уточняющие вопросы: {e}")
            
            try:
                suggested_refinements = await self._suggest_refinements(
                    request.query, combined_results
                )
            except RuntimeError as e:
                logger.warning(f"Не удалось сгенерировать предложения: {e}")

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
        db_search_request = SearchRequest(
            query=request.query,
            top_k=request.top_k * 2,
            source=request.source,
            tags=request.tags,
            date_from=request.date_from,
            date_to=request.date_to,
        )
        db_response = self.adapter.search(db_search_request)
        db_results = db_response.results

        artifact_results_raw = self.artifacts_reader.search_artifacts(
            query=request.query,
            artifact_types=request.artifact_types,
            limit=request.top_k * 2,
        )

        artifact_results = []
        for artifact_result in artifact_results_raw:
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
        """Объединение результатов с настраиваемыми весами."""
        total_weight = db_weight + artifact_weight
        if total_weight > 0:
            db_weight = db_weight / total_weight
            artifact_weight = artifact_weight / total_weight

        all_results: Dict[str, SearchResultItem] = {}

        # Обрабатываем результаты из БД с прямым взвешиванием
        if db_results:
            for result in db_results:
                weighted_score = result.score * db_weight
                result.score = weighted_score
                all_results[result.record_id] = result

        # Обрабатываем результаты из артифактов с прямым взвешиванием
        if artifact_results:
            for result in artifact_results:
                weighted_score = result.score * artifact_weight

                if result.record_id in all_results:
                    # Если результат уже есть, суммируем скоры
                    all_results[result.record_id].score += weighted_score
                else:
                    result.score = weighted_score
                    all_results[result.record_id] = result

        sorted_results = sorted(
            all_results.values(), key=lambda x: x.score, reverse=True
        )
        return sorted_results[:top_k]

    def _calculate_confidence(self, results: List[SearchResultItem]) -> float:
        """Вычисление уверенности в релевантности результатов."""
        if not results:
            return 0.0

        avg_score = sum(r.score for r in results) / len(results)
        count_factor = min(len(results) / 10.0, 1.0)

        confidence = (avg_score * 0.7 + count_factor * 0.3)
        return min(confidence, 1.0)

    async def _refine_query(
        self, original_query: str, feedback: List[Any]
    ) -> Optional[str]:
        """Уточнение запроса на основе обратной связи."""
        llm_client = self._get_llm_client()
        if not llm_client:
            return None

        relevant_items = [f for f in feedback if f.relevance == "relevant"]
        irrelevant_items = [f for f in feedback if f.relevance == "irrelevant"]

        prompt = f"""Ты помощник для улучшения поисковых запросов. Пользователь искал: "{original_query}"

Обратная связь:
- Релевантные результаты: {len(relevant_items)}
- Нерелевантные результаты: {len(irrelevant_items)}

Помоги улучшить поисковый запрос, чтобы найти больше релевантных результатов и меньше нерелевантных.

Верни только улучшенный запрос, без дополнительных объяснений. Если запрос уже хорош, верни его без изменений."""

        from ..config import get_settings
        settings = get_settings()
        max_tokens = settings.large_context_max_tokens
        
        try:
            async with llm_client:
                refined = await llm_client.generate_summary(
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=max_tokens,
                    top_p=0.9,
                    presence_penalty=0.1,
                )

                refined = refined.strip().strip('"').strip("'")
                return refined if refined and refined != original_query else None
        except Exception as e:
            logger.warning(f"Ошибка при рефайнинге запроса: {e}", exc_info=True)
            return None

    async def _generate_clarifying_questions(
        self, query: str, results: List[SearchResultItem]
    ) -> List[str]:
        """Генерация уточняющих вопросов для запроса.
        
        Returns:
            Список уточняющих вопросов
            
        Raises:
            RuntimeError: Если LMQL не настроен или произошла ошибка
        """
        if not self.lmql_adapter:
            raise RuntimeError(
                "LMQL не настроен. Установите MEMORY_MCP_USE_LMQL=true и настройте модель"
            )

        results_preview = "\n".join(
            [f"- {r.content[:100]}..." for r in results[:5]]
        )

        try:
            lmql_query_str = self._create_lmql_questions_query(query, results_preview)
            response_data = await self.lmql_adapter.execute_json_query(
                prompt=f"""Ты помощник для улучшения поиска. Пользователь искал: "{query}"

Текущие результаты поиска:
{results_preview}

Сгенерируй 2-3 уточняющих вопроса, которые помогут пользователю лучше сформулировать запрос.""",
                json_schema="[QUESTIONS]",
                constraints="""
                    2 <= len(QUESTIONS) <= 3 and
                    all(isinstance(q, str) for q in QUESTIONS)
                """,
                temperature=0.5,
                max_tokens=512,
            )

            if isinstance(response_data, list) and all(
                isinstance(q, str) for q in response_data
            ):
                return response_data[:3]
            else:
                raise RuntimeError(f"Неожиданный формат ответа LMQL: {response_data}")

        except Exception as e:
            logger.error(f"Ошибка при генерации уточняющих вопросов через LMQL: {e}")
            raise RuntimeError(f"Ошибка генерации уточняющих вопросов: {e}") from e

    def _create_lmql_questions_query(self, query: str, results_preview: str) -> str:
        """Создание LMQL запроса для генерации уточняющих вопросов.
        
        Args:
            query: Поисковый запрос
            results_preview: Превью результатов поиска
            
        Returns:
            Строка с LMQL запросом (для логирования/отладки)
        """
        # Этот метод возвращает строку для отладки, фактический запрос выполняется в execute_json_query
        return f"LMQL query for clarifying questions: {query[:50]}..."

    async def _suggest_refinements(
        self, query: str, results: List[SearchResultItem]
    ) -> List[str]:
        """Генерация предложений по уточнению запроса.
        
        Returns:
            Список альтернативных формулировок запроса
            
        Raises:
            RuntimeError: Если LMQL не настроен или произошла ошибка
        """
        if not self.lmql_adapter:
            raise RuntimeError(
                "LMQL не настроен. Установите MEMORY_MCP_USE_LMQL=true и настройте модель"
            )

        try:
            lmql_query_str = self._create_lmql_refinements_query(query)
            response_data = await self.lmql_adapter.execute_json_query(
                prompt=f"""Ты помощник для улучшения поиска. Пользователь искал: "{query}"

Предложи 2-3 альтернативных формулировки этого запроса, которые могут дать лучшие результаты.""",
                json_schema="[REFINEMENTS]",
                constraints="""
                    2 <= len(REFINEMENTS) <= 3 and
                    all(isinstance(r, str) for r in REFINEMENTS)
                """,
                temperature=0.5,
                max_tokens=512,
            )

            if isinstance(response_data, list) and all(
                isinstance(r, str) for r in response_data
            ):
                return response_data[:3]
            else:
                raise RuntimeError(f"Неожиданный формат ответа LMQL: {response_data}")

        except Exception as e:
            logger.error(f"Ошибка при генерации предложений через LMQL: {e}")
            raise RuntimeError(f"Ошибка генерации предложений: {e}") from e

    def _create_lmql_refinements_query(self, query: str) -> str:
        """Создание LMQL запроса для генерации предложений по уточнению.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Строка с LMQL запросом (для логирования/отладки)
        """
        # Этот метод возвращает строку для отладки, фактический запрос выполняется в execute_json_query
        return f"LMQL query for refinements: {query[:50]}..."

    def _boost_results_with_graph_connections(
        self, results: List[SearchResultItem], query: str
    ) -> List[SearchResultItem]:
        """Повышение релевантности результатов на основе графа связей."""
        if not self.connection_builder or not self.connection_builder.graph:
            return results

        query_entities = self.entity_enricher.extract_entities_from_query(query)
        if not query_entities:
            return results

        entity_names = [e.get("value", "") for e in query_entities if e.get("value")]

        boosted_results = []
        for result in results:
            result_id = result.record_id
            
            connections = self.connection_builder.find_connections(
                query_entities=entity_names,
                result_id=result_id,
                max_paths=3,
                max_depth=3,
            )

            boost_factor = 1.0
            if connections:
                from ..config import get_settings
                settings = get_settings()
                
                min_path_length = min(c.path_length for c in connections)
                
                if min_path_length == 1:
                    boost_factor = settings.search_boost_path_1
                elif min_path_length == 2:
                    boost_factor = settings.search_boost_path_2
                elif min_path_length == 3:
                    boost_factor = settings.search_boost_path_3
                
                max_strength = max(c.path_strength for c in connections)
                boost_factor += max_strength * settings.search_boost_strength_multiplier
                
                logger.debug(
                    f"Буст для результата {result_id}: "
                    f"путь={min_path_length}, сила={max_strength:.2f}, "
                    f"фактор={boost_factor:.2f}"
                )

            result.score *= boost_factor
            boosted_results.append(result)

        boosted_results.sort(key=lambda x: x.score, reverse=True)
        
        return boosted_results

    def _get_personalization_context(self) -> Optional[Dict[str, Any]]:
        """Получение контекста персонализации на основе истории сессий."""
        try:
            feedback_history = self.session_store.get_feedback_for_learning(limit=100)
            
            if not feedback_history:
                return None

            relevant_count = 0
            irrelevant_count = 0
            db_preference = 0.0
            artifact_preference = 0.0
            total_feedback = 0

            for feedback in feedback_history:
                relevance = feedback.get("relevance", "")
                result_id = feedback.get("result_id", "")
                artifact_path = feedback.get("artifact_path")
                
                result_type = ""
                if artifact_path or (result_id and result_id.startswith("artifact:")):
                    result_type = "artifact"
                elif result_id:
                    result_type = "db_record"
                
                if relevance == "relevant":
                    relevant_count += 1
                    if result_type == "db_record":
                        db_preference += 1.0
                    elif result_type == "artifact":
                        artifact_preference += 1.0
                elif relevance == "irrelevant":
                    irrelevant_count += 1
                
                total_feedback += 1

            if total_feedback == 0:
                return None

            if db_preference + artifact_preference > 0:
                db_ratio = db_preference / (db_preference + artifact_preference)
                artifact_ratio = artifact_preference / (db_preference + artifact_preference)
                
                adapted_db_weight = 0.4 + (db_ratio * 0.4)
                adapted_artifact_weight = 0.4 + (artifact_ratio * 0.4)
            else:
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

