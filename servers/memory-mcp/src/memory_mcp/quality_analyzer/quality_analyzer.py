#!/usr/bin/env python3
"""
Основной класс для анализа качества индексации и поиска

Координирует работу всех компонентов системы анализа качества:
- Генерация тестовых запросов
- Анализ релевантности через LLM
- Расчет метрик и создание отчетов
- Управление историей анализов
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from ..core.langchain_adapters import LangChainEmbeddingAdapter, build_langchain_embeddings_from_env
from ..search import HybridSearchManager
from .core import (
    HistoryManager,
    MetricsCalculator,
    QueryGenerator,
    RelevanceAnalyzer,
    ReportGenerator,
)
from .models import (
    BatchResultSchema,
    QuerySchema,
    RelevanceAnalysisSchema,
    SearchResultSchema,
)
from .utils import BatchConfig, BatchManager, log_error

logger = logging.getLogger(__name__)


class QualityAnalyzer:
    """Основной класс для анализа качества индексации и поиска"""

    def __init__(
        self,
        llm_model: str = "gpt-oss-20b:latest",
        llm_base_url: str = "http://localhost:1234",
        max_context_tokens: int = 131072,  # Для gpt-oss-20b
        temperature: float = 0.1,
        max_response_tokens: int = 131072,  # Для gpt-oss-20b (максимальный лимит)
        thinking_level: Optional[str] = None,
        reports_dir: Path = Path("artifacts/reports"),
        history_dir: Path = Path("quality_analysis_history"),
        reports_subdir: Optional[str] = "quality_analysis",
        results_per_query: int = 10,
        qdrant_url: Optional[str] = None,
        search_collection: str = "chat_messages",
        hybrid_alpha: float = 0.6,
        batch_max_size: int = 10,
        system_prompt_reserve: float = 0.2,
        max_query_tokens: int = 6000,
    ):
        """
        Инициализация анализатора качества

        Args:
            llm_model: Модель LLM для анализа (используется через LangChain)
            llm_base_url: URL LM Studio сервера
            max_context_tokens: Максимальное количество токенов в контексте
            temperature: Температура для генерации
            max_response_tokens: Максимальное количество токенов ответа
            thinking_level: Уровень мышления (thinking)
            reports_dir: Директория с отчетами
            history_dir: Директория для хранения истории анализов
            qdrant_url: URL Qdrant сервера (если None, берется из настроек)
        """
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.max_context_tokens = max_context_tokens
        self.reports_dir = reports_dir
        self.history_dir = history_dir
        self.results_per_query = results_per_query
        self.qdrant_url = qdrant_url
        self.search_collection = search_collection
        self.hybrid_alpha = hybrid_alpha
        self.thinking_level = thinking_level

        # Создаем компоненты
        self.query_generator = QueryGenerator()
        self.relevance_analyzer = RelevanceAnalyzer(
            model_name=llm_model,
            base_url=llm_base_url,
            max_context_tokens=max_context_tokens,
            temperature=temperature,
            max_response_tokens=max_response_tokens,
            thinking_level=thinking_level,
        )
        self.metrics_calculator = MetricsCalculator()
        self.report_generator = ReportGenerator(
            reports_dir,
            quality_subdir=reports_subdir,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            temperature=temperature,
            max_tokens=max_response_tokens,
            thinking_level=thinking_level,
        )
        self.history_manager = HistoryManager(history_dir)
        self.batch_manager = BatchManager(
            BatchConfig(
                max_context_tokens=max_context_tokens,
                system_prompt_reserve=system_prompt_reserve,
                max_batch_size=batch_max_size,
                max_query_tokens=max_query_tokens,
            )
        )

        # Компоненты поиска и эмбеддингов
        # Используем LangChain для эмбеддингов
        self._embedding_client = build_langchain_embeddings_from_env()
        if self._embedding_client is None:
            raise ValueError(
                "Не удалось инициализировать LangChain Embeddings. "
                "Убедитесь, что LangChain установлен и настройки эмбеддингов корректны."
            )
        self._qdrant_client = None
        self._search_manager: Optional[HybridSearchManager] = None

        # Создаем директории если не существуют
        self.history_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Инициализирован QualityAnalyzer (модель: {llm_model}, Qdrant: {qdrant_url})")

    async def analyze_chat_quality(
        self,
        chat_name: str,
        chat_data: List[Dict[str, Any]],
        test_queries: Optional[List[Dict[str, Any]]] = None,
        batch_size: Optional[int] = None,
        max_queries: Optional[int] = None,
        custom_queries: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Анализ качества для конкретного чата

        Args:
            chat_name: Название чата
            chat_data: Данные чата (сообщения, сессии, задачи)
            test_queries: Предустановленные тестовые запросы
            batch_size: Размер батча для обработки

        Returns:
            Результаты анализа качества
        """
        logger.info(f"Начинаем анализ качества для чата: {chat_name}")

        # Генерируем тестовые запросы если не предоставлены
        if max_queries is None:
            max_queries = 20

        if not test_queries:
            logger.info("Генерация тестовых запросов...")

            generated_queries: List[Dict[str, Any]] = []

            if custom_queries:
                logger.info(
                    "Добавление пользовательских запросов (%d шт.)",
                    len(custom_queries),
                )
                generated_queries.extend(
                    self.query_generator.generate_custom_queries(
                        chat_name, custom_queries
                    )
                )

            remaining_slots = max(0, max_queries - len(generated_queries))

            if remaining_slots > 0:
                logger.info(
                    "Генерация дополнительных запросов (%d шт.)",
                    remaining_slots,
                )
                generated_queries.extend(
                    await self.query_generator.generate_queries_for_chat(
                        chat_name,
                        chat_data,
                        max_queries=remaining_slots,
                    )
                )
            elif generated_queries:
                logger.info(
                    "Количество пользовательских запросов превышает лимит (%d) — лишние будут отброшены",
                    max_queries,
                )
                generated_queries = generated_queries[:max_queries]

            test_queries = generated_queries

        self.batch_manager.normalize_queries(test_queries)
        logger.info(f"Подготовлено {len(test_queries)} тестовых запросов")

        if batch_size:
            batches = [
                test_queries[i : i + batch_size]
                for i in range(0, len(test_queries), batch_size)
            ]
            logger.info("Используется фиксированный размер батча %d", batch_size)
        else:
            batches = list(self.batch_manager.split(test_queries))
            logger.info(
                "Размеры батчей выбраны динамически (всего %d батчей)", len(batches)
            )

        all_results = []
        captured_errors: List[str] = []
        for index, batch in enumerate(batches, start=1):
            logger.info("Обработка батча %d/%d", index, len(batches))

            batch_results = await self._process_query_batch(chat_name, batch)
            all_results.extend(batch_results)
            batch_errors = [
                item.get("error") for item in batch_results if item.get("error")
            ]
            captured_errors.extend(error for error in batch_errors if error)

        # Рассчитываем метрики
        logger.info("Расчет метрик качества...")
        metrics = self.metrics_calculator.calculate_metrics(all_results)

        logger.info("Запрос рекомендаций у LLM...")
        try:
            llm_recommendations = (
                await self.report_generator.generate_llm_recommendations(
                    chat_name,
                    metrics,
                    all_results,
                )
            )
        except Exception as exc:  # pragma: no cover - зависит от внешнего сервиса
            logger.warning("Не удалось получить рекомендации через LLM: %s", exc)
            llm_recommendations = []

        # Создаем отчет
        logger.info("Создание отчета...")
        report = self.report_generator.generate_chat_report(
            chat_name,
            all_results,
            metrics,
            llm_recommendations=llm_recommendations,
        )

        # Сохраняем историю
        logger.info("Сохранение истории анализа...")
        analysis_record = {
            "chat_name": chat_name,
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(test_queries),
            "results": all_results,
            "metrics": metrics,
            "ai_recommendations": llm_recommendations,
            "report": report,
        }
        if captured_errors:
            analysis_record["errors"] = captured_errors

        self.history_manager.save_analysis(analysis_record)

        logger.info(f"Анализ качества для чата {chat_name} завершен")

        return {
            "chat_name": chat_name,
            "total_queries": len(test_queries),
            "results": all_results,
            "metrics": metrics,
            "report": report,
            "ai_recommendations": llm_recommendations,
            "errors": captured_errors,
        }

    async def analyze_multiple_chats(
        self,
        chats_data: Dict[str, List[Dict[str, Any]]],
        selected_chats: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        max_queries: Optional[int] = None,
        custom_queries: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Анализ качества для нескольких чатов

        Args:
            chats_data: Данные всех чатов
            selected_chats: Список чатов для анализа (если None - все чаты)

        Returns:
            Результаты анализа для всех чатов
        """
        logger.info("Начинаем анализ качества для нескольких чатов")

        if selected_chats:
            chats_to_analyze = {
                name: data
                for name, data in chats_data.items()
                if name in selected_chats
            }
        else:
            chats_to_analyze = chats_data

        logger.info(f"Будет проанализировано чатов: {len(chats_to_analyze)}")

        # Анализируем каждый чат
        chat_results = {}
        overall_errors: Dict[str, List[str]] = {}

        for chat_name, chat_data in chats_to_analyze.items():
            try:
                result = await self.analyze_chat_quality(
                    chat_name,
                    chat_data,
                    batch_size=batch_size,
                    max_queries=max_queries,
                    custom_queries=custom_queries,
                )
                chat_results[chat_name] = result
                if result.get("errors"):
                    overall_errors[chat_name] = result["errors"]
            except Exception as e:
                log_error(f"Ошибка анализа чата {chat_name}", e)
                chat_results[chat_name] = {
                    "error": str(e),
                    "chat_name": chat_name,
                }

        # Рассчитываем общие метрики
        logger.info("Расчет общих метрик...")
        overall_metrics = self.metrics_calculator.calculate_overall_metrics(
            chat_results
        )

        # Создаем общий отчет
        logger.info("Создание общего отчета...")
        overall_report = self.report_generator.generate_overall_report(
            chat_results, overall_metrics
        )

        # Сохраняем общую историю
        logger.info("Сохранение общей истории анализа...")
        overall_record = {
            "timestamp": datetime.now().isoformat(),
            "total_chats": len(chats_to_analyze),
            "chat_results": chat_results,
            "overall_metrics": overall_metrics,
            "overall_report": overall_report,
        }
        if overall_errors:
            overall_record["errors"] = overall_errors

        self.history_manager.save_overall_analysis(overall_record)

        logger.info("Анализ качества для нескольких чатов завершен")

        return {
            "total_chats": len(chats_to_analyze),
            "chat_results": chat_results,
            "overall_metrics": overall_metrics,
            "overall_report": overall_report,
        }

    async def _process_query_batch(
        self,
        chat_name: str,
        batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Обработка батча тестовых запросов

        Args:
            chat_name: Название чата
            batch: Батч тестовых запросов

        Returns:
            Результаты анализа для батча
        """
        logger.debug(f"Обработка батча из {len(batch)} запросов для чата {chat_name}")

        batch_results = []

        for query_data in batch:
            try:
                if "chat_name" not in query_data:
                    query_data["chat_name"] = chat_name

                try:
                    query_model = QuerySchema(**query_data)
                except ValidationError as exc:
                    log_error("Ошибка валидации запроса", exc)
                    batch_results.append(
                        {
                            "query": query_data,
                            "search_results": [],
                            "relevance_analysis": {
                                "overall_score": 0.0,
                                "individual_scores": [],
                                "problems": {},
                                "explanation": "Запрос не прошел валидацию",
                                "recommendations": [],
                            },
                            "timestamp": datetime.now().isoformat(),
                            "error": f"Query validation error: {exc}",
                        }
                    )
                    continue

                search_results = await self._perform_search(query_data)

                search_models: List[SearchResultSchema] = []
                for item in search_results:
                    try:
                        search_models.append(
                            SearchResultSchema(
                                id=item.get("id") or item.get("result_id", ""),
                                text=item.get("text") or item.get("content", ""),
                                score=float(item.get("score", 0.0)),
                                query_id=query_model.id,
                                metadata=item.get("metadata", {}),
                            )
                        )
                    except ValidationError as exc:
                        log_error("Ошибка валидации результата поиска", exc)

                relevance_analysis = await self.relevance_analyzer.analyze_relevance(
                    query_data, search_results
                )

                try:
                    relevance_model = RelevanceAnalysisSchema(**relevance_analysis)
                except ValidationError as exc:
                    log_error("Ошибка валидации анализа релевантности", exc)
                    relevance_model = RelevanceAnalysisSchema(
                        overall_score=0.0,
                        individual_scores=[],
                        problems={},
                        explanation="Анализ релевантности не прошел валидацию",
                        recommendations=[],
                    )

                batch_schema = BatchResultSchema(
                    query=query_model,
                    search_results=search_models,
                    relevance_analysis=relevance_model,
                    timestamp=datetime.now(),
                )

                batch_results.append(batch_schema.model_dump(mode="json"))

            except Exception as e:
                log_error("Ошибка обработки запроса", e)
                batch_results.append(
                    {
                        "query": query_data,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return batch_results

    async def _perform_search(self, query_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Выполнение поиска через memory_mcp

        Args:
            query_data: Данные тестового запроса

        Returns:
            Результаты поиска
        """
        # Здесь будет интеграция с существующей системой поиска memory_mcp
        # Пока возвращаем заглушку
        logger.debug(f"Выполнение поиска для запроса: {query_data.get('query', 'N/A')}")
        query_text = query_data.get("query", "").strip()
        if not query_text:
            logger.warning("Получен пустой текст запроса, результаты поиска пропущены")
            return []

        search_engine = self._get_search_engine()
        if not search_engine:
            raise RuntimeError(
                "Поисковый движок недоступен для анализа качества. "
                "Проверьте конфигурацию search_engine."
            )

        query_embedding = await self._generate_query_embedding(query_text)
        if query_embedding is None:
            raise RuntimeError(
                f"Не удалось получить эмбеддинг для запроса '{query_text}'. "
                "Проверьте конфигурацию embedding_service."
            )

        chat_name = query_data.get("chat_name") or query_data.get("chat")
        where_filter = {"chat": chat_name} if chat_name else None

        try:
            results = search_engine.search(
                query=query_text,
                query_embedding=query_embedding,
                top_k=self.results_per_query,
                where=where_filter,
                use_hybrid=True,
            )
        except Exception as exc:
            log_error("Ошибка выполнения поиска", exc)
            return []

        return results

    async def _generate_query_embedding(self, query_text: str) -> Optional[List[float]]:
        """Создание эмбеддинга для поискового запроса."""
        try:
            async with self._embedding_client:
                embeddings = await self._embedding_client.generate_embeddings(
                    [query_text]
                )
            if embeddings:
                return embeddings[0]
        except Exception as exc:
            log_error("Ошибка генерации эмбеддинга запроса", exc)
            return None

    def _get_search_engine(self):
        """Получение поискового движка для коллекции сообщений через Qdrant."""
        if self._search_manager is None:
            try:
                from qdrant_client import QdrantClient
                from ..memory.qdrant_collections import QdrantCollectionsManager
                
                # Получаем URL Qdrant из настроек, если не указан
                qdrant_url = self.qdrant_url
                if qdrant_url is None:
                    from ..config import get_settings
                    settings = get_settings()
                    qdrant_url = settings.get_qdrant_url()
                
                if not qdrant_url:
                    logger.warning("Qdrant URL не настроен, поисковый движок недоступен")
                    return None
                
                qdrant_client = QdrantClient(url=qdrant_url, check_compatibility=False)
                collections_manager = QdrantCollectionsManager(url=qdrant_url)
                self._qdrant_client = qdrant_client
                
                # HybridSearchManager может работать с Qdrant через адаптер
                # Пока используем прямой доступ к Qdrant
                self._search_manager = collections_manager
            except Exception as exc:
                log_error("Не удалось инициализировать Qdrant клиент", exc)
                self._search_manager = None
                return None

        try:
            # Возвращаем коллекцию из Qdrant
            if hasattr(self._search_manager, 'get_collection'):
                return self._search_manager.get_collection(self.search_collection)
            return self._search_manager
        except Exception as exc:
            log_error("Не удалось получить поисковый движок", exc)
            return None

    async def get_analysis_history(
        self,
        chat_name: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Получение истории анализов

        Args:
            chat_name: Фильтр по чату
            limit: Максимальное количество записей

        Returns:
            История анализов
        """
        return self.history_manager.get_history(chat_name, limit)

    async def compare_analyses(
        self,
        analysis_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Сравнение нескольких анализов

        Args:
            analysis_ids: ID анализов для сравнения

        Returns:
            Результаты сравнения
        """
        return self.history_manager.compare_analyses(analysis_ids)

    async def generate_improvement_recommendations(
        self,
        analysis_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Генерация рекомендаций по улучшению

        Args:
            analysis_results: Результаты анализа качества

        Returns:
            Список рекомендаций
        """
        logger.info("Генерация рекомендаций по улучшению...")

        recommendations = []

        # Анализируем метрики и генерируем рекомендации
        metrics = analysis_results.get("metrics", {})

        # Рекомендации по проблемам индексации
        if metrics.get("indexing_issues", 0) > 0:
            recommendations.append(
                {
                    "type": "indexing",
                    "priority": "high",
                    "title": "Проблемы с индексацией",
                    "description": f"Обнаружено {metrics['indexing_issues']} проблем с индексацией",
                    "suggestions": [
                        "Проверить настройки индексации",
                        "Увеличить размер чанков для длинных сообщений",
                        "Проверить фильтрацию сообщений",
                    ],
                }
            )

        # Рекомендации по проблемам поиска
        if metrics.get("search_issues", 0) > 0:
            recommendations.append(
                {
                    "type": "search",
                    "priority": "medium",
                    "title": "Проблемы с поиском",
                    "description": f"Обнаружено {metrics['search_issues']} проблем с поиском",
                    "suggestions": [
                        "Настроить веса гибридного поиска",
                        "Улучшить токенизацию для русского языка",
                        "Проверить пороги релевантности",
                    ],
                }
            )

        # Рекомендации по проблемам контекста
        if metrics.get("context_issues", 0) > 0:
            recommendations.append(
                {
                    "type": "context",
                    "priority": "medium",
                    "title": "Проблемы с контекстом",
                    "description": f"Обнаружено {metrics['context_issues']} проблем с контекстом",
                    "suggestions": [
                        "Улучшить группировку сообщений в сессии",
                        "Настроить параметры кластеризации",
                        "Проверить качество саммаризации",
                    ],
                }
            )

        logger.info(f"Сгенерировано {len(recommendations)} рекомендаций")
        return recommendations
