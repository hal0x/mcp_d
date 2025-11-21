#!/usr/bin/env python3
"""
Калькулятор метрик качества

Рассчитывает сравнительные метрики качества индексации и поиска:
- Базовые метрики (средняя оценка, процент успешных поисков)
- Детальные метрики по типам запросов и чатам
- Сравнительные метрики с предыдущими запусками
- Статистический анализ проблем
"""

import logging
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import (
    AnalysisResult,
    BatchResultSchema,
    QualityMetrics,
    QuerySchema,
    RelevanceAnalysisSchema,
    SearchResultSchema,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Калькулятор метрик качества"""

    def __init__(self):
        """Инициализация калькулятора метрик"""
        self.metric_weights = {
            "factual": 1.0,
            "contextual": 1.2,
            "analytical": 1.5,
            "custom": 1.0,
        }

        logger.info("Инициализирован MetricsCalculator")

    def calculate_metrics(
        self,
        analysis_results: List[Any],
        historical_data: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Расчет метрик качества

        Args:
            analysis_results: Результаты анализа релевантности
            historical_data: Исторические данные для сравнения

        Returns:
            Рассчитанные метрики
        """
        logger.info(f"Расчет метрик для {len(analysis_results)} результатов анализа")

        if not analysis_results:
            return self._create_empty_metrics()

        normalized: List[BatchResultSchema] = []
        for result in analysis_results:
            normalized.extend(self._to_batch_results(result))

        basic_metrics = self._calculate_basic_metrics(normalized)

        type_metrics = self._calculate_type_metrics(normalized)

        problem_metrics = self._calculate_problem_metrics(normalized)

        temporal_metrics = self._calculate_temporal_metrics(normalized)

        historical_normalized: List[BatchResultSchema] = []
        if historical_data:
            for item in historical_data:
                historical_normalized.extend(self._to_batch_results(item))

        comparative_metrics = self._calculate_comparative_metrics(
            normalized,
            historical_normalized if historical_normalized else None,
        )

        # Объединяем все метрики
        metrics_model = QualityMetrics(
            average_score=basic_metrics.get("average_score", 0.0),
            median_score=basic_metrics.get("median_score", 0.0),
            success_rate=basic_metrics.get("success_rate", 0.0),
            total_queries=basic_metrics.get("total_queries", 0),
            successful_queries=basic_metrics.get("successful_queries", 0),
            details={
                "basic": basic_metrics,
                "by_type": type_metrics,
                "problems": problem_metrics,
                "temporal": temporal_metrics,
                "comparative": comparative_metrics,
            },
        )

        logger.info("Расчет метрик завершен")
        return metrics_model.model_dump()

    def _to_batch_results(self, result: Any) -> List[BatchResultSchema]:
        if isinstance(result, BatchResultSchema):
            return [result]

        if isinstance(result, dict):
            try:
                return [BatchResultSchema(**result)]
            except Exception as exc:  # pragma: no cover - логирование
                logger.error("Не удалось интерпретировать результат анализа: %s", exc)
                return []

        if isinstance(result, AnalysisResult):
            batch_results: List[BatchResultSchema] = []

            score_map = {score.query_id: score for score in result.relevance_scores}
            results_by_query: Dict[str, List[SearchResultSchema]] = {}
            for search_result in result.search_results:
                try:
                    schema = SearchResultSchema(
                        id=search_result.result_id,
                        text=search_result.content,
                        score=search_result.score,
                        query_id=search_result.query_id,
                        metadata=search_result.metadata,
                    )
                    results_by_query.setdefault(search_result.query_id, []).append(
                        schema
                    )
                except Exception as exc:  # pragma: no cover
                    logger.error("Ошибка преобразования результата поиска: %s", exc)

            for query in result.queries:
                score = score_map.get(query.query_id)
                if not score:
                    continue

                try:
                    query_schema = QuerySchema(
                        id=query.query_id,
                        query=query.text,
                        type=query.query_type,
                        chat_name=query.chat_name,
                        entity=query.entity,
                        timeframe=query.timeframe,
                        generated_at=query.generated_at,
                    )

                    relevance_schema = RelevanceAnalysisSchema(
                        overall_score=score.overall_score,
                        individual_scores=score.individual_scores,
                        problems=score.problems,
                        explanation=score.explanation,
                        recommendations=score.recommendations,
                    )

                    batch_results.append(
                        BatchResultSchema(
                            query=query_schema,
                            search_results=results_by_query.get(query.query_id, []),
                            relevance_analysis=relevance_schema,
                            timestamp=result.timestamp,
                        )
                    )
                except Exception as exc:  # pragma: no cover
                    logger.error("Ошибка преобразования Query -> BatchResult: %s", exc)

            return batch_results

        return []

    def _calculate_basic_metrics(
        self, analysis_results: List[BatchResultSchema]
    ) -> Dict[str, Any]:
        """
        Расчет базовых метрик

        Args:
            analysis_results: Результаты анализа

        Returns:
            Базовые метрики
        """
        scores = []
        successful_searches = 0
        total_queries = len(analysis_results)

        for result in analysis_results:
            overall_score = result.relevance_analysis.overall_score

            if overall_score > 0:
                scores.append(overall_score)
                if overall_score >= 5:  # Порог успешного поиска
                    successful_searches += 1

        if not scores:
            return {
                "average_score": 0.0,
                "median_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "success_rate": 0.0,
                "total_queries": total_queries,
                "successful_queries": 0,
            }

        return {
            "average_score": statistics.mean(scores),
            "median_score": statistics.median(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "std_deviation": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "success_rate": successful_searches / total_queries,
            "total_queries": total_queries,
            "successful_queries": successful_searches,
        }

    def _calculate_type_metrics(
        self, analysis_results: List[BatchResultSchema]
    ) -> Dict[str, Any]:
        """
        Расчет метрик по типам запросов

        Args:
            analysis_results: Результаты анализа

        Returns:
            Метрики по типам
        """
        type_stats = {}

        for result in analysis_results:
            query_type = result.query.type
            overall_score = result.relevance_analysis.overall_score

            if query_type not in type_stats:
                type_stats[query_type] = {
                    "scores": [],
                    "count": 0,
                    "successful": 0,
                }

            type_stats[query_type]["scores"].append(overall_score)
            type_stats[query_type]["count"] += 1
            if overall_score >= 5:
                type_stats[query_type]["successful"] += 1

        # Рассчитываем метрики для каждого типа
        type_metrics = {}
        for query_type, stats in type_stats.items():
            if stats["scores"]:
                type_metrics[query_type] = {
                    "average_score": statistics.mean(stats["scores"]),
                    "median_score": statistics.median(stats["scores"]),
                    "success_rate": stats["successful"] / stats["count"],
                    "total_queries": stats["count"],
                    "successful_queries": stats["successful"],
                }
            else:
                type_metrics[query_type] = {
                    "average_score": 0.0,
                    "median_score": 0.0,
                    "success_rate": 0.0,
                    "total_queries": stats["count"],
                    "successful_queries": 0,
                }

        return type_metrics

    def _calculate_problem_metrics(
        self, analysis_results: List[BatchResultSchema]
    ) -> Dict[str, Any]:
        """
        Расчет метрик проблем

        Args:
            analysis_results: Результаты анализа

        Returns:
            Метрики проблем
        """
        total_problems = {
            "indexing": 0,
            "search": 0,
            "context": 0,
        }

        problem_details = {
            "indexing": [],
            "search": [],
            "context": [],
        }

        for result in analysis_results:
            problems = result.relevance_analysis.problems
            query_data = result.query

            for problem_type, count in problems.items():
                if problem_type in total_problems:
                    total_problems[problem_type] += count

                    # Сохраняем детали проблем
                    if count > 0:
                        if isinstance(query_data, dict):
                            query_text = query_data.get("query", "")
                            query_type = query_data.get("type", "unknown")
                        else:
                            query_text = getattr(query_data, "query", None) or getattr(
                                query_data, "text", ""
                            )
                            query_type = getattr(query_data, "type", "unknown")

                        problem_details[problem_type].append(
                            {
                                "query": query_text,
                                "query_type": query_type,
                                "count": count,
                                "score": result.relevance_analysis.overall_score,
                            }
                        )

        # Рассчитываем статистику проблем
        problem_stats = {}
        for problem_type, count in total_problems.items():
            problem_stats[problem_type] = {
                "total_count": count,
                "affected_queries": len(problem_details[problem_type]),
                "average_score_when_problem": 0.0,
            }

            if problem_details[problem_type]:
                scores = [detail["score"] for detail in problem_details[problem_type]]
                problem_stats[problem_type][
                    "average_score_when_problem"
                ] = statistics.mean(scores)

        return {
            "total_problems": total_problems,
            "problem_stats": problem_stats,
            "problem_details": problem_details,
        }

    def _calculate_temporal_metrics(
        self, analysis_results: List[BatchResultSchema]
    ) -> Dict[str, Any]:
        """
        Расчет временных метрик

        Args:
            analysis_results: Результаты анализа

        Returns:
            Временные метрики
        """
        # Группируем по времени выполнения
        hourly_stats = {}

        for result in analysis_results:
            timestamp = result.timestamp
            hour = timestamp.hour

            if hour not in hourly_stats:
                hourly_stats[hour] = {
                    "scores": [],
                    "count": 0,
                }

            overall_score = result.relevance_analysis.overall_score

            hourly_stats[hour]["scores"].append(overall_score)
            hourly_stats[hour]["count"] += 1

        # Рассчитываем метрики по часам
        temporal_metrics = {}
        for hour, stats in hourly_stats.items():
            if stats["scores"]:
                temporal_metrics[f"hour_{hour}"] = {
                    "average_score": statistics.mean(stats["scores"]),
                    "query_count": stats["count"],
                }

        return temporal_metrics

    def _calculate_comparative_metrics(
        self,
        current_results: List[BatchResultSchema],
        historical_data: Optional[List[BatchResultSchema]],
    ) -> Dict[str, Any]:
        """
        Расчет сравнительных метрик

        Args:
            current_results: Текущие результаты
            historical_data: Исторические данные

        Returns:
            Сравнительные метрики
        """
        if not historical_data:
            return {
                "trend": "no_data",
                "improvement": 0.0,
                "comparison_available": False,
            }

        # Рассчитываем текущие метрики
        current_metrics = self._calculate_basic_metrics(current_results)
        current_avg_score = current_metrics.get("average_score", 0.0)
        current_success_rate = current_metrics.get("success_rate", 0.0)

        # Рассчитываем исторические метрики
        historical_scores = []
        historical_successful = 0
        historical_total = 0

        for historical_result in historical_data:
            overall_score = historical_result.relevance_analysis.overall_score

            historical_scores.append(overall_score)
            historical_total += 1
            if overall_score >= 5:
                historical_successful += 1

        if not historical_scores:
            return {
                "trend": "no_historical_data",
                "improvement": 0.0,
                "comparison_available": False,
            }

        historical_avg_score = statistics.mean(historical_scores)
        historical_success_rate = historical_successful / historical_total

        # Рассчитываем улучшения
        score_improvement = current_avg_score - historical_avg_score
        success_rate_improvement = current_success_rate - historical_success_rate

        # Определяем тренд
        if score_improvement > 0.5:
            trend = "improving"
        elif score_improvement < -0.5:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "score_improvement": score_improvement,
            "success_rate_improvement": success_rate_improvement,
            "current_avg_score": current_avg_score,
            "historical_avg_score": historical_avg_score,
            "current_success_rate": current_success_rate,
            "historical_success_rate": historical_success_rate,
            "comparison_available": True,
        }

    def calculate_overall_metrics(self, chat_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Расчет общих метрик для нескольких чатов

        Args:
            chat_results: Результаты анализа для всех чатов

        Returns:
            Общие метрики
        """
        logger.info(f"Расчет общих метрик для {len(chat_results)} чатов")

        all_scores = []
        all_problems = {"indexing": 0, "search": 0, "context": 0}
        chat_metrics = {}

        for chat_name, chat_result in chat_results.items():
            if "error" in chat_result:
                continue

            results = chat_result.get("results", [])
            metrics = chat_result.get("metrics", {})

            # Собираем оценки
            for result in results:
                relevance_analysis = result.get("relevance_analysis", {})
                overall_score = relevance_analysis.get("overall_score", 0)
                if overall_score > 0:
                    all_scores.append(overall_score)

            # Собираем проблемы
            problem_metrics = metrics.get("problems", {})
            total_problems = problem_metrics.get("total_problems", {})
            for problem_type, count in total_problems.items():
                all_problems[problem_type] += count

            # Сохраняем метрики чата
            chat_metrics[chat_name] = {
                "average_score": metrics.get("basic", {}).get("average_score", 0.0),
                "success_rate": metrics.get("basic", {}).get("success_rate", 0.0),
                "total_queries": metrics.get("basic", {}).get("total_queries", 0),
            }

        # Рассчитываем общие метрики
        overall_metrics = {
            "total_chats": len(chat_results),
            "successful_chats": len(
                [c for c in chat_results.values() if "error" not in c]
            ),
            "average_score": statistics.mean(all_scores) if all_scores else 0.0,
            "median_score": statistics.median(all_scores) if all_scores else 0.0,
            "total_problems": all_problems,
            "chat_metrics": chat_metrics,
        }

        logger.info("Расчет общих метрик завершен")
        return overall_metrics

    def _create_empty_metrics(self) -> Dict[str, Any]:
        """
        Создание пустых метрик

        Returns:
            Пустые метрики
        """
        return {
            "basic": {
                "average_score": 0.0,
                "median_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "std_deviation": 0.0,
                "success_rate": 0.0,
                "total_queries": 0,
                "successful_queries": 0,
            },
            "by_type": {},
            "problems": {
                "total_problems": {"indexing": 0, "search": 0, "context": 0},
                "problem_stats": {},
                "problem_details": {},
            },
            "temporal": {},
            "comparative": {
                "trend": "no_data",
                "improvement": 0.0,
                "comparison_available": False,
            },
            "calculated_at": datetime.now().isoformat(),
        }
