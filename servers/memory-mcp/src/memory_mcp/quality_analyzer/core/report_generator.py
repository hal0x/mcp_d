#!/usr/bin/env python3
"""
Генератор отчетов анализа качества

Создает Markdown отчеты с результатами анализа качества:
- Отчеты по отдельным чатам
- Общие отчеты по всем чатам
- Сравнительные отчеты с историей
- Рекомендации по улучшению
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.lmql_adapter import LMQLAdapter, build_lmql_adapter_from_env

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Генератор отчетов анализа качества"""

    def __init__(
        self,
        reports_dir: Path = Path("artifacts/reports"),
        quality_subdir: Optional[str] = "quality_analysis",
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 131072,  # Для gpt-oss-20b (максимальный лимит)
        thinking_level: Optional[str] = None,
        lmql_adapter: Optional[LMQLAdapter] = None,
    ):
        """
        Инициализация генератора отчетов

        Args:
            reports_dir: Директория для сохранения отчетов
            quality_subdir: Поддиректория для отчетов качества
            llm_model: Модель LLM для генерации рекомендаций
            llm_base_url: URL LM Studio сервера
            temperature: Температура для генерации
            max_tokens: Максимальное количество токенов
            thinking_level: Уровень мышления (thinking)
            lmql_adapter: Опциональный LMQL адаптер для структурированной генерации.
                         Если не указан, создается из настроек окружения.
        """
        self.reports_dir = reports_dir
        if quality_subdir:
            self.quality_reports_dir = reports_dir / quality_subdir
        else:
            self.quality_reports_dir = reports_dir

        self.quality_reports_dir.mkdir(parents=True, exist_ok=True)

        from ...core.langchain_adapters import LangChainLLMAdapter
        from .templates import DEFAULT_PROMPTS_DIR, PromptTemplateManager

        self.report_template_manager = PromptTemplateManager(
            Path(__file__).resolve().parent.parent / "templates" / "reports"
        )

        self.prompt_manager = PromptTemplateManager(DEFAULT_PROMPTS_DIR)

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.embedding_client: Optional[LangChainLLMAdapter] = None
        self.thinking_level = thinking_level

        if llm_model and llm_base_url:
            # Используем LM Studio Server
            self.embedding_client = LangChainLLMAdapter(
                model_name=llm_model,
                base_url=llm_base_url,
            )

        # Инициализация LMQL адаптера
        try:
            self.lmql_adapter = lmql_adapter or build_lmql_adapter_from_env()
        except RuntimeError:
            self.lmql_adapter = None
            logger.debug("LMQL адаптер не настроен для ReportGenerator")

        logger.info(
            "Инициализирован ReportGenerator (директория: %s)",
            self.quality_reports_dir,
        )

    def generate_chat_report(
        self,
        chat_name: str,
        analysis_results: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        llm_recommendations: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Генерация отчета по конкретному чату

        Args:
            chat_name: Название чата
            analysis_results: Результаты анализа
            metrics: Рассчитанные метрики

        Returns:
            Markdown отчет
        """
        logger.info(f"Генерация отчета для чата: {chat_name}")

        basic_section = self._format_basic_section(metrics)
        type_section = self._format_type_section(metrics)
        problem_section = self._format_problem_section(metrics)
        comparative_section = self._format_comparative_section(metrics)
        details_section = self._format_details_section(analysis_results)
        recommendations_section = self._format_recommendations_section(
            metrics,
            llm_recommendations=llm_recommendations,
        )

        context = {
            "chat_name": chat_name,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "basic_section": basic_section,
            "type_section": type_section,
            "problem_section": problem_section,
            "comparative_section": comparative_section,
            "details_section": details_section,
            "recommendations_section": recommendations_section,
        }

        try:
            report_content = self.report_template_manager.format(
                "main_report", **context
            )
        except Exception as exc:  # pragma: no cover - fallback
            logger.error("Ошибка генерации отчета по шаблону: %s", exc)
            report_content = self._fallback_chat_report(
                chat_name,
                metrics,
                analysis_results,
            )

        self._save_chat_report(chat_name, report_content)

        logger.info("Отчет для чата %s сохранен", chat_name)
        return report_content

    def generate_overall_report(
        self,
        chat_results: Dict[str, Any],
        overall_metrics: Dict[str, Any],
    ) -> str:
        """
        Генерация общего отчета по всем чатам

        Args:
            chat_results: Результаты анализа для всех чатов
            overall_metrics: Общие метрики

        Returns:
            Общий Markdown отчет
        """
        logger.info("Генерация общего отчета")

        summary_section = self._format_overall_basic(overall_metrics)
        chat_table = self._format_chat_table(overall_metrics)
        problem_summary = self._format_overall_problems(overall_metrics)
        recommendations_section = self._format_overall_recommendations(overall_metrics)

        context = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary_section": summary_section,
            "chat_table": chat_table,
            "problem_summary": problem_summary,
            "recommendations_section": recommendations_section,
        }

        try:
            report_content = self.report_template_manager.format(
                "summary_report", **context
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Ошибка генерации общего отчета: %s", exc)
            report_content = self._fallback_overall_report(overall_metrics)

        self._save_overall_report(report_content)

        logger.info("Общий отчет сохранен")
        return report_content

    async def generate_llm_recommendations(
        self,
        chat_name: str,
        metrics: Dict[str, Any],
        analysis_results: List[Dict[str, Any]],
        max_problem_examples: int = 5,
    ) -> List[Dict[str, Any]]:
        """Получение рекомендаций через LLM."""

        payload = self._build_recommendation_payload(
            chat_name,
            metrics,
            analysis_results,
            limit=max_problem_examples,
        )

        if not payload:
            logger.debug("Недостаточно данных для генерации LLM-рекомендаций")
            return []

        # Используем LMQL для структурированной генерации рекомендаций
        if self.lmql_adapter:
            logger.debug("Используется LMQL для генерации рекомендаций")
            return await self._generate_recommendations_with_lmql(
                chat_name, payload
            )

        # Fallback на старую реализацию, если LMQL недоступен
        if not self.embedding_client:
            logger.debug("Генератор рекомендаций LM Studio не настроен")
            return []

        try:
            prompt = self.prompt_manager.format(
                "quality_recommendations_base",
                chat_name=chat_name,
                metrics_json=json.dumps(payload, ensure_ascii=False, indent=2),
            )
        except Exception as exc:
            logger.error("Не удалось подготовить промпт рекомендаций: %s", exc)
            return []

        if not prompt.strip():
            logger.warning("Промпт рекомендаций пуст — пропускаем генерацию")
            return []

        try:
            async with self.embedding_client:
                response = await self.embedding_client.generate_summary(
                    prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
        except Exception as exc:  # pragma: no cover - внешнее API
            logger.warning("Не удалось получить рекомендации от LM Studio: %s", exc)
            return []

        return self._parse_llm_recommendations(response)

    async def _generate_recommendations_with_lmql(
        self, chat_name: str, payload: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Генерация рекомендаций с использованием LMQL.
        
        Args:
            chat_name: Название чата
            payload: Данные для анализа
            
        Returns:
            Список рекомендаций
            
        Raises:
            RuntimeError: Если произошла ошибка при выполнении запроса
        """
        try:
            prompt = self.prompt_manager.format(
                "quality_recommendations_base",
                chat_name=chat_name,
                metrics_json=json.dumps(payload, ensure_ascii=False, indent=2),
            )

            if not prompt.strip():
                logger.warning("Промпт рекомендаций пуст — пропускаем генерацию")
                return []

            # Создаем JSON схему для рекомендаций
            json_schema = """[{
    "title": "[TITLE]",
    "description": "[DESCRIPTION]",
    "suggestions": [SUGGESTIONS],
    "priority": "[PRIORITY]"
}]"""

            # Ограничения для переменных
            constraints = """
    len([RECOMMENDATIONS]) <= 10 and
    all(isinstance(r, dict) for r in [RECOMMENDATIONS]) and
    all("title" in r and "description" in r for r in [RECOMMENDATIONS]) and
    all(r.get("priority") in ["high", "medium", "low", "ai"] for r in [RECOMMENDATIONS] if "priority" in r)
"""

            # Выполняем LMQL запрос
            result = await self.lmql_adapter.execute_json_query(
                prompt=prompt,
                json_schema=json_schema,
                constraints=constraints,
                temperature=self.temperature,
                max_tokens=min(self.max_tokens, 4096),  # Ограничиваем для рекомендаций
            )

            if not result:
                return []

            # Валидация и нормализация результата
            if isinstance(result, list):
                return self._normalize_llm_entries(result)
            elif isinstance(result, dict) and "recommendations" in result:
                recommendations = result["recommendations"]
                if isinstance(recommendations, list):
                    return self._normalize_llm_entries(recommendations)

            logger.warning(f"Неожиданный формат ответа LMQL: {type(result)}")
            return []

        except Exception as e:
            logger.error(f"Ошибка при использовании LMQL для генерации рекомендаций: {e}")
            # Fallback на старую реализацию при ошибке
            if self.embedding_client:
                try:
                    prompt = self.prompt_manager.format(
                        "quality_recommendations_base",
                        chat_name=chat_name,
                        metrics_json=json.dumps(payload, ensure_ascii=False, indent=2),
                    )
                    async with self.embedding_client:
                        response = await self.embedding_client.generate_summary(
                            prompt,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                    return self._parse_llm_recommendations(response)
                except Exception as fallback_exc:
                    logger.warning(f"Fallback также не удался: {fallback_exc}")
            return []

    def _get_type_display_name(self, query_type: str) -> str:
        """Получение отображаемого имени типа запроса"""
        type_names = {
            "factual": "Фактологические запросы",
            "contextual": "Контекстные запросы",
            "analytical": "Аналитические запросы",
            "custom": "Пользовательские запросы",
        }
        return type_names.get(query_type, query_type.title())

    def _get_problem_display_name(self, problem_type: str) -> str:
        """Получение отображаемого имени типа проблемы"""
        problem_names = {
            "indexing": "Проблемы индексации",
            "search": "Проблемы поиска",
            "context": "Проблемы контекста",
        }
        return problem_names.get(problem_type, problem_type.title())

    def _get_trend_display_name(self, trend: str) -> str:
        """Получение отображаемого имени тренда"""
        trend_names = {
            "improving": "📈 Улучшение",
            "declining": "📉 Ухудшение",
            "stable": "➡️ Стабильно",
            "no_data": "❓ Нет данных",
            "no_historical_data": "❓ Нет исторических данных",
        }
        return trend_names.get(trend, trend.title())

    def _format_basic_section(self, metrics: Dict[str, Any]) -> str:
        basic = metrics.get("basic", {})
        lines = [
            f"- **Средняя оценка релевантности:** {basic.get('average_score', 0):.2f}/10",
            f"- **Медианная оценка:** {basic.get('median_score', 0):.2f}/10",
            f"- **Процент успешных поисков:** {basic.get('success_rate', 0)*100:.1f}%",
            f"- **Всего тестовых запросов:** {basic.get('total_queries', 0)}",
            f"- **Успешных запросов:** {basic.get('successful_queries', 0)}",
        ]
        return "\n".join(lines)

    def _format_type_section(self, metrics: Dict[str, Any]) -> str:
        type_metrics = metrics.get("by_type", {})
        if not type_metrics:
            return "_Нет данных по типам запросов._"

        blocks = []
        for query_type, values in type_metrics.items():
            block = [
                f"### {self._get_type_display_name(query_type)}",
                f"- **Средняя оценка:** {values.get('average_score', 0):.2f}/10",
                f"- **Процент успеха:** {values.get('success_rate', 0)*100:.1f}%",
                f"- **Количество запросов:** {values.get('total_queries', 0)}",
                "",
            ]
            blocks.append("\n".join(block))
        return "\n".join(blocks)

    def _format_problem_section(self, metrics: Dict[str, Any]) -> str:
        problem_metrics = metrics.get("problems", {})
        total_problems = problem_metrics.get("total_problems", {})
        if not any(total_problems.values()):
            return "✅ **Проблем не обнаружено**"

        lines = [
            "### Общая статистика проблем",
            f"- **Проблемы индексации:** {total_problems.get('indexing', 0)}",
            f"- **Проблемы поиска:** {total_problems.get('search', 0)}",
            f"- **Проблемы контекста:** {total_problems.get('context', 0)}",
            "",
        ]

        details = problem_metrics.get("problem_details", {})
        for problem_type, items in details.items():
            if not items:
                continue
            lines.append(f"### {self._get_problem_display_name(problem_type)}")
            for detail in items[:5]:
                lines.append(
                    f"- **Запрос:** {detail.get('query', 'N/A')} — {detail.get('score', 0):.1f}/10"
                )
            lines.append("")

        return "\n".join(lines)

    def _format_comparative_section(self, metrics: Dict[str, Any]) -> str:
        comparative = metrics.get("comparative", {})
        if not comparative.get("comparison_available"):
            return "_Исторические данные отсутствуют._"

        lines = [
            "## 📈 Сравнение с предыдущими анализами",
            f"- **Тренд:** {self._get_trend_display_name(comparative.get('trend', 'no_data'))}",
            f"- **Δ средней оценки:** {comparative.get('score_improvement', 0):+.2f}",
            f"- **Δ процента успеха:** {comparative.get('success_rate_improvement', 0)*100:+.1f}%",
        ]
        return "\n".join(lines)

    def _format_details_section(self, analysis_results: List[Dict[str, Any]]) -> str:
        blocks = []
        for idx, result in enumerate(analysis_results[:10], start=1):
            query_data = result.get("query", {})
            relevance = result.get("relevance_analysis", {})
            block_lines = [
                f"### Запрос {idx}",
                f"**Текст:** {query_data.get('query', 'N/A')}",
                f"**Тип:** {query_data.get('type', 'unknown')}",
                f"**Оценка:** {relevance.get('overall_score', 0):.1f}/10",
                f"**Объяснение:** {relevance.get('explanation', 'N/A')}",
            ]
            recs = relevance.get("recommendations", [])
            if recs:
                block_lines.append("**Рекомендации:**")
                block_lines.extend(f"- {rec}" for rec in recs)
            blocks.append("\n".join(block_lines))

        return "\n\n".join(blocks) if blocks else "_Подробные результаты отсутствуют._"

    def _format_recommendations_section(
        self,
        metrics: Dict[str, Any],
        llm_recommendations: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        recommendations = self._generate_improvement_recommendations(metrics)

        if llm_recommendations:
            recommendations.extend(self._normalize_llm_entries(llm_recommendations))

        if not recommendations:
            return "✅ **Специальных рекомендаций нет**"

        blocks = []
        for rec in recommendations:
            block = [
                f"### {rec.get('title', 'Рекомендация')}",
                rec.get("description", ""),
            ]

            priority = rec.get("priority")
            if priority:
                block.append(f"**Приоритет:** {priority}")

            block.append("**Предлагаемые действия:**")

            suggestions = rec.get("suggestions", [])
            if isinstance(suggestions, str):
                suggestions = [suggestions]

            if not suggestions:
                block.append("- (нет конкретных действий)")
            else:
                block.extend(f"- {suggestion}" for suggestion in suggestions)

            blocks.append("\n".join(filter(None, block)))

        return "\n\n".join(blocks)

    def _build_recommendation_payload(
        self,
        chat_name: str,
        metrics: Dict[str, Any],
        analysis_results: List[Dict[str, Any]],
        limit: int = 5,
    ) -> Dict[str, Any]:
        basic_metrics = self._get_metric_section(metrics, "basic")
        problem_metrics = self._get_metric_section(metrics, "problems")
        comparative_metrics = self._get_metric_section(metrics, "comparative")

        if not basic_metrics:
            return {}

        sorted_results = sorted(
            analysis_results,
            key=lambda item: item.get("relevance_analysis", {}).get(
                "overall_score", 0.0
            ),
        )

        low_score_examples: List[Dict[str, Any]] = []
        for result in sorted_results[:limit]:
            query_data = result.get("query", {})
            relevance = result.get("relevance_analysis", {})
            low_score_examples.append(
                {
                    "query": query_data.get("query"),
                    "query_type": query_data.get("type"),
                    "overall_score": relevance.get("overall_score"),
                    "problems": relevance.get("problems", {}),
                    "explanation": relevance.get("explanation"),
                    "recommendations": relevance.get("recommendations", []),
                }
            )

        problem_details = (
            problem_metrics.get("problem_details", {}) if problem_metrics else {}
        )
        limited_problem_details = {
            key: details[:limit] for key, details in problem_details.items()
        }

        return {
            "chat_name": chat_name,
            "basic_metrics": basic_metrics,
            "problem_summary": problem_metrics.get("total_problems", {})
            if problem_metrics
            else {},
            "comparative": comparative_metrics,
            "low_scores": low_score_examples,
            "problem_details": limited_problem_details,
        }

    def _parse_llm_recommendations(self, response: str) -> List[Dict[str, Any]]:
        if not response:
            return []

        response = response.strip()
        if not response:
            return []

        import re

        parsed_entries: List[Dict[str, Any]] = []
        match = re.search(r"\[[\s\S]*\]", response)

        if match:
            json_blob = match.group(0)
            try:
                data = json.loads(json_blob)
                if isinstance(data, list):
                    parsed_entries = [item for item in data if isinstance(item, dict)]
            except json.JSONDecodeError:
                logger.warning("Ответ LLM с рекомендациями содержит невалидный JSON")

        if not parsed_entries:
            text = response.replace("```", "").strip()
            if not text:
                return []

            suggestions = [
                line.strip("-• ") for line in text.splitlines() if line.strip()
            ]

            return [
                {
                    "title": "Рекомендации от LLM",
                    "description": suggestions[0] if suggestions else text[:200],
                    "suggestions": suggestions,
                    "priority": "ai",
                }
            ]

        return self._normalize_llm_entries(parsed_entries)

    def _normalize_llm_entries(
        self, entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []

        for entry in entries:
            title = entry.get("title") or entry.get("name") or "Рекомендация"
            description = entry.get("description") or entry.get("summary") or ""

            suggestions = (
                entry.get("suggestions") or entry.get("actions") or entry.get("steps")
            )
            if isinstance(suggestions, str):
                suggestions = [
                    item.strip("-• ")
                    for item in suggestions.splitlines()
                    if item.strip()
                ]
            elif suggestions is None:
                suggestions = []

            priority = entry.get("priority") or entry.get("impact")
            if isinstance(priority, (dict, list)):
                priority = None

            normalized.append(
                {
                    "title": title,
                    "description": description,
                    "suggestions": suggestions,
                    "priority": priority,
                }
            )

        return normalized

    def _get_metric_section(
        self, metrics: Dict[str, Any], section: str
    ) -> Dict[str, Any]:
        if section in metrics:
            value = metrics.get(section)
            return value if isinstance(value, dict) else {}

        details = metrics.get("details")
        if isinstance(details, dict):
            sub_value = details.get(section)
            if isinstance(sub_value, dict):
                return sub_value

        return {}

    def _format_overall_basic(self, overall_metrics: Dict[str, Any]) -> str:
        lines = [
            f"- **Средняя оценка:** {overall_metrics.get('average_score', 0):.2f}/10",
            f"- **Медианная оценка:** {overall_metrics.get('median_score', 0):.2f}/10",
            f"- **Проанализировано чатов:** {overall_metrics.get('total_chats', 0)}",
            f"- **Успешно обработано:** {overall_metrics.get('successful_chats', 0)}",
        ]
        return "\n".join(lines)

    def _format_chat_table(self, overall_metrics: Dict[str, Any]) -> str:
        chat_metrics = overall_metrics.get("chat_metrics", {})
        if not chat_metrics:
            return "_Нет детализированных данных по чатам._"

        header = "| Чат | Средняя оценка | Процент успеха | Запросов |\n|-----|----------------|----------------|----------|"
        rows = [
            f"| {chat_name} | {vals.get('average_score', 0):.2f} | {vals.get('success_rate', 0)*100:.1f}% | {vals.get('total_queries', 0)} |"
            for chat_name, vals in chat_metrics.items()
        ]
        return "\n".join([header, *rows])

    def _format_overall_problems(self, overall_metrics: Dict[str, Any]) -> str:
        total_problems = overall_metrics.get("total_problems", {})
        if not any(total_problems.values()):
            return "✅ **Общие проблемы не обнаружены**"
        return "\n".join(
            [
                f"- Индексация: {total_problems.get('indexing', 0)}",
                f"- Поиск: {total_problems.get('search', 0)}",
                f"- Контекст: {total_problems.get('context', 0)}",
            ]
        )

    def _format_overall_recommendations(self, overall_metrics: Dict[str, Any]) -> str:
        recs = self._generate_general_recommendations(overall_metrics)
        if not recs:
            return (
                "✅ **Система работает стабильно, дополнительные действия не требуются**"
            )
        blocks = []
        for rec in recs:
            block = [
                f"### {rec.get('title', 'Рекомендация')}",
                rec.get("description", ""),
                "**Предлагаемые действия:**",
            ]
            block.extend(f"- {suggestion}" for suggestion in rec.get("suggestions", []))
            blocks.append("\n".join(block))
        return "\n\n".join(blocks)

    def _fallback_chat_report(
        self,
        chat_name: str,
        metrics: Dict[str, Any],
        analysis_results: List[Dict[str, Any]],
        llm_recommendations: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        logger.warning("Используется fallback генерация отчета для %s", chat_name)
        return "\n".join(
            [
                f"# Анализ качества поиска - {chat_name}",
                f"**Дата анализа:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                self._format_basic_section(metrics),
                self._format_type_section(metrics),
                self._format_problem_section(metrics),
                self._format_comparative_section(metrics),
                self._format_details_section(analysis_results),
                self._format_recommendations_section(
                    metrics,
                    llm_recommendations=llm_recommendations,
                ),
            ]
        )

    def _fallback_overall_report(self, overall_metrics: Dict[str, Any]) -> str:
        logger.warning("Используется fallback генерация сводного отчета")
        return "\n".join(
            [
                "# Общий анализ качества поиска",
                f"**Дата анализа:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                self._format_overall_basic(overall_metrics),
                self._format_chat_table(overall_metrics),
                self._format_overall_problems(overall_metrics),
                self._format_overall_recommendations(overall_metrics),
            ]
        )

    def _generate_improvement_recommendations(
        self, metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Генерация рекомендаций по улучшению для конкретного чата"""
        recommendations = []

        basic_metrics = metrics.get("basic", {})
        average_score = basic_metrics.get("average_score", 0)
        basic_metrics.get("success_rate", 0)

        problem_metrics = metrics.get("problems", {})
        total_problems = problem_metrics.get("total_problems", {})

        # Рекомендации по низким оценкам
        if average_score < 5:
            recommendations.append(
                {
                    "title": "Низкое качество поиска",
                    "description": f"Средняя оценка релевантности составляет {average_score:.2f}/10, что указывает на серьезные проблемы.",
                    "suggestions": [
                        "Проверить настройки индексации",
                        "Улучшить качество эмбеддингов",
                        "Настроить параметры поиска",
                        "Проверить фильтрацию результатов",
                    ],
                }
            )

        # Рекомендации по проблемам индексации
        if total_problems.get("indexing", 0) > 0:
            recommendations.append(
                {
                    "title": "Проблемы с индексацией",
                    "description": f"Обнаружено {total_problems['indexing']} проблем с индексацией.",
                    "suggestions": [
                        "Проверить процесс индексации сообщений",
                        "Увеличить размер чанков для длинных сообщений",
                        "Улучшить фильтрацию нерелевантных сообщений",
                        "Проверить обработку специальных символов",
                    ],
                }
            )

        # Рекомендации по проблемам поиска
        if total_problems.get("search", 0) > 0:
            recommendations.append(
                {
                    "title": "Проблемы с поиском",
                    "description": f"Обнаружено {total_problems['search']} проблем с поиском.",
                    "suggestions": [
                        "Настроить веса гибридного поиска",
                        "Улучшить токенизацию для русского языка",
                        "Проверить пороги релевантности",
                        "Оптимизировать алгоритм ранжирования",
                    ],
                }
            )

        # Рекомендации по проблемам контекста
        if total_problems.get("context", 0) > 0:
            recommendations.append(
                {
                    "title": "Проблемы с контекстом",
                    "description": f"Обнаружено {total_problems['context']} проблем с контекстом.",
                    "suggestions": [
                        "Улучшить группировку сообщений в сессии",
                        "Настроить параметры кластеризации",
                        "Проверить качество саммаризации",
                        "Увеличить контекстную информацию в результатах",
                    ],
                }
            )

        return recommendations

    def _generate_general_recommendations(
        self, overall_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Генерация общих рекомендаций"""
        recommendations = []

        average_score = overall_metrics.get("average_score", 0)
        total_problems = overall_metrics.get("total_problems", {})

        # Общие рекомендации по качеству
        if average_score < 6:
            recommendations.append(
                {
                    "title": "Общее улучшение качества системы",
                    "description": f"Общая оценка качества составляет {average_score:.2f}/10, что требует системных улучшений.",
                    "suggestions": [
                        "Провести полный аудит системы индексации",
                        "Обновить модели эмбеддингов",
                        "Настроить параметры поиска для всех чатов",
                        "Внедрить мониторинг качества в реальном времени",
                    ],
                }
            )

        # Рекомендации по проблемам
        total_problem_count = sum(total_problems.values())
        if total_problem_count > 0:
            recommendations.append(
                {
                    "title": "Системное решение проблем",
                    "description": f"Обнаружено {total_problem_count} проблем в системе.",
                    "suggestions": [
                        "Создать план устранения проблем по приоритету",
                        "Внедрить автоматические тесты качества",
                        "Настроить алерты при снижении качества",
                        "Регулярно проводить анализ качества",
                    ],
                }
            )

        return recommendations

    def _save_chat_report(self, chat_name: str, report_content: str):
        """Сохранение отчета по чату"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{chat_name}_quality_analysis_{timestamp}.md"
        filepath = self.quality_reports_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Отчет сохранен: {filepath}")

    def _save_overall_report(self, report_content: str):
        """Сохранение общего отчета"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"overall_quality_analysis_{timestamp}.md"
        filepath = self.quality_reports_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Общий отчет сохранен: {filepath}")
