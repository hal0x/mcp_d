"""Интеграционные тесты для ReportGenerator с LMQL."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from memory_mcp.quality_analyzer.core.report_generator import ReportGenerator
from memory_mcp.core.lmql_adapter import LMQLAdapter


class TestReportGeneratorLMQL:
    """Тесты для ReportGenerator с LMQL."""

    @pytest.fixture
    def mock_lmql_adapter(self):
        """Мок для LMQL адаптера."""
        adapter = MagicMock(spec=LMQLAdapter)
        adapter.available.return_value = True
        adapter.execute_json_query = AsyncMock()
        return adapter

    @pytest.fixture
    def report_generator(self, mock_lmql_adapter):
        """Создает генератор отчетов с моком LMQL."""
        return ReportGenerator(
            reports_dir=Path("test_reports"),
            lmql_adapter=mock_lmql_adapter,
        )

    @pytest.fixture
    def sample_metrics(self):
        """Пример метрик для тестирования."""
        return {
            "basic": {
                "average_score": 7.5,
                "median_score": 8.0,
                "success_rate": 0.85,
                "total_queries": 20,
                "successful_queries": 17,
            },
            "by_type": {
                "factual": {"average_score": 8.0},
                "analytical": {"average_score": 7.0},
            },
        }

    @pytest.fixture
    def sample_analysis_results(self):
        """Пример результатов анализа."""
        return [
            {
                "query": {"query": "test query", "type": "factual"},
                "relevance_analysis": {
                    "overall_score": 8.0,
                    "problems": {"indexing": 0, "search": 0, "context": 0},
                    "explanation": "Результаты релевантны",
                    "recommendations": [],
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_generate_llm_recommendations_with_lmql_success(
        self, report_generator, mock_lmql_adapter, sample_metrics, sample_analysis_results
    ):
        """Тест успешной генерации рекомендаций через LMQL."""
        mock_lmql_adapter.execute_json_query.return_value = [
            {
                "title": "Улучшить индексацию",
                "description": "Рекомендуется улучшить индексацию данных",
                "suggestions": ["проверить настройки", "обновить индексы"],
                "priority": "high",
            },
            {
                "title": "Оптимизировать поиск",
                "description": "Можно оптимизировать алгоритм поиска",
                "suggestions": ["настроить веса", "улучшить релевантность"],
                "priority": "medium",
            },
        ]

        recommendations = await report_generator.generate_llm_recommendations(
            chat_name="test_chat",
            metrics=sample_metrics,
            analysis_results=sample_analysis_results,
        )

        assert len(recommendations) == 2
        assert recommendations[0]["title"] == "Улучшить индексацию"
        assert recommendations[0]["priority"] == "high"
        assert len(recommendations[0]["suggestions"]) == 2

    @pytest.mark.asyncio
    async def test_generate_llm_recommendations_without_lmql(
        self, sample_metrics, sample_analysis_results
    ):
        """Тест fallback на старую реализацию при отсутствии LMQL."""
        generator = ReportGenerator(
            reports_dir=Path("test_reports"),
            lmql_adapter=None,
            llm_model=None,  # Нет LLM клиента
        )

        recommendations = await generator.generate_llm_recommendations(
            chat_name="test_chat",
            metrics=sample_metrics,
            analysis_results=sample_analysis_results,
        )

        # Должен вернуть пустой список, так как нет ни LMQL, ни LLM клиента
        assert recommendations == []

    @pytest.mark.asyncio
    async def test_generate_llm_recommendations_lmql_error(
        self, report_generator, mock_lmql_adapter, sample_metrics, sample_analysis_results
    ):
        """Тест обработки ошибки LMQL."""
        mock_lmql_adapter.execute_json_query.side_effect = RuntimeError("LMQL error")

        recommendations = await report_generator.generate_llm_recommendations(
            chat_name="test_chat",
            metrics=sample_metrics,
            analysis_results=sample_analysis_results,
        )

        # Должен вернуть пустой список при ошибке и отсутствии fallback
        assert recommendations == []

    @pytest.mark.asyncio
    async def test_generate_llm_recommendations_empty_payload(
        self, report_generator, mock_lmql_adapter
    ):
        """Тест обработки пустого payload."""
        recommendations = await report_generator.generate_llm_recommendations(
            chat_name="test_chat",
            metrics={},
            analysis_results=[],
        )

        assert recommendations == []
        # LMQL не должен вызываться при пустом payload
        mock_lmql_adapter.execute_json_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_llm_recommendations_invalid_format(
        self, report_generator, mock_lmql_adapter, sample_metrics, sample_analysis_results
    ):
        """Тест обработки невалидного формата ответа LMQL."""
        # LMQL возвращает не список, а словарь
        mock_lmql_adapter.execute_json_query.return_value = {
            "recommendations": [
                {
                    "title": "Тест",
                    "description": "Описание",
                    "suggestions": [],
                    "priority": "low",
                }
            ]
        }

        recommendations = await report_generator.generate_llm_recommendations(
            chat_name="test_chat",
            metrics=sample_metrics,
            analysis_results=sample_analysis_results,
        )

        # Должен обработать словарь с ключом "recommendations"
        assert len(recommendations) == 1
        assert recommendations[0]["title"] == "Тест"

