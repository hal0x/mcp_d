"""Интеграционные тесты для RelevanceAnalyzer с LMQL."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memory_mcp.quality_analyzer.core.relevance_analyzer import RelevanceAnalyzer
from memory_mcp.core.lmql_adapter import LMQLAdapter


class TestRelevanceAnalyzerLMQL:
    """Тесты для RelevanceAnalyzer с LMQL."""

    @pytest.fixture
    def mock_lmql_adapter(self):
        """Мок для LMQL адаптера."""
        adapter = MagicMock(spec=LMQLAdapter)
        adapter.available.return_value = True
        adapter.execute_json_query = AsyncMock()
        return adapter

    @pytest.fixture
    def sample_search_results(self):
        """Пример результатов поиска."""
        return [
            {
                "text": "Пример текста 1",
                "score": 0.9,
                "metadata": {"chat": "test_chat", "date": "2024-01-01"},
            },
            {
                "text": "Пример текста 2",
                "score": 0.8,
                "metadata": {"chat": "test_chat", "date": "2024-01-02"},
            },
        ]

    @pytest.mark.asyncio
    async def test_analyze_relevance_with_lmql_success(self, mock_lmql_adapter, sample_search_results):
        """Тест успешного анализа релевантности через LMQL."""
        mock_lmql_adapter.execute_json_query.return_value = {
            "overall_score": 8.5,
            "individual_scores": [9.0, 8.0],
            "problems": {
                "indexing": 0,
                "search": 0,
                "context": 0,
            },
            "explanation": "Результаты релевантны запросу",
            "recommendations": [],
        }

        analyzer = RelevanceAnalyzer(
            model_name="test-model",
            base_url="http://localhost:1234",
            lmql_adapter=mock_lmql_adapter,
        )

        query_data = {"query": "test query", "type": "factual"}
        result = await analyzer.analyze_relevance(query_data, sample_search_results)

        assert result["overall_score"] == 8.5
        assert len(result["individual_scores"]) == 2
        assert result["individual_scores"][0] == 9.0
        assert "explanation" in result

    @pytest.mark.asyncio
    async def test_analyze_relevance_without_lmql(self, sample_search_results):
        """Тест ошибки при отсутствии LMQL адаптера."""
        analyzer = RelevanceAnalyzer(
            model_name="test-model",
            base_url="http://localhost:1234",
            lmql_adapter=None,
        )

        query_data = {"query": "test query", "type": "factual"}
        
        with pytest.raises(RuntimeError, match="LMQL не настроен"):
            await analyzer.analyze_relevance(query_data, sample_search_results)

    @pytest.mark.asyncio
    async def test_analyze_relevance_lmql_error(self, mock_lmql_adapter, sample_search_results):
        """Тест ошибки при исключении в LMQL."""
        mock_lmql_adapter.execute_json_query.side_effect = RuntimeError("LMQL error")

        analyzer = RelevanceAnalyzer(
            model_name="test-model",
            base_url="http://localhost:1234",
            lmql_adapter=mock_lmql_adapter,
        )

        query_data = {"query": "test query", "type": "factual"}
        
        with pytest.raises(RuntimeError, match="Ошибка анализа релевантности через LMQL"):
            await analyzer.analyze_relevance(query_data, sample_search_results)

