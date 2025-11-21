"""Интеграционные тесты для QueryIntentAnalyzer с LMQL."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from memory_mcp.search.query_intent_analyzer import QueryIntentAnalyzer, QueryIntent
from memory_mcp.core.adapters.lmql_adapter import LMQLAdapter


class TestQueryIntentAnalyzerLMQL:
    """Тесты для QueryIntentAnalyzer с LMQL."""

    @pytest.fixture
    def mock_lmql_adapter(self):
        """Мок для LMQL адаптера."""
        adapter = MagicMock(spec=LMQLAdapter)
        adapter.available.return_value = True
        adapter.execute_json_query = AsyncMock()
        return adapter

    @pytest.mark.asyncio
    async def test_analyze_intent_with_lmql_success(self, mock_lmql_adapter):
        """Тест успешного анализа намерения через LMQL."""
        mock_lmql_adapter.execute_json_query.return_value = {
            "intent_type": "informational",
            "confidence": 0.85,
            "reasoning": "Пользователь ищет информацию",
        }

        analyzer = QueryIntentAnalyzer(lmql_adapter=mock_lmql_adapter)
        result = await analyzer.analyze_intent("что такое криптовалюты")

        assert isinstance(result, QueryIntent)
        assert result.intent_type == "informational"
        assert result.confidence == 0.85
        assert result.recommended_db_weight == 0.5
        assert result.recommended_artifact_weight == 0.5

    @pytest.mark.asyncio
    async def test_analyze_intent_without_lmql(self):
        """Тест ошибки при отсутствии LMQL адаптера."""
        analyzer = QueryIntentAnalyzer(lmql_adapter=None)

        with pytest.raises(RuntimeError, match="LMQL не настроен"):
            await analyzer.analyze_intent("test query")

    @pytest.mark.asyncio
    async def test_analyze_intent_lmql_error(self, mock_lmql_adapter):
        """Тест ошибки при исключении в LMQL."""
        mock_lmql_adapter.execute_json_query.side_effect = RuntimeError("LMQL error")

        analyzer = QueryIntentAnalyzer(lmql_adapter=mock_lmql_adapter)

        with pytest.raises(RuntimeError, match="Ошибка анализа намерения через LMQL"):
            await analyzer.analyze_intent("test query")

    @pytest.mark.asyncio
    async def test_analyze_intent_empty_query(self, mock_lmql_adapter):
        """Тест обработки пустого запроса."""
        analyzer = QueryIntentAnalyzer(lmql_adapter=mock_lmql_adapter)
        result = await analyzer.analyze_intent("")

        assert isinstance(result, QueryIntent)
        assert result.intent_type == "informational"
        assert result.confidence == 0.5

