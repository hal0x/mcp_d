"""Интеграционные тесты для QueryUnderstandingEngine с LMQL."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memory_mcp.search.query_understanding import QueryUnderstandingEngine, QueryUnderstanding
from memory_mcp.core.adapters.lmql_adapter import LMQLAdapter


class TestQueryUnderstandingLMQL:
    """Тесты для QueryUnderstandingEngine с LMQL."""

    @pytest.fixture
    def mock_lmql_adapter(self):
        """Мок для LMQL адаптера."""
        adapter = MagicMock(spec=LMQLAdapter)
        adapter.available.return_value = True
        adapter.execute_json_query = AsyncMock()
        return adapter

    @pytest.mark.asyncio
    async def test_understand_query_with_lmql_success(self, mock_lmql_adapter):
        """Тест успешного понимания запроса через LMQL."""
        mock_lmql_adapter.execute_json_query.return_value = {
            "sub_queries": ["криптовалюты", "блокчейн"],
            "implicit_requirements": ["актуальная информация"],
            "alternative_formulations": ["криптовалюты и блокчейн технологии"],
            "key_concepts": ["криптовалюты", "блокчейн"],
            "concept_relationships": {
                "криптовалюты": ["блокчейн", "технологии"]
            },
            "enhanced_query": "криптовалюты и блокчейн технологии",
        }

        engine = QueryUnderstandingEngine(lmql_adapter=mock_lmql_adapter)
        result = await engine.understand_query("криптовалюты блокчейн")

        assert isinstance(result, QueryUnderstanding)
        assert result.original_query == "криптовалюты блокчейн"
        assert len(result.sub_queries) == 2
        assert "криптовалюты" in result.sub_queries
        assert len(result.key_concepts) == 2

    @pytest.mark.asyncio
    async def test_understand_query_without_lmql(self):
        """Тест ошибки при отсутствии LMQL адаптера."""
        engine = QueryUnderstandingEngine(lmql_adapter=None)
        
        with pytest.raises(RuntimeError, match="LMQL не настроен"):
            await engine.understand_query("test query")

    @pytest.mark.asyncio
    async def test_understand_query_lmql_error(self, mock_lmql_adapter):
        """Тест ошибки при исключении в LMQL."""
        mock_lmql_adapter.execute_json_query.side_effect = RuntimeError("LMQL error")

        engine = QueryUnderstandingEngine(lmql_adapter=mock_lmql_adapter)
        
        with pytest.raises(RuntimeError, match="Ошибка понимания запроса через LMQL"):
            await engine.understand_query("test query")

