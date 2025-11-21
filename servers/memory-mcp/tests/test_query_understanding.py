"""Unit-тесты для QueryUnderstandingEngine."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from memory_mcp.search.query_understanding import QueryUnderstandingEngine, QueryUnderstanding
from memory_mcp.core.lmql_adapter import LMQLAdapter


@pytest.fixture
def mock_lmql_adapter():
    """Мок для LMQL адаптера."""
    adapter = MagicMock(spec=LMQLAdapter)
    adapter.available.return_value = True
    adapter.execute_json_query = AsyncMock()
    return adapter


@pytest.fixture
def understanding_engine(mock_lmql_adapter):
    """Создает движок понимания запросов с моком LMQL."""
    return QueryUnderstandingEngine(lmql_adapter=mock_lmql_adapter)


@pytest.mark.asyncio
async def test_understand_query_simple(understanding_engine, mock_lmql_adapter):
    """Тест понимания простого запроса."""
    mock_lmql_adapter.execute_json_query.return_value = {
        "sub_queries": ["блокчейн"],
        "implicit_requirements": [],
        "alternative_formulations": ["технология блокчейн"],
        "key_concepts": ["блокчейн"],
        "concept_relationships": {},
        "enhanced_query": "блокчейн",
    }
    
        understanding = await understanding_engine.understand_query("блокчейн")
        
        assert understanding.original_query == "блокчейн"
        assert len(understanding.sub_queries) > 0
        assert "блокчейн" in understanding.sub_queries[0]


@pytest.mark.asyncio
async def test_understand_query_complex(understanding_engine, mock_lmql_adapter):
    """Тест понимания сложного запроса."""
    mock_lmql_adapter.execute_json_query.return_value = {
        "sub_queries": ["блокчейн", "криптовалюты"],
        "implicit_requirements": ["технология", "применение"],
        "alternative_formulations": ["технология блокчейн", "криптовалютные технологии"],
        "key_concepts": ["блокчейн", "криптовалюты", "технология"],
        "concept_relationships": {
            "блокчейн": ["криптовалюты", "технология"]
        },
        "enhanced_query": "блокчейн и криптовалюты технология применение",
    }
    
        understanding = await understanding_engine.understand_query("блокчейн и криптовалюты")
        
        assert understanding.original_query == "блокчейн и криптовалюты"
        # Должна быть декомпозиция
        assert len(understanding.sub_queries) >= 1
    assert len(understanding.key_concepts) > 0


@pytest.mark.asyncio
async def test_understand_query_with_lmql(understanding_engine, mock_lmql_adapter):
    """Тест понимания запроса с использованием LMQL."""
    mock_lmql_adapter.execute_json_query.return_value = {
        "sub_queries": ["блокчейн", "криптовалюты"],
        "implicit_requirements": ["технология", "применение"],
        "alternative_formulations": ["технология блокчейн", "криптовалютные технологии"],
        "key_concepts": ["блокчейн", "криптовалюты", "технология"],
        "concept_relationships": {
            "блокчейн": ["криптовалюты", "технология"]
        },
        "enhanced_query": "блокчейн и криптовалюты технология применение",
    }
        
        understanding = await understanding_engine.understand_query("блокчейн и криптовалюты")
        
        assert len(understanding.sub_queries) == 2
        assert len(understanding.key_concepts) > 0
        assert understanding.enhanced_query != understanding.original_query


@pytest.mark.asyncio
async def test_understand_query_empty(understanding_engine):
    """Тест обработки пустого запроса."""
    understanding = await understanding_engine.understand_query("")
    
    assert understanding.original_query == ""
    assert len(understanding.sub_queries) == 0
    assert len(understanding.key_concepts) == 0


@pytest.mark.asyncio
async def test_understand_query_without_lmql():
    """Тест ошибки при отсутствии LMQL адаптера."""
    engine = QueryUnderstandingEngine(lmql_adapter=None)
    
    with pytest.raises(RuntimeError, match="LMQL не настроен"):
        await engine.understand_query("test query")


@pytest.mark.asyncio
async def test_understand_query_lmql_error(understanding_engine, mock_lmql_adapter):
    """Тест ошибки при исключении в LMQL."""
    mock_lmql_adapter.execute_json_query.side_effect = RuntimeError("LMQL error")
    
    with pytest.raises(RuntimeError, match="Ошибка понимания запроса через LMQL"):
        await understanding_engine.understand_query("test query")
