"""Unit-тесты для QueryUnderstandingEngine."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memory_mcp.search.query_understanding import QueryUnderstandingEngine, QueryUnderstanding


@pytest.fixture
def understanding_engine():
    """Создает движок понимания запросов."""
    return QueryUnderstandingEngine()


@pytest.mark.asyncio
async def test_understand_query_simple(understanding_engine):
    """Тест понимания простого запроса."""
    with patch.object(
        understanding_engine, "_get_llm_client", return_value=None
    ):
        understanding = await understanding_engine.understand_query("блокчейн")
        
        assert understanding.original_query == "блокчейн"
        assert len(understanding.sub_queries) > 0
        assert "блокчейн" in understanding.sub_queries[0]


@pytest.mark.asyncio
async def test_understand_query_complex(understanding_engine):
    """Тест понимания сложного запроса."""
    with patch.object(
        understanding_engine, "_get_llm_client", return_value=None
    ):
        understanding = await understanding_engine.understand_query("блокчейн и криптовалюты")
        
        assert understanding.original_query == "блокчейн и криптовалюты"
        # Должна быть декомпозиция
        assert len(understanding.sub_queries) >= 1


@pytest.mark.asyncio
async def test_understand_query_with_llm(understanding_engine):
    """Тест понимания запроса с использованием LLM."""
    mock_llm = MagicMock()
    mock_llm.generate_summary = AsyncMock(return_value='''{
        "sub_queries": ["блокчейн", "криптовалюты"],
        "implicit_requirements": ["технология", "применение"],
        "alternative_formulations": ["технология блокчейн", "криптовалютные технологии"],
        "key_concepts": ["блокчейн", "криптовалюты", "технология"],
        "concept_relationships": {
            "блокчейн": ["криптовалюты", "технология"]
        },
        "enhanced_query": "блокчейн и криптовалюты технология применение"
    }''')
    
    with patch.object(
        understanding_engine, "_get_llm_client", return_value=mock_llm
    ), patch.object(
        understanding_engine, "_parse_llm_response"
    ) as mock_parse:
        mock_parse.return_value = QueryUnderstanding(
            original_query="блокчейн и криптовалюты",
            sub_queries=["блокчейн", "криптовалюты"],
            implicit_requirements=["технология", "применение"],
            alternative_formulations=["технология блокчейн"],
            key_concepts=["блокчейн", "криптовалюты"],
            concept_relationships={"блокчейн": ["криптовалюты"]},
            enhanced_query="блокчейн и криптовалюты технология применение",
        )
        
        understanding = await understanding_engine.understand_query("блокчейн и криптовалюты")
        
        assert len(understanding.sub_queries) == 2
        assert len(understanding.key_concepts) > 0
        assert understanding.enhanced_query != understanding.original_query


def test_simple_understanding(understanding_engine):
    """Тест простого понимания без LLM."""
    understanding = understanding_engine._simple_understanding("блокчейн и криптовалюты")
    
    assert understanding.original_query == "блокчейн и криптовалюты"
    assert len(understanding.sub_queries) >= 1
    assert len(understanding.key_concepts) > 0


def test_simple_understanding_single_query(understanding_engine):
    """Тест простого понимания для одного запроса."""
    understanding = understanding_engine._simple_understanding("блокчейн")
    
    assert len(understanding.sub_queries) == 1
    assert understanding.sub_queries[0] == "блокчейн"

