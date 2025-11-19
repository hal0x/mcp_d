"""Unit-тесты для QueryIntentAnalyzer."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memory_mcp.search.query_intent_analyzer import QueryIntentAnalyzer, QueryIntent


@pytest.fixture
def intent_analyzer():
    """Создает анализатор намерений."""
    return QueryIntentAnalyzer()


@pytest.mark.asyncio
async def test_analyze_intent_informational(intent_analyzer):
    """Тест анализа информационного намерения."""
    with patch.object(
        intent_analyzer, "_get_llm_client", return_value=None
    ):
        intent = await intent_analyzer.analyze_intent("что такое блокчейн")
        
        assert intent.intent_type in ["informational", "analytical"]
        assert intent.confidence > 0.0
        assert intent.recommended_db_weight > 0.0
        assert intent.recommended_artifact_weight > 0.0


@pytest.mark.asyncio
async def test_analyze_intent_transactional(intent_analyzer):
    """Тест анализа транзакционного намерения."""
    with patch.object(
        intent_analyzer, "_get_llm_client", return_value=None
    ):
        intent = await intent_analyzer.analyze_intent("как создать кошелек")
        
        assert intent.intent_type == "transactional"
        assert intent.recommended_db_weight > intent.recommended_artifact_weight


@pytest.mark.asyncio
async def test_analyze_intent_navigational(intent_analyzer):
    """Тест анализа навигационного намерения."""
    with patch.object(
        intent_analyzer, "_get_llm_client", return_value=None
    ):
        intent = await intent_analyzer.analyze_intent("где найти информацию о биткоине")
        
        assert intent.intent_type == "navigational"
        assert intent.recommended_db_weight > intent.recommended_artifact_weight


@pytest.mark.asyncio
async def test_analyze_intent_with_llm(intent_analyzer):
    """Тест анализа намерения с использованием LLM."""
    mock_llm = MagicMock()
    mock_llm.generate_summary = AsyncMock(return_value='{"intent_type": "informational", "confidence": 0.9, "reasoning": "test"}')
    
    with patch.object(
        intent_analyzer, "_get_llm_client", return_value=mock_llm
    ), patch.object(
        intent_analyzer, "_parse_llm_response"
    ) as mock_parse:
        mock_parse.return_value = QueryIntent(
            intent_type="informational",
            confidence=0.9,
            recommended_db_weight=0.6,
            recommended_artifact_weight=0.4,
        )
        
        intent = await intent_analyzer.analyze_intent("test query")
        
        assert intent.intent_type == "informational"
        assert intent.confidence == 0.9


def test_generate_recommendations(intent_analyzer):
    """Тест генерации рекомендаций для разных типов намерений."""
    # Информационный поиск
    rec_info = intent_analyzer._generate_recommendations("informational", "test")
    assert rec_info["db_weight"] == 0.5
    assert rec_info["artifact_weight"] == 0.5
    
    # Транзакционный поиск
    rec_trans = intent_analyzer._generate_recommendations("transactional", "test")
    assert rec_trans["db_weight"] > rec_trans["artifact_weight"]
    
    # Навигационный поиск
    rec_nav = intent_analyzer._generate_recommendations("navigational", "test")
    assert rec_nav["db_weight"] > rec_nav["artifact_weight"]
    
    # Аналитический поиск
    rec_anal = intent_analyzer._generate_recommendations("analytical", "test")
    assert rec_anal["artifact_weight"] > rec_anal["db_weight"]


def test_heuristic_intent_analysis(intent_analyzer):
    """Тест эвристического анализа намерения."""
    # Информационный
    intent = intent_analyzer._heuristic_intent_analysis("блокчейн технология")
    assert intent.intent_type == "informational"
    
    # Транзакционный
    intent = intent_analyzer._heuristic_intent_analysis("как сделать что-то")
    assert intent.intent_type == "transactional"
    
    # Навигационный
    intent = intent_analyzer._heuristic_intent_analysis("где найти информацию")
    assert intent.intent_type == "navigational"
    
    # Аналитический
    intent = intent_analyzer._heuristic_intent_analysis("почему работает блокчейн")
    assert intent.intent_type == "analytical"

