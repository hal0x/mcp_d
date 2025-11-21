"""Unit-тесты для QueryIntentAnalyzer."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from memory_mcp.search.query_intent_analyzer import QueryIntentAnalyzer, QueryIntent
from memory_mcp.core.adapters.lmql_adapter import LMQLAdapter


@pytest.fixture
def mock_lmql_adapter():
    """Мок для LMQL адаптера."""
    adapter = MagicMock(spec=LMQLAdapter)
    adapter.available.return_value = True
    adapter.execute_json_query = AsyncMock()
    return adapter


@pytest.fixture
def intent_analyzer(mock_lmql_adapter):
    """Создает анализатор намерений с моком LMQL."""
    return QueryIntentAnalyzer(lmql_adapter=mock_lmql_adapter)


@pytest.mark.asyncio
async def test_analyze_intent_informational(intent_analyzer, mock_lmql_adapter):
    """Тест анализа информационного намерения."""
    mock_lmql_adapter.execute_json_query.return_value = {
        "intent_type": "informational",
        "confidence": 0.9,
        "reasoning": "Пользователь ищет информацию",
    }
    
    intent = await intent_analyzer.analyze_intent("что такое блокчейн")
    
    assert intent.intent_type == "informational"
    assert intent.confidence > 0.0
    assert intent.recommended_db_weight > 0.0
    assert intent.recommended_artifact_weight > 0.0


@pytest.mark.asyncio
async def test_analyze_intent_transactional(intent_analyzer, mock_lmql_adapter):
    """Тест анализа транзакционного намерения."""
    mock_lmql_adapter.execute_json_query.return_value = {
        "intent_type": "transactional",
        "confidence": 0.85,
        "reasoning": "Пользователь хочет выполнить действие",
    }
    
    intent = await intent_analyzer.analyze_intent("как создать кошелек")
    
    assert intent.intent_type == "transactional"
    assert intent.recommended_db_weight > intent.recommended_artifact_weight


@pytest.mark.asyncio
async def test_analyze_intent_navigational(intent_analyzer, mock_lmql_adapter):
    """Тест анализа навигационного намерения."""
    mock_lmql_adapter.execute_json_query.return_value = {
        "intent_type": "navigational",
        "confidence": 0.8,
        "reasoning": "Пользователь ищет место",
    }
    
    intent = await intent_analyzer.analyze_intent("где найти информацию о биткоине")
    
    assert intent.intent_type == "navigational"
    assert intent.recommended_db_weight > intent.recommended_artifact_weight


@pytest.mark.asyncio
async def test_analyze_intent_with_lmql(intent_analyzer, mock_lmql_adapter):
    """Тест анализа намерения с использованием LMQL."""
    mock_lmql_adapter.execute_json_query.return_value = {
        "intent_type": "informational",
        "confidence": 0.9,
        "reasoning": "test",
    }
    
    intent = await intent_analyzer.analyze_intent("test query")
    
    assert intent.intent_type == "informational"
    assert intent.confidence == 0.9


@pytest.mark.asyncio
async def test_analyze_intent_without_lmql():
    """Тест ошибки при отсутствии LMQL адаптера."""
    analyzer = QueryIntentAnalyzer(lmql_adapter=None)
    
    with pytest.raises(RuntimeError, match="LMQL не настроен"):
        await analyzer.analyze_intent("test query")


@pytest.mark.asyncio
async def test_analyze_intent_empty_query(intent_analyzer, mock_lmql_adapter):
    """Тест обработки пустого запроса."""
    intent = await intent_analyzer.analyze_intent("")
    
    assert isinstance(intent, QueryIntent)
    assert intent.intent_type == "informational"
    assert intent.confidence == 0.5


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


