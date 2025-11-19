"""Интеграционные тесты для smart_search."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_mcp.mcp.schema import SmartSearchRequest, SearchFeedback
from memory_mcp.memory.artifacts_reader import ArtifactsReader
from memory_mcp.mcp.adapters import MemoryServiceAdapter
from memory_mcp.search import SmartSearchEngine, SearchSessionStore


@pytest.fixture
def mock_adapter():
    """Создает мок адаптера памяти."""
    adapter = MagicMock(spec=MemoryServiceAdapter)
    adapter.search.return_value = MagicMock(
        results=[],
        total_matches=0,
    )
    # Добавляем атрибут graph для EntityContextEnricher
    adapter.graph = MagicMock()
    return adapter


@pytest.fixture
def temp_artifacts_dir():
    """Создает временную директорию с артифактами."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = Path(tmpdir) / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "chat_contexts").mkdir()
        (artifacts_dir / "chat_contexts" / "test.md").write_text(
            "Test content about cryptocurrencies and blockchain."
        )
        yield artifacts_dir


@pytest.fixture
def temp_db_path():
    """Создает временный путь к БД сессий."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sessions.db"
        yield db_path


@pytest.fixture
def artifacts_reader(temp_artifacts_dir):
    """Создает читатель артифактов."""
    reader = ArtifactsReader(artifacts_dir=temp_artifacts_dir)
    reader.scan_artifacts_directory()
    return reader


@pytest.fixture
def session_store(temp_db_path):
    """Создает хранилище сессий."""
    return SearchSessionStore(db_path=temp_db_path)


@pytest.fixture
def smart_search_engine(mock_adapter, artifacts_reader, session_store):
    """Создает интерактивный поисковый движок."""
    return SmartSearchEngine(
        adapter=mock_adapter,
        artifacts_reader=artifacts_reader,
        session_store=session_store,
        min_confidence=0.5,
    )


@pytest.mark.asyncio
async def test_smart_search_basic(smart_search_engine):
    """Тест базового поиска."""
    request = SmartSearchRequest(query="cryptocurrencies", top_k=10)

    # Мокаем LLM компоненты для избежания реальных вызовов
    with patch.object(
        smart_search_engine.query_understanding, "understand_query", new_callable=AsyncMock
    ) as mock_understanding, patch.object(
        smart_search_engine.intent_analyzer, "analyze_intent", new_callable=AsyncMock
    ) as mock_intent:
        from memory_mcp.search.query_understanding import QueryUnderstanding
        from memory_mcp.search.query_intent_analyzer import QueryIntent
        
        mock_understanding.return_value = QueryUnderstanding(
            original_query="cryptocurrencies",
            sub_queries=["cryptocurrencies"],
            implicit_requirements=[],
            alternative_formulations=[],
            key_concepts=["cryptocurrencies"],
            concept_relationships={},
            enhanced_query="cryptocurrencies",
        )
        mock_intent.return_value = QueryIntent(
            intent_type="informational",
            confidence=0.8,
            recommended_db_weight=0.6,
            recommended_artifact_weight=0.4,
        )
        
        response = await smart_search_engine.search(request)

        assert response.session_id is not None
        assert response.confidence_score >= 0.0
        assert response.confidence_score <= 1.0
        assert isinstance(response.results, list)
        assert response.artifacts_found >= 0
        assert response.db_records_found >= 0


@pytest.mark.asyncio
async def test_smart_search_with_feedback(smart_search_engine):
    """Тест поиска с обратной связью."""
    # Первый запрос
    request1 = SmartSearchRequest(query="test query", top_k=10)
    response1 = await smart_search_engine.search(request1)

    # Второй запрос с feedback
    feedback = [
        SearchFeedback(
            record_id="result1",
            relevance="relevant",
            comment="Good result",
        )
    ]
    request2 = SmartSearchRequest(
        query="test query",
        session_id=response1.session_id,
        feedback=feedback,
        top_k=10,
    )

    # Мокаем LLM для рефайнинга
    with patch.object(
        smart_search_engine, "_refine_query", new_callable=AsyncMock
    ) as mock_refine:
        mock_refine.return_value = "refined test query"
        response2 = await smart_search_engine.search(request2)

        assert response2.session_id == response1.session_id
        # Проверяем, что feedback был обработан
        session = smart_search_engine.session_store.get_session(response2.session_id)
        assert len(session["feedback"]) > 0


@pytest.mark.asyncio
async def test_smart_search_clarifying_questions(smart_search_engine):
    """Тест генерации уточняющих вопросов."""
    request = SmartSearchRequest(query="test", clarify=True, top_k=10)

    # Мокаем LLM для генерации вопросов
    with patch.object(
        smart_search_engine,
        "_generate_clarifying_questions",
        new_callable=AsyncMock,
    ) as mock_questions:
        mock_questions.return_value = ["Question 1?", "Question 2?"]
        response = await smart_search_engine.search(request)

        assert response.clarifying_questions is not None
        assert len(response.clarifying_questions) == 2


def test_smart_search_combine_results(smart_search_engine):
    """Тест объединения результатов из БД и артифактов."""
    from memory_mcp.mcp.schema import SearchResultItem
    from datetime import datetime, timezone

    db_results = [
        SearchResultItem(
            record_id="db1",
            score=0.9,
            content="DB result 1",
            source="telegram",
            timestamp=datetime.now(timezone.utc),
        )
    ]

    artifact_results = [
        SearchResultItem(
            record_id="artifact:test.md",
            score=0.8,
            content="Artifact result 1",
            source="artifact",
            timestamp=datetime.now(timezone.utc),
        )
    ]

    combined = smart_search_engine._combine_results(
        db_results, artifact_results, top_k=10
    )

    assert len(combined) == 2
    # Проверяем, что результаты отсортированы по score
    scores = [r.score for r in combined]
    assert scores == sorted(scores, reverse=True)
    
    # Тест с кастомными весами
    combined_custom = smart_search_engine._combine_results(
        db_results, artifact_results, top_k=10,
        db_weight=0.8, artifact_weight=0.2
    )
    assert len(combined_custom) == 2


def test_smart_search_calculate_confidence(smart_search_engine):
    """Тест вычисления confidence score."""
    from memory_mcp.mcp.schema import SearchResultItem
    from datetime import datetime, timezone

    # Высокие scores
    high_results = [
        SearchResultItem(
            record_id=f"r{i}",
            score=0.9,
            content=f"Result {i}",
            source="test",
            timestamp=datetime.now(timezone.utc),
        )
        for i in range(5)
    ]

    confidence_high = smart_search_engine._calculate_confidence(high_results)
    assert confidence_high > 0.5

    # Низкие scores
    low_results = [
        SearchResultItem(
            record_id=f"r{i}",
            score=0.1,
            content=f"Result {i}",
            source="test",
            timestamp=datetime.now(timezone.utc),
        )
        for i in range(2)
    ]

    confidence_low = smart_search_engine._calculate_confidence(low_results)
    assert confidence_low < confidence_high

    # Пустые результаты
    confidence_empty = smart_search_engine._calculate_confidence([])
    assert confidence_empty == 0.0

