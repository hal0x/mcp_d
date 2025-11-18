"""Тесты для SearchSessionStore."""

import tempfile
from pathlib import Path

import pytest

from memory_mcp.search.search_session_store import SearchSessionStore


@pytest.fixture
def temp_db_path():
    """Создает временный путь к БД для тестов."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sessions.db"
        yield db_path


def test_session_store_create_session(temp_db_path):
    """Тест создания новой сессии."""
    store = SearchSessionStore(db_path=temp_db_path)
    session_id = store.create_session("test query")

    assert session_id is not None
    assert len(session_id) > 0

    session = store.get_session(session_id)
    assert session is not None
    assert session["original_query"] == "test query"


def test_session_store_add_refined_query(temp_db_path):
    """Тест добавления уточненного запроса."""
    store = SearchSessionStore(db_path=temp_db_path)
    session_id = store.create_session("original query")

    query_id = store.add_refined_query(session_id, "refined query")
    assert query_id is not None

    session = store.get_session(session_id)
    assert len(session["queries"]) == 2
    assert session["queries"][1]["query_text"] == "refined query"
    assert session["queries"][1]["query_type"] == "refined"


def test_session_store_save_results(temp_db_path):
    """Тест сохранения результатов поиска."""
    store = SearchSessionStore(db_path=temp_db_path)
    session_id = store.create_session("test query")

    results = [
        {
            "record_id": "result1",
            "artifact_path": None,
            "score": 0.9,
            "content": "Test content 1",
        },
        {
            "record_id": "result2",
            "artifact_path": "/path/to/artifact",
            "score": 0.8,
            "content": "Test content 2",
        },
    ]

    store.save_results(session_id, None, results)

    session = store.get_session(session_id)
    assert len(session["results"]) == 2
    assert session["results"][0]["result_id"] == "result1"
    assert session["results"][1]["result_id"] == "result2"


def test_session_store_add_feedback(temp_db_path):
    """Тест добавления обратной связи."""
    store = SearchSessionStore(db_path=temp_db_path)
    session_id = store.create_session("test query")

    store.add_feedback(
        session_id, "result1", "relevant", artifact_path=None, comment="Good result"
    )
    store.add_feedback(
        session_id, "result2", "irrelevant", artifact_path="/path/to/artifact"
    )

    session = store.get_session(session_id)
    assert len(session["feedback"]) == 2
    assert session["feedback"][0]["relevance"] == "relevant"
    assert session["feedback"][1]["relevance"] == "irrelevant"


def test_session_store_get_feedback_for_learning(temp_db_path):
    """Тест получения обратной связи для обучения."""
    store = SearchSessionStore(db_path=temp_db_path)
    session_id = store.create_session("test query")

    store.add_feedback(session_id, "result1", "relevant")
    store.add_feedback(session_id, "result2", "irrelevant")

    feedback = store.get_feedback_for_learning(limit=10)
    assert len(feedback) >= 2
    assert all("original_query" in f for f in feedback)

