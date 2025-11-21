"""Тесты для SessionSummarizer с LMQL интеграцией."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, Optional

from memory_mcp.analysis.summarization.session.summarizer import SessionSummarizer
from memory_mcp.core.adapters.lmql_adapter import LMQLAdapter
from memory_mcp.core.adapters.langchain_adapters import LangChainLLMAdapter


@pytest.fixture
def mock_lmql_adapter():
    """Мок LMQL адаптера."""
    adapter = MagicMock(spec=LMQLAdapter)
    adapter.available = MagicMock(return_value=True)
    return adapter


@pytest.fixture
def mock_llm_client():
    """Мок LLM клиента."""
    client = MagicMock(spec=LangChainLLMAdapter)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def sample_session():
    """Пример сессии для тестирования."""
    return {
        "session_id": "test_session_1",
        "chat": "TestChat",
        "participants": ["user1", "user2"],
        "time_range_bkk": "2024-01-01 10:00 - 11:00",
        "start_time_utc": "2024-01-01T03:00:00Z",
        "end_time_utc": "2024-01-01T04:00:00Z",
        "messages": [
            {
                "id": 1,
                "text": "Привет, как дела?",
                "author": "user1",
                "timestamp": "2024-01-01T03:00:00Z",
            },
            {
                "id": 2,
                "text": "Отлично, спасибо!",
                "author": "user2",
                "timestamp": "2024-01-01T03:01:00Z",
            },
        ],
    }


@pytest.mark.asyncio
async def test_generate_summary_with_lmql_channel_mode(mock_lmql_adapter, mock_llm_client):
    """Тест генерации саммаризации через LMQL для канала."""
    # Настраиваем мок LMQL адаптера
    mock_lmql_adapter.execute_json_query = AsyncMock(
        return_value={
            "context": "Тестовый контекст для канала",
            "key_points": ["Тезис 1", "Тезис 2"],
            "important_items": ["Важный пункт"],
            "risks": ["Риск 1"],
        }
    )

    summarizer = SessionSummarizer(
        embedding_client=mock_llm_client,
        lmql_adapter=mock_lmql_adapter,
    )

    prompt = "Проанализируй следующий разговор..."
    result = await summarizer._generate_summary_with_lmql(
        prompt=prompt,
        chat_mode="channel",
        language="ru",
    )

    assert result is not None
    assert result["context"] == "Тестовый контекст для канала"
    assert len(result["key_points"]) == 2
    assert len(result["important_items"]) == 1
    assert len(result["risks"]) == 1
    assert result["discussion"] == []  # Для канала discussion пустой
    assert result["decisions"] == []  # Для канала decisions пустой

    # Проверяем, что LMQL был вызван с правильными параметрами
    mock_lmql_adapter.execute_json_query.assert_called_once()
    call_args = mock_lmql_adapter.execute_json_query.call_args
    assert call_args.kwargs["prompt"] == prompt
    assert call_args.kwargs["temperature"] == 0.3
    assert call_args.kwargs["max_tokens"] == 30000
    assert "context" in call_args.kwargs["json_schema"]
    assert "key_points" in call_args.kwargs["json_schema"]


@pytest.mark.asyncio
async def test_generate_summary_with_lmql_group_mode(mock_lmql_adapter, mock_llm_client):
    """Тест генерации саммаризации через LMQL для группы."""
    # Настраиваем мок LMQL адаптера
    mock_lmql_adapter.execute_json_query = AsyncMock(
        return_value={
            "context": "Тестовый контекст для группы",
            "discussion": ["Обсуждение 1", "Обсуждение 2"],
            "decisions": ["Решение 1"],
            "risks": ["Риск 1"],
        }
    )

    summarizer = SessionSummarizer(
        embedding_client=mock_llm_client,
        lmql_adapter=mock_lmql_adapter,
    )

    prompt = "Проанализируй следующий разговор..."
    result = await summarizer._generate_summary_with_lmql(
        prompt=prompt,
        chat_mode="group",
        language="ru",
    )

    assert result is not None
    assert result["context"] == "Тестовый контекст для группы"
    assert len(result["discussion"]) == 2
    assert len(result["decisions"]) == 1
    assert len(result["risks"]) == 1
    assert result["key_points"] == []  # Для группы key_points пустой
    assert result["important_items"] == []  # Для группы important_items пустой

    # Проверяем, что LMQL был вызван с правильными параметрами
    mock_lmql_adapter.execute_json_query.assert_called_once()
    call_args = mock_lmql_adapter.execute_json_query.call_args
    assert "discussion" in call_args.kwargs["json_schema"]
    assert "decisions" in call_args.kwargs["json_schema"]


@pytest.mark.asyncio
async def test_generate_summary_with_lmql_error_handling(mock_lmql_adapter, mock_llm_client):
    """Тест обработки ошибок при использовании LMQL."""
    # Настраиваем мок LMQL адаптера для выброса ошибки
    mock_lmql_adapter.execute_json_query = AsyncMock(
        side_effect=RuntimeError("LMQL error")
    )

    summarizer = SessionSummarizer(
        embedding_client=mock_llm_client,
        lmql_adapter=mock_lmql_adapter,
    )

    prompt = "Проанализируй следующий разговор..."
    result = await summarizer._generate_summary_with_lmql(
        prompt=prompt,
        chat_mode="group",
        language="ru",
    )

    # При ошибке должен вернуться None
    assert result is None


@pytest.mark.asyncio
async def test_generate_summary_with_lmql_no_adapter(mock_llm_client):
    """Тест генерации саммаризации без LMQL адаптера."""
    summarizer = SessionSummarizer(
        embedding_client=mock_llm_client,
        lmql_adapter=None,
    )

    prompt = "Проанализируй следующий разговор..."
    result = await summarizer._generate_summary_with_lmql(
        prompt=prompt,
        chat_mode="group",
        language="ru",
    )

    # Без адаптера должен вернуться None
    assert result is None


@pytest.mark.asyncio
async def test_summarize_session_with_lmql_fallback(
    mock_lmql_adapter, mock_llm_client, sample_session
):
    """Тест саммаризации сессии с использованием LMQL и fallback."""
    # Настраиваем мок LMQL адаптера для успешного ответа
    mock_lmql_adapter.execute_json_query = AsyncMock(
        return_value={
            "context": "Тестовый контекст",
            "discussion": ["Обсуждение 1"],
            "decisions": [],
            "risks": [],
        }
    )

    summarizer = SessionSummarizer(
        embedding_client=mock_llm_client,
        lmql_adapter=mock_lmql_adapter,
        enable_quality_check=False,  # Отключаем проверку качества для упрощения теста
        enable_iterative_refinement=False,  # Отключаем итеративное улучшение
    )

    # Мокаем вспомогательные методы
    with patch.object(summarizer, "_prepare_conversation_text", return_value="Test conversation"):
        with patch.object(
            summarizer, "_create_summarization_prompt", return_value="Test prompt"
        ):
            with patch.object(
                summarizer, "_ensure_summary_completeness", return_value=({}, False)
            ):
                with patch.object(
                    summarizer, "_extract_action_items", return_value=[]
                ):
                    with patch.object(
                        summarizer, "_format_links_artifacts", return_value=[]
                    ):
                        with patch.object(
                            summarizer, "_build_canonical_summary", return_value=({}, {})
                        ):
                            result = await summarizer.summarize_session(sample_session)

    # Проверяем, что LMQL был вызван
    mock_lmql_adapter.execute_json_query.assert_called_once()
    assert result is not None


@pytest.mark.asyncio
async def test_summarize_session_with_lmql_fallback_to_llm(
    mock_lmql_adapter, mock_llm_client, sample_session
):
    """Тест саммаризации сессии с fallback на LLM при ошибке LMQL."""
    # Настраиваем мок LMQL адаптера для выброса ошибки
    mock_lmql_adapter.execute_json_query = AsyncMock(
        side_effect=RuntimeError("LMQL error")
    )

    # Настраиваем мок LLM клиента для fallback
    mock_llm_client.generate_summary = AsyncMock(
        return_value="## Контекст\nТестовый контекст\n\n## Ход дискуссии\n- Обсуждение 1"
    )

    summarizer = SessionSummarizer(
        embedding_client=mock_llm_client,
        lmql_adapter=mock_lmql_adapter,
        enable_quality_check=False,
        enable_iterative_refinement=False,
    )

    # Мокаем вспомогательные методы
    with patch.object(summarizer, "_prepare_conversation_text", return_value="Test conversation"):
        with patch.object(
            summarizer, "_create_summarization_prompt", return_value="Test prompt"
        ):
            with patch.object(
                summarizer, "_ensure_summary_completeness", return_value=({}, False)
            ):
                with patch.object(
                    summarizer, "_extract_action_items", return_value=[]
                ):
                    with patch.object(
                        summarizer, "_format_links_artifacts", return_value=[]
                    ):
                        with patch.object(
                            summarizer, "_build_canonical_summary", return_value=({}, {})
                        ):
                            result = await summarizer.summarize_session(sample_session)

    # Проверяем, что LMQL был вызван и затем произошел fallback на LLM
    mock_lmql_adapter.execute_json_query.assert_called_once()
    mock_llm_client.generate_summary.assert_called_once()
    assert result is not None

