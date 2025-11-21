"""Тесты для LargeContextProcessor."""

import pytest
from datetime import datetime, timedelta

from memory_mcp.analysis.aggregation.large_context_processor import LargeContextProcessor


def create_test_message(text: str, days_ago: int = 0) -> dict:
    """Создает тестовое сообщение."""
    date = datetime.now() - timedelta(days=days_ago)
    return {
        "text": text,
        "date": date.isoformat(),
        "from": {"display": "TestUser"},
    }


@pytest.mark.asyncio
async def test_estimate_tokens():
    """Тест оценки токенов."""
    processor = LargeContextProcessor()
    assert processor.estimate_tokens("Hello world") > 0


@pytest.mark.asyncio
async def test_process_single_request():
    """Тест обработки одного запроса."""
    processor = LargeContextProcessor(max_tokens=10000)
    messages = [create_test_message(f"Message {i}") for i in range(10)]
    
    # Без LLM клиента результат будет пустым, но структура должна быть правильной
    result = await processor._process_single_request(messages, "test_chat", None)
    assert "summary" in result
    assert "groups" in result
    assert "tokens_used" in result


def test_format_messages_for_llm():
    """Тест форматирования сообщений для LLM."""
    processor = LargeContextProcessor()
    messages = [
        create_test_message("Test message 1"),
        create_test_message("Test message 2"),
    ]
    formatted = processor._format_messages_for_llm(messages)
    assert "Test message 1" in formatted
    assert "Test message 2" in formatted


def test_combine_summaries():
    """Тест объединения саммаризаций."""
    processor = LargeContextProcessor()
    summaries = [
        {"group_index": 1, "summary": "Summary 1"},
        {"group_index": 2, "summary": "Summary 2"},
    ]
    combined = processor._combine_summaries(summaries)
    assert "Summary 1" in combined
    assert "Summary 2" in combined

