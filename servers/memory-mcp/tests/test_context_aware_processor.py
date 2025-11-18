"""Тесты для ContextAwareProcessor."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from memory_mcp.analysis.context_aware_processor import ContextAwareProcessor
from memory_mcp.analysis.large_context_processor import LargeContextProcessor


def create_test_message(text: str, days_ago: int = 0) -> dict:
    """Создает тестовое сообщение."""
    date = datetime.now() - timedelta(days=days_ago)
    return {
        "text": text,
        "date": date.isoformat(),
        "from": {"display": "TestUser"},
    }


@pytest.mark.asyncio
async def test_process_without_smart_search():
    """Тест обработки без smart_search."""
    large_processor = LargeContextProcessor(enable_hierarchical=False)
    context_processor = ContextAwareProcessor(
        large_processor, enable_smart_search=False
    )
    
    messages = [create_test_message("Test message")]
    result = await context_processor.process_with_context_awareness(
        messages, "test_chat"
    )
    
    assert "summary" in result or "detailed_summaries" in result
    assert result.get("context_insights") is None


@pytest.mark.asyncio
async def test_extract_context_query():
    """Тест извлечения поискового запроса из сообщений."""
    large_processor = LargeContextProcessor()
    context_processor = ContextAwareProcessor(large_processor)
    
    messages = [
        create_test_message("Bitcoin price discussion"),
        create_test_message("Ethereum analysis"),
    ]
    query = context_processor._extract_context_query(messages, "test_chat")
    assert len(query) > 0
    assert isinstance(query, str)


def test_extract_topics_from_text():
    """Тест извлечения тем из текста."""
    large_processor = LargeContextProcessor()
    context_processor = ContextAwareProcessor(large_processor)
    
    text = "Bitcoin and Ethereum are cryptocurrencies. Bitcoin is popular."
    topics = context_processor._extract_topics_from_text(text)
    assert len(topics) > 0
    assert isinstance(topics, list)


def test_enhance_prompt_with_context():
    """Тест улучшения промпта с контекстом."""
    large_processor = LargeContextProcessor()
    context_processor = ContextAwareProcessor(large_processor)
    
    original_prompt = "Create summary"
    context_insights = {
        "key_topics": ["Bitcoin", "Ethereum"],
        "confidence": 0.8,
    }
    
    enhanced = context_processor._enhance_prompt_with_context(
        original_prompt, context_insights, "test_chat"
    )
    
    assert "Bitcoin" in enhanced or "Ethereum" in enhanced
    assert len(enhanced) > len(original_prompt)


def test_create_default_prompt():
    """Тест создания стандартного промпта."""
    large_processor = LargeContextProcessor()
    context_processor = ContextAwareProcessor(large_processor)
    
    prompt = context_processor._create_default_prompt("test_chat")
    assert "test_chat" in prompt
    assert len(prompt) > 0

