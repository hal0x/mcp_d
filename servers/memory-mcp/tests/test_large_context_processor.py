"""Тесты для LargeContextProcessor."""

import pytest
from datetime import datetime, timedelta

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
async def test_estimate_tokens():
    """Тест оценки токенов."""
    processor = LargeContextProcessor()
    assert processor.estimate_tokens("Hello world") > 0


@pytest.mark.asyncio
async def test_should_use_hierarchical():
    """Тест определения необходимости иерархической обработки."""
    processor = LargeContextProcessor(
        hierarchical_threshold=1000, enable_hierarchical=True
    )
    
    # Маленький контекст
    small_messages = [create_test_message("Test") for _ in range(10)]
    assert not processor.should_use_hierarchical(small_messages)
    
    # Большой контекст (создаем много длинных сообщений)
    large_messages = [
        create_test_message("A" * 200, days_ago=i) for i in range(1000)
    ]
    # Проверяем, что оценка токенов больше порога
    tokens = processor.estimate_messages_tokens(large_messages)
    if tokens > 1000:
        assert processor.should_use_hierarchical(large_messages)


@pytest.mark.asyncio
async def test_process_single_request():
    """Тест обработки одного запроса."""
    processor = LargeContextProcessor(
        max_tokens=10000, enable_hierarchical=False
    )
    messages = [create_test_message(f"Message {i}") for i in range(10)]
    
    # Без LLM клиента результат будет пустым, но структура должна быть правильной
    result = await processor._process_single_request(messages, "test_chat", None)
    assert "summary" in result
    assert "groups" in result
    assert "tokens_used" in result


@pytest.mark.asyncio
async def test_compress_context():
    """Тест сжатия контекста."""
    processor = LargeContextProcessor()
    messages = [
        create_test_message("A" * 100, days_ago=i) for i in range(100)
    ]
    compressed = processor._compress_context(messages, target_tokens=1000)
    assert len(compressed) > 0
    assert processor.estimate_tokens(compressed) <= 2000  # С запасом


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


def test_cache_key_generation():
    """Тест генерации ключа кэша."""
    processor = LargeContextProcessor(enable_caching=True)
    messages = [create_test_message("Test")]
    key1 = processor._get_cache_key(messages, "chat1")
    key2 = processor._get_cache_key(messages, "chat1")
    assert key1 == key2  # Одинаковые входные данные должны давать одинаковый ключ
    
    key3 = processor._get_cache_key(messages, "chat2")
    assert key1 != key3  # Разные чаты должны давать разные ключи


def test_clear_cache():
    """Тест очистки кэша."""
    processor = LargeContextProcessor(enable_caching=True)
    processor._cache["test_key"] = {"test": "data"}
    assert len(processor._cache) > 0
    processor.clear_cache()
    assert len(processor._cache) == 0

