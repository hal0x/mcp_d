"""Тесты для AdaptiveMessageGrouper."""

import pytest
from datetime import datetime, timedelta

from memory_mcp.analysis.adaptive_message_grouper import AdaptiveMessageGrouper


def create_test_message(text: str, days_ago: int = 0) -> dict:
    """Создает тестовое сообщение."""
    date = datetime.now() - timedelta(days=days_ago)
    return {
        "text": text,
        "date": date.isoformat(),
        "from": {"display": "TestUser"},
    }


def test_estimate_tokens():
    """Тест оценки токенов."""
    grouper = AdaptiveMessageGrouper()
    assert grouper.estimate_tokens("Hello world") > 0
    assert grouper.estimate_tokens("") == 0


def test_estimate_message_tokens():
    """Тест оценки токенов сообщения."""
    grouper = AdaptiveMessageGrouper()
    msg = create_test_message("Test message")
    tokens = grouper.estimate_message_tokens(msg)
    assert tokens > 0


def test_group_small_context():
    """Тест группировки маленького контекста."""
    grouper = AdaptiveMessageGrouper(max_tokens=126000)
    messages = [create_test_message(f"Message {i}") for i in range(10)]
    groups = grouper.group_messages_adaptively(messages, "test_chat")
    assert len(groups) == 1  # Все должно поместиться в одну группу


def test_group_large_context():
    """Тест группировки большого контекста."""
    grouper = AdaptiveMessageGrouper(max_tokens=1000)  # Маленький лимит для теста
    # Создаем много длинных сообщений
    messages = [
        create_test_message("A" * 100, days_ago=i) for i in range(100)
    ]
    groups = grouper.group_messages_adaptively(messages, "test_chat")
    assert len(groups) > 1  # Должно быть несколько групп


def test_temporal_grouping():
    """Тест временной группировки."""
    grouper = AdaptiveMessageGrouper(strategy="temporal", max_tokens=10000)
    messages = [
        create_test_message(f"Message {i}", days_ago=i) for i in range(30)
    ]
    groups = grouper.group_messages_adaptively(messages, "test_chat")
    assert len(groups) > 0


def test_quantitative_grouping():
    """Тест количественной группировки."""
    grouper = AdaptiveMessageGrouper(strategy="quantitative", max_tokens=10000)
    messages = [create_test_message(f"Message {i}") for i in range(50)]
    groups = grouper.group_messages_adaptively(messages, "test_chat")
    assert len(groups) > 0


def test_merge_small_groups():
    """Тест объединения маленьких групп."""
    grouper = AdaptiveMessageGrouper(
        max_tokens=10000, min_group_size_tokens=1000
    )
    # Создаем несколько маленьких групп
    groups = [
        [create_test_message(f"Msg {i}") for i in range(5)]
        for _ in range(5)
    ]
    merged = grouper._merge_small_groups(groups)
    assert len(merged) <= len(groups)  # Должно быть меньше или равно


def test_activity_estimation():
    """Тест оценки активности."""
    grouper = AdaptiveMessageGrouper()
    # Высокая активность
    high_activity = [create_test_message(f"Msg {i}", days_ago=i) for i in range(100)]
    assert grouper._estimate_activity(high_activity) in ["high", "medium", "low"]
    
    # Низкая активность
    low_activity = [create_test_message(f"Msg {i}", days_ago=i*10) for i in range(5)]
    assert grouper._estimate_activity(low_activity) in ["high", "medium", "low"]

