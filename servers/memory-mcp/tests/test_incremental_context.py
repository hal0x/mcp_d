#!/usr/bin/env python3
"""
Тесты для расширенного контекста при доиндексации
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from memory_mcp.analysis.context.incremental_context_manager import IncrementalContextManager
from memory_mcp.analysis.summarization.session.summarizer import SessionSummarizer


class TestIncrementalContextManager:
    """Тесты для IncrementalContextManager"""

    def test_empty_context(self):
        """Тест пустого контекста"""
        manager = IncrementalContextManager()
        context = manager._empty_context()

        assert context["previous_messages_count"] == 0
        assert context["previous_sessions_count"] == 0
        assert context["chat_context"] is None
        assert context["previous_messages"] == []
        assert context["previous_sessions"] == []

    def test_should_use_extended_context(self):
        """Тест определения необходимости расширенного контекста"""
        manager = IncrementalContextManager()

        # Малая сессия - должна использовать расширенный контекст
        small_session = [{"text": "test"}] * 5
        assert manager.should_use_extended_context(
            small_session, min_messages_threshold=10
        )

        # Большая сессия - не должна использовать расширенный контекст
        large_session = [{"text": "test"}] * 20
        assert not manager.should_use_extended_context(
            large_session, min_messages_threshold=10
        )

    def test_format_context_for_prompt(self):
        """Тест форматирования контекста для промпта"""
        manager = IncrementalContextManager()

        # Тестовый расширенный контекст
        extended_context = {
            "chat_context": "Тестовый контекст чата",
            "previous_sessions": [
                {
                    "session_id": "session-1",
                    "summary": "Краткое описание предыдущей сессии",
                }
            ],
            "previous_messages": [{"text": "Предыдущее сообщение для контекста"}],
        }

        formatted = manager.format_context_for_prompt(
            extended_context, max_context_length=1000
        )

        assert "## Образ чата" in formatted
        assert "Тестовый контекст чата" in formatted
        assert "## Предыдущие сессии" in formatted
        assert "session-1" in formatted
        assert "## Предыдущие сообщения" in formatted
        assert "Предыдущее сообщение" in formatted

    def test_format_context_empty(self):
        """Тест форматирования пустого контекста"""
        manager = IncrementalContextManager()

        empty_context = manager._empty_context()
        formatted = manager.format_context_for_prompt(empty_context)

        assert formatted == ""

    def test_parse_message_time(self):
        """Тест парсинга времени сообщения"""
        manager = IncrementalContextManager()

        # Тест с корректным временем
        message = {"date_utc": "2025-10-06T18:00:00+00:00"}
        parsed_time = manager._parse_message_time(message)
        assert isinstance(parsed_time, datetime)

        # Тест с некорректным временем
        message_invalid = {"date_utc": "invalid"}
        parsed_time_invalid = manager._parse_message_time(message_invalid)
        assert isinstance(parsed_time_invalid, datetime)  # Должен вернуть текущее время

    def test_get_session_start_time(self):
        """Тест получения времени начала сессии"""
        manager = IncrementalContextManager()

        # Тест с сообщениями
        messages = [
            {"date_utc": "2025-10-06T18:00:00+00:00"},
            {"date_utc": "2025-10-06T19:00:00+00:00"},
            {"date_utc": "2025-10-06T17:00:00+00:00"},  # Самое раннее
        ]

        start_time = manager._get_session_start_time(messages)
        expected_time = datetime.fromisoformat("2025-10-06T17:00:00+00:00")
        assert start_time == expected_time

        # Тест с пустым списком
        empty_messages = []
        start_time_empty = manager._get_session_start_time(empty_messages)
        assert isinstance(start_time_empty, datetime)


class TestSessionSummarizerWithExtendedContext:
    """Тесты для SessionSummarizer с расширенным контекстом"""

    def test_create_summarization_prompt_with_extended_context(self):
        """Тест создания промпта с расширенным контекстом"""
        summarizer = SessionSummarizer()

        # Тестовые данные
        conversation_text = "Тестовый разговор"
        chat = "Test Chat"
        language = "ru"
        session = {"session_id": "test-session"}
        chat_mode = "group"
        previous_context = {
            "previous_sessions_count": 0,
            "recent_context": "",
            "ongoing_decisions": [],
            "open_risks": [],
            "key_links": [],
            "session_timeline": [],
        }

        # Расширенный контекст
        extended_context = {
            "previous_messages_count": 5,
            "chat_context": "Тестовый контекст чата",
            "previous_sessions": [
                {"session_id": "prev-session", "summary": "Предыдущая сессия"}
            ],
            "previous_messages": [{"text": "Предыдущее сообщение"}],
        }

        prompt = summarizer._create_summarization_prompt(
            conversation_text,
            chat,
            language,
            session,
            chat_mode,
            previous_context,
            extended_context,
        )

        # Проверяем наличие расширенного контекста в промпте
        assert "## Расширенный контекст (для малой сессии)" in prompt
        assert "Тестовый контекст чата" in prompt
        assert "Предыдущее сообщение" in prompt
        assert "При малом количестве сообщений используй расширенный контекст" in prompt

    def test_create_summarization_prompt_without_extended_context(self):
        """Тест создания промпта без расширенного контекста"""
        summarizer = SessionSummarizer()

        # Тестовые данные
        conversation_text = "Тестовый разговор"
        chat = "Test Chat"
        language = "ru"
        session = {"session_id": "test-session"}
        chat_mode = "group"
        previous_context = {
            "previous_sessions_count": 0,
            "recent_context": "",
            "ongoing_decisions": [],
            "open_risks": [],
            "key_links": [],
            "session_timeline": [],
        }

        prompt = summarizer._create_summarization_prompt(
            conversation_text,
            chat,
            language,
            session,
            chat_mode,
            previous_context,
            None,  # Без расширенного контекста
        )

        # Проверяем отсутствие расширенного контекста в промпте
        assert "## Расширенный контекст (для малой сессии)" not in prompt
        assert (
            "При малом количестве сообщений используй расширенный контекст"
            not in prompt
        )

    def test_create_summarization_prompt_with_empty_extended_context(self):
        """Тест создания промпта с пустым расширенным контекстом"""
        summarizer = SessionSummarizer()

        # Тестовые данные
        conversation_text = "Тестовый разговор"
        chat = "Test Chat"
        language = "ru"
        session = {"session_id": "test-session"}
        chat_mode = "group"
        previous_context = {
            "previous_sessions_count": 0,
            "recent_context": "",
            "ongoing_decisions": [],
            "open_risks": [],
            "key_links": [],
            "session_timeline": [],
        }

        # Пустой расширенный контекст
        empty_extended_context = {
            "previous_messages_count": 0,
            "chat_context": None,
            "previous_sessions": [],
            "previous_messages": [],
        }

        prompt = summarizer._create_summarization_prompt(
            conversation_text,
            chat,
            language,
            session,
            chat_mode,
            previous_context,
            empty_extended_context,
        )

        # Проверяем отсутствие расширенного контекста в промпте
        assert "## Расширенный контекст (для малой сессии)" not in prompt


class TestIntegration:
    """Интеграционные тесты"""

    def test_incremental_context_manager_integration(self):
        """Тест интеграции IncrementalContextManager"""
        # Создаем временную директорию для тестов
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создаем тестовые файлы
            context_dir = Path(temp_dir) / "chat_contexts"
            context_dir.mkdir()

            # Создаем тестовый контекст чата
            context_file = context_dir / "Test Chat_context.md"
            context_content = """# Test Chat Context

## Образ чата
Это тестовый чат для проверки функциональности расширенного контекста.

## Последние сессии
- session-1: Тестовая сессия
"""
            context_file.write_text(context_content, encoding="utf-8")

            # Создаем менеджер контекста
            manager = IncrementalContextManager(qdrant_url=None)  # Тест без Qdrant

            # Тестовые сообщения
            test_messages = [
                {
                    "text": "Тестовое сообщение 1",
                    "date_utc": "2025-10-06T18:00:00+00:00",
                },
                {
                    "text": "Тестовое сообщение 2",
                    "date_utc": "2025-10-06T19:00:00+00:00",
                },
            ]

            # Получаем расширенный контекст
            context = manager.get_extended_context_for_session(
                "Test Chat", test_messages, context_window_hours=24
            )

            # Проверяем результат
            assert (
                context["previous_messages_count"] == 0
            )  # Нет предыдущих сообщений в ChromaDB
            assert (
                context["previous_sessions_count"] == 0
            )  # Нет предыдущих сессий в ChromaDB
            assert (
                context["chat_context"] is not None
            )  # Контекст чата должен быть загружен
            assert "тестовый чат" in context["chat_context"].lower()

    def test_session_summarizer_with_extended_context(self):
        """Тест SessionSummarizer с расширенным контекстом"""
        summarizer = SessionSummarizer()

        # Проверяем что incremental_context_manager инициализирован
        assert hasattr(summarizer, "incremental_context_manager")
        assert summarizer.incremental_context_manager is not None

        # Проверяем что это экземпляр IncrementalContextManager
        assert isinstance(
            summarizer.incremental_context_manager, IncrementalContextManager
        )


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"])
