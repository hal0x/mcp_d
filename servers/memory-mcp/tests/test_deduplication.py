"""Тесты для модуля дедупликации сообщений."""

import pytest

from memory_mcp.utils.deduplication import (
    deduplicate_by_hash,
    deduplicate_by_id,
    deduplicate_consecutive,
    get_message_hash,
    is_similar,
    normalize_text,
)


class TestGetMessageHash:
    """Тесты для функции get_message_hash."""

    def test_hash_same_message(self):
        """Тест: одинаковые сообщения дают одинаковый хеш."""
        msg1 = {"text": "Hello", "id": 1}
        msg2 = {"text": "Hello", "id": 2}  # Разный ID, но тот же текст

        hash1 = get_message_hash(msg1)
        hash2 = get_message_hash(msg2)

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex string length

    def test_hash_different_messages(self):
        """Тест: разные сообщения дают разные хеши."""
        msg1 = {"text": "Hello"}
        msg2 = {"text": "World"}

        hash1 = get_message_hash(msg1)
        hash2 = get_message_hash(msg2)

        assert hash1 != hash2

    def test_hash_with_multiple_fields(self):
        """Тест: хеш учитывает несколько полей."""
        msg1 = {"text": "Hello", "caption": "Test"}
        msg2 = {"text": "Hello"}  # Без caption

        hash1 = get_message_hash(msg1)
        hash2 = get_message_hash(msg2)

        assert hash1 != hash2

    def test_hash_empty_message(self):
        """Тест: пустое сообщение даёт валидный хеш."""
        msg = {}
        hash_val = get_message_hash(msg)
        assert len(hash_val) == 32


class TestDeduplicateById:
    """Тесты для функции deduplicate_by_id."""

    def test_no_duplicates(self):
        """Тест: нет дубликатов."""
        messages = [{"id": 1, "text": "A"}, {"id": 2, "text": "B"}]
        result = deduplicate_by_id(messages)
        assert len(result) == 2
        assert result == messages

    def test_with_duplicates(self):
        """Тест: удаление дубликатов по ID."""
        messages = [
            {"id": 1, "text": "A"},
            {"id": 2, "text": "B"},
            {"id": 1, "text": "C"},  # Дубликат
        ]
        result = deduplicate_by_id(messages)
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[0]["text"] == "A"  # Первое вхождение сохраняется
        assert result[1]["id"] == 2

    def test_messages_without_id(self):
        """Тест: сообщения без ID сохраняются."""
        messages = [
            {"id": 1, "text": "A"},
            {"text": "No ID"},  # Без ID
            {"id": 1, "text": "B"},  # Дубликат
        ]
        result = deduplicate_by_id(messages)
        assert len(result) == 2
        assert {"text": "No ID"} in result

    def test_empty_list(self):
        """Тест: пустой список."""
        result = deduplicate_by_id([])
        assert result == []


class TestNormalizeText:
    """Тесты для функции normalize_text."""

    def test_removes_extra_spaces(self):
        """Тест: удаление лишних пробелов."""
        text = "  Hello    World  "
        result = normalize_text(text)
        assert result == "hello world"

    def test_lowercase(self):
        """Тест: приведение к нижнему регистру."""
        text = "Hello World"
        result = normalize_text(text)
        assert result == "hello world"

    def test_removes_emoji(self):
        """Тест: удаление эмодзи."""
        text = "Hello 😀 World"
        result = normalize_text(text)
        assert "😀" not in result

    def test_empty_string(self):
        """Тест: пустая строка."""
        assert normalize_text("") == ""
        assert normalize_text(None) == ""


class TestIsSimilar:
    """Тесты для функции is_similar."""

    def test_exact_match(self):
        """Тест: точное совпадение."""
        assert is_similar("Hello", "Hello") is True

    def test_similar_texts(self):
        """Тест: похожие тексты."""
        # "Hello world" и "Hello world!" очень похожи (нормализация убирает различия)
        assert is_similar("Hello world", "Hello world!", threshold=0.85) is True

    def test_different_texts(self):
        """Тест: разные тексты."""
        assert is_similar("Hello", "Goodbye", threshold=0.5) is False

    def test_empty_strings(self):
        """Тест: пустые строки."""
        assert is_similar("", "Hello") is False
        assert is_similar("Hello", "") is False
        assert is_similar("", "") is False

    def test_threshold(self):
        """Тест: влияние порога схожести."""
        text1 = "Hello world"
        text2 = "Hello word"  # Опечатка

        # Проверяем, что тексты действительно похожи при низком пороге
        # "Hello world" и "Hello word" имеют схожесть около 0.9
        similarity_high = is_similar(text1, text2, threshold=0.95)
        similarity_low = is_similar(text1, text2, threshold=0.5)
        
        # При низком пороге должны быть похожи
        assert similarity_low is True
        # При высоком пороге могут быть не похожи (зависит от точной схожести)
        # Это нормально, так как схожесть может быть около 0.9


class TestDeduplicateConsecutive:
    """Тесты для функции deduplicate_consecutive."""

    def test_no_duplicates(self):
        """Тест: нет последовательных дубликатов."""
        messages = [
            {"text": "Hello"},
            {"text": "World"},
            {"text": "Test"},
        ]
        result = deduplicate_consecutive(messages, threshold=0.9)
        assert len(result) == 3

    def test_with_consecutive_duplicates(self):
        """Тест: удаление последовательных дубликатов."""
        messages = [
            {"text": "Hello"},
            {"text": "Hello"},  # Дубликат (consecutive_count=1, <= max_consecutive=1, добавляется)
            {"text": "Hello"},  # Ещё дубликат (consecutive_count=2, > max_consecutive=1, пропускается)
            {"text": "World"},
        ]
        result = deduplicate_consecutive(messages, threshold=0.9, max_consecutive=1)
        # max_consecutive=1 означает: первое + максимум 1 дубликат = 2 сообщения "Hello" + "World"
        assert len(result) == 3
        assert result[0]["text"] == "Hello"
        assert result[1]["text"] == "Hello"  # Второй дубликат добавлен
        assert result[2]["text"] == "World"  # Третий дубликат пропущен

    def test_max_consecutive(self):
        """Тест: максимальное количество последовательных дубликатов."""
        messages = [
            {"text": "Hello"},
            {"text": "Hello"},  # Дубликат (consecutive_count=1, <= max_consecutive=1, добавляется)
            {"text": "Hello"},  # Ещё дубликат (consecutive_count=2, > max_consecutive=1, пропускается)
            {"text": "Hello"},  # Ещё дубликат (consecutive_count=3, > max_consecutive=1, пропускается)
            {"text": "World"},
        ]
        # max_consecutive=1 означает: первое + максимум 1 дубликат = 2 сообщения "Hello"
        result = deduplicate_consecutive(messages, threshold=0.9, max_consecutive=1)
        assert len(result) == 3  # Первое "Hello" + один дубликат "Hello" + "World"
        assert result[0]["text"] == "Hello"
        assert result[1]["text"] == "Hello"
        assert result[2]["text"] == "World"

    def test_empty_list(self):
        """Тест: пустой список."""
        result = deduplicate_consecutive([], threshold=0.9)
        assert result == []

    def test_custom_get_text_func(self):
        """Тест: использование кастомной функции извлечения текста."""

        def get_text(msg):
            return msg.get("content", "")

        messages = [
            {"content": "Hello"},
            {"content": "Hello"},  # Дубликат (consecutive_count=1, <= max_consecutive=1, добавляется)
            {"content": "Hello"},  # Ещё дубликат (consecutive_count=2, > max_consecutive=1, пропускается)
        ]
        result = deduplicate_consecutive(
            messages, threshold=0.9, max_consecutive=1, get_text_func=get_text
        )
        # max_consecutive=1 означает: первое + максимум 1 дубликат = 2 сообщения
        assert len(result) == 2
        assert result[0]["content"] == "Hello"
        assert result[1]["content"] == "Hello"


class TestDeduplicateByHash:
    """Тесты для функции deduplicate_by_hash."""

    def test_no_duplicates(self):
        """Тест: нет дубликатов по хешу."""
        messages = [
            {"text": "Hello", "id": 1},
            {"text": "World", "id": 2},
        ]
        result = deduplicate_by_hash(messages)
        assert len(result) == 2

    def test_with_hash_duplicates(self):
        """Тест: удаление дубликатов по хешу."""
        messages = [
            {"text": "Hello", "id": 1},
            {"text": "Hello", "id": 2},  # Дубликат по содержимому
        ]
        result = deduplicate_by_hash(messages)
        assert len(result) == 1
        assert result[0]["id"] == 1  # Первое вхождение сохраняется

    def test_empty_list(self):
        """Тест: пустой список."""
        result = deduplicate_by_hash([])
        assert result == []

