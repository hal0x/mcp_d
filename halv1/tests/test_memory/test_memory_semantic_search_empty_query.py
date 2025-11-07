"""Тесты для semantic_search с пустыми запросами."""

import pytest
from unittest.mock import MagicMock
from memory import UnifiedMemory


class TestMemorySemanticSearchEmpty:
    def setup_method(self):
        self.memory = UnifiedMemory()
        self.memory._embed = MagicMock(return_value=[0.1, 0.2, 0.3])

    def test_empty_string_returns_empty_list(self):
        """Пустая строка возвращает пустой список."""
        result = self.memory.semantic_search("")
        assert result == []
        self.memory._embed.assert_not_called()

    def test_whitespace_only_returns_empty_list(self):
        """Строка из пробелов возвращает пустой список.""" 
        result = self.memory.semantic_search("   ")
        assert result == []
        self.memory._embed.assert_not_called()

    def test_none_query_handling(self):
        """Обработка None как запроса."""
        try:
            result = self.memory.semantic_search(None)
            assert result == []
        except (TypeError, AttributeError):
            pass  # Может упасть на None

    def test_tabs_and_newlines_empty(self):
        """Табы и переносы строк считаются пустыми."""
        result = self.memory.semantic_search("\t\n\r")
        assert result == []
        self.memory._embed.assert_not_called()
