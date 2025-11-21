"""Интеграционные тесты для SemanticRegrouper с LMQL."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from memory_mcp.analysis.semantic_regrouper import SemanticRegrouper
from memory_mcp.core.adapters.lmql_adapter import LMQLAdapter


class TestSemanticRegrouperLMQL:
    """Тесты для SemanticRegrouper с LMQL."""

    @pytest.fixture
    def mock_lmql_adapter(self):
        """Мок для LMQL адаптера."""
        adapter = MagicMock(spec=LMQLAdapter)
        adapter.available.return_value = True
        adapter.execute_json_query = AsyncMock()
        return adapter

    @pytest.fixture
    def sample_sessions(self):
        """Пример сессий для тестов."""
        return [
            {
                "session_id": "session_1",
                "messages": [{"text": "Сообщение 1", "from": {"display": "User1"}}],
                "time_range": "2024-01-01",
                "window": "day",
            },
            {
                "session_id": "session_2",
                "messages": [{"text": "Сообщение 2", "from": {"display": "User2"}}],
                "time_range": "2024-01-02",
                "window": "day",
            },
        ]

    @pytest.mark.asyncio
    async def test_regroup_sessions_with_lmql_success(
        self, mock_lmql_adapter, sample_sessions
    ):
        """Тест успешной перегруппировки через LMQL."""
        mock_lmql_adapter.execute_json_query.return_value = {
            "groups": [
                {
                    "group_id": "group_1",
                    "theme": "Общая тема",
                    "rationale": "Сессии связаны по смыслу",
                    "session_ids": ["session_1", "session_2"],
                    "combined_summary": "Объединенная сводка",
                }
            ]
        }

        regrouper = SemanticRegrouper(lmql_adapter=mock_lmql_adapter)
        result = await regrouper.regroup_sessions(sample_sessions, "test_chat")

        assert len(result) == 1
        assert result[0]["regrouped"] is True
        assert "session_1" in result[0]["original_session_ids"]
        assert "session_2" in result[0]["original_session_ids"]

    @pytest.mark.asyncio
    async def test_regroup_sessions_without_lmql(self, sample_sessions):
        """Тест ошибки при отсутствии LMQL адаптера."""
        regrouper = SemanticRegrouper(lmql_adapter=None)

        with pytest.raises(RuntimeError, match="LMQL не настроен"):
            await regrouper.regroup_sessions(sample_sessions, "test_chat")

    @pytest.mark.asyncio
    async def test_regroup_sessions_lmql_error(
        self, mock_lmql_adapter, sample_sessions
    ):
        """Тест ошибки при исключении в LMQL."""
        mock_lmql_adapter.execute_json_query.side_effect = RuntimeError("LMQL error")

        regrouper = SemanticRegrouper(lmql_adapter=mock_lmql_adapter)

        # При ошибке LMQL метод возвращает исходные сессии как отдельные группы
        result = await regrouper.regroup_sessions(sample_sessions, "test_chat")
        
        # Проверяем, что вернулись исходные сессии
        assert len(result) == len(sample_sessions)
        assert all("session_id" in group for group in result)

    @pytest.mark.asyncio
    async def test_regroup_sessions_empty_list(self, mock_lmql_adapter):
        """Тест обработки пустого списка сессий."""
        regrouper = SemanticRegrouper(lmql_adapter=mock_lmql_adapter)
        result = await regrouper.regroup_sessions([], "test_chat")

        assert result == []

    @pytest.mark.asyncio
    async def test_regroup_sessions_single_session(
        self, mock_lmql_adapter, sample_sessions
    ):
        """Тест обработки одной сессии."""
        regrouper = SemanticRegrouper(lmql_adapter=mock_lmql_adapter)
        result = await regrouper.regroup_sessions([sample_sessions[0]], "test_chat")

        assert result == [sample_sessions[0]]

