"""Интеграционные тесты для ClusterSummarizer с LMQL."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from memory_mcp.analysis.cluster_summarizer import ClusterSummarizer
from memory_mcp.core.lmql_adapter import LMQLAdapter


class TestClusterSummarizerLMQL:
    """Тесты для ClusterSummarizer с LMQL."""

    @pytest.fixture
    def mock_lmql_adapter(self):
        """Мок для LMQL адаптера."""
        adapter = MagicMock(spec=LMQLAdapter)
        adapter.available.return_value = True
        adapter.execute_json_query = AsyncMock()
        return adapter

    @pytest.fixture
    def mock_embedding_client(self):
        """Мок для embedding клиента."""
        return MagicMock()

    @pytest.fixture
    def sample_cluster(self):
        """Пример кластера для тестов."""
        return {
            "cluster_id": "cluster_1",
            "sessions": [
                {
                    "session_id": "session_1",
                    "topics": [{"title": "Тема 1", "summary": "Описание темы 1"}],
                    "messages": [{"text": "Сообщение 1"}],
                }
            ],
            "session_count": 1,
            "dominant_chat": "test_chat",
        }

    @pytest.mark.asyncio
    async def test_summarize_cluster_with_lmql_success(
        self, mock_lmql_adapter, mock_embedding_client, sample_cluster
    ):
        """Тест успешной генерации сводки через LMQL."""
        mock_lmql_adapter.execute_json_query.return_value = {
            "title": "Главная тема кластера",
            "description": "Краткое описание кластера",
            "key_insights": ["Инсайт 1", "Инсайт 2", "Инсайт 3"],
        }

        summarizer = ClusterSummarizer(
            embedding_client=mock_embedding_client, lmql_adapter=mock_lmql_adapter
        )
        result = await summarizer.summarize_cluster(sample_cluster)

        assert "title" in result
        assert "description" in result
        assert "key_insights" in result
        assert len(result["key_insights"]) == 3

    @pytest.mark.asyncio
    async def test_summarize_cluster_without_lmql(
        self, mock_embedding_client, sample_cluster
    ):
        """Тест ошибки при отсутствии LMQL адаптера."""
        summarizer = ClusterSummarizer(
            embedding_client=mock_embedding_client, lmql_adapter=None
        )

        with pytest.raises(RuntimeError, match="LMQL не настроен"):
            await summarizer.summarize_cluster(sample_cluster)

    @pytest.mark.asyncio
    async def test_summarize_cluster_lmql_error(
        self, mock_lmql_adapter, mock_embedding_client, sample_cluster
    ):
        """Тест ошибки при исключении в LMQL."""
        mock_lmql_adapter.execute_json_query.side_effect = RuntimeError("LMQL error")

        summarizer = ClusterSummarizer(
            embedding_client=mock_embedding_client, lmql_adapter=mock_lmql_adapter
        )

        with pytest.raises(RuntimeError, match="Ошибка генерации сводки через LMQL"):
            await summarizer.summarize_cluster(sample_cluster)

    @pytest.mark.asyncio
    async def test_summarize_cluster_empty_sessions(
        self, mock_lmql_adapter, mock_embedding_client
    ):
        """Тест обработки пустого кластера."""
        empty_cluster = {
            "cluster_id": "empty_cluster",
            "sessions": [],
            "session_count": 0,
        }

        summarizer = ClusterSummarizer(
            embedding_client=mock_embedding_client, lmql_adapter=mock_lmql_adapter
        )
        result = await summarizer.summarize_cluster(empty_cluster)

        assert result["title"] == "Пустой кластер empty_cluster"
        assert result["description"] == "Нет сессий для анализа"

