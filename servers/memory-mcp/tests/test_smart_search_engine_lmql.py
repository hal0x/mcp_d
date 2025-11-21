"""Интеграционные тесты для SmartSearchEngine с LMQL."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from memory_mcp.search.smart_search_engine import SmartSearchEngine
from memory_mcp.mcp.schema import SearchResultItem
from memory_mcp.core.lmql_adapter import LMQLAdapter


class TestSmartSearchEngineLMQL:
    """Тесты для SmartSearchEngine с LMQL."""

    @pytest.fixture
    def mock_lmql_adapter(self):
        """Мок для LMQL адаптера."""
        adapter = MagicMock(spec=LMQLAdapter)
        adapter.available.return_value = True
        adapter.execute_json_query = AsyncMock()
        return adapter

    @pytest.fixture
    def mock_adapter(self):
        """Мок для MemoryServiceAdapter."""
        return MagicMock()

    @pytest.fixture
    def mock_artifacts_reader(self):
        """Мок для ArtifactsReader."""
        return MagicMock()

    @pytest.fixture
    def mock_session_store(self):
        """Мок для SearchSessionStore."""
        return MagicMock()

    @pytest.fixture
    def sample_results(self):
        """Пример результатов поиска."""
        from datetime import datetime, timezone
        return [
            SearchResultItem(
                record_id="result_1",
                content="Содержимое результата 1",
                score=0.9,
                source="telegram",
                timestamp=datetime.now(timezone.utc),
            ),
            SearchResultItem(
                record_id="result_2",
                content="Содержимое результата 2",
                score=0.8,
                source="telegram",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

    @pytest.mark.asyncio
    async def test_generate_clarifying_questions_with_lmql_success(
        self, mock_lmql_adapter, mock_adapter, mock_artifacts_reader, mock_session_store, sample_results
    ):
        """Тест успешной генерации уточняющих вопросов через LMQL."""
        mock_lmql_adapter.execute_json_query.return_value = [
            "Вопрос 1?",
            "Вопрос 2?",
            "Вопрос 3?",
        ]

        engine = SmartSearchEngine(
            adapter=mock_adapter,
            artifacts_reader=mock_artifacts_reader,
            session_store=mock_session_store,
            lmql_adapter=mock_lmql_adapter,
        )
        result = await engine._generate_clarifying_questions("test query", sample_results)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(q, str) for q in result)

    @pytest.mark.asyncio
    async def test_generate_clarifying_questions_without_lmql(
        self, mock_adapter, mock_artifacts_reader, mock_session_store, sample_results
    ):
        """Тест ошибки при отсутствии LMQL адаптера."""
        engine = SmartSearchEngine(
            adapter=mock_adapter,
            artifacts_reader=mock_artifacts_reader,
            session_store=mock_session_store,
            lmql_adapter=None,
        )

        with pytest.raises(RuntimeError, match="LMQL не настроен"):
            await engine._generate_clarifying_questions("test query", sample_results)

    @pytest.mark.asyncio
    async def test_suggest_refinements_with_lmql_success(
        self, mock_lmql_adapter, mock_adapter, mock_artifacts_reader, mock_session_store, sample_results
    ):
        """Тест успешной генерации предложений через LMQL."""
        mock_lmql_adapter.execute_json_query.return_value = [
            "Вариант 1",
            "Вариант 2",
            "Вариант 3",
        ]

        engine = SmartSearchEngine(
            adapter=mock_adapter,
            artifacts_reader=mock_artifacts_reader,
            session_store=mock_session_store,
            lmql_adapter=mock_lmql_adapter,
        )
        result = await engine._suggest_refinements("test query", sample_results)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(r, str) for r in result)

    @pytest.mark.asyncio
    async def test_suggest_refinements_without_lmql(
        self, mock_adapter, mock_artifacts_reader, mock_session_store, sample_results
    ):
        """Тест ошибки при отсутствии LMQL адаптера."""
        engine = SmartSearchEngine(
            adapter=mock_adapter,
            artifacts_reader=mock_artifacts_reader,
            session_store=mock_session_store,
            lmql_adapter=None,
        )

        with pytest.raises(RuntimeError, match="LMQL не настроен"):
            await engine._suggest_refinements("test query", sample_results)

    @pytest.mark.asyncio
    async def test_generate_clarifying_questions_lmql_error(
        self, mock_lmql_adapter, mock_adapter, mock_artifacts_reader, mock_session_store, sample_results
    ):
        """Тест ошибки при исключении в LMQL."""
        mock_lmql_adapter.execute_json_query.side_effect = RuntimeError("LMQL error")

        engine = SmartSearchEngine(
            adapter=mock_adapter,
            artifacts_reader=mock_artifacts_reader,
            session_store=mock_session_store,
            lmql_adapter=mock_lmql_adapter,
        )

        with pytest.raises(RuntimeError, match="Ошибка генерации уточняющих вопросов"):
            await engine._generate_clarifying_questions("test query", sample_results)

