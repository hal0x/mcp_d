"""Интеграционные тесты для EntityDictionary с LMQL."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from memory_mcp.analysis.entity_dictionary import EntityDictionary
from memory_mcp.core.lmql_adapter import LMQLAdapter


class TestEntityDictionaryLMQL:
    """Тесты для EntityDictionary с LMQL."""

    @pytest.fixture
    def mock_lmql_adapter(self):
        """Мок для LMQL адаптера."""
        adapter = MagicMock(spec=LMQLAdapter)
        adapter.available.return_value = True
        adapter.execute_validation_query = AsyncMock()
        return adapter

    @pytest.fixture
    def entity_dict(self, tmp_path, mock_lmql_adapter):
        """Создание экземпляра EntityDictionary для тестов."""
        storage_path = tmp_path / "entity_dictionaries"
        return EntityDictionary(
            storage_path=storage_path,
            enable_llm_validation=True,
            lmql_adapter=mock_lmql_adapter,
        )

    @pytest.mark.asyncio
    async def test_validate_entity_with_lmql_success_yes(self, entity_dict, mock_lmql_adapter):
        """Тест успешной валидации сущности через LMQL (ДА)."""
        mock_lmql_adapter.execute_validation_query.return_value = "ДА"

        result = await entity_dict._validate_entity_with_llm_async(
            entity_type="persons",
            normalized_value="иван",
            original_value="Иван",
        )

        assert result is True
        mock_lmql_adapter.execute_validation_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_entity_with_lmql_success_no(self, entity_dict, mock_lmql_adapter):
        """Тест успешной валидации сущности через LMQL (НЕТ)."""
        mock_lmql_adapter.execute_validation_query.return_value = "НЕТ"

        result = await entity_dict._validate_entity_with_llm_async(
            entity_type="persons",
            normalized_value="стол",
            original_value="стол",
        )

        assert result is False
        mock_lmql_adapter.execute_validation_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_entity_without_lmql(self, tmp_path):
        """Тест работы без LMQL адаптера (fallback на обычный LLM)."""
        entity_dict = EntityDictionary(
            storage_path=tmp_path / "entity_dictionaries",
            enable_llm_validation=True,
            lmql_adapter=None,
        )

        with patch.object(entity_dict, "_get_llm_client") as mock_get_client:
            mock_llm_client = MagicMock()
            mock_llm_client.__aenter__ = AsyncMock(return_value=mock_llm_client)
            mock_llm_client.__aexit__ = AsyncMock(return_value=None)
            mock_llm_client.generate_summary = AsyncMock(return_value="ДА")
            mock_get_client.return_value = mock_llm_client

            result = await entity_dict._validate_entity_with_llm_async(
                entity_type="persons",
                normalized_value="иван",
                original_value="Иван",
            )

            assert result is True
            mock_llm_client.generate_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_entity_lmql_error(self, entity_dict, mock_lmql_adapter):
        """Тест ошибки при исключении в LMQL."""
        mock_lmql_adapter.execute_validation_query.side_effect = RuntimeError("LMQL error")

        with pytest.raises(RuntimeError, match="LMQL error"):
            await entity_dict._validate_entity_with_llm_async(
                entity_type="persons",
                normalized_value="иван",
                original_value="Иван",
            )

