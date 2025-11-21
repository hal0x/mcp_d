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
        from unittest.mock import patch, MagicMock
        storage_path = tmp_path / "entity_dictionaries"
        entity_dict = EntityDictionary(
            storage_path=storage_path,
            enable_llm_validation=True,
            lmql_adapter=mock_lmql_adapter,
        )
        # Мокируем _get_llm_client, чтобы не требовался реальный LLM клиент
        entity_dict._get_llm_client = MagicMock(return_value=None)
        return entity_dict

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

    @pytest.mark.asyncio
    async def test_classify_entity_type_with_lmql_success(self, entity_dict, mock_lmql_adapter):
        """Тест успешной классификации типа сущности через LMQL."""
        mock_lmql_adapter.execute_json_query = AsyncMock()
        mock_lmql_adapter.execute_json_query.return_value = {
            "type": "persons",
            "is_new": False,
        }

        result = await entity_dict._classify_entity_type(
            value="Иван",
            normalized_value="иван",
            context="Иван работает в компании",
        )

        assert result == "persons"
        mock_lmql_adapter.execute_json_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_entity_type_with_lmql_new_type(self, entity_dict, mock_lmql_adapter):
        """Тест классификации нового типа сущности через LMQL."""
        mock_lmql_adapter.execute_json_query = AsyncMock()
        mock_lmql_adapter.execute_json_query.return_value = {
            "type": "games",
            "is_new": True,
        }

        result = await entity_dict._classify_entity_type(
            value="Minecraft",
            normalized_value="minecraft",
        )

        assert result == "games"
        assert "games" in entity_dict.dynamic_entity_types
        mock_lmql_adapter.execute_json_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_entity_type_without_lmql(self, tmp_path):
        """Тест классификации без LMQL (fallback на обычный LLM)."""
        entity_dict = EntityDictionary(
            storage_path=tmp_path / "entity_dictionaries",
            enable_llm_validation=True,
            lmql_adapter=None,
        )

        with patch.object(entity_dict, "_get_llm_client") as mock_get_client:
            mock_llm_client = MagicMock()
            mock_llm_client.__aenter__ = AsyncMock(return_value=mock_llm_client)
            mock_llm_client.__aexit__ = AsyncMock(return_value=None)
            mock_llm_client.generate_summary = AsyncMock(
                return_value='{"type": "persons", "is_new": false}'
            )
            mock_get_client.return_value = mock_llm_client

            result = await entity_dict._classify_entity_type(
                value="Иван",
                normalized_value="иван",
            )

            assert result == "persons"
            mock_llm_client.generate_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_descriptions_batch_with_lmql_success(
        self, entity_dict, mock_lmql_adapter
    ):
        """Тест успешной батч-генерации описаний через LMQL."""
        mock_lmql_adapter.execute_json_query = AsyncMock()
        mock_lmql_adapter.execute_json_query.return_value = {
            "иван": "Русское мужское имя, распространенное в России",
            "мария": "Русское женское имя, популярное в России",
        }

        candidates = [
            {
                "entity_type": "persons",
                "normalized_value": "иван",
                "original_value": "Иван",
                "all_contexts": [{"content": "Иван работает программистом"}],
            },
            {
                "entity_type": "persons",
                "normalized_value": "мария",
                "original_value": "Мария",
                "all_contexts": [{"content": "Мария учится в университете"}],
            },
        ]

        mock_llm_client = MagicMock()
        results = await entity_dict._generate_descriptions_batch_single(
            candidates, mock_llm_client
        )

        assert len(results) == 2
        assert "иван" in results
        assert "мария" in results
        assert "Русское мужское имя" in results["иван"]
        mock_lmql_adapter.execute_json_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_descriptions_batch_without_lmql(self, tmp_path):
        """Тест батч-генерации описаний без LMQL (fallback на обычный LLM)."""
        entity_dict = EntityDictionary(
            storage_path=tmp_path / "entity_dictionaries",
            enable_llm_validation=True,
            lmql_adapter=None,
        )

        mock_llm_client = MagicMock()
        mock_llm_client.__aenter__ = AsyncMock(return_value=mock_llm_client)
        mock_llm_client.__aexit__ = AsyncMock(return_value=None)
        mock_llm_client.generate_summary = AsyncMock(
            return_value='{"иван": "Русское мужское имя"}'
        )

        candidates = [
            {
                "entity_type": "persons",
                "normalized_value": "иван",
                "original_value": "Иван",
                "all_contexts": [],
            }
        ]

        results = await entity_dict._generate_descriptions_batch_single(
            candidates, mock_llm_client
        )

        assert len(results) == 1
        assert "иван" in results
        mock_llm_client.generate_summary.assert_called_once()

