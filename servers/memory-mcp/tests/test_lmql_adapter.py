"""Unit тесты для LMQLAdapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memory_mcp.core.adapters.lmql_adapter import LMQLAdapter, build_lmql_adapter_from_env


class TestLMQLAdapter:
    """Тесты для LMQLAdapter."""

    @pytest.fixture
    def mock_lmql(self):
        """Мок для модуля lmql."""
        with patch("memory_mcp.core.lmql_adapter.lmql") as mock:
            mock.run = AsyncMock()
            yield mock

    def test_init_with_lmstudio(self, mock_lmql):
        """Тест инициализации адаптера с LM Studio бэкендом."""
        adapter = LMQLAdapter(
            model="gpt-oss-20b",
            backend="lmstudio",
            base_url="http://localhost:1234",
        )
        assert adapter.model == "gpt-oss-20b"
        assert adapter.backend == "lmstudio"
        assert adapter.model_identifier == "openai/gpt-oss-20b"
        assert adapter.api_config == {"api_base": "http://localhost:1234"}

    def test_init_with_openai(self, mock_lmql):
        """Тест инициализации адаптера с OpenAI бэкендом."""
        adapter = LMQLAdapter(
            model="gpt-3.5-turbo",
            backend="openai",
        )
        assert adapter.model == "gpt-3.5-turbo"
        assert adapter.backend == "openai"
        assert adapter.model_identifier == "openai/gpt-3.5-turbo"
        assert adapter.api_config == {}

    def test_init_without_lmql(self):
        """Тест инициализации без установленного LMQL."""
        with patch("memory_mcp.core.lmql_adapter.lmql", None):
            with pytest.raises(ImportError, match="LMQL не установлен"):
                LMQLAdapter(model="test", backend="openai")

    def test_available(self, mock_lmql):
        """Тест проверки доступности LMQL."""
        adapter = LMQLAdapter(model="test", backend="openai")
        assert adapter.available() is True

    def test_available_without_lmql(self):
        """Тест проверки доступности без LMQL."""
        with patch("memory_mcp.core.lmql_adapter.lmql", None):
            adapter = LMQLAdapter.__new__(LMQLAdapter)
            adapter.lmql = None
            assert adapter.available() is False

    @pytest.mark.asyncio
    async def test_execute_query_success(self, mock_lmql):
        """Тест успешного выполнения запроса."""
        mock_result = MagicMock()
        mock_result.variables = {"result": "test"}
        mock_lmql.run.return_value = [mock_result]

        adapter = LMQLAdapter(model="test", backend="openai")
        result = await adapter.execute_query("test query")

        assert result == "{'result': 'test'}"
        mock_lmql.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_query_error(self, mock_lmql):
        """Тест обработки ошибки при выполнении запроса."""
        mock_lmql.run.side_effect = Exception("Test error")

        adapter = LMQLAdapter(model="test", backend="openai")
        
        # При ошибке должен быть выброшен RuntimeError
        with pytest.raises(RuntimeError, match="Ошибка выполнения LMQL запроса"):
            await adapter.execute_query("test query")

    @pytest.mark.asyncio
    async def test_execute_json_query_success(self, mock_lmql):
        """Тест успешного выполнения JSON запроса."""
        mock_result = MagicMock()
        mock_result.variables = '{"key": "value"}'
        mock_lmql.run.return_value = [mock_result]

        adapter = LMQLAdapter(model="test", backend="openai")
        result = await adapter.execute_json_query(
            prompt="test",
            json_schema='{"key": "[VALUE]"}',
        )

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_execute_json_query_invalid_json(self, mock_lmql):
        """Тест обработки невалидного JSON."""
        mock_result = MagicMock()
        mock_result.variables = "invalid json"
        mock_lmql.run.return_value = [mock_result]

        adapter = LMQLAdapter(model="test", backend="openai")
        
        # При невалидном JSON должен быть выброшен RuntimeError
        with pytest.raises(RuntimeError, match="Не удалось извлечь JSON"):
            await adapter.execute_json_query(
                prompt="test",
                json_schema='{"key": "[VALUE]"}',
            )

    @pytest.mark.asyncio
    async def test_execute_validation_query_success(self, mock_lmql):
        """Тест успешного выполнения валидационного запроса."""
        mock_result = MagicMock()
        mock_result.variables = "ДА"
        mock_lmql.run.return_value = [mock_result]

        adapter = LMQLAdapter(model="test", backend="openai")
        result = await adapter.execute_validation_query(
            prompt="test",
            valid_responses=["ДА", "НЕТ"],
        )

        assert result == "ДА"

    @pytest.mark.asyncio
    async def test_execute_validation_query_invalid_response(self, mock_lmql):
        """Тест обработки невалидного ответа валидации."""
        mock_result = MagicMock()
        mock_result.variables = "INVALID"
        mock_lmql.run.return_value = [mock_result]

        adapter = LMQLAdapter(model="test", backend="openai")
        
        # При невалидном ответе должен быть выброшен RuntimeError
        with pytest.raises(RuntimeError, match="Невалидный результат валидации"):
            await adapter.execute_validation_query(
                prompt="test",
                valid_responses=["ДА", "НЕТ"],
            )

    def test_extract_json_from_result(self, mock_lmql):
        """Тест извлечения JSON из результата."""
        adapter = LMQLAdapter(model="test", backend="openai")

        # Тест с чистым JSON
        result = adapter._extract_json_from_result('{"key": "value"}')
        assert result == '{"key": "value"}'

        # Тест с markdown code block
        result = adapter._extract_json_from_result('```json\n{"key": "value"}\n```')
        assert result == '{"key": "value"}'

        # Тест с текстом вокруг JSON
        result = adapter._extract_json_from_result('Some text {"key": "value"} more text')
        assert result == '{"key": "value"}'

    def test_extract_validation_result(self, mock_lmql):
        """Тест извлечения результата валидации."""
        adapter = LMQLAdapter(model="test", backend="openai")

        # Тест с точным совпадением
        result = adapter._extract_validation_result("ДА", ["ДА", "НЕТ"])
        assert result == "ДА"

        # Тест с совпадением в верхнем регистре
        result = adapter._extract_validation_result("да", ["ДА", "НЕТ"])
        assert result == "ДА"

        # Тест с невалидным ответом
        with pytest.raises(RuntimeError, match="Невалидный результат валидации"):
            adapter._extract_validation_result("INVALID", ["ДА", "НЕТ"])


class TestBuildLMQLAdapterFromEnv:
    """Тесты для функции build_lmql_adapter_from_env."""

    @patch("memory_mcp.core.lmql_adapter.get_settings")
    @patch("memory_mcp.core.lmql_adapter.LMQLAdapter")
    def test_build_with_lmql_enabled(self, mock_adapter_class, mock_get_settings):
        """Тест создания адаптера при включенном LMQL."""
        mock_settings = MagicMock()
        mock_settings.use_lmql = True
        mock_settings.lmql_model = "test-model"
        mock_settings.lmql_backend = "lmstudio"
        mock_settings.lmstudio_host = "localhost"
        mock_settings.lmstudio_port = 1234
        mock_get_settings.return_value = mock_settings

        mock_adapter_instance = MagicMock()
        mock_adapter_class.return_value = mock_adapter_instance

        result = build_lmql_adapter_from_env()

        assert result == mock_adapter_instance
        mock_adapter_class.assert_called_once_with(
            model="test-model",
            backend="lmstudio",
            base_url="http://localhost:1234",
        )

    @patch("memory_mcp.core.lmql_adapter.get_settings")
    def test_build_with_lmql_disabled(self, mock_get_settings):
        """Тест ошибки при отключенном LMQL."""
        mock_settings = MagicMock()
        mock_settings.use_lmql = False
        mock_get_settings.return_value = mock_settings

        with pytest.raises(RuntimeError, match="LMQL отключен в настройках"):
            build_lmql_adapter_from_env()

    @patch("memory_mcp.core.lmql_adapter.get_settings")
    def test_build_without_model(self, mock_get_settings):
        """Тест ошибки при отсутствии модели."""
        mock_settings = MagicMock()
        mock_settings.use_lmql = True
        mock_settings.lmql_model = None
        mock_settings.lmstudio_llm_model = None
        mock_get_settings.return_value = mock_settings

        with pytest.raises(RuntimeError, match="модель не указана"):
            build_lmql_adapter_from_env()

    @patch("memory_mcp.core.lmql_adapter.get_settings")
    @patch("memory_mcp.core.lmql_adapter.LMQLAdapter")
    def test_build_with_fallback_to_lmstudio_model(self, mock_adapter_class, mock_get_settings):
        """Тест использования lmstudio_llm_model при отсутствии lmql_model."""
        mock_settings = MagicMock()
        mock_settings.use_lmql = True
        mock_settings.lmql_model = None
        mock_settings.lmstudio_llm_model = "fallback-model"
        mock_settings.lmql_backend = "lmstudio"
        mock_settings.lmstudio_host = "localhost"
        mock_settings.lmstudio_port = 1234
        mock_get_settings.return_value = mock_settings

        mock_adapter_instance = MagicMock()
        mock_adapter_class.return_value = mock_adapter_instance

        result = build_lmql_adapter_from_env()

        assert result == mock_adapter_instance
        mock_adapter_class.assert_called_once_with(
            model="fallback-model",
            backend="lmstudio",
            base_url="http://localhost:1234",
        )

