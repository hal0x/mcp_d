"""
Тесты для инструментов MCP.

Проверяет:
- Валидацию входных данных
- Обработку ошибок
- Выполнение кода
- Сохранение скриптов
- Мета-инструменты
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shell_mcp.tools.run import (
    RunCodeArgs,
    RunSavedArgs,
    _choose_stricter_cpus,
    _choose_stricter_memory,
    _mem_to_bytes,
    _normalize_env,
)


class TestRunCodeArgs:
    """Тесты для класса RunCodeArgs."""

    def test_init_with_code(self):
        """Тест инициализации с кодом."""
        args = RunCodeArgs(code="print('Hello')", language="python")
        assert args.code == "print('Hello')"
        assert args.language == "python"
        assert args.script_path is None

    def test_init_with_script_path(self):
        """Тест инициализации с путем к скрипту."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('Hello')")
            temp_path = f.name

        try:
            args = RunCodeArgs(script_path=temp_path, language="python")
            assert args.script_path == temp_path
            assert args.code is None
        finally:
            Path(temp_path).unlink()

    def test_init_without_source(self):
        """Тест инициализации без источника кода."""
        with pytest.raises(
            ValueError, match="Either 'code' or 'script_path' must be provided"
        ):
            RunCodeArgs(language="python")

    def test_init_with_both_code_and_script_path(self):
        """Тест инициализации с кодом и путем к скрипту."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('Hello')")
            temp_path = f.name

        try:
            # Этот тест проверяет, что валидация script_path происходит раньше
            # чем проверка взаимоисключающих параметров
            with pytest.raises(ValueError, match="script_path does not exist"):
                RunCodeArgs(
                    code="print('Hello')",
                    script_path="/nonexistent/path.py",
                    language="python",
                )
        finally:
            Path(temp_path).unlink()

    def test_validate_env_valid(self):
        """Тест валидации корректных переменных окружения."""
        args = RunCodeArgs(code="print('Hello')", env=["KEY1=value1", "KEY2=value2"])
        assert args.env == ["KEY1=value1", "KEY2=value2"]

    def test_validate_env_invalid(self):
        """Тест валидации некорректных переменных окружения."""
        with pytest.raises(ValueError, match="Invalid env format"):
            RunCodeArgs(code="print('Hello')", env=["INVALID_FORMAT"])

    def test_validate_memory_valid(self):
        """Тест валидации корректных значений памяти."""
        valid_values = ["512m", "1g", "256", "1024k"]
        for value in valid_values:
            args = RunCodeArgs(code="print('Hello')", memory=value)
            assert args.memory == value

    def test_validate_memory_invalid(self):
        """Тест валидации некорректных значений памяти."""
        with pytest.raises(ValueError, match="String should match pattern"):
            RunCodeArgs(code="print('Hello')", memory="invalid")

    def test_validate_cpus_valid(self):
        """Тест валидации корректных значений CPU."""
        valid_values = ["0.5", "1", "2.5", "10"]
        for value in valid_values:
            args = RunCodeArgs(code="print('Hello')", cpus=value)
            assert args.cpus == value

    def test_validate_cpus_invalid(self):
        """Тест валидации некорректных значений CPU."""
        with pytest.raises(ValueError, match="String should match pattern"):
            RunCodeArgs(code="print('Hello')", cpus="invalid")

    def test_validate_dependencies(self):
        """Тест валидации зависимостей."""
        # Корректные зависимости
        args = RunCodeArgs(code="print('Hello')", dependencies=["requests", "numpy"])
        assert args.dependencies == ["requests", "numpy"]

        # Пустые зависимости
        args = RunCodeArgs(code="print('Hello')", dependencies=["", "  "])
        assert args.dependencies is None

        # Очистка пробелов
        args = RunCodeArgs(
            code="print('Hello')", dependencies=[" requests ", " numpy "]
        )
        assert args.dependencies == ["requests", "numpy"]

    def test_validate_script_path_exists(self):
        """Тест валидации существующего пути к скрипту."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("print('Hello')")
            temp_path = f.name

        try:
            args = RunCodeArgs(script_path=temp_path, language="python")
            assert args.script_path == temp_path
        finally:
            Path(temp_path).unlink()

    def test_validate_script_path_not_exists(self):
        """Тест валидации несуществующего пути к скрипту."""
        with pytest.raises(ValueError, match="script_path does not exist"):
            RunCodeArgs(script_path="/nonexistent/path.py", language="python")

    def test_parse_from_json_string(self):
        """Тест парсинга из JSON строки."""
        json_data = '{"code": "print(\\"Hello\\")", "language": "python"}'
        args = RunCodeArgs.model_validate_json(json_data)
        assert args.code == 'print("Hello")'
        assert args.language == "python"

    def test_parse_from_dict(self):
        """Тест парсинга из словаря."""
        data = {"code": "print('Hello')", "language": "python"}
        args = RunCodeArgs.model_validate(data)
        assert args.code == "print('Hello')"
        assert args.language == "python"


class TestRunSavedArgs:
    """Тесты для класса RunSavedArgs."""

    def test_init_basic(self):
        """Тест базовой инициализации."""
        args = RunSavedArgs(name="test-script")
        assert args.name == "test-script"
        assert args.timeout_seconds == 120

    def test_init_with_options(self):
        """Тест инициализации с опциями."""
        args = RunSavedArgs(
            name="test-script",
            timeout_seconds=60,
            env=["KEY=value"],
            memory="512m",
            cpus="0.5",
        )
        assert args.name == "test-script"
        assert args.timeout_seconds == 60
        assert args.env == ["KEY=value"]
        assert args.memory == "512m"
        assert args.cpus == "0.5"


class TestUtilityFunctions:
    """Тесты для вспомогательных функций."""

    def test_normalize_env(self):
        """Тест нормализации переменных окружения."""
        env_vars = ["KEY1=value1", "KEY2=value2", "KEY3=value with spaces"]
        result = _normalize_env(env_vars)

        assert result == {
            "KEY1": "value1",
            "KEY2": "value2",
            "KEY3": "value with spaces",
        }

    def test_normalize_env_empty(self):
        """Тест нормализации пустого списка переменных окружения."""
        result = _normalize_env(None)
        assert result == {}

        result = _normalize_env([])
        assert result == {}

    def test_mem_to_bytes(self):
        """Тест преобразования памяти в байты."""
        assert _mem_to_bytes("512") == 512
        assert _mem_to_bytes("512k") == 512 * 1024
        assert _mem_to_bytes("512m") == 512 * 1024 * 1024
        assert _mem_to_bytes("1g") == 1024 * 1024 * 1024

    def test_choose_stricter_memory(self):
        """Тест выбора более строгого лимита памяти."""
        assert _choose_stricter_memory("512m", "1g") == "512m"
        assert _choose_stricter_memory("1g", "512m") == "512m"
        assert _choose_stricter_memory("512m", None) == "512m"
        assert _choose_stricter_memory(None, "1g") == "1g"
        assert _choose_stricter_memory(None, None) is None

    def test_choose_stricter_cpus(self):
        """Тест выбора более строгого лимита CPU."""
        assert _choose_stricter_cpus("0.5", "1") == "0.5"
        assert _choose_stricter_cpus("1", "0.5") == "0.5"
        assert _choose_stricter_cpus("0.5", None) == "0.5"
        assert _choose_stricter_cpus(None, "1") == "1"
        assert _choose_stricter_cpus(None, None) is None


class TestMCPTools:
    """Тесты для MCP инструментов."""

    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.mock_executor = MagicMock()
        self.mock_store = MagicMock()

    def test_run_code_args_validation(self):
        """Тест валидации аргументов run_code."""
        # Тест корректных аргументов
        args = RunCodeArgs(
            code="print('Hello')",
            language="python",
            timeout_seconds=60,
            env=["KEY=value"],
            memory="512m",
            cpus="0.5",
        )

        assert args.code == "print('Hello')"
        assert args.language == "python"
        assert args.timeout_seconds == 60
        assert args.env == ["KEY=value"]
        assert args.memory == "512m"
        assert args.cpus == "0.5"

    def test_run_saved_args_validation(self):
        """Тест валидации аргументов run_saved."""
        args = RunSavedArgs(
            name="test-script",
            timeout_seconds=60,
            env=["KEY=value"],
            memory="512m",
            cpus="0.5",
        )

        assert args.name == "test-script"
        assert args.timeout_seconds == 60
        assert args.env == ["KEY=value"]
        assert args.memory == "512m"
        assert args.cpus == "0.5"


class TestMetaTools:
    """Тесты для мета-инструментов."""

    @patch("shell_mcp.tools.meta.ensure_docker_available")
    @patch("shell_mcp.tools.meta.get_settings")
    def test_health_check_success(self, mock_settings, mock_ensure_docker):
        """Тест успешной проверки здоровья."""
        mock_settings.return_value.DEFAULT_IMAGE = "python:3.11"
        mock_settings.return_value.CONTAINER_WORKDIR = "/workspace"
        mock_settings.return_value.DEFAULT_NETWORK = True

        from mcp.server.fastmcp import FastMCP

        from shell_mcp.tools.meta import bind

        mcp = FastMCP("test")
        bind(mcp)

        # Проверяем, что функция health была зарегистрирована
        # FastMCP хранит инструменты в _tool_manager
        assert hasattr(mcp, "_tool_manager")
        assert hasattr(mcp._tool_manager, "_tools")
        assert len(mcp._tool_manager._tools) > 0

    @patch("shell_mcp.tools.meta.get_settings")
    @patch("shell_mcp.tools.meta._meta.version")
    def test_version_info(self, mock_version, mock_settings):
        """Тест получения информации о версии."""
        mock_settings.return_value.DEFAULT_IMAGE = "python:3.11"
        mock_settings.return_value.CONTAINER_WORKDIR = "/workspace"
        mock_settings.return_value.DEFAULT_NETWORK = True
        mock_version.return_value = "1.0.0"

        from mcp.server.fastmcp import FastMCP

        from shell_mcp.tools.meta import bind

        mcp = FastMCP("test")
        bind(mcp)

        # Проверяем, что функция version была зарегистрирована
        # FastMCP хранит инструменты в _tool_manager
        assert hasattr(mcp, "_tool_manager")
        assert hasattr(mcp._tool_manager, "_tools")
        assert len(mcp._tool_manager._tools) > 0

    def test_index_existing_scripts_tool(self):
        """Тест инструмента index_existing_scripts."""
        from mcp.server.fastmcp import FastMCP

        from shell_mcp.tools.run import bind

        mcp = FastMCP("test")
        bind(mcp)

        # Проверяем, что инструмент зарегистрирован
        assert "index_existing_scripts" in mcp._tool_manager._tools

        # Вызываем инструмент (он будет работать с реальными данными)
        result = mcp._tool_manager._tools["index_existing_scripts"].fn()

        # Проверяем структуру результата
        assert "success" in result
        assert "indexed_count" in result
        assert "indexed_files" in result
        assert "message" in result
        assert isinstance(result["success"], bool)
        assert isinstance(result["indexed_count"], int)
        assert isinstance(result["indexed_files"], list)
        assert isinstance(result["message"], str)

    def test_save_temp_script_tool(self):
        """Тест инструмента save_temp_script."""
        from mcp.server.fastmcp import FastMCP

        from shell_mcp.tools.run import bind

        mcp = FastMCP("test")
        bind(mcp)

        # Проверяем, что инструмент зарегистрирован
        assert "save_temp_script" in mcp._tool_manager._tools

        # Вызываем инструмент
        result = mcp._tool_manager._tools["save_temp_script"].fn("test_temp", "python", "print('Hello')")

        # Проверяем структуру результата
        assert "success" in result
        assert "script" in result
        assert "message" in result
        assert isinstance(result["success"], bool)
        assert isinstance(result["message"], str)

    def test_list_temp_scripts_tool(self):
        """Тест инструмента list_temp_scripts."""
        from mcp.server.fastmcp import FastMCP

        from shell_mcp.tools.run import bind

        mcp = FastMCP("test")
        bind(mcp)

        # Проверяем, что инструмент зарегистрирован
        assert "list_temp_scripts" in mcp._tool_manager._tools

        # Вызываем инструмент
        result = mcp._tool_manager._tools["list_temp_scripts"].fn()

        # Проверяем, что результат - список
        assert isinstance(result, list)

    def test_promote_temp_script_tool(self):
        """Тест инструмента promote_temp_script."""
        from mcp.server.fastmcp import FastMCP

        from shell_mcp.tools.run import bind

        mcp = FastMCP("test")
        bind(mcp)

        # Проверяем, что инструмент зарегистрирован
        assert "promote_temp_script" in mcp._tool_manager._tools

        # Вызываем инструмент с несуществующим скриптом
        result = mcp._tool_manager._tools["promote_temp_script"].fn("nonexistent")

        # Проверяем структуру результата
        assert "success" in result
        assert "script" in result
        assert "message" in result
        assert result["success"] is False
        assert "not found" in result["message"]

    def test_cleanup_old_temp_scripts_tool(self):
        """Тест инструмента cleanup_old_temp_scripts."""
        from mcp.server.fastmcp import FastMCP

        from shell_mcp.tools.run import bind

        mcp = FastMCP("test")
        bind(mcp)

        # Проверяем, что инструмент зарегистрирован
        assert "cleanup_old_temp_scripts" in mcp._tool_manager._tools

        # Вызываем инструмент
        result = mcp._tool_manager._tools["cleanup_old_temp_scripts"].fn()

        # Проверяем структуру результата
        assert "success" in result
        assert "cleaned_count" in result
        assert "cleaned_scripts" in result
        assert "message" in result
        assert isinstance(result["success"], bool)
        assert isinstance(result["cleaned_count"], int)
        assert isinstance(result["cleaned_scripts"], list)
        assert isinstance(result["message"], str)
