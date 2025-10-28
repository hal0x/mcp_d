"""
Тесты для DockerExecutor.

Проверяет:
- Создание и управление контейнерами
- Выполнение кода
- Очистка ресурсов
- Обработка ошибок
- Ограничения ресурсов
"""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shell_mcp.services.docker_executor import (
    DockerExecutor,
    DockerUnavailableError,
    ExecutionResult,
    ensure_docker_available,
    prepare_command,
    resolve_language,
)


class TestDockerExecutor:
    """Тесты для класса DockerExecutor."""

    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.executor = DockerExecutor()

    def test_init_default_values(self):
        """Тест инициализации с значениями по умолчанию."""
        executor = DockerExecutor()
        assert executor.default_image == "python:3.11"
        assert executor.default_network is True

    def test_init_custom_values(self):
        """Тест инициализации с пользовательскими значениями."""
        executor = DockerExecutor(default_image="node:20-slim", default_network=False)
        assert executor.default_image == "node:20-slim"
        assert executor.default_network is False

    @patch("shell_mcp.services.docker_executor.subprocess.run")
    def test_ensure_image_pull_policy_always(self, mock_run):
        """Тест принудительного обновления образа."""
        mock_run.return_value = MagicMock()

        with patch("shell_mcp.services.docker_executor.get_settings") as mock_settings:
            mock_settings.return_value.MEMORY = None
            mock_settings.return_value.CPUS = None
            mock_settings.return_value.READONLY_FS = False
            mock_settings.return_value.USER_ID = None
            mock_settings.return_value.GROUP_ID = None
            mock_settings.return_value.PULL_POLICY = "always"
            mock_settings.return_value.CONTAINER_WORKDIR = "/workspace"

            executor = DockerExecutor()
            executor.ensure_image("python:3.11")

            # Проверяем, что был вызван docker pull
            mock_run.assert_called_with(
                ["docker", "pull", "python:3.11"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    @patch("shell_mcp.services.docker_executor.subprocess.run")
    def test_ensure_image_pull_policy_if_not_present(self, mock_run):
        """Тест обновления образа только если отсутствует."""
        # Мокаем inspect - образ не найден
        mock_inspect = MagicMock()
        mock_inspect.returncode = 1
        mock_run.return_value = mock_inspect

        with patch("shell_mcp.services.docker_executor.get_settings") as mock_settings:
            mock_settings.return_value.PULL_POLICY = "if-not-present"

            executor = DockerExecutor()
            executor.ensure_image("python:3.11")

            # Проверяем, что был вызван docker pull
            mock_run.assert_called_with(
                ["docker", "pull", "python:3.11"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


class TestUtilityFunctions:
    """Тесты для вспомогательных функций."""

    def test_resolve_language_python(self):
        """Тест определения языка Python."""
        result = resolve_language("python")
        assert result["filename"] == "main.py"
        assert result["command"] == ("python", "main.py")

    def test_resolve_language_bash(self):
        """Тест определения языка Bash."""
        result = resolve_language("bash")
        assert result["filename"] == "script.sh"
        assert result["command"] == ("bash", "script.sh")

    def test_resolve_language_node(self):
        """Тест определения языка Node.js."""
        result = resolve_language("node")
        assert result["filename"] == "app.mjs"
        assert result["command"] == ("node", "app.mjs")
        assert result["image"] == "node:20-slim"

    def test_resolve_language_unknown(self):
        """Тест определения неизвестного языка."""
        result = resolve_language("unknown")
        assert result["filename"] == "snippet.txt"
        assert result["command"] == ("cat", "snippet.txt")

    def test_prepare_command_default(self):
        """Тест подготовки команды по умолчанию."""
        command = prepare_command(None, "script.py", ("python", "script.py"))
        assert command == ["python", "script.py"]

    def test_prepare_command_custom(self):
        """Тест подготовки пользовательской команды."""
        command = prepare_command(
            "python -m pytest {script}", "test.py", ("python", "test.py")
        )
        assert command == ["python", "-m", "pytest", "test.py"]

    def test_prepare_command_with_script_placeholder(self):
        """Тест подготовки команды с плейсхолдером скрипта."""
        command = prepare_command(
            "bash {script} --verbose", "script.sh", ("bash", "script.sh")
        )
        assert command == ["bash", "script.sh", "--verbose"]


class TestDockerAvailability:
    """Тесты проверки доступности Docker."""

    @patch("shell_mcp.services.docker_executor.subprocess.run")
    def test_ensure_docker_available_success(self, mock_run):
        """Тест успешной проверки доступности Docker."""
        mock_run.return_value = MagicMock()

        # Не должно вызывать исключение
        ensure_docker_available()

        mock_run.assert_called_with(
            ["docker", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    @patch("shell_mcp.services.docker_executor.subprocess.run")
    def test_ensure_docker_available_failure(self, mock_run):
        """Тест неудачной проверки доступности Docker."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker")

        with pytest.raises(DockerUnavailableError):
            ensure_docker_available()

    @patch("shell_mcp.services.docker_executor.subprocess.run")
    def test_ensure_docker_available_not_found(self, mock_run):
        """Тест когда Docker не найден."""
        mock_run.side_effect = FileNotFoundError()

        with pytest.raises(DockerUnavailableError):
            ensure_docker_available()


class TestExecutionResult:
    """Тесты для класса ExecutionResult."""

    def test_execution_result_creation(self):
        """Тест создания результата выполнения."""
        result = ExecutionResult(
            stdout="Hello World",
            stderr="",
            exit_code=0,
            timed_out=False,
            image="python:3.11",
            command=["python", "main.py"],
            network_enabled=True,
            artifacts=["output.txt"],
        )

        assert result.stdout == "Hello World"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.timed_out is False
        assert result.image == "python:3.11"
        assert result.command == ["python", "main.py"]
        assert result.network_enabled is True
        assert result.artifacts == ["output.txt"]

    def test_execution_result_to_dict(self):
        """Тест преобразования результата в словарь."""
        result = ExecutionResult(
            stdout="Hello World",
            stderr="",
            exit_code=0,
            timed_out=False,
            image="python:3.11",
            command=["python", "main.py"],
            network_enabled=True,
            artifacts=["output.txt"],
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["stdout"] == "Hello World"
        assert result_dict["stderr"] == ""
        assert result_dict["exit_code"] == 0
        assert result_dict["timed_out"] is False
        assert result_dict["image"] == "python:3.11"
        assert result_dict["command"] == ["python", "main.py"]
        assert result_dict["network_enabled"] is True
        assert result_dict["artifacts"] == ["output.txt"]


# Интеграционные тесты (требуют Docker)
class TestDockerExecutorIntegration:
    """Интеграционные тесты DockerExecutor (требуют Docker)."""

    @pytest.mark.integration
    def test_run_simple_python_code(self):
        """Тест выполнения простого Python кода."""
        executor = DockerExecutor()

        result = executor.run(
            code="print('Hello from Docker!')", language="python", timeout=30
        )

        assert result.exit_code == 0
        assert "Hello from Docker!" in result.stdout
        assert result.timed_out is False
        assert result.image == "python:3.11"

    @pytest.mark.integration
    def test_run_with_dependencies(self):
        """Тест выполнения кода с зависимостями."""
        executor = DockerExecutor()

        result = executor.run(
            code="import requests; print('requests imported successfully')",
            language="python",
            dependencies=["requests"],
            timeout=60,
        )

        assert result.exit_code == 0
        assert "requests imported successfully" in result.stdout

    @pytest.mark.integration
    def test_run_with_artifacts(self):
        """Тест выполнения кода с созданием артефактов."""
        executor = DockerExecutor()

        code = """
import os
os.makedirs('artifacts', exist_ok=True)
with open('artifacts/test.txt', 'w') as f:
    f.write('Test artifact')
print('Artifact created')
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            result = executor.run(
                code=code, language="python", out_artifacts_path=temp_dir, timeout=30
            )

            assert result.exit_code == 0
            assert "Artifact created" in result.stdout
            assert len(result.artifacts) > 0

            # Проверяем, что артефакт был скопирован
            artifact_path = Path(temp_dir) / "test.txt"
            assert artifact_path.exists()
            assert artifact_path.read_text() == "Test artifact"

    @pytest.mark.integration
    def test_run_timeout(self):
        """Тест таймаута выполнения."""
        executor = DockerExecutor()

        result = executor.run(
            code="import time; time.sleep(5)", language="python", timeout=2
        )

        assert result.timed_out is True
        assert result.exit_code is None
        assert "timed out" in result.stderr.lower()

    @pytest.mark.integration
    def test_run_with_network_disabled(self):
        """Тест выполнения с отключенной сетью."""
        executor = DockerExecutor(default_network=False)

        result = executor.run(
            code="import urllib.request; urllib.request.urlopen('http://example.com')",
            language="python",
            timeout=10,
        )

        # Должен завершиться с ошибкой из-за отсутствия сети
        assert result.exit_code != 0
        assert result.network_enabled is False
