from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from executor import create_executor
from executor.docker_executor import DockerExecutor
from executor.mcp_shell_executor import MCPShellExecutor


def test_create_executor_always_returns_docker() -> None:
    """Тест: factory всегда возвращает Docker executor для безопасности."""
    executor = create_executor("unknown")
    assert isinstance(executor, DockerExecutor)


def test_create_executor_docker() -> None:
    """Тест: factory возвращает Docker executor."""
    executor = create_executor("docker")
    assert isinstance(executor, DockerExecutor)


def test_create_executor_shell_mcp() -> None:
    """Тест: factory поддерживает shell-mcp провайдера."""
    executor = create_executor("shell-mcp")
    assert isinstance(executor, MCPShellExecutor)


def test_create_executor_with_venv_path() -> None:
    """Тест: factory принимает venv_path для совместимости."""
    executor = create_executor("docker", venv_path="test_venv")
    assert isinstance(executor, DockerExecutor)
