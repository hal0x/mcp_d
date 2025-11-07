import shutil
from pathlib import Path

import pytest

from executor import create_executor
from .docker_utils import check_docker_available, create_test_docker_workdir


@pytest.fixture
def docker_executor():
    """Provide a shell-mcp backed executor or skip when Docker is unavailable."""
    if not check_docker_available():
        pytest.skip("Docker недоступен")
    return create_executor("shell-mcp", "venv")


@pytest.fixture
def docker_workdir():
    """Provide and cleanup a temporary working directory for Docker tests."""
    work_dir = create_test_docker_workdir()
    try:
        yield work_dir
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)
