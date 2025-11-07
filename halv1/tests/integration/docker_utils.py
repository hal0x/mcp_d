import subprocess
from pathlib import Path

import pytest


def check_docker_available() -> bool:
    """Return True if Docker engine is available on this host.

    Uses `docker info` which succeeds when the daemon is reachable.
    """
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def skip_if_docker_unavailable() -> None:
    """Skip the current test if Docker is not available."""
    if not check_docker_available():
        pytest.skip("Docker недоступен")


def create_test_docker_workdir() -> Path:
    """Create a temporary working directory for Docker-related tests.

    DockerExecutor uses `tmp_docker_work` by default, but tests may
    want their own workspace for additional artifacts.
    """
    work_dir = Path("tmp_docker_work_test")
    work_dir.mkdir(exist_ok=True)
    return work_dir

