"""Executor interface with basic implementation and error handling."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


class ExecutionError(Exception):
    """Raised when code execution fails."""


@dataclass
class ExecutionResult:
    """Outcome of executing user supplied code."""

    stdout: str
    stderr: str
    files: Dict[str, bytes]
    returncode: int = 0


@dataclass
class ToolPolicy:
    """Resource limits for tool execution."""

    max_wall_time_s: float | None = None
    max_mem_mb: int | None = None
    cpu_quota: float | None = None
    network_mode: str = "none"
    network_proxy: str | None = None
    userns: str | None = None
    cap_drop: str = "ALL"
    seccomp_profile: str = "security/seccomp_profile.json"
    apparmor_profile: str | None = "security/apparmor_profile"


class CodeExecutor(ABC):
    """Interface for executing code snippets."""

    @abstractmethod
    def execute(self, code: str, policy: ToolPolicy | None = None) -> ExecutionResult:
        """Execute ``code`` and return its result."""


class SimpleCodeExecutor(CodeExecutor):
    """Executes Python code using ``exec`` and captures errors."""

    def execute(
        self, code: str, policy: ToolPolicy | None = None
    ) -> ExecutionResult:  # pragma: no cover - simple wrapper
        local: Dict[str, Any] = {}
        try:
            exec(code, {}, local)
        except Exception as exc:  # noqa: BLE001 - broad for example purposes
            raise ExecutionError(str(exc)) from exc
        return ExecutionResult(str(local.get("result", "")), "", {}, 0)


class SubprocessCodeExecutor(CodeExecutor):
    """Execute code in a separate Python process.

    The subprocess is constrained by ``timeout`` seconds and ``memory_limit``
    bytes. Standard output and error are captured and returned.
    """

    def __init__(
        self,
        timeout: float = 1.0,
        memory_limit: int = 50_000_000,
        cpu_limit: float | None = None,
        use_container: bool = False,
        container_runtime: str = "docker",
    ) -> None:
        """Create an executor for running untrusted Python code.

        Parameters
        ----------
        timeout:
            Maximum wall-clock time in seconds for execution.
        memory_limit:
            Maximum memory usage in bytes.
        cpu_limit:
            Optional CPU quota used when running in a container.
        use_container:
            When ``True`` the code is executed inside a Docker/Podman
            container with network disabled.
        container_runtime:
            Binary name of the container engine to use (``docker`` or
            ``podman``).
        """

        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.use_container = use_container
        self.container_runtime = container_runtime

    def _limit_resources(self) -> None:
        """Apply resource limits in the child process."""
        try:
            import resource

            resource.setrlimit(
                resource.RLIMIT_AS, (self.memory_limit, self.memory_limit)
            )
            cpu_time = max(1, int(self.timeout))
            if self.cpu_limit is not None:
                cpu_time = min(cpu_time, int(self.cpu_limit))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_time, cpu_time))
        except Exception:  # pragma: no cover - best effort on non-Unix
            pass

    def execute(  # type: ignore[override]
        self,
        code: str,
        packages: List[str] | None = None,
        policy: ToolPolicy | None = None,
    ) -> ExecutionResult:
        """Run ``code`` in a restricted subprocess.

        Returns an :class:`ExecutionResult` containing ``stdout``,
        ``stderr`` and a mapping of created ``files``. Raises
        :class:`ExecutionError` if the code fails or exceeds the timeout.
        """

        original_timeout = self.timeout
        original_memory_limit = self.memory_limit
        original_cpu_limit = self.cpu_limit

        if policy:
            if policy.max_wall_time_s is not None:
                self.timeout = policy.max_wall_time_s
            if policy.max_mem_mb is not None:
                self.memory_limit = int(policy.max_mem_mb * 1024 * 1024)
            if policy.cpu_quota is not None:
                self.cpu_limit = policy.cpu_quota

        work_dir: Path | None = None
        try:
            # Safety checks are handled by container isolation; only syntax is validated
            try:
                import ast

                ast.parse(code)
            except SyntaxError as exc:
                raise ExecutionError(str(exc)) from exc

            work_dir, pkg_dir, env = self._prepare_environment(packages)
            if self.use_container:
                # В контейнере используем стандартный путь для пакетов
                container_prefix = "import sys; sys.path.insert(0, r'/workspace');"
                if packages:
                    container_prefix += "sys.path.insert(0, r'/tmp/packages');"
                code = container_prefix + code
                completed = self._run_in_container(code, env, work_dir, pkg_dir)
            else:
                # На хосте используем реальный путь к пакетам
                if packages:
                    code = f"import sys; sys.path.insert(0, r'{pkg_dir}');" + code
                completed = self._run_locally(code, env, work_dir)
            if completed.returncode != 0:
                raise ExecutionError(completed.stderr.strip())
            files = self._collect_outputs(work_dir)
            return ExecutionResult(
                completed.stdout,
                completed.stderr,
                files,
                completed.returncode,
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - rare
            raise ExecutionError("Execution timed out") from exc
        finally:
            if work_dir is not None:
                shutil.rmtree(work_dir.parent, ignore_errors=True)
            self.timeout = original_timeout
            self.memory_limit = original_memory_limit
            self.cpu_limit = original_cpu_limit

    def _prepare_environment(
        self, packages: List[str] | None
    ) -> tuple[Path, Path, dict[str, str]]:
        tmp_root = Path(tempfile.mkdtemp())
        work_dir = tmp_root / "work"
        pkg_dir = tmp_root / "packages"
        work_dir.mkdir()
        pkg_dir.mkdir()
        if packages:
            self._install_packages(packages, pkg_dir)
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{pkg_dir}{os.pathsep}" + env.get("PYTHONPATH", "")
        return work_dir, pkg_dir, env

    def _run_in_container(
        self, code: str, env: dict[str, str], work_dir: Path, pkg_dir: Path
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            self.container_runtime,
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            f"{self.memory_limit}b",
            "-v",
            f"{pkg_dir}:/tmp/packages",
            "-v",
            f"{work_dir}:/workspace",
            "--workdir",
            "/workspace",
            "-e",
            "PYTHONPATH=/tmp/packages",
        ]
        if self.cpu_limit is not None:
            cmd += ["--cpus", str(self.cpu_limit)]
        cmd += ["python:3.11-slim", "python", "-I", "-c", code]
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=env,
        )

    def _run_locally(
        self, code: str, env: dict[str, str], work_dir: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=str(work_dir),
            env=env,
            preexec_fn=self._limit_resources if hasattr(os, "fork") else None,
        )

    def _collect_outputs(self, work_dir: Path) -> Dict[str, bytes]:
        files: Dict[str, bytes] = {}
        for root, _, filenames in os.walk(work_dir):
            for name in filenames:
                path = Path(root) / name
                rel = path.relative_to(work_dir)
                files[str(rel)] = path.read_bytes()
        return files

    def _install_packages(self, packages: List[str], target: Path) -> None:
        """Install ``packages`` into ``target`` directory with limits."""

        try:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-input",
                "-q",
                "--target",
                str(target),
            ] + packages
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(target),
                preexec_fn=self._limit_resources if hasattr(os, "fork") else None,
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - rare
            raise ExecutionError("Package installation timed out") from exc

        if completed.returncode != 0:
            raise ExecutionError(completed.stderr.strip())
