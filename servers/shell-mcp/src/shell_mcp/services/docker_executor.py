from __future__ import annotations

import shlex
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..config import get_settings


class DockerUnavailableError(RuntimeError):
    """Raised when Docker is not installed or inaccessible."""


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    image: str
    command: list[str]
    network_enabled: bool
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "image": self.image,
            "command": self.command,
            "network_enabled": self.network_enabled,
            "artifacts": self.artifacts,
        }


_LANGUAGE_CONFIG: dict[str, dict[str, object]] = {
    "python": {"filename": "main.py", "command": ("python", "main.py")},
    "bash": {"filename": "script.sh", "command": ("bash", "script.sh")},
    "sh": {"filename": "script.sh", "command": ("sh", "script.sh")},
    "shell": {"filename": "script.sh", "command": ("sh", "script.sh")},
    "node": {
        "filename": "app.mjs",
        "command": ("node", "app.mjs"),
        "image": "node:20-slim",
    },
}
_DEFAULT_LANGUAGE: dict[str, object] = {
    "filename": "snippet.txt",
    "command": ("cat", "snippet.txt"),
}


def ensure_docker_available() -> None:
    try:
        subprocess.run(
            ["docker", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise DockerUnavailableError(
            "Docker is required but not available. Install Docker and ensure it is running."
        ) from exc


def resolve_language(language: str) -> dict[str, object]:
    lang_key = (language or "").strip().lower()
    if lang_key in _LANGUAGE_CONFIG:
        return _LANGUAGE_CONFIG[lang_key]
    return _DEFAULT_LANGUAGE


def prepare_command(
    command: str | None, filename: str, default_cmd: Sequence[str]
) -> list[str]:
    if command is None:
        return list(default_cmd)
    tokens = shlex.split(command)
    return [filename if token == "{script}" else token for token in tokens]


class DockerExecutor:
    def __init__(
        self,
        *,
        default_image: str | None = None,
        default_network: bool | None = None,
    ) -> None:
        s = get_settings()
        self.default_image = default_image or s.DEFAULT_IMAGE
        self.default_network = (
            s.DEFAULT_NETWORK if default_network is None else default_network
        )

    def ensure_image(self, image: str) -> None:
        s = get_settings()
        if s.PULL_POLICY == "always":
            subprocess.run(
                ["docker", "pull", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        inspect = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if inspect.returncode == 0:
            return
        subprocess.run(
            ["docker", "pull", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def run(
        self,
        *,
        code: str,
        language: str,
        image: str | None = None,
        command: str | None = None,
        network: bool | None = None,
        timeout: int = 120,
        env: dict[str, str] | None = None,
        memory: str | None = None,
        cpus: str | None = None,
        readonly_fs: bool | None = None,
        dependencies: list[str] | None = None,
        out_artifacts_path: str | None = None,
    ) -> ExecutionResult:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        ensure_docker_available()

        s = get_settings()
        lang_spec = resolve_language(language)
        filename = lang_spec["filename"]
        default_cmd = lang_spec["command"]
        lang_image = lang_spec.get("image") if isinstance(lang_spec, dict) else None
        image_name = image or (
            lang_image if isinstance(lang_image, str) else self.default_image
        )
        network_enabled = self.default_network if network is None else network

        self.ensure_image(image_name)

        container_cmd = prepare_command(command, filename, default_cmd)  # type: ignore[arg-type]

        env_values = env or {}

        # Используем домашнюю директорию для совместимости с macOS Docker

        home_dir = Path.home()
        temp_dir = home_dir / "mcp-temp"
        temp_dir.mkdir(exist_ok=True)

        # Создаем уникальную поддиректорию для этого выполнения
        import uuid

        unique_dir = temp_dir / f"exec-{uuid.uuid4().hex[:8]}"
        unique_dir.mkdir(exist_ok=True)

        try:
            host_dir = unique_dir
            artifacts_dir = host_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            script_path = host_dir / str(filename)
            script_path.write_text(code, encoding="utf-8")

            docker_cmd: list[str] = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{host_dir}:{s.CONTAINER_WORKDIR}",
                "--workdir",
                s.CONTAINER_WORKDIR,
            ]

            if not network_enabled:
                docker_cmd.extend(["--network", "none"])

            # Optional resource/security flags
            mem_eff = memory or s.MEMORY
            cpus_eff = cpus or s.CPUS
            readonly_eff = (
                s.READONLY_FS if readonly_fs is None else (s.READONLY_FS or readonly_fs)
            )

            if mem_eff:
                docker_cmd.extend(["--memory", mem_eff])
            if cpus_eff:
                docker_cmd.extend(["--cpus", cpus_eff])
            if readonly_eff:
                docker_cmd.append("--read-only")
            if s.USER_ID is not None and s.GROUP_ID is not None:
                docker_cmd.extend(["--user", f"{s.USER_ID}:{s.GROUP_ID}"])

            for key, value in env_values.items():
                docker_cmd.extend(["-e", f"{key}={value}"])

            docker_cmd.append(image_name)

            if dependencies:
                install_cmd = (
                    "pip install --disable-pip-version-check --root-user-action=ignore "
                    + " ".join(shlex.quote(dep) for dep in dependencies)
                )
                run_seq = " ".join(shlex.quote(part) for part in container_cmd)
                container_cmd = ["/bin/sh", "-lc", f"{install_cmd} && {run_seq}"]

            docker_cmd.extend(container_cmd)

            def _collect_and_export(destination: str | None) -> list[str]:
                results: list[str] = []
                if artifacts_dir.exists():
                    for artifact in artifacts_dir.rglob("*"):
                        if artifact.is_file():
                            results.append(str(artifact.relative_to(artifacts_dir)))
                if destination and results:
                    dest_root = Path(destination).expanduser()
                    if not dest_root.is_absolute():
                        dest_root = Path.cwd() / dest_root
                    dest_root.mkdir(parents=True, exist_ok=True)
                    for artifact in artifacts_dir.rglob("*"):
                        if artifact.is_file():
                            rel = artifact.relative_to(artifacts_dir)
                            target = dest_root / rel
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(artifact, target)
                return results

            try:
                process = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                artifacts = _collect_and_export(out_artifacts_path)
                return ExecutionResult(
                    stdout=str(process.stdout) if process.stdout is not None else "",
                    stderr=str(process.stderr) if process.stderr is not None else "",
                    exit_code=process.returncode,
                    timed_out=False,
                    image=image_name,
                    command=container_cmd,
                    network_enabled=network_enabled,
                    artifacts=artifacts,
                )
            except subprocess.TimeoutExpired as exc:
                artifacts = _collect_and_export(out_artifacts_path)
                return ExecutionResult(
                    stdout=str(exc.stdout) if exc.stdout is not None else "",
                    stderr=(
                        str(exc.stderr)
                        if exc.stderr is not None
                        else "" + "\nExecution timed out."
                    ),
                    exit_code=None,
                    timed_out=True,
                    image=image_name,
                    command=container_cmd,
                    network_enabled=network_enabled,
                    artifacts=artifacts,
                )
        finally:
            # Очищаем временную директорию
            if unique_dir.exists():
                shutil.rmtree(unique_dir, ignore_errors=True)
