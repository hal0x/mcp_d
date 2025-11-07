"""Factory for choosing the appropriate code executor."""

from __future__ import annotations

from .docker_executor import DockerExecutor
from .code_executor import SubprocessCodeExecutor, CodeExecutor
from .mcp_executor import MCPCodeExecutor
from .config_loader import config_loader


def create_executor(provider: str, venv_path: str = "venv") -> CodeExecutor:
    """Return an executor instance based on *provider*.

    Parameters
    ----------
    provider:
        Name of the executor provider. Supported values: "docker", "local", "mcp", "shell-mcp".
        Defaults to "docker".
    venv_path:
        Path to a virtual environment (kept for compatibility but not used).
    """
    prov = (provider or "docker").lower()
    if prov == "local":
        # Use a safe subprocess-based executor when local execution is requested
        return SubprocessCodeExecutor()
    if prov == "shell-mcp":
        from .mcp_shell_executor import MCPShellExecutor

        settings = config_loader.load_settings()
        executor_cfg = settings.get("executor", {})
        return MCPShellExecutor(config=executor_cfg)
    if prov == "mcp":
        settings = config_loader.load_settings()
        executor_cfg = settings.get("executor", {})
        mcp_cfg = executor_cfg.get("mcp", {})
        return MCPCodeExecutor(config=mcp_cfg)
    # Default to Docker for isolation with configuration
    config = config_loader.merge_configs()
    return DockerExecutor(config=config)


__all__ = ["create_executor"]
