from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SHELL_MCP_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8070
    log_level: str = "INFO"
    debug: bool = False

    DEFAULT_IMAGE: str = "python:3.11"
    CONTAINER_WORKDIR: str = "/workspace"
    DEFAULT_NETWORK: bool = True  # сеть включена по умолчанию

    # Optional resource/security toggles (not enforced by default)
    MEMORY: str | None = None  # e.g., "512m"
    CPUS: str | None = None  # e.g., "0.5" or "1"
    READONLY_FS: bool = False
    USER_ID: int | None = None
    GROUP_ID: int | None = None
    PULL_POLICY: str = "if-not-present"  # or "always"
    MAX_CONCURRENCY: int | None = (
        3  # limit parallel run_code_simple executions (default 3)
    )
    SAVED_SCRIPTS_DIR: str = "/scripts"
    TEMP_SCRIPTS_DIR: str = "/scripts/tmp"
    TEMP_SCRIPT_MAX_AGE_DAYS: int = 3  # максимальный возраст временных скриптов в днях


def get_settings() -> Settings:
    # Back-compat with old env names if provided
    import os

    default_image = os.environ.get("MCP_DOCKER_IMAGE")
    if default_image:
        os.environ.setdefault("SHELL_MCP_DEFAULT_IMAGE", default_image)
    workdir = os.environ.get("MCP_CONTAINER_WORKDIR")
    if workdir:
        os.environ.setdefault("SHELL_MCP_CONTAINER_WORKDIR", workdir)

    return Settings()
