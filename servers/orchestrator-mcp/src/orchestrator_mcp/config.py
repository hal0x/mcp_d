"""Configuration for Orchestrator MCP."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for orchestrator."""

    model_config = SettingsConfigDict(
        env_prefix="ORCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    supervisor_url: str = "http://localhost:8001"
    policy_url: str = "http://localhost:8002"
    learning_url: str = "http://localhost:8003"
    default_timeout: float = 30.0
    log_level: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
