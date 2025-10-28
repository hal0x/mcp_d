"""Configuration for Policy MCP."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for Policy MCP."""

    model_config = SettingsConfigDict(
        env_prefix="POLICY_", env_file=".env", env_file_encoding="utf-8"
    )

    # Database
    DB_URL: str = "postgresql+asyncpg://policy:policy@localhost:5432/policy"
    DB_POOL_SIZE: int | None = Field(default=None, ge=1, description="Primary connection pool size")
    DB_MAX_OVERFLOW: int | None = Field(default=None, ge=0, description="Max overflow for pool connections")
    DB_ECHO: bool = Field(default=False, description="Enable SQL echo for debugging")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Structlog log level")


def get_settings() -> Settings:
    """Get configuration settings."""
    return Settings()
