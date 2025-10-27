"""Configuration for Policy MCP."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for Policy MCP."""

    model_config = SettingsConfigDict(
        env_prefix="POLICY_", env_file=".env", env_file_encoding="utf-8"
    )

    # Database
    DB_URL: str = "postgresql+asyncpg://policy:policy@localhost:5432/policy"
    
    # Logging
    LOG_LEVEL: str = "INFO"


def get_settings() -> Settings:
    """Get configuration settings."""
    return Settings()

