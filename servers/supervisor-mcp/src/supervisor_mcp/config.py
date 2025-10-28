"""Configuration for supervisor MCP server."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Supervisor MCP server settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="SUPERVISOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Database settings
    db_url: str = "postgresql+asyncpg://supervisor:supervisor@localhost:5432/supervisor"
    db_pool_size: int | None = None
    db_max_overflow: int | None = None
    db_echo: bool = False
    
    # Redis settings
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl_seconds: int = 60
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Logging
    log_level: str = "INFO"
    
    # Health check settings
    health_check_interval: int = 30  # seconds
    health_check_timeout: float = 5.0  # seconds
    
    # Metrics settings
    metrics_retention_days: int = 30
    aggregation_windows: list[str] = ["7d", "30d"]
    aggregation_refresh_seconds: int = 300
    
    # Alert settings
    alert_cooldown_minutes: int = 5


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
