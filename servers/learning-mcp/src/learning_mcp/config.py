"""Configuration for learning MCP server."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Learning MCP server settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="LEARNING_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Supervisor MCP integration
    supervisor_url: str = "http://localhost:8001"
    supervisor_timeout: float = 30.0
    
    # Policy MCP integration
    policy_url: str = "http://localhost:8002"
    policy_timeout: float = 10.0
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8003
    
    # Logging
    log_level: str = "INFO"
    
    # Learning parameters
    min_samples: int = 100  # Минимум образцов для обучения
    confidence_threshold: float = 0.7  # Порог уверенности
    validation_split: float = 0.2  # Доля данных для валидации
    
    # Training windows
    default_window: str = "7d"
    available_windows: list[str] = ["7d", "30d"]
    online_learning_window: str = "1d"
    online_learning_interval_seconds: int = 900
    ab_test_threshold: float = 0.01

    # Optimization targets
    default_metric: str = "success_rate"
    available_metrics: list[str] = [
        "success_rate",
        "avg_execution_time",
        "error_rate",
        "resource_efficiency"
    ]
    
    # Pattern detection
    pattern_min_frequency: int = 5  # Минимальная частота паттерна
    pattern_confidence_threshold: float = 0.6


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
