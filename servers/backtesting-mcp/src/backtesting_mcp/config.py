"""Configuration helpers for the backtesting MCP server."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BacktestingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BACKTEST_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )
    # Server settings
    host: str = Field(default="0.0.0.0", description="Host to bind the server to")
    port: int = Field(default=8082, description="Port to bind the server to")
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Data settings
    data_root: Path = Field(
        default=Path("data"), description="Root directory for cached market data"
    )
    default_timeframe: str = Field(default="1h")
    preferred_source: str = Field(
        default="binance",
        description="Preferred market data source: binance|tradingview|both",
    )
    binance_url: str = Field(
        default="http://localhost:8000", description="URL of binance-mcp server"
    )
    tradingview_url: str = Field(
        default="http://localhost:8060", description="URL of tradingview-mcp server"
    )
    request_timeout: float = Field(
        default=15.0, description="HTTP timeout for upstream MCP requests"
    )
    optuna_db_host: str = Field(
        default="postgres",
        description="Hostname for Optuna PostgreSQL storage",
    )
    optuna_db_port: int = Field(
        default=5432,
        description="Port for Optuna PostgreSQL storage",
    )
    optuna_db_name: str = Field(
        default="tradingview_scanners",
        description="Database name for Optuna studies",
    )
    optuna_db_user: str = Field(
        default="tradingview",
        description="Database user for Optuna PostgreSQL storage",
    )
    optuna_db_password: str = Field(
        default="tradingview",
        description="Database password for Optuna PostgreSQL storage",
    )
    optuna_n_jobs: int = Field(
        default=1,
        description="Parallel optimization jobs for Optuna",
    )
    optuna_sampler: str = Field(
        default="TPE",
        description="Default Optuna sampler: TPE|Random|Grid|CmaEs",
    )
    optuna_pruner: str = Field(
        default="median",
        description="Default Optuna pruner: median|percentile|none",
    )

    def get_optuna_storage_url(self) -> str:
        """Construct Optuna storage URL for PostgreSQL backend."""
        return (
            f"postgresql://{self.optuna_db_user}:{self.optuna_db_password}"
            f"@{self.optuna_db_host}:{self.optuna_db_port}/{self.optuna_db_name}"
        )


def get_settings() -> BacktestingSettings:
    return BacktestingSettings()


__all__ = ["BacktestingSettings", "get_settings"]
