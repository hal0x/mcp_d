"""Configuration helpers for TradingView MCP server."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings sourced from environment variables."""

    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    log_level: str = Field(default="INFO", description="Logging level")
    default_transport: Literal["stdio", "streamable-http"] = Field(
        default="streamable-http", description="Default transport protocol"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    redis_enabled: bool = Field(default=True, description="Enable Redis caching layer")
    redis_url: Optional[str] = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    redis_socket_timeout: float = Field(default=0.5, ge=0.0, le=5.0, description="Redis socket timeout in seconds")
    cache_ttl_market_seconds: int = Field(default=60, ge=1, le=900, description="TTL for market data cache")
    cache_ttl_metadata_seconds: int = Field(default=300, ge=30, le=3600, description="TTL for metadata cache")
    tv_requests_per_min: int = Field(default=90, ge=1, le=600, description="TradingView requests per minute")
    tv_burst_limit: int = Field(default=20, ge=1, le=200, description="Maximum burst size for TradingView requests")
    supervisor_metrics_enabled: bool = Field(default=False, description="Enable Supervisor MCP telemetry")
    supervisor_url: Optional[str] = Field(default=None, description="Supervisor MCP ingestion endpoint")
    supervisor_timeout: Optional[float] = Field(default=2.5, ge=0.1, le=10.0, description="Supervisor request timeout in seconds")
    supervisor_actor: Optional[str] = Field(default="tradingview-mcp", description="Actor name for Supervisor facts")

    model_config = SettingsConfigDict(
        env_prefix="TRADINGVIEW_MCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("log_level", mode="before")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        level = v.upper()
        if level not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return level

    @field_validator("default_transport", mode="before")
    def validate_transport(cls, v):
        """Validate transport protocol."""
        if v not in {"stdio", "streamable-http"}:
            raise ValueError(f"Invalid transport: {v}. Must be 'stdio' or 'streamable-http'")
        return v


@lru_cache
def get_settings() -> Settings:
    """Load settings from environment variables."""
    _ensure_env_aliases()
    return Settings()


__all__ = ["Settings", "get_settings"]


def _ensure_env_aliases() -> None:
    """Populate new-style variables from legacy names when not explicitly provided."""
    alias_map = {
        "TRADINGVIEW_MCP_HOST": ["HOST"],
        "TRADINGVIEW_MCP_PORT": ["PORT"],
        "TRADINGVIEW_MCP_LOG_LEVEL": ["LOG_LEVEL"],
        "TRADINGVIEW_MCP_DEFAULT_TRANSPORT": ["DEFAULT_TRANSPORT"],
        "TRADINGVIEW_MCP_REDIS_URL": ["REDIS_URL", "CACHE_URL"],
        "TRADINGVIEW_MCP_SUPERVISOR_URL": ["SUPERVISOR_URL"],
    }

    for target, candidates in alias_map.items():
        if os.getenv(target) is not None:
            continue
        for legacy in candidates:
            value = os.getenv(legacy)
            if value is not None:
                os.environ[target] = value
                break
