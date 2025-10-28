"""Configuration helpers for TradingView MCP server."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

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
    }

    for target, candidates in alias_map.items():
        if os.getenv(target) is not None:
            continue
        for legacy in candidates:
            value = os.getenv(legacy)
            if value is not None:
                os.environ[target] = value
                break
