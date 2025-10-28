"""Конфигурация MCP сервера TG Dump на базе Pydantic settings."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки MCP сервера TG Dump, загружаемые из переменных окружения."""

    chroma_path: str = Field("./chroma_db", description="Путь к базе ChromaDB")
    chats_path: str = Field("./chats", description="Путь к директории с чатами")
    artifacts_path: str = Field("./artifacts", description="Путь к артефактам")

    host: str = Field(
        "127.0.0.1", description="HTTP хост для streamable-http транспорта"
    )
    port: int = Field(8000, description="HTTP порт для streamable-http транспорта")
    log_level: str = Field("INFO", description="Уровень логирования")
    transport: Literal["stdio", "streamable-http"] = Field(
        "stdio", description="Транспорт для запуска MCP сервера"
    )
    debug: bool = Field(False, description="Включение расширенного логгирования")

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_MCP_",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """Кэшированный доступ к настройкам."""
    _apply_env_aliases()
    return Settings()


__all__ = ["Settings", "get_settings"]


def _apply_env_aliases() -> None:
    """Обеспечивает обратную совместимость со старыми именами переменных."""
    alias_map = {
        "MEMORY_MCP_HOST": ["TG_DUMP_HOST", "HOST"],
        "MEMORY_MCP_PORT": ["TG_DUMP_PORT", "PORT"],
        "MEMORY_MCP_LOG_LEVEL": ["MEMORY_LOG_LEVEL", "TG_DUMP_LOG_LEVEL", "LOG_LEVEL"],
        "MEMORY_MCP_TRANSPORT": ["TG_DUMP_TRANSPORT", "TRANSPORT"],
        "MEMORY_MCP_DB_PATH": ["MEMORY_DB_PATH"],
    }

    for target, candidates in alias_map.items():
        if os.getenv(target):
            continue
        for legacy in candidates:
            value = os.getenv(legacy)
            if value:
                os.environ[target] = value
                break
