#!/usr/bin/env python3
"""Загрузка и хранение конфигурации анализа качества.

DEPRECATED: Используйте QualityAnalysisSettings из memory_mcp.config.
Этот модуль оставлен для обратной совместимости.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import QualityAnalysisSettings, get_quality_analysis_settings

logger = logging.getLogger(__name__)


@dataclass
class QualityAnalysisConfig:
    """
    Конфигурация модуля анализа качества.

    DEPRECATED: Используйте QualityAnalysisSettings из memory_mcp.config.
    Этот класс оставлен для обратной совместимости и является оберткой над QualityAnalysisSettings.
    """

    ollama_model: str = "gpt-oss-20b:latest"
    ollama_base_url: str = "http://localhost:11434"
    max_context_tokens: int = 131072  # Для gpt-oss-20b
    temperature: float = 0.1
    max_response_tokens: int = 131072  # Для gpt-oss-20b (максимальный лимит)
    batch_size: int | None = None
    max_queries_per_chat: int = 20
    thinking_level: str | None = None
    reports_dir: Path = Path("artifacts/reports")
    quality_reports_subdir: str = "quality_analysis"
    history_dir: Path = Path("quality_analysis_history")
    thresholds: dict[str, Any] = field(default_factory=dict)
    custom_queries_path: Path | None = None
    chroma_path: Path = Path("chroma_db")
    search_collection: str = "chat_messages"
    hybrid_alpha: float = 0.6
    results_per_query: int = 10
    chats_dir: Path = Path("chats")
    batch_max_size: int = 10
    system_prompt_reserve: float = 0.2
    max_query_tokens: int = 6000

    @classmethod
    def from_settings(cls, settings: QualityAnalysisSettings) -> "QualityAnalysisConfig":
        """Создает QualityAnalysisConfig из QualityAnalysisSettings."""
        return cls(
            ollama_model=settings.ollama_model,
            ollama_base_url=settings.ollama_base_url,
            max_context_tokens=settings.max_context_tokens,
            temperature=settings.temperature,
            max_response_tokens=settings.max_response_tokens,
            batch_size=settings.batch_size,
            max_queries_per_chat=settings.max_queries_per_chat,
            thinking_level=settings.thinking_level,
            reports_dir=settings.reports_dir,
            quality_reports_subdir=settings.quality_reports_subdir,
            history_dir=settings.history_dir,
            thresholds=settings.thresholds,
            custom_queries_path=settings.custom_queries_path,
            chroma_path=settings.chroma_path,
            search_collection=settings.search_collection,
            hybrid_alpha=settings.hybrid_alpha,
            results_per_query=settings.results_per_query,
            chats_dir=settings.chats_dir,
            batch_max_size=settings.batch_max_size,
            system_prompt_reserve=settings.system_prompt_reserve,
            max_query_tokens=settings.max_query_tokens,
        )

    @classmethod
    def load_from_path(cls, config_path: Path) -> QualityAnalysisConfig:
        """
        Загрузка конфигурации из файла.

        DEPRECATED: Используйте get_quality_analysis_settings() из memory_mcp.config.
        Этот метод оставлен для обратной совместимости.

        Если файл отсутствует или поврежден, возвращаются значения по умолчанию.
        """
        settings = get_quality_analysis_settings(config_path)
        return cls.from_settings(settings)


def load_config(config_path: Path | None) -> QualityAnalysisConfig:
    """
    Утилита для загрузки конфигурации с учетом стандартного расположения.

    DEPRECATED: Используйте get_quality_analysis_settings() из memory_mcp.config.
    Эта функция оставлена для обратной совместимости.
    """
    settings = get_quality_analysis_settings(config_path)
    return QualityAnalysisConfig.from_settings(settings)
