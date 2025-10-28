#!/usr/bin/env python3
"""Загрузка и хранение конфигурации анализа качества."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QualityAnalysisConfig:
    """Конфигурация модуля анализа качества."""

    ollama_model: str = "gpt-oss-20b:latest"
    ollama_base_url: str = "http://localhost:11434"
    max_context_tokens: int = 8192
    temperature: float = 0.1
    max_response_tokens: int = 1000
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
    def load_from_path(cls, config_path: Path) -> QualityAnalysisConfig:
        """Загрузка конфигурации из файла.

        Если файл отсутствует или поврежден, возвращаются значения по умолчанию.
        """
        if not config_path or not config_path.exists():
            logger.debug(
                "Файл конфигурации анализа качества не найден: %s", config_path
            )
            return cls()

        try:
            with open(config_path, encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception as exc:  # pragma: no cover — логирование вместо падения
            logger.error("Ошибка чтения конфигурации качества: %s", exc)
            return cls()

        qa_section = config_data.get("quality_analysis", {})
        ollama_section = qa_section.get("ollama", {})
        query_section = qa_section.get("query_generation", {})
        analysis_section = qa_section.get("analysis", {})
        reporting_section = qa_section.get("reporting", {})

        def _resolve_path(path_value: str | None) -> Path | None:
            if not path_value:
                return None
            path_obj = Path(path_value)
            if not path_obj.is_absolute():
                candidate = (config_path.parent / path_obj).resolve()
                if candidate.exists():
                    return candidate
                path_obj = (Path.cwd() / path_obj).resolve()
            return path_obj

        reports_dir_raw = reporting_section.get("reports_dir", "artifacts/reports")
        reports_path_resolved = _resolve_path(reports_dir_raw)
        if reports_path_resolved is None:
            reports_path_resolved = (config_path.parent / "artifacts/reports").resolve()

        subdir_value = reporting_section.get("subdir")
        subdir_explicit = subdir_value is not None
        if subdir_value == "":
            quality_subdir = None
        else:
            quality_subdir = subdir_value

        if subdir_explicit:
            reports_base_dir = reports_path_resolved
        else:
            parsed_reports = Path(reports_dir_raw)
            if parsed_reports.parent not in (Path("."), Path("")):
                quality_subdir = parsed_reports.name or "quality_analysis"
                base_candidate = parsed_reports.parent
                reports_base_dir = (
                    _resolve_path(str(base_candidate)) or reports_path_resolved.parent
                )
            else:
                quality_subdir = "quality_analysis"
                reports_base_dir = reports_path_resolved

        history_path = _resolve_path(
            reporting_section.get("history_dir", "quality_analysis_history")
        )
        if history_path is None:
            history_path = (config_path.parent / "quality_analysis_history").resolve()

        custom_queries_path = query_section.get("custom_queries_file")
        custom_queries_file = (
            _resolve_path(custom_queries_path) if custom_queries_path else None
        )

        thinking_value: str | None = ollama_section.get("thinking_level")
        if thinking_value is None and "thinking" in ollama_section:
            maybe_thinking = ollama_section.get("thinking")
            if isinstance(maybe_thinking, dict):
                thinking_value = maybe_thinking.get("level")
            elif isinstance(maybe_thinking, str):
                thinking_value = maybe_thinking

        batch_size_value = analysis_section.get("batch_size")
        if isinstance(batch_size_value, int) and batch_size_value <= 0:
            batch_size_value = None

        chroma_path_raw = analysis_section.get("chroma_path")
        chroma_path = (
            _resolve_path(chroma_path_raw)
            if chroma_path_raw
            else (config_path.parent / "chroma_db").resolve()
        )

        if not isinstance(reports_base_dir, Path):
            reports_base_dir = Path(reports_base_dir)
        reports_base_dir = reports_base_dir.resolve()

        if not isinstance(history_path, Path):
            history_path = Path(history_path)
        history_path = history_path.resolve()

        chats_dir_raw = analysis_section.get("chats_dir")
        chats_dir_candidate = (
            _resolve_path(chats_dir_raw)
            if chats_dir_raw
            else (config_path.parent / "chats").resolve()
        )

        if chats_dir_candidate is None or not chats_dir_candidate.exists():
            fallback = (Path.cwd() / "chats").resolve()
            if fallback.exists():
                chats_dir_candidate = fallback
            else:
                chats_dir_candidate = Path("chats").resolve()

        chats_dir = chats_dir_candidate

        return cls(
            ollama_model=ollama_section.get("model", cls.ollama_model),
            ollama_base_url=ollama_section.get("base_url", cls.ollama_base_url),
            max_context_tokens=ollama_section.get(
                "max_context_tokens", cls.max_context_tokens
            ),
            temperature=ollama_section.get("temperature", cls.temperature),
            max_response_tokens=ollama_section.get(
                "max_tokens", cls.max_response_tokens
            ),
            batch_size=batch_size_value,
            max_queries_per_chat=query_section.get(
                "max_queries_per_chat", cls.max_queries_per_chat
            ),
            thinking_level=thinking_value,
            reports_dir=reports_base_dir,
            quality_reports_subdir=quality_subdir,
            history_dir=history_path,
            thresholds=qa_section.get("thresholds", {}),
            custom_queries_path=custom_queries_file,
            chroma_path=chroma_path,
            search_collection=analysis_section.get(
                "search_collection", cls.search_collection
            ),
            hybrid_alpha=analysis_section.get("hybrid_alpha", cls.hybrid_alpha),
            results_per_query=analysis_section.get(
                "results_per_query", cls.results_per_query
            ),
            chats_dir=chats_dir,
            batch_max_size=analysis_section.get("batch_max_size", cls.batch_max_size),
            system_prompt_reserve=analysis_section.get(
                "system_prompt_reserve", cls.system_prompt_reserve
            ),
            max_query_tokens=analysis_section.get(
                "max_query_tokens", cls.max_query_tokens
            ),
        )


def load_config(config_path: Path | None) -> QualityAnalysisConfig:
    """Утилита для загрузки конфигурации с учетом стандартного расположения."""
    if config_path and config_path.exists():
        return QualityAnalysisConfig.load_from_path(config_path)

    default_path = Path("config/quality_analysis.json")
    return QualityAnalysisConfig.load_from_path(default_path)
