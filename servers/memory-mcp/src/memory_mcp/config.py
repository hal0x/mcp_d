"""Конфигурация MCP сервера TG Dump на базе Pydantic settings."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки MCP сервера TG Dump, загружаемые из переменных окружения."""

    chroma_path: str = Field("./chroma_db", description="Путь к базе ChromaDB")
    chats_path: str = Field("./chats", description="Путь к директории с чатами")
    artifacts_path: str = Field("./artifacts", description="Путь к артефактам")
    input_path: str = Field("input", description="Путь к директории input с новыми сообщениями")
    background_indexing_enabled: bool = Field(False, description="Включить фоновую индексацию при старте")
    background_indexing_interval: int = Field(60, description="Интервал проверки input директории в секундах")

    host: str = Field(
        "127.0.0.1", description="HTTP хост для streamable-http транспорта"
    )
    port: int = Field(8000, description="HTTP порт для streamable-http транспорта")
    log_level: str = Field("INFO", description="Уровень логирования")
    transport: Literal["stdio", "streamable-http"] = Field(
        "stdio", description="Транспорт для запуска MCP сервера"
    )
    debug: bool = Field(False, description="Включение расширенного логгирования")
    
    # LM Studio настройки
    lmstudio_host: str = Field(
        "127.0.0.1", description="Хост LM Studio Server"
    )
    lmstudio_port: int = Field(1234, description="Порт LM Studio Server")
    lmstudio_model: str = Field(
        "text-embedding-qwen3-embedding-0.6b", description="Модель для эмбеддингов в LM Studio"
    )
    lmstudio_llm_model: str | None = Field(
        "gpt-oss-20b", description="Модель LLM для генерации текста в LM Studio (если None, используется Ollama)"
    )
    
    # Дополнительные настройки
    db_path: str = Field("data/memory_graph.db", description="Путь к SQLite базе данных")
    embeddings_url: SecretStr | None = Field(None, description="URL сервиса эмбеддингов (приоритет над LM Studio)")
    qdrant_url: SecretStr | None = Field(None, description="URL Qdrant векторной базы данных")
    
    # Настройки больших контекстов
    large_context_max_tokens: int = Field(
        131072, description="Максимальный контекст модели (токенов)"
    )
    large_context_prompt_reserve: int = Field(
        5000, description="Резерв токенов для промпта"
    )
    large_context_hierarchical_threshold: int = Field(
        100000, description="Порог для включения иерархической обработки (токенов)"
    )
    large_context_enable_hierarchical: bool = Field(
        True, description="Включить иерархическую обработку"
    )
    large_context_use_smart_search: bool = Field(
        True, description="Использовать smart_search для анализа контекста"
    )
    large_context_cache_size: int = Field(
        100, description="Размер кэша для промежуточных результатов"
    )

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_MCP_",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )
    
    def get_embeddings_url(self) -> str | None:
        """Получить URL эмбеддингов как строку (безопасно)."""
        if self.embeddings_url is None:
            return None
        return self.embeddings_url.get_secret_value()
    
    def get_qdrant_url(self) -> str | None:
        """Получить URL Qdrant как строку (безопасно)."""
        if self.qdrant_url is None:
            return None
        return self.qdrant_url.get_secret_value()


class QualityAnalysisSettings(BaseSettings):
    """Настройки модуля анализа качества, загружаемые из переменных окружения."""

    # Ollama настройки
    ollama_model: str = Field("gpt-oss-20b:latest", description="Модель Ollama")
    ollama_base_url: str = Field("http://localhost:11434", description="URL Ollama сервера")
    max_context_tokens: int = Field(131072, description="Максимальное количество токенов контекста (для gpt-oss-20b)")
    temperature: float = Field(0.1, description="Температура для генерации")
    max_response_tokens: int = Field(131072, description="Максимальное количество токенов ответа (для gpt-oss-20b)")
    thinking_level: str | None = Field(None, description="Уровень мышления (thinking)")

    # Настройки генерации запросов
    max_queries_per_chat: int = Field(20, description="Максимальное количество запросов на чат")
    batch_size: int | None = Field(None, description="Размер батча (None = автоматический)")

    # Настройки анализа
    search_collection: str = Field("chat_messages", description="Коллекция для поиска")
    hybrid_alpha: float = Field(0.6, description="Коэффициент гибридного поиска")
    results_per_query: int = Field(10, description="Количество результатов на запрос")
    batch_max_size: int = Field(10, description="Максимальный размер батча")
    system_prompt_reserve: float = Field(0.2, description="Резерв для системного промпта")
    max_query_tokens: int = Field(6000, description="Максимальное количество токенов запроса")

    # Пути (поддерживают относительные и абсолютные)
    reports_dir: Path = Field(Path("artifacts/reports"), description="Директория для отчетов")
    quality_reports_subdir: str = Field("quality_analysis", description="Поддиректория для отчетов качества")
    history_dir: Path = Field(Path("quality_analysis_history"), description="Директория для истории")
    chroma_path: Path = Field(Path("chroma_db"), description="Путь к ChromaDB")
    chats_dir: Path = Field(Path("chats"), description="Директория с чатами")
    custom_queries_path: Path | None = Field(None, description="Путь к файлу с кастомными запросами")

    # Пороги (хранятся как JSON строка или dict)
    thresholds: dict[str, Any] = Field(default_factory=dict, description="Пороги для оценки качества")

    @field_validator("reports_dir", "history_dir", "chroma_path", "chats_dir", "custom_queries_path", mode="before")
    @classmethod
    def validate_paths(cls, v: Any) -> Any:
        """Валидация путей - конвертация строк в Path."""
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("thresholds", mode="before")
    @classmethod
    def validate_thresholds(cls, v: Any) -> dict[str, Any]:
        """Валидация порогов - поддержка JSON строки."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        if isinstance(v, dict):
            return v
        return {}

    model_config = SettingsConfigDict(
        env_prefix="QUALITY_ANALYSIS_",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    @classmethod
    def load_from_json(cls, config_path: Path | None = None) -> "QualityAnalysisSettings":
        """
        Загружает настройки из JSON файла (для обратной совместимости).

        Args:
            config_path: Путь к JSON файлу. Если None, используется config/quality_analysis.json

        Returns:
            Экземпляр QualityAnalysisSettings
        """
        if config_path is None:
            config_path = Path("config/quality_analysis.json")

        if not config_path.exists():
            return cls()

        try:
            with open(config_path, encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception:
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

        # Обработка путей
        reports_dir_raw = reporting_section.get("reports_dir", "artifacts/reports")
        reports_path_resolved = _resolve_path(reports_dir_raw) or Path(reports_dir_raw)

        subdir_value = reporting_section.get("subdir")
        quality_subdir = subdir_value if subdir_value is not None else "quality_analysis"

        history_path = _resolve_path(reporting_section.get("history_dir", "quality_analysis_history")) or Path("quality_analysis_history")
        chroma_path = _resolve_path(analysis_section.get("chroma_path")) or Path("chroma_db")
        chats_dir = _resolve_path(analysis_section.get("chats_dir")) or Path("chats")
        custom_queries_path = _resolve_path(query_section.get("custom_queries_file"))

        # Создаем настройки из JSON
        return cls(
            ollama_model=ollama_section.get("model", "gpt-oss-20b:latest"),
            ollama_base_url=ollama_section.get("base_url", "http://localhost:11434"),
            max_context_tokens=ollama_section.get("max_context_tokens", 131072),
            temperature=ollama_section.get("temperature", 0.1),
            max_response_tokens=ollama_section.get("max_tokens", 131072),
            thinking_level=ollama_section.get("thinking_level"),
            max_queries_per_chat=query_section.get("max_queries_per_chat", 20),
            batch_size=analysis_section.get("batch_size") if analysis_section.get("batch_size") and analysis_section.get("batch_size") > 0 else None,
            search_collection=analysis_section.get("search_collection", "chat_messages"),
            hybrid_alpha=analysis_section.get("hybrid_alpha", 0.6),
            results_per_query=analysis_section.get("results_per_query", 10),
            batch_max_size=analysis_section.get("batch_max_size", 10),
            system_prompt_reserve=analysis_section.get("system_prompt_reserve", 0.2),
            max_query_tokens=analysis_section.get("max_query_tokens", 6000),
            reports_dir=reports_path_resolved,
            quality_reports_subdir=quality_subdir,
            history_dir=history_path,
            chroma_path=chroma_path,
            chats_dir=chats_dir,
            custom_queries_path=custom_queries_path,
            thresholds=qa_section.get("thresholds", {}),
        )


@lru_cache
def get_settings() -> Settings:
    """Кэшированный доступ к настройкам."""
    _apply_env_aliases()
    return Settings()


@lru_cache
def get_quality_analysis_settings(config_path: Path | None = None) -> QualityAnalysisSettings:
    """
    Кэшированный доступ к настройкам анализа качества.

    Стратегия загрузки настроек:
    1. Загружаем настройки из переменных окружения (Pydantic Settings)
    2. Если доступен JSON файл, загружаем настройки из него
    3. Объединяем: для каждого поля, если значение из env равно дефолтному,
       заменяем его значением из JSON (env имеет приоритет только для явно заданных значений)

    Args:
        config_path: Путь к JSON файлу конфигурации (опционально)

    Returns:
        Экземпляр QualityAnalysisSettings
    """
    try:
        # Загружаем настройки из переменных окружения
        settings = QualityAnalysisSettings()
        
        # Если доступен JSON файл, объединяем настройки
        json_config_path = config_path or Path("config/quality_analysis.json")
        if json_config_path.exists():
            json_settings = QualityAnalysisSettings.load_from_json(config_path)
            
            # Объединяем настройки: env имеет приоритет только для явно заданных значений
            # Если значение из env равно дефолтному, используем значение из JSON
            for field_name in QualityAnalysisSettings.model_fields:
                env_value = getattr(settings, field_name)
                json_value = getattr(json_settings, field_name)
                default_value = QualityAnalysisSettings.model_fields[field_name].default
                
                # Если значение из env равно дефолтному, значит оно не было задано явно
                # В этом случае используем значение из JSON
                if env_value == default_value:
                    setattr(settings, field_name, json_value)
                # Иначе оставляем значение из env (оно было задано явно)
        
        return settings
    except Exception:
        # Fallback на JSON при ошибке загрузки из env
        return QualityAnalysisSettings.load_from_json(config_path)


__all__ = ["Settings", "QualityAnalysisSettings", "get_settings", "get_quality_analysis_settings"]


def _apply_env_aliases() -> None:
    """
    Обеспечивает обратную совместимость со старыми именами переменных.
    
    Логика работы:
    - Если целевая переменная (например, MEMORY_MCP_HOST) уже установлена, она используется как есть.
    - Если целевая переменная не установлена, проверяются старые имена (алиасы) в порядке приоритета.
    - Первое найденное значение из алиасов копируется в целевую переменную.
    
    Приоритет: новые переменные (MEMORY_MCP_*) > старые переменные (TG_DUMP_*, MEMORY_*, и т.д.)
    """
    alias_map = {
        "MEMORY_MCP_HOST": ["TG_DUMP_HOST", "HOST"],
        "MEMORY_MCP_PORT": ["TG_DUMP_PORT", "PORT"],
        "MEMORY_MCP_LOG_LEVEL": ["MEMORY_LOG_LEVEL", "TG_DUMP_LOG_LEVEL", "LOG_LEVEL"],
        "MEMORY_MCP_TRANSPORT": ["TG_DUMP_TRANSPORT", "TRANSPORT"],
        "MEMORY_MCP_DB_PATH": ["MEMORY_DB_PATH"],
        "MEMORY_MCP_LMSTUDIO_HOST": ["LMSTUDIO_HOST"],
        "MEMORY_MCP_LMSTUDIO_PORT": ["LMSTUDIO_PORT"],
        "MEMORY_MCP_LMSTUDIO_MODEL": ["LMSTUDIO_MODEL"],
        "MEMORY_MCP_EMBEDDINGS_URL": ["EMBEDDINGS_URL"],
        "MEMORY_MCP_QDRANT_URL": ["QDRANT_URL"],
        "MEMORY_MCP_LARGE_CONTEXT_MAX_TOKENS": ["LARGE_CONTEXT_MAX_TOKENS"],
        "MEMORY_MCP_LARGE_CONTEXT_ENABLE_HIERARCHICAL": ["LARGE_CONTEXT_ENABLE_HIERARCHICAL"],
        "MEMORY_MCP_LARGE_CONTEXT_USE_SMART_SEARCH": ["LARGE_CONTEXT_USE_SMART_SEARCH"],
    }

    for target, candidates in alias_map.items():
        # Если целевая переменная уже установлена, пропускаем
        if os.getenv(target):
            continue
        # Ищем значение в старых переменных (алиасах)
        for legacy in candidates:
            value = os.getenv(legacy)
            if value:
                os.environ[target] = value
                break
