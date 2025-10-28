#!/usr/bin/env python3
"""
Модуль анализа качества индексации и поиска

Система автоматического тестирования качества работы memory_mcp через:
- Генерацию тестовых запросов на основе реальных данных
- Анализ релевантности результатов через Ollama
- Сравнительные метрики и рекомендации по улучшению
"""

from .core import (
    HistoryManager,
    MetricsCalculator,
    QueryGenerator,
    RelevanceAnalyzer,
    ReportGenerator,
)
from .quality_analyzer import QualityAnalyzer
from .utils import log_error, safe_execute, swallow_errors

__all__ = [
    "QueryGenerator",
    "RelevanceAnalyzer",
    "MetricsCalculator",
    "ReportGenerator",
    "HistoryManager",
    "QualityAnalyzer",
    "log_error",
    "safe_execute",
    "swallow_errors",
]


def __getattr__(name):  # pragma: no cover - простая ленивость
    if name in {"QualityAnalysisError", "run_quality_analysis", "main"}:
        from . import main as _main

        globals().update(
            {
                "QualityAnalysisError": _main.QualityAnalysisError,
                "run_quality_analysis": _main.run_quality_analysis,
                "main": _main.main,
            }
        )

        if "QualityAnalysisError" not in __all__:
            __all__.extend(["QualityAnalysisError", "run_quality_analysis", "main"])
        return globals()[name]
    raise AttributeError(f"module 'memory_mcp.quality_analyzer' has no attribute {name!r}")
