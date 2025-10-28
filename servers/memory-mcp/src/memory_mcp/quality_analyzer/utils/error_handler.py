#!/usr/bin/env python3
"""Утилиты для централизованной обработки ошибок в анализаторе качества."""

import logging
from contextlib import contextmanager
from typing import Callable, Generator, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def log_error(context: str, error: Exception) -> None:
    """Логирует ошибку с указанием контекста."""
    logger.error("%s: %s", context, error, exc_info=True)


def safe_execute(
    callback: Callable[[], T], context: str, fallback: Optional[T] = None
) -> Optional[T]:
    """Безопасно выполняет функцию, возвращая запасной результат при ошибке."""
    try:
        return callback()
    except Exception as exc:  # pragma: no cover - логирование вместо падения
        log_error(context, exc)
        return fallback


@contextmanager
def swallow_errors(context: str) -> Generator[None, None, None]:
    """Контекстный менеджер для подавления исключений с предварительным логированием."""
    try:
        yield
    except Exception as exc:  # pragma: no cover - логирование вместо падения
        log_error(context, exc)
