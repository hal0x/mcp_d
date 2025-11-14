#!/usr/bin/env python3
"""Утилиты для централизованной обработки ошибок."""

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


def format_error_message(exc: Exception, prefix: str = "Ошибка") -> str:
    """
    Форматирует сообщение об ошибке с единообразным префиксом.

    Обеспечивает консистентное форматирование ошибок для клиентов.
    Если сообщение уже содержит префикс "Ошибка" или "Error", не добавляет его повторно.

    Args:
        exc: Исключение для форматирования
        prefix: Префикс для сообщения об ошибке (по умолчанию "Ошибка")

    Returns:
        Отформатированное сообщение об ошибке

    Example:
        >>> format_error_message(ValueError("Неверное значение"))
        "Ошибка: Неверное значение"
        >>> format_error_message(ValueError("Ошибка: уже есть префикс"))
        "Ошибка: уже есть префикс"
    """
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    prefix_lowered = prefix.lower()

    # Проверяем, не начинается ли сообщение уже с префикса
    if not (lowered.startswith(prefix_lowered) or lowered.startswith("error")):
        message = f"{prefix}: {message}"

    return message
