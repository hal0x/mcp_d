"""Утилиты для оптимизации размера контекста.

Унифицированные методы для обрезки и сжатия контекста в разных частях системы.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    """
    Обрезает текст до максимальной длины.

    Args:
        text: Текст для обрезки
        max_chars: Максимальная длина в символах
        suffix: Суффикс, добавляемый при обрезке

    Returns:
        Обрезанный текст
    """
    if not text or len(text) <= max_chars:
        return text

    if max_chars <= len(suffix):
        return suffix[:max_chars]

    return text[: max_chars - len(suffix)] + suffix


def truncate_text_from_end(text: str, max_chars: int, suffix: str = "...") -> str:
    """
    Обрезает текст с конца до максимальной длины (берет последние N символов).

    Args:
        text: Текст для обрезки
        max_chars: Максимальная длина в символах
        suffix: Суффикс, добавляемый при обрезке

    Returns:
        Обрезанный текст (последние N символов)
    """
    if not text or len(text) <= max_chars:
        return text

    if max_chars <= len(suffix):
        return suffix[:max_chars]

    return suffix + text[-(max_chars - len(suffix)) :]


def optimize_context_dict(
    context: Dict[str, Any],
    max_size: int,
    text_fields: Optional[List[str]] = None,
    truncate_from_end: bool = False,
) -> Dict[str, Any]:
    """
    Оптимизирует размер контекста в словаре, обрезая текстовые поля.

    Args:
        context: Словарь с контекстом
        max_size: Максимальный размер для каждого текстового поля
        text_fields: Список полей для оптимизации (если None, используется список по умолчанию)
        truncate_from_end: Если True, обрезает с конца, иначе с начала

    Returns:
        Оптимизированный словарь
    """
    if text_fields is None:
        text_fields = ["recent_context", "chat_context", "context"]

    optimized = context.copy()
    truncate_func = truncate_text_from_end if truncate_from_end else truncate_text

    for field in text_fields:
        if field in optimized and optimized[field]:
            if isinstance(optimized[field], str):
                if len(optimized[field]) > max_size:
                    optimized[field] = truncate_func(optimized[field], max_size)

    return optimized


def compress_messages_context(
    messages: List[Dict[str, Any]],
    target_tokens: int,
    avg_tokens_per_char: float = 0.25,
    format_func: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
) -> str:
    """
    Сжимает контекст сообщений до целевого количества токенов.

    Стратегия:
    - Берет первые 30% и последние 30% сообщений
    - Если все еще слишком много, обрезает длинные сообщения

    Args:
        messages: Список сообщений
        target_tokens: Целевое количество токенов
        avg_tokens_per_char: Среднее количество токенов на символ (~4 символа на токен)
        format_func: Функция для форматирования сообщений в текст.
                    Если None, используется стандартное форматирование.

    Returns:
        Сжатый текст контекста
    """
    if not messages:
        return ""

    # Функция форматирования по умолчанию
    if format_func is None:

        def default_format(msgs: List[Dict[str, Any]]) -> str:
            lines = []
            for msg in msgs:
                author = msg.get("from") or msg.get("author") or "Unknown"
                text = msg.get("text") or msg.get("content") or ""
                date = msg.get("date") or msg.get("timestamp") or ""
                if text:
                    lines.append(f"[{date}] {author}: {text}")
            return "\n".join(lines)

        format_func = default_format

    # Оцениваем размер всего контекста
    full_text = format_func(messages)
    total_tokens = int(len(full_text) * avg_tokens_per_char)

    if total_tokens <= target_tokens:
        return full_text

    # Берем первые 30% и последние 30% сообщений
    first_count = int(len(messages) * 0.3)
    last_count = int(len(messages) * 0.3)
    selected = messages[:first_count] + messages[-last_count:]

    # Форматируем выбранные сообщения
    compressed_text = format_func(selected)
    compressed_tokens = int(len(compressed_text) * avg_tokens_per_char)

    # Если все еще слишком много, обрезаем длинные сообщения
    if compressed_tokens > target_tokens:
        target_chars = int(target_tokens / avg_tokens_per_char)
        ratio = target_chars / len(compressed_text)
        max_chars_per_msg = int(200 * ratio)  # Примерно 200 символов на сообщение

        compressed_lines = []
        for msg in selected:
            author = msg.get("from") or msg.get("author") or "Unknown"
            text = msg.get("text") or msg.get("content") or ""
            date = msg.get("date") or msg.get("timestamp") or ""

            if len(text) > max_chars_per_msg:
                text = text[:max_chars_per_msg] + "..."

            if text:
                compressed_lines.append(f"[{date}] {author}: {text}")

        compressed_text = "\n".join(compressed_lines)

    return compressed_text


def truncate_embedding_text(
    context_text: str,
    message_text: str,
    max_tokens: int,
    avg_tokens_per_char: float = 0.25,
    context_prefix: str = "",
    message_prefix: str = "[CURRENT]: ",
) -> Tuple[str, str]:
    """
    Обрезает контекст и текст сообщения для эмбеддингов с учетом лимита токенов.

    Args:
        context_text: Текст контекста
        message_text: Текст сообщения
        max_tokens: Максимальное количество токенов
        avg_tokens_per_char: Среднее количество токенов на символ
        context_prefix: Префикс для контекста (если есть)
        message_prefix: Префикс для сообщения

    Returns:
        Кортеж (обрезанный контекст, обрезанное сообщение)
    """
    # Оценка токенов
    estimated_tokens = int((len(context_text) + len(message_text)) * avg_tokens_per_char)

    if estimated_tokens <= max_tokens:
        return context_text, message_text

    # Сначала обрезаем контекст, если он слишком длинный
    max_context_chars = int(max_tokens * 0.2 * (1 / avg_tokens_per_char))  # 20% от лимита
    if len(context_text) > max_context_chars:
        context_text = truncate_text(context_text, max_context_chars)

    # Затем обрезаем основное сообщение
    # Учитываем длину контекста + префиксы
    context_tokens = int(len(context_text) * avg_tokens_per_char)
    prefix_tokens = int((len(context_prefix) + len(message_prefix)) * avg_tokens_per_char)
    remaining_tokens = max_tokens - context_tokens - prefix_tokens

    if remaining_tokens > 0:
        remaining_chars = int(remaining_tokens * (1 / avg_tokens_per_char))
        if len(message_text) > remaining_chars:
            message_text = truncate_text(message_text, remaining_chars)
    else:
        # Если контекст уже занимает почти весь лимит, используем только сообщение
        max_msg_chars = int(max_tokens * (1 / avg_tokens_per_char) - 10)
        message_text = truncate_text(message_text, max_msg_chars)

    return context_text, message_text

