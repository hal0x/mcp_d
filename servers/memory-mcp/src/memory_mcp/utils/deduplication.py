"""Утилиты для дедупликации сообщений.

Централизованные функции для удаления дубликатов сообщений различными способами:
- По ID сообщения
- По хешу содержимого
- По схожести текста (последовательные дубликаты)
"""

import hashlib
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional


def get_message_hash(message: Dict) -> str:
    """
    Получение хеша сообщения для дедупликации.

    Хеш создаётся на основе ключевых полей сообщения: text, caption, file_name, sticker_emoji.
    Это позволяет находить дубликаты даже если ID различаются.

    Args:
        message: Словарь с данными сообщения

    Returns:
        MD5 хеш сообщения в виде hex-строки

    Example:
        >>> msg = {"text": "Hello", "id": 123}
        >>> hash1 = get_message_hash(msg)
        >>> hash2 = get_message_hash(msg)
        >>> hash1 == hash2
        True
    """
    content = ""
    if isinstance(message, dict):
        # Собираем ключевые поля для хеширования
        fields = ["text", "caption", "file_name", "sticker_emoji"]
        for field in fields:
            if field in message and message[field]:
                content += str(message[field])

    return hashlib.md5(content.encode("utf-8")).hexdigest()


def deduplicate_by_id(messages: List[Dict]) -> List[Dict]:
    """
    Дедупликация сообщений по полю 'id'.

    Удаляет сообщения с одинаковым ID, оставляя первое вхождение.

    Args:
        messages: Список сообщений для дедупликации

    Returns:
        Список уникальных сообщений (по ID)

    Example:
        >>> msgs = [{"id": 1, "text": "A"}, {"id": 2, "text": "B"}, {"id": 1, "text": "C"}]
        >>> deduplicate_by_id(msgs)
        [{"id": 1, "text": "A"}, {"id": 2, "text": "B"}]
    """
    seen_ids = set()
    unique_messages = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        if "id" in message:
            msg_id = str(message["id"])
            if msg_id not in seen_ids:
                seen_ids.add(msg_id)
                unique_messages.append(message)
        else:
            # Сообщения без ID добавляем как есть
            unique_messages.append(message)

    return unique_messages


def normalize_text(text: Optional[str]) -> str:
    """
    Нормализация текста для сравнения.

    Убирает лишние пробелы, эмодзи и приводит к нижнему регистру.

    Args:
        text: Исходный текст

    Returns:
        Нормализованный текст

    Example:
        >>> normalize_text("  Hello  World!  ")
        "hello world!"
    """
    if not text:
        return ""

    # Убираем лишние пробелы
    normalized = re.sub(r"\s+", " ", text.strip())

    # Убираем эмодзи для более точного сравнения
    normalized = re.sub(r"[\U00010000-\U0010ffff]", "", normalized)

    # Приводим к нижнему регистру
    normalized = normalized.lower()

    return normalized


def is_similar(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """
    Проверка схожести двух текстов.

    Использует SequenceMatcher для вычисления коэффициента схожести.
    Тексты считаются похожими, если коэффициент >= threshold.

    Args:
        text1: Первый текст
        text2: Второй текст
        threshold: Порог схожести (0.0-1.0), по умолчанию 0.85

    Returns:
        True если тексты похожи (схожесть >= threshold)

    Example:
        >>> is_similar("Hello world", "Hello world!", threshold=0.9)
        True
        >>> is_similar("Hello", "Goodbye", threshold=0.5)
        False
    """
    if not text1 or not text2:
        return False

    # Точное совпадение
    if text1 == text2:
        return True

    # Нормализуем тексты для сравнения
    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)

    # Проверка через SequenceMatcher
    similarity = SequenceMatcher(None, norm_text1, norm_text2).ratio()
    return similarity >= threshold


def deduplicate_consecutive(
    messages: List[Dict],
    threshold: float = 0.85,
    max_consecutive: int = 1,
    get_text_func: Optional[callable] = None,
) -> List[Dict]:
    """
    Дедупликация последовательных похожих сообщений.

    Удаляет сообщения, которые идут подряд и имеют схожий текст.
    Оставляет первое сообщение из группы похожих.

    Args:
        messages: Список сообщений для дедупликации
        threshold: Порог схожести текстов (0.0-1.0)
        max_consecutive: Максимальное количество подряд идущих похожих сообщений
        get_text_func: Функция для извлечения текста из сообщения.
                      Если None, используется встроенная логика.

    Returns:
        Список сообщений без последовательных дубликатов

    Example:
        >>> msgs = [
        ...     {"text": "Hello"},
        ...     {"text": "Hello!"},
        ...     {"text": "World"},
        ... ]
        >>> deduplicate_consecutive(msgs, threshold=0.9)
        [{"text": "Hello"}, {"text": "World"}]
    """
    if not messages:
        return []

    def _get_text(msg: Dict) -> str:
        """Извлечение текста из сообщения."""
        if get_text_func:
            return get_text_func(msg) or ""
        # Стандартная логика извлечения текста
        text_parts = []
        if "text" in msg and msg["text"]:
            text_parts.append(str(msg["text"]))
        if "caption" in msg and msg["caption"]:
            text_parts.append(str(msg["caption"]))
        return " ".join(text_parts)

    deduplicated = []
    prev_text = None
    consecutive_count = 0

    for msg in messages:
        if not isinstance(msg, dict):
            deduplicated.append(msg)
            continue

        text = _get_text(msg)
        normalized_text = normalize_text(text)

        # Если текст пустой, всегда добавляем (может быть медиа)
        if not normalized_text:
            deduplicated.append(msg)
            prev_text = None
            consecutive_count = 0
            continue

        # Проверяем похожесть с предыдущим
        if prev_text and is_similar(normalized_text, prev_text, threshold):
            consecutive_count += 1

            # Пропускаем, если превышен лимит
            if consecutive_count > max_consecutive:
                # Добавляем аннотацию о пропущенных сообщениях
                if consecutive_count == max_consecutive + 1:
                    # Добавляем маркер только один раз
                    last_msg = deduplicated[-1]
                    if not last_msg.get("_duplicate_marker"):
                        last_msg["_duplicate_count"] = 1
                        last_msg["_duplicate_marker"] = True
                        last_msg["_duplicate_variants"] = [text[:200]]
                else:
                    # Увеличиваем счётчик и сохраняем вариацию
                    last_msg = deduplicated[-1]
                    last_msg["_duplicate_count"] = (
                        last_msg.get("_duplicate_count", 1) + 1
                    )
                    # Сохраняем до 5 вариаций для анализа
                    variants = last_msg.get("_duplicate_variants", [])
                    if len(variants) < 5 and text[:200] not in variants:
                        variants.append(text[:200])
                        last_msg["_duplicate_variants"] = variants
                continue
            else:
                # В пределах лимита - добавляем сообщение
                deduplicated.append(msg)
                prev_text = normalized_text
        else:
            # Не похоже на предыдущее - сбрасываем счётчик и добавляем
            consecutive_count = 0
            deduplicated.append(msg)
            prev_text = normalized_text

    return deduplicated


def deduplicate_by_hash(messages: List[Dict]) -> List[Dict]:
    """
    Дедупликация сообщений по хешу содержимого.

    Удаляет сообщения с одинаковым хешем (вычисленным через get_message_hash),
    оставляя первое вхождение.

    Args:
        messages: Список сообщений для дедупликации

    Returns:
        Список уникальных сообщений (по хешу)

    Example:
        >>> msgs = [
        ...     {"text": "Hello", "id": 1},
        ...     {"text": "Hello", "id": 2},  # Дубликат по содержимому
        ... ]
        >>> deduplicate_by_hash(msgs)
        [{"text": "Hello", "id": 1}]
    """
    seen_hashes = set()
    unique_messages = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        msg_hash = get_message_hash(message)
        if msg_hash not in seen_hashes:
            seen_hashes.add(msg_hash)
            unique_messages.append(message)

    return unique_messages

