#!/usr/bin/env python3
"""
Утилита для валидации и очистки URL
"""

import logging
import re
from typing import List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Регулярное выражение для поиска URL (включая IPv6 с квадратными скобками)
URL_PATTERN = re.compile(
    r'https?://(?:\[[^\]]+\]|[^\s<>"{}|\\^`\[\]]+)(?::[0-9]+)?(?:/[^\s<>"{}|\\^`]*)?',
    re.IGNORECASE,
)

# Паттерны для некорректных IPv6 URL - более точные
INVALID_IPV6_PATTERNS = [
    r"https?://\[[^\]]*\]$",  # IPv6 в квадратных скобках без порта (точное совпадение)
    r"https?://\[[^\]]*\]:[0-9]+$",  # IPv6 с портом (точное совпадение)
    r"https?://[0-9a-fA-F:]+$",  # IPv6 без квадратных скобок (точное совпадение)
]


def is_valid_url(url: str) -> bool:
    """
    Проверяет валидность URL

    Args:
        url: URL для проверки

    Returns:
        True если URL валиден, False иначе
    """
    try:
        parsed = urlparse(url)

        # Проверяем базовые компоненты
        if not parsed.scheme or not parsed.netloc:
            return False

        # Проверяем схему
        if parsed.scheme not in ["http", "https"]:
            return False

        # Проверяем на некорректные IPv6 паттерны
        for pattern in INVALID_IPV6_PATTERNS:
            if re.match(pattern, url):
                logger.debug(f"Обнаружен некорректный IPv6 URL: {url}")
                return False

        return True

    except Exception as e:
        logger.debug(f"Ошибка при валидации URL {url}: {e}")
        return False


def clean_url(url: str) -> Optional[str]:
    """
    Очищает и исправляет URL

    Args:
        url: Исходный URL

    Returns:
        Очищенный URL или None если не удалось исправить
    """
    try:
        # Убираем лишние символы
        url = url.strip()

        # Проверяем валидность
        if is_valid_url(url):
            return url

        # Пытаемся исправить распространенные проблемы
        # Убираем некорректные IPv6 адреса
        for pattern in INVALID_IPV6_PATTERNS:
            if re.match(pattern, url):
                logger.warning(f"Удаляем некорректный IPv6 URL: {url}")
                return None

        # Пытаемся исправить URL без схемы
        if not url.startswith(("http://", "https://")):
            fixed_url = "https://" + url
            if is_valid_url(fixed_url):
                return fixed_url

        return None

    except Exception as e:
        logger.debug(f"Ошибка при очистке URL {url}: {e}")
        return None


def extract_and_clean_urls(text: str) -> List[str]:
    """
    Извлекает и очищает все URL из текста

    Args:
        text: Исходный текст

    Returns:
        Список валидных URL
    """
    urls = URL_PATTERN.findall(text)
    valid_urls = []

    for url in urls:
        cleaned_url = clean_url(url)
        if cleaned_url:
            valid_urls.append(cleaned_url)
        else:
            logger.debug(f"Пропущен некорректный URL: {url}")

    return valid_urls


def sanitize_text_for_embedding(text: str) -> tuple[str, List[str]]:
    """
    Очищает текст от некорректных URL для безопасной передачи в эмбеддинг

    Args:
        text: Исходный текст

    Returns:
        Tuple[очищенный_текст, список_замененных_url]
    """
    if not text:
        return text, []

    # Находим все URL
    urls = URL_PATTERN.findall(text)
    replaced_urls = []

    # Заменяем некорректные URL на placeholder
    cleaned_text = text
    for url in urls:
        if not is_valid_url(url):
            logger.debug(f"Заменяем некорректный URL на placeholder: {url}")
            cleaned_text = cleaned_text.replace(url, "[INVALID_URL]")
            replaced_urls.append(url)

    return cleaned_text, replaced_urls


def validate_embedding_text(text: str) -> tuple[str, List[str]]:
    """
    Валидирует и нормализует текст перед отправкой в эмбеддинг

    Args:
        text: Текст для валидации

    Returns:
        Tuple[валидированный_текст, список_замененных_url]
    """
    if not text:
        return text, []

    # Импортируем нормализатор (ленивый импорт для избежания циклических зависимостей)
    try:
        from .advanced_text_normalizer import normalize_message_text
    except ImportError:
        # Если модуль недоступен, используем базовую валидацию
        return _basic_validate_embedding_text(text)

    # Применяем расширенную нормализацию
    normalization_result = normalize_message_text(text)
    normalized_text = normalization_result.normalized_text

    # Очищаем от некорректных URL
    cleaned_text, replaced_urls = sanitize_text_for_embedding(normalized_text)

    # Проверяем длину
    max_length = 16000  # Лимит для эмбеддингов
    if len(cleaned_text) > max_length:
        logger.warning(
            f"Текст слишком длинный для эмбеддинга: {len(cleaned_text)} символов"
        )
        # Обрезаем с учетом троеточия
        cleaned_text = cleaned_text[: max_length - 3] + "..."

    return cleaned_text, replaced_urls


def _basic_validate_embedding_text(text: str) -> tuple[str, List[str]]:
    """
    Базовая валидация текста (fallback если нормализатор недоступен)

    Args:
        text: Текст для валидации

    Returns:
        Tuple[валидированный_текст, список_замененных_url]
    """
    if not text:
        return text, []

    # Очищаем от некорректных URL
    cleaned_text, replaced_urls = sanitize_text_for_embedding(text)

    # Проверяем длину
    max_length = 16000  # Лимит для эмбеддингов
    if len(cleaned_text) > max_length:
        logger.warning(
            f"Текст слишком длинный для эмбеддинга: {len(cleaned_text)} символов"
        )
        # Обрезаем с учетом троеточия
        cleaned_text = cleaned_text[: max_length - 3] + "..."

    return cleaned_text, replaced_urls
