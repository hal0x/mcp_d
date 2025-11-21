#!/usr/bin/env python3
"""
Утилиты для работы с текстом для session_summarizer
"""

import re
from typing import Any, Dict, List, Optional

from ....utils.datetime_utils import format_datetime_display, parse_datetime_utc


def strip_markdown(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    cleaned = re.sub(r"[`*_#>|~]+", " ", cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_text_for_display(text: str) -> str:
    """
    Нормализация текста для проверки вариаций

    Args:
        text: Исходный текст

    Returns:
        Нормализованный текст
    """
    if not text:
        return ""
    # Убираем пунктуацию и пробелы, приводим к нижнему регистру
    normalized = re.sub(r"[^\w\s]", "", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text.strip()
    return text[: max_len - 3].rstrip() + "..."


def normalize_summary(text: str, max_chars: int = 200) -> str:
    normalized = strip_markdown(text)
    if len(normalized) <= max_chars:
        return normalized
    truncated = normalized[:max_chars].rsplit(" ", 1)[0]
    return truncated.rstrip(",:;") + "..."


def clean_bullet(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"^[-*]\s*", "", cleaned)
    cleaned = cleaned.replace("- [ ]", "").replace("- [x]", "").strip()
    cleaned = cleaned.replace("•", "").strip()
    cleaned = strip_markdown(cleaned)
    return cleaned


def build_topic_title(summary_text: str) -> str:
    clean_text = strip_markdown(summary_text)
    words = [w for w in re.split(r"\s+", clean_text) if w]
    if not words:
        return "Основная тема"
    max_words = min(max(len(words), 5), 8)
    title_words = words[:max_words]
    title = " ".join(title_words)
    title = title.strip().rstrip(":;,.")
    return title[:80].capitalize()


def build_claim_summary(message_text: str, fallback: str) -> str:
    candidate = normalize_summary(message_text, max_chars=240)
    if len(candidate.split()) >= 5:
        return candidate
    return normalize_summary(fallback, max_chars=240)


def prepare_conversation_text(
    messages: List[Dict[str, Any]],
    max_messages: int = 50,
    max_chars: int = 16000,
) -> str:
    """
    Подготовка текста разговора для саммаризации

    Args:
        messages: Список сообщений
        max_messages: Максимальное количество сообщений для включения

    Returns:
        Форматированный текст разговора
    """
    from .message import extract_message_text, format_author_name, message_key
    
    # Берём первые и последние сообщения, если слишком много
    selected_messages = messages
    if len(messages) > max_messages:
        # Берём первые 40% и последние 60%
        first_count = int(max_messages * 0.4)
        last_count = max_messages - first_count
        selected_messages = messages[:first_count] + messages[-last_count:]

    text_parts = []
    for _i, msg in enumerate(selected_messages):
        # Извлекаем информацию о сообщении
        text = msg.get("text", "").strip()
        if not text:
            continue

        # Извлекаем автора
        from_user = msg.get("from", {})
        if isinstance(from_user, dict):
            author = from_user.get("username") or from_user.get(
                "display", "Unknown"
            )
        else:
            author = str(from_user) if from_user else "Unknown"

        # Извлекаем время
        date_str = msg.get("date_utc") or msg.get("date", "")
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo

            if date_str:
                dt = parse_datetime_utc(date_str, use_zoneinfo=True)
                if dt:
                    time_str = dt.astimezone(ZoneInfo("Asia/Bangkok")).strftime("%H:%M")
                else:
                    time_str = "??:??"
            else:
                time_str = "??:??"
        except Exception:
            time_str = "??:??"

        # Ограничиваем длину текста
        if len(text) > 300:
            text = text[:300] + "..."

        # Добавляем информацию о дублях, если есть
        duplicate_info = ""
        if msg.get("_duplicate_marker"):
            dup_count = msg.get("_duplicate_count", 0)
            variants = msg.get("_duplicate_variants", [])

            if dup_count > 0:
                duplicate_info = f" [повторено {dup_count + 1}x"
                # Показываем вариации если они есть и отличаются
                if variants and len(variants) > 0:
                    # Проверяем, есть ли значимые различия
                    base_normalized = normalize_text_for_display(text)
                    has_variations = any(
                        normalize_text_for_display(v) != base_normalized
                        for v in variants
                    )
                    if has_variations:
                        duplicate_info += ", есть вариации"
                duplicate_info += "]"

        text_parts.append(f"[{time_str}] {author}: {text}{duplicate_info}")

    result = "\n".join(text_parts)

    # Ограничиваем общую длину текста для контроля размера промпта
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... (текст обрезан)"

    return result

