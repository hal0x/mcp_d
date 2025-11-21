#!/usr/bin/env python3
"""
Проверка полноты саммаризаций и дополнение fallback данными для session_summarizer
"""

from typing import Any, Dict, List, Tuple

from ..utils.session_summarizer_chat_utils import detect_chat_mode
from .session_summarizer_prompt_fallback import build_fallback_structure


def ensure_summary_completeness(
    messages: List[Dict[str, Any]],
    chat: str,
    structure: Dict[str, Any],
    strict_mode: bool = False,
) -> Tuple[Dict[str, Any], bool]:
    """
    Проверяет и при необходимости дополняет саммаризацию эвристическими данными.

    Args:
        messages: Исходные сообщения сессии
        chat: Название чата
        structure: Структура, полученная от LLM
        strict_mode: Если True, выбрасывает исключение вместо fallback

    Returns:
        (дополненная структура, признак использования fallback)
    """
    # Безопасное извлечение context (может быть строкой или списком)
    context_raw = structure.get("context") or ""
    if isinstance(context_raw, list):
        context_text = "\n".join(context_raw).strip()
    else:
        context_text = context_raw.strip()

    key_points = structure.get("key_points") or []
    important_items = structure.get("important_items") or []
    discussion = structure.get("discussion") or []
    decisions = structure.get("decisions") or []
    risks = structure.get("risks") or []

    # Определяем тип чата
    chat_mode = detect_chat_mode(messages)

    needs_context = len(context_text) < 40
    needs_discussion = len(discussion) < 2
    needs_decisions = len(decisions) == 0
    needs_risks = len(risks) == 0

    # Для каналов всегда заполняем ключевые тезисы и важные моменты
    if chat_mode == "channel":
        needs_key_points = len(key_points) == 0
        needs_important_items = len(important_items) == 0
    else:
        needs_key_points = False
        needs_important_items = False

    if not any(
        [
            needs_context,
            needs_key_points,
            needs_important_items,
            needs_discussion,
            needs_decisions,
            needs_risks,
        ]
    ):
        return structure, False

    if strict_mode:
        raise RuntimeError(
            f"LLM вернул пустую или неполную структуру саммаризации для чата '{chat}'. "
            "Проверьте конфигурацию LLM клиента."
        )

    fallback = build_fallback_structure(messages, chat, chat_mode)
    patched = dict(structure)

    if needs_context:
        patched["context"] = fallback["context"]
    if needs_key_points:
        patched["key_points"] = fallback.get("key_points", [])
    if needs_important_items:
        patched["important_items"] = fallback.get("important_items", [])
    if needs_discussion:
        patched["discussion"] = fallback["discussion"]
    if needs_decisions:
        patched["decisions"] = fallback["decisions"]
    if needs_risks:
        patched["risks"] = fallback["risks"]

    return patched, True

