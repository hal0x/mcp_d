#!/usr/bin/env python3
"""
Fallback структуры для session_summarizer
"""

from typing import Any, Dict, List

from ..utils.session_summarizer_chat_utils import (
    collect_participants,
    detect_chat_mode,
    select_key_messages,
    select_messages_with_keywords,
)
from ..utils.session_summarizer_message_utils import format_message_bullet
from ..utils.session_summarizer_text_utils import truncate_text


def build_fallback_structure(
    messages: List[Dict[str, Any]], chat: str, chat_mode: str
) -> Dict[str, Any]:
    """
    Формирует эвристическую саммаризацию, если LLM дал пустой результат.
    """
    valid_messages = [m for m in messages if (m.get("text") or "").strip()]

    if not valid_messages:
        if chat_mode == "channel":
            return {
                "context": f"Автосаммаризация (fallback) для канала {chat}: сообщений не найдено.",
                "key_points": ["- Данных для анализа не обнаружено"],
                "important_items": ["- Данных для анализа не обнаружено"],
                "discussion": ["- Данных для анализа не обнаружено"],
                "decisions": [],
                "risks": [
                    "- Автопроверка: риски не обнаружены; требуется ручная проверка."
                ],
            }
        else:
            return {
                "context": f"Автосаммаризация (fallback) для чата {chat}: сообщений не найдено.",
                "discussion": ["- Данных для анализа не обнаружено"],
                "decisions": [
                    "- [ ] Автопроверка: действия не обнаружены; требуется ручная проверка."
                ],
                "risks": [
                    "- Автопроверка: риски не обнаружены; требуется ручная проверка."
                ],
            }

    participants = collect_participants(valid_messages)
    first_text = truncate_text(valid_messages[0].get("text", ""), 220)
    last_text = truncate_text(valid_messages[-1].get("text", ""), 220)
    context_lines = [
        f'Автосаммаризация (fallback) по чату "{chat}". Сообщений: {len(valid_messages)}.',
    ]
    if participants:
        context_lines.append(f"Активные участники: {', '.join(participants)}")
    if first_text:
        context_lines.append(f"Старт обсуждения: {first_text}")
    if last_text and last_text != first_text:
        context_lines.append(f"Финал обсуждения: {last_text}")

    # Для канала больше упор на ключевые публикации, для группы — на реплики
    discussion_limit = 5 if chat_mode == "channel" else 4
    discussion_msgs = select_key_messages(
        valid_messages, limit=discussion_limit
    )
    if len(discussion_msgs) < 2:
        discussion_msgs = valid_messages[: min(2, len(valid_messages))]
    discussion_lines = [
        format_message_bullet(msg, prefix="- ") for msg in discussion_msgs
    ]

    decision_msgs = select_messages_with_keywords(
        valid_messages,
        keywords=[
            "нужно",
            "надо",
            "должн",
            "давайте",
            "решим",
            "todo",
            "should",
            "must",
            "plan",
        ],
        limit=3,
    )
    if chat_mode == "channel":
        # В каналах решения обычно неуместны — делаем пустой список или заметку
        decision_lines = (
            []
            if decision_msgs == []
            else [
                format_message_bullet(m, prefix="- ") for m in decision_msgs
            ]
        )
    elif decision_msgs:
        decision_lines = [
            format_message_bullet(msg, prefix="- [ ] ")
            for msg in decision_msgs
        ]
    else:
        decision_lines = [
            "- [ ] Автопроверка: явных действий не зафиксировано; требуется ручная проверка."
        ]

    risk_msgs = select_messages_with_keywords(
        valid_messages,
        keywords=["риск", "проблем", "опас", "сомн", "issue", "блок", "concern"],
        limit=3,
    )
    if risk_msgs:
        risk_lines = [
            format_message_bullet(msg, prefix="- ") for msg in risk_msgs
        ]
    else:
        risk_lines = [
            "- Автопроверка: явных рисков в сообщениях не найдено; проверить вручную."
        ]

    # Для каналов добавляем ключевые тезисы и важные моменты
    if chat_mode == "channel":
        # Извлекаем ключевые тезисы из первых сообщений
        key_points = []
        for msg in valid_messages[:3]:
            text = truncate_text(msg.get("text", ""), 100)
            if text:
                key_points.append(f"- {text}")

        # Извлекаем важные моменты (ссылки, даты, имена, ключевые слова)
        important_items = []
        important_keywords = [
            "важно",
            "критично",
            "срочно",
            "внимание",
            "attention",
            "important",
            "critical",
            "urgent",
            "required",
            "must",
            "should",
            "update",
            "upgrade",
            "vote",
            "voting",
            "action required",
            "mandatory",
            "scheduled",
            "deadline",
            "breaking",
            "announcement",
        ]

        for msg in valid_messages:
            text = msg.get("text", "")
            text_lower = text.lower()

            # Проверяем ключевые слова
            if any(keyword in text_lower for keyword in important_keywords):
                important_items.append(f"- {truncate_text(text, 80)}")
            # Также добавляем сообщения с датами и временем (часто важные объявления)
            elif any(
                pattern in text
                for pattern in ["UTC", "GMT", "at ", "on ", "2024", "2025"]
            ):
                important_items.append(f"- {truncate_text(text, 80)}")

            if len(important_items) >= 3:
                break

        # Если ничего не найдено, берем первые сообщения как важные
        if not important_items:
            for msg in valid_messages[:2]:
                text = truncate_text(msg.get("text", ""), 80)
                if text:
                    important_items.append(f"- {text}")

        # Если все еще пусто, добавляем заметку
        if not important_items:
            important_items = [
                "- Автопроверка: важные моменты не выделены; требуется ручная проверка."
            ]

        return {
            "context": "\n".join(context_lines),
            "key_points": key_points[:5],
            "important_items": important_items[:5],
            "discussion": discussion_lines,
            "decisions": decision_lines,
            "risks": risk_lines,
        }
    else:
        return {
            "context": "\n".join(context_lines),
            "discussion": discussion_lines,
            "decisions": decision_lines,
            "risks": risk_lines,
        }

