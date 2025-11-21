#!/usr/bin/env python3
"""
Утилиты для работы с чатами и участниками для session_summarizer
"""

from collections import Counter
from typing import Any, Dict, List

from ....utils.datetime_utils import parse_datetime_utc


def detect_chat_mode(messages: List[Dict[str, Any]]) -> str:
    """
    Определяет тип чата: 'channel' или 'group'.

    Логика:
    - Если у большинства сообщений нет поля 'from' или оно None → канал
    - Если есть много разных авторов → группа
    - Если один автор доминирует → канал
    """
    if not messages:
        return "group"

    # Подсчитываем сообщения без автора (характерно для каналов)
    messages_without_author = 0
    authors = []

    for m in messages:
        fr = m.get("from")
        if fr is None or not fr:
            messages_without_author += 1
        else:
            name = (
                fr.get("username") or fr.get("display") or fr.get("id") or "unknown"
            )
            authors.append(str(name))

    total_messages = len(messages)
    messages_with_author = total_messages - messages_without_author

    # Если больше 70% сообщений без автора → канал
    if messages_without_author / total_messages > 0.7:
        return "channel"

    # Если сообщений с авторами мало, но они есть
    if messages_with_author < 5:
        return "group"

    # Анализируем авторов
    total = len([a for a in authors if a != "unknown"])
    if total == 0:
        return "group"

    cnt = Counter(a for a in authors if a != "unknown")
    top, top_count = cnt.most_common(1)[0]
    top_share = top_count / total
    unique = len(cnt)

    # Канал, если автор один/почти один, и сообщений достаточно
    if (top_share >= 0.85 and unique <= 3 and total >= 5) or unique == 1:
        return "channel"

    return "group"


def map_profile(chat_mode: str, messages: List[Dict[str, Any]]) -> str:
    unique_authors = set()
    for msg in messages:
        author = msg.get("from") or {}
        username = (
            author.get("username") or author.get("display") or author.get("id")
        )
        if username:
            unique_authors.add(str(username))
    if len(unique_authors) <= 1 or chat_mode == "channel":
        return "broadcast"
    return "group-project"


def collect_participants(messages: List[Dict[str, Any]]) -> List[str]:
    participants = Counter()
    for msg in messages:
        author = msg.get("from") or {}
        display = author.get("display") or author.get("username")
        if not display:
            user_id = author.get("id") if isinstance(author, dict) else None
            display = f"user_{user_id}" if user_id else "unknown"
        participants[display] += 1
    top = [name for name, _ in participants.most_common(5)]
    return top


def select_key_messages(
    messages: List[Dict[str, Any]], limit: int
) -> List[Dict[str, Any]]:
    scored: List[tuple[float, Dict[str, Any]]] = []
    for msg in messages:
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        score = len(text)
        for reaction in msg.get("reactions", []) or []:
            score += reaction.get("count", 0) * 5
        scored.append((score, msg))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_msgs = [msg for _, msg in scored[:limit]]
    top_msgs.sort(key=lambda m: (m.get("date_utc") or m.get("date") or ""))
    return top_msgs


def select_messages_with_keywords(
    messages: List[Dict[str, Any]], keywords: List[str], limit: int
) -> List[Dict[str, Any]]:
    matched = []
    keywords_lower = [kw.lower() for kw in keywords]

    for msg in messages:
        text = (msg.get("text") or "").lower()
        if not text:
            continue
        if any(kw in text for kw in keywords_lower):
            matched.append(msg)

    matched.sort(key=lambda m: (m.get("date_utc") or m.get("date") or ""))
    return matched[:limit]


def format_message_bullet(msg: Dict[str, Any], prefix: str = "- ") -> str:
    from .message import format_message_time, format_author_name
    from .text import truncate_text
    
    time_str = format_message_time(msg)
    author = format_author_name(msg)
    text = truncate_text((msg.get("text") or "").replace("\n", " "), 220)
    return f"{prefix}[{time_str}] {author}: {text}"


def derive_time_span(messages: List[Dict[str, Any]]) -> str:
    from .message import build_message_envelope
    
    if not messages:
        return ""
    envelopes = [
        build_message_envelope(msg, fallback_index=idx)
        for idx, msg in enumerate(messages)
        if msg.get("text")
    ]
    if not envelopes:
        if messages:
            envelopes = [
                build_message_envelope(messages[0], fallback_index=0),
                build_message_envelope(
                    messages[-1], fallback_index=len(messages) - 1
                ),
            ]
    start = envelopes[0].get("ts_bkk")
    end = envelopes[-1].get("ts_bkk")
    if not start or not end:
        return ""

    try:
        start_dt = parse_datetime_utc(start, return_none_on_error=True, use_zoneinfo=True)
        end_dt = parse_datetime_utc(end, return_none_on_error=True, use_zoneinfo=True)
        if start_dt and end_dt:
            return f"{start_dt.strftime('%Y-%m-%d %H:%M')} – {end_dt.strftime('%H:%M')} BKK"
        return ""
    except Exception:
        return ""

