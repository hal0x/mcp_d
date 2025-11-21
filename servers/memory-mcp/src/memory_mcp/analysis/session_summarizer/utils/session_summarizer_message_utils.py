#!/usr/bin/env python3
"""
Утилиты для работы с сообщениями для session_summarizer
"""

from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from ...utils.datetime_utils import format_datetime_display, parse_datetime_utc


def message_key(msg: Dict[str, Any], fallback_index: Optional[int] = None) -> str:
    """Генерирует стабильный ключ сообщения для привязки служебных данных."""
    from .session_summarizer_text_utils import extract_message_text
    
    candidate = msg.get("id")
    if candidate is None:
        candidate = msg.get("message_id") or msg.get("msg_id")
    if candidate is None and fallback_index is not None:
        candidate = f"auto_{fallback_index:05d}"
    if candidate is None:
        text = extract_message_text(msg)
        candidate = f"auto_{abs(hash((msg.get('date_utc'), text))) % 10**8:08d}"
    return str(candidate)


def extract_message_text(msg: Dict[str, Any]) -> str:
    """Возвращает текст сообщения с учётом форматированных структур Telegram."""
    text = msg.get("text", "")
    if isinstance(text, list):
        parts: List[str] = []
        for part in text:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        text = "".join(parts)
    return text if isinstance(text, str) else str(text)


def format_author_name(msg: Dict[str, Any]) -> str:
    author = msg.get("from") or {}
    return author.get("display") or author.get("username") or "автор"


def format_message_time(msg: Dict[str, Any]) -> str:
    """Форматирует время сообщения для отображения."""
    date = msg.get("date_utc") or msg.get("date")
    return format_datetime_display(date, format_type="time", fallback="??:??")


def build_message_envelope(
    msg: Dict[str, Any],
    fallback_index: Optional[int] = None,
    msg_key: Optional[str] = None,
) -> Dict[str, Any]:
    original_ts = msg.get("date_utc") or msg.get("date") or ""

    try:
        dt_utc = parse_datetime_utc(original_ts, return_none_on_error=True, use_zoneinfo=True)
    except Exception:
        dt_utc = None

    dt_bkk = dt_utc.astimezone(ZoneInfo("Asia/Bangkok")) if dt_utc else None

    key = msg_key or message_key(msg, fallback_index)

    return {
        "id": msg.get("id"),
        "key": key,
        "ts_utc": dt_utc.isoformat() if dt_utc else "",
        "ts_bkk": dt_bkk.isoformat() if dt_bkk else "",
        "author": format_author_name(msg),
        "text": (msg.get("text") or "").strip(),
        "raw": msg,
    }


def message_identifier(envelope: Dict[str, Any]) -> Optional[str]:
    if envelope.get("id") is not None:
        return str(envelope.get("id"))
    if envelope.get("key") is not None:
        return str(envelope.get("key"))
    return None


def lookup_message_envelope(
    message_id: str, aux_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    index_map = aux_data.get("message_index_map", {})
    message_map = aux_data.get("message_map", [])
    key = str(message_id)
    idx = index_map.get(key)
    if idx is None or idx < 0 or idx >= len(message_map):
        return None
    return message_map[idx]


def expand_message_ids(
    seed_ids: List[str],
    aux_data: Dict[str, Any],
    target_count: int,
) -> List[str]:
    message_map = aux_data.get("message_map", [])
    index_map = aux_data.get("message_index_map", {})
    if not message_map:
        return seed_ids

    indices = [
        index_map.get(mid) for mid in seed_ids if index_map.get(mid) is not None
    ]
    if not indices:
        return seed_ids

    left = min(indices)
    right = max(indices)
    while (right - left + 1) < target_count and (
        left > 0 or right < len(message_map) - 1
    ):
        if left > 0:
            left -= 1
        if (right - left + 1) >= target_count:
            break
        if right < len(message_map) - 1:
            right += 1

    expanded: List[str] = []
    for idx in range(left, right + 1):
        identifier = message_identifier(message_map[idx])
        if identifier:
            expanded.append(identifier)
        if len(expanded) >= target_count:
            break

    return expanded


def collect_segment_by_ids(
    message_ids: List[str], aux_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    message_map = aux_data.get("message_map", [])
    index_map = aux_data.get("message_index_map", {})
    segment: List[Dict[str, Any]] = []
    for mid in message_ids:
        idx = index_map.get(mid)
        if idx is not None and 0 <= idx < len(message_map):
            segment.append(message_map[idx])
    return segment


def find_message_id_for_text(
    text: str, messages: List[Dict[str, Any]]
) -> Optional[str]:
    from .session_summarizer_text_utils import strip_markdown
    
    plain = strip_markdown(text).lower()
    if not plain:
        return None
    snippet = plain[:40]
    for idx, msg in enumerate(messages):
        body = strip_markdown(msg.get("text", "")).lower()
        if snippet and snippet in body:
            msg_id = msg.get("id")
            if msg_id is not None:
                return str(msg_id)
            return message_key(msg, fallback_index=idx)
    return None

