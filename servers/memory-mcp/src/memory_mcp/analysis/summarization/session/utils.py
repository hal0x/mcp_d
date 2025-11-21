#!/usr/bin/env python3
"""
Утилиты для session_summarizer
Объединяет функциональность из utils/action.py, utils/chat.py, utils/domain.py,
utils/message.py, utils/text.py, utils/topic.py
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from ....utils.processing.datetime_utils import format_datetime_display, parse_datetime_utc
from .constants import (
    CRYPTO_EXCHANGES,
    CRYPTO_TERMS,
    CRYPTO_TICKERS,
    GEOPOLITICS_PATTERNS,
    SCI_TECH_PATTERNS,
    SCI_TECH_TERMS,
)


# ============================================================================
# Из utils/text.py
# ============================================================================

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


# ============================================================================
# Из utils/message.py
# ============================================================================

def message_key(msg: Dict[str, Any], fallback_index: Optional[int] = None) -> str:
    """Генерирует стабильный ключ сообщения для привязки служебных данных."""
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


# ============================================================================
# Из utils/chat.py
# ============================================================================

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
    time_str = format_message_time(msg)
    author = format_author_name(msg)
    text = truncate_text((msg.get("text") or "").replace("\n", " "), 220)
    return f"{prefix}[{time_str}] {author}: {text}"


def derive_time_span(messages: List[Dict[str, Any]]) -> str:
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


# ============================================================================
# Из utils/topic.py
# ============================================================================

def split_messages_for_topics(
    message_map: List[Dict[str, Any]], topic_count: int
) -> List[List[Dict[str, Any]]]:
    if topic_count <= 0 or not message_map:
        return []
    if topic_count == 1:
        return [message_map]

    total = len(message_map)
    base = max(1, total // topic_count)
    segments: List[List[Dict[str, Any]]] = []
    start_idx = 0

    for i in range(topic_count):
        end_idx = start_idx + base
        if i == topic_count - 1:
            end_idx = total
        segment = message_map[start_idx:end_idx]
        if not segment and message_map:
            segment = [message_map[min(start_idx, total - 1)]]
        segments.append(segment)
        start_idx = end_idx

    return segments


def topic_time_span(
    segment: List[Dict[str, Any]], fallback_span: str
) -> str:
    if not segment:
        return fallback_span
    start = segment[0].get("ts_bkk")
    end = segment[-1].get("ts_bkk")

    try:
        start_dt = parse_datetime_utc(start, return_none_on_error=True, use_zoneinfo=True) if start else None
        end_dt = parse_datetime_utc(end, return_none_on_error=True, use_zoneinfo=True) if end else None
        if start_dt and end_dt:
            if start_dt.date() == end_dt.date():
                return f"{start_dt.strftime('%Y-%m-%d %H:%M')} – {end_dt.strftime('%H:%M')} BKK"
            return f"{start_dt.strftime('%Y-%m-%d %H:%M')} – {end_dt.strftime('%Y-%m-%d %H:%M')} BKK"
    except Exception:
        return fallback_span
    return fallback_span


def guess_topic_title(
    topics: List[Dict[str, Any]], text: Optional[str]
) -> str:
    if not topics:
        return ""
    if not text:
        return topics[0].get("title", "")
    lowered = text.lower()
    for topic in topics:
        summary_text = topic.get("summary", "").lower()
        if summary_text and lowered in summary_text or summary_text in lowered:
            return topic.get("title", "")
    return topics[0].get("title", "")


def build_minimal_topic(
    message_map: List[Dict[str, Any]],
    legacy_summary: Dict[str, Any],
    fallback_span: str,
) -> Dict[str, Any]:
    if message_map:
        anchor = message_map[0]
        summary_candidate = anchor.get("text", "") or legacy_summary.get(
            "context", "Краткий обзор"
        )
        message_id = anchor.get("id") or anchor.get("key")
        message_ids = [str(message_id)] if message_id is not None else []
        segment = [anchor]
    else:
        summary_candidate = legacy_summary.get("context", "Краткий обзор сессии")
        message_ids = []
        segment = []

    summary_text = normalize_summary(summary_candidate or "Краткий обзор")
    title = build_topic_title(summary_text)
    time_span = topic_time_span(segment, fallback_span)

    return {
        "title": title,
        "time_span": time_span,
        "message_ids": message_ids,
        "summary": summary_text,
    }


# ============================================================================
# Из utils/domain.py
# ============================================================================

def detect_domain_addons(keyed_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Определяет доменные аддоны и дополнительные метки для сообщений."""
    addons: set[str] = set()
    asset_tags: set[str] = set()
    geo_tags: set[str] = set()
    per_message: Dict[str, Dict[str, Any]] = {}

    for item in keyed_messages:
        key = item["key"]
        text = item["text"]
        lowered = text.lower()
        uppered = text.upper()

        entry: Dict[str, Any] = {}

        # --- Crypto ---
        ticker_hits = {
            ticker
            for ticker in CRYPTO_TICKERS
            if re.search(rf"\b{re.escape(ticker)}\b", uppered)
        }
        exchange_hits = {
            exch.upper() for exch in CRYPTO_EXCHANGES if exch in lowered
        }
        keyword_hit = any(term in lowered for term in CRYPTO_TERMS)

        if ticker_hits or exchange_hits or keyword_hit:
            addons.add("crypto")
            combined_tags = sorted(ticker_hits | exchange_hits)
            if combined_tags:
                entry["asset_tags"] = combined_tags
                asset_tags.update(combined_tags)

        # --- Sci-tech ---
        sci_term_hit = any(term in lowered for term in SCI_TECH_TERMS)
        sci_pattern_hit = any(pattern.search(text) for pattern in SCI_TECH_PATTERNS)
        if sci_term_hit or sci_pattern_hit:
            addons.add("sci-tech")
            entry["sci_markers"] = True

        # --- Geopolitics ---
        geo_hits = {
            label for pattern, label in GEOPOLITICS_PATTERNS if pattern.search(text)
        }
        if geo_hits:
            addons.add("geopolitics")
            geo_sorted = sorted(geo_hits)
            entry["geo_entities"] = geo_sorted
            geo_tags.update(geo_sorted)

        if entry:
            per_message[key] = entry

    return {
        "addons": addons,
        "asset_tags": sorted(asset_tags),
        "geo_entities": sorted(geo_tags),
        "by_key": per_message,
    }


def flatten_entities(entities: Dict[str, Any]) -> List[str]:
    if not entities:
        return []
    buckets = ["mentions", "tickers", "organizations", "people", "locations"]
    result = []
    for bucket in buckets:
        for item in entities.get(bucket, [])[:5]:
            value = item.get("value") if isinstance(item, dict) else str(item)
            if value and value not in result:
                result.append(value)
    return result[:20]


def build_attachments(artifacts: List[str]) -> List[str]:
    attachments = []
    for artifact in artifacts or []:
        if not artifact:
            continue
        if artifact.startswith("http"):
            attachments.append(f"link:{artifact.split()[0]}")
        elif artifact.startswith("📎"):
            attachments.append(f"doc:{artifact.replace('📎', '').strip()}")
        else:
            attachments.append(f"link:{artifact}")
    return attachments[:20]


def format_links_artifacts(entities: Dict[str, Any]) -> List[str]:
    """
    Форматирование ссылок и артефактов

    Args:
        entities: Извлечённые сущности

    Returns:
        Список ссылок и артефактов
    """
    artifacts: List[str] = []
    seen: set[str] = set()

    links = entities.get("links", []) or entities.get("urls", [])
    for link_info in links[:15]:
        raw = link_info.get("value", "")
        count = link_info.get("count", 1)
        normalized = normalize_attachment("link", raw, count=count)
        if normalized and normalized not in seen:
            artifacts.append(normalized)
            seen.add(normalized)

    files = entities.get("files", [])
    for file_info in files[:10]:
        raw = file_info.get("value", "")
        count = file_info.get("count", 1)
        normalized = normalize_attachment("doc", raw, count=count)
        if normalized and normalized not in seen:
            artifacts.append(normalized)
            seen.add(normalized)

    return artifacts


def normalize_attachment(
    kind: str, value: str, *, count: int = 1
) -> Optional[str]:
    if not value:
        return None
    clean_value = value.strip()
    clean_value = clean_value.strip('*_"')

    if kind == "link":
        clean_value = sanitize_url(clean_value)
        if not clean_value:
            return None
    else:
        clean_value = re.sub(r"\s+", " ", clean_value)

    suffix = f" ({count}x)" if count and count > 1 else ""
    return f"{kind}:{clean_value}{suffix}"


def sanitize_url(url: str) -> Optional[str]:
    candidate = url.strip()
    candidate = candidate.split()[0]
    candidate = candidate.rstrip(").,;")
    if candidate.startswith("["):
        candidate = candidate.strip("[]")
    if candidate.startswith("www."):
        candidate = f"https://{candidate}"

    parsed = urlparse(candidate)
    if not parsed.scheme:
        candidate = f"https://{candidate}"
        parsed = urlparse(candidate)
    if not parsed.netloc:
        return None
    normalized = parsed._replace(fragment="").geturl()
    return normalized


# ============================================================================
# Из utils/action.py
# ============================================================================

def parse_action_item(decision_text: str) -> Optional[Dict[str, Any]]:
    """
    Парсинг Action Item из текста решения

    Args:
        decision_text: Текст решения

    Returns:
        Словарь с данными Action Item или None
    """
    # Извлекаем компоненты
    item = {
        "text": decision_text,
        "confidence": 0.8,  # По умолчанию высокая уверенность из саммаризации
        "owner": None,
        "due": None,
        "priority": "P2",
    }

    # Ищем владельца (owner)
    owner_match = re.search(r"owner:\s*@?(\w+)", decision_text, re.IGNORECASE)
    if owner_match:
        item["owner"] = "@" + owner_match.group(1)

    # Ищем срок (due)
    due_match = re.search(
        r"due:\s*([0-9\-:T ]+(?:BKK|UTC)?)", decision_text, re.IGNORECASE
    )
    if due_match:
        item["due"] = due_match.group(1).strip()

    # Ищем приоритет
    pri_match = re.search(r"pri:\s*(P[123])", decision_text, re.IGNORECASE)
    if pri_match:
        item["priority"] = pri_match.group(1).upper()

    return item


async def extract_action_items(
    messages: List[Dict[str, Any]], summary_structure: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Извлечение Action Items из сессии

    Args:
        messages: Список сообщений
        summary_structure: Структура саммаризации

    Returns:
        Список Action Items с confidence >= 0.7
    """
    action_items = []

    # Берём решения из саммаризации
    decisions = summary_structure.get("decisions", [])

    for decision in decisions:
        # Парсим структуру решения
        item = parse_action_item(decision)
        if item and item.get("confidence", 0) >= 0.7:
            action_items.append(item)

    # Также ищем маркеры действий в последних сообщениях
    last_messages = messages[-10:] if len(messages) > 10 else messages
    for msg in last_messages:
        text = msg.get("text", "").lower()
        if any(
            marker in text
            for marker in [
                "решили",
                "next",
                "todo",
                "нужно",
                "надо",
                "следующий шаг",
            ]
        ):
            # Извлекаем как потенциальное действие
            item = {
                "text": msg.get("text", ""),
                "confidence": 0.75,
                "owner": None,
                "due": None,
                "priority": "P2",
            }
            action_items.append(item)

    return action_items[:5]  # Максимум 5 действий


def create_action_from_decision(
    decision: Dict[str, Any],
    summary: Dict[str, Any],
    aux_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not decision:
        return None
    text = decision.get("text") if isinstance(decision, dict) else str(decision)
    if not text:
        return None
    action = {
        "text": text,
        "owner": decision.get("owner") or decision.get("assignee"),
        "due_raw": decision.get("due_raw") or decision.get("due"),
        "due": decision.get("due"),
        "priority": decision.get("priority") or "normal",
        "status": decision.get("status") or "open",
    }
    msg_id = find_message_id_for_text(text, aux_data.get("messages", []))
    if msg_id:
        action["msg_id"] = msg_id
    action["topic_title"] = guess_topic_title(summary.get("topics", []), text)
    return action


def create_risk_entry(
    risk_entry: Any,
    summary: Dict[str, Any],
    aux_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not risk_entry:
        return None
    if isinstance(risk_entry, dict):
        text = risk_entry.get("text") or risk_entry.get("description", "")
        likelihood = risk_entry.get("likelihood") or "medium"
        impact = risk_entry.get("impact") or "medium"
        mitigation = risk_entry.get("mitigation")
    else:
        text = str(risk_entry)
        likelihood = "medium"
        impact = "medium"
        mitigation = None
    if not text:
        return None
    risk = {
        "text": text,
        "likelihood": likelihood,
        "impact": impact,
    }
    if mitigation:
        risk["mitigation"] = mitigation
    msg_id = find_message_id_for_text(text, aux_data.get("messages", []))
    if msg_id:
        risk["msg_id"] = msg_id
    risk["topic_title"] = guess_topic_title(summary.get("topics", []), text)
    return risk


def derive_rationale(
    profile: str, actions: List[Dict[str, Any]], risks: List[Dict[str, Any]]
) -> str:
    if profile == "broadcast":
        if not actions:
            return "news_channel_no_actions"
        if not risks:
            return "no_risks_detected"
        return "author_opinion_no_tasks"
    else:
        if actions:
            return "project_session_with_actions"
        if not risks:
            return "no_risks_detected"
        return "threads_not_applicable"


def is_small_session(
    messages_total: int, topics: List[Dict[str, Any]]
) -> bool:
    return messages_total < 5 or len(topics) < 2


def apply_small_session_policy(
    topics: List[Dict[str, Any]],
    claims: List[Dict[str, Any]],
    discussion: List[Dict[str, Any]],
    actions: List[Dict[str, Any]],
    risks: List[Dict[str, Any]],
    rationale: str,
    message_map: List[Dict[str, Any]],
    legacy_summary: Dict[str, Any],
    fallback_span: str,
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    str,
    Dict[str, Any],
]:
    policy_flags = ["small_session"]

    normalized_topics = topics[:1]
    if not normalized_topics:
        normalized_topics = [
            build_minimal_topic(message_map, legacy_summary, fallback_span)
        ]
    else:
        primary_topic = normalized_topics[0]
        if not primary_topic.get("message_ids") and message_map:
            anchor = message_map[0]
            msg_identifier = anchor.get("id") or anchor.get("key")
            if msg_identifier is not None:
                primary_topic["message_ids"] = [str(msg_identifier)]
        if not primary_topic.get("time_span"):
            primary_topic["time_span"] = topic_time_span(
                message_map[:1], fallback_span
            )

    primary_topic_title = normalized_topics[0].get("title", "Краткая сессия")

    normalized_claims: List[Dict[str, Any]] = []
    for claim in claims[:3]:
        claim_copy = dict(claim)
        claim_copy["topic_title"] = primary_topic_title
        normalized_claims.append(claim_copy)

    normalized_discussion = discussion[:3]

    rationale_override = "insufficient_evidence"

    policy_info = {
        "score_cap": 60.0,
        "policy_flags": policy_flags,
    }

    return (
        normalized_topics,
        normalized_claims,
        normalized_discussion,
        [],
        [],
        rationale_override,
        policy_info,
    )


