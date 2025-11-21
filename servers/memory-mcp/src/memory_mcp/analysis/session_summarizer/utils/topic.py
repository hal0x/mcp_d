#!/usr/bin/env python3
"""
Утилиты для работы с топиками для session_summarizer
"""

from typing import Any, Dict, List, Optional

from ...utils.datetime_utils import parse_datetime_utc
from .message import message_identifier
from .text import (
    build_topic_title,
    normalize_summary,
    strip_markdown,
)


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

