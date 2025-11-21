#!/usr/bin/env python3
"""
Методы генерации для session_summarizer (topics, claims, discussion, actions, risks)
"""

import re
from typing import Any, Dict, List, Optional

from ..utils.session_summarizer_utils import (
    build_claim_summary,
    build_message_envelope,
    build_topic_title,
    clean_bullet,
    find_message_id_for_text,
    flatten_entities,
    message_identifier,
    normalize_summary,
    select_key_messages,
    split_messages_for_topics,
    topic_time_span,
    truncate_text,
)


def generate_topics(
    profile: str,
    legacy_summary: Dict[str, Any],
    message_map: List[Dict[str, Any]],
    fallback_span: str,
) -> List[Dict[str, Any]]:
    if profile == "broadcast":
        source = (
            legacy_summary.get("key_points")
            or legacy_summary.get("discussion")
            or []
        )
    else:
        source = (
            legacy_summary.get("discussion")
            or legacy_summary.get("key_points")
            or []
        )

    cleaned = []
    for item in source:
        normalized = clean_bullet(item)
        if normalized:
            cleaned.append(normalized)

    if len(cleaned) < 2:
        context_sentences = re.split(
            r"[\.?!]\s+", legacy_summary.get("context", "")
        )
        for sentence in context_sentences:
            from .session_summarizer_utils import strip_markdown
            sentence = strip_markdown(sentence).strip()
            if len(sentence.split()) >= 4 and sentence not in cleaned:
                cleaned.append(sentence)
            if len(cleaned) >= 2:
                break

    cleaned = cleaned[:6]
    if not cleaned:
        cleaned = [legacy_summary.get("context", "") or "Основная тема сессии"]

    segments = split_messages_for_topics(message_map, len(cleaned))
    topics = []

    for idx, text in enumerate(cleaned):
        segment = segments[idx] if idx < len(segments) else []
        message_ids = []
        for item in segment:
            message_id = item.get("id")
            if message_id is None:
                message_id = item.get("key")
            if message_id is not None:
                message_ids.append(str(message_id))
        span = topic_time_span(segment, fallback_span)
        title = build_topic_title(text)
        summary_text = normalize_summary(text)
        topics.append(
            {
                "title": title,
                "time_span": span,
                "message_ids": message_ids,
                "summary": summary_text,
            }
        )

    return topics


def generate_claims(
    profile: str,
    topics: List[Dict[str, Any]],
    message_map: List[Dict[str, Any]],
    entities: Dict[str, Any],
    addons_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not topics:
        return []

    entity_candidates = flatten_entities(entities)[:4]
    claims: List[Dict[str, Any]] = []
    modality = "media" if profile == "broadcast" else "internal"
    source = modality

    per_message_addons = addons_info.get("by_key", {}) if addons_info else {}
    global_asset_tags = addons_info.get("asset_tags", []) if addons_info else []
    global_geo_tags = addons_info.get("geo_entities", []) if addons_info else []
    addons_set = addons_info.get("addons", set()) if addons_info else set()

    for topic in topics:
        linked_messages = [
            item
            for item in message_map
            if str(item.get("id")) in topic.get("message_ids", [])
        ]
        if not linked_messages:
            anchor = message_map[0] if message_map else None
            if not anchor:
                continue
            summary_text = build_claim_summary(
                anchor.get("text", ""), topic.get("summary", "")
            )
            summary_text = summary_text[:160]  # Ограничение для избежания штрафа
            claim = {
                "ts": anchor.get("ts_bkk", ""),
                "source": source,
                "modality": modality,
                "credibility": "medium",
                "entities": entity_candidates,
                "summary": summary_text,
                "msg_id": str(anchor.get("id"))
                if anchor.get("id") is not None
                else "",
                "topic_title": topic.get("title"),
            }
            apply_addon_metadata_to_claim(
                claim,
                per_message_addons.get(anchor.get("key", "")),
                addons_set,
                global_asset_tags,
                global_geo_tags,
            )
            claims.append(claim)
            continue

        for item in linked_messages[:3]:
            summary_text = build_claim_summary(
                item.get("text", ""), topic.get("summary", "")
            )
            if not summary_text:
                continue
            summary_text = summary_text[:160]  # Ограничение для избежания штрафа
            claim = {
                "ts": item.get("ts_bkk", ""),
                "source": source,
                "modality": modality,
                "credibility": "medium",
                "entities": entity_candidates,
                "summary": summary_text,
                "msg_id": str(item.get("id")) if item.get("id") is not None else "",
                "topic_title": topic.get("title"),
            }
            apply_addon_metadata_to_claim(
                claim,
                per_message_addons.get(item.get("key", "")),
                addons_set,
                global_asset_tags,
                global_geo_tags,
            )
            claims.append(claim)
            if len(claims) >= 20:
                return claims

    credibility_order = {"high": 0, "medium": 1, "low": 2}
    sorted_claims = sorted(
        claims,
        key=lambda item: (
            credibility_order.get(item.get("credibility", "medium"), 1),
            item.get("ts", ""),
        ),
    )
    return sorted_claims[:20]


def apply_addon_metadata_to_claim(
    claim: Dict[str, Any],
    message_addon: Optional[Dict[str, Any]],
    active_addons: set,
    global_asset_tags: List[str],
    global_geo_tags: List[str],
) -> None:
    """Обогащает claim доменными метаданными, если они доступны."""

    if message_addon:
        asset_tags = message_addon.get("asset_tags")
        if asset_tags:
            claim["asset_tags"] = asset_tags

        geo_entities = message_addon.get("geo_entities")
        if geo_entities:
            claim["geo_scope"] = geo_entities

        if message_addon.get("sci_markers"):
            claim["field"] = "sci-tech"

    # Если у конкретного сообщения нет тегов, но есть глобальные — добавляем мягко
    if "asset_tags" not in claim and global_asset_tags:
        claim["asset_tags"] = global_asset_tags

    if "geo_scope" not in claim and global_geo_tags:
        claim["geo_scope"] = global_geo_tags

    if "field" not in claim and "sci-tech" in active_addons:
        claim["field"] = "sci-tech"


def generate_discussion(
    profile: str,
    message_map: List[Dict[str, Any]],
    limit_override: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not message_map:
        return []

    limit = limit_override or (6 if profile == "broadcast" else 8)
    min_items = 3 if profile == "broadcast" else 4
    key_messages = select_key_messages(
        [item["raw"] for item in message_map], limit=limit
    )

    timeline = []
    for idx, msg in enumerate(key_messages[:limit]):
        envelope = build_message_envelope(msg, fallback_index=idx)
        quote = truncate_text(envelope["text"].replace("\n", " "), 220)
        if not quote:
            continue
        msg_identifier = envelope.get("id")
        if msg_identifier is None:
            msg_identifier = envelope.get("key")
        timeline.append(
            {
                "ts": envelope.get("ts_bkk", ""),
                "author": envelope.get("author", ""),
                "msg_id": str(msg_identifier) if msg_identifier is not None else "",
                "quote": quote,
            }
        )

    if len(timeline) < min_items and message_map:
        filler = message_map[:min_items]
        for envelope in filler:
            quote = truncate_text(envelope["text"], 220)
            if not quote:
                continue
            msg_identifier = envelope.get("id")
            if msg_identifier is None:
                msg_identifier = envelope.get("key")
            timeline.append(
                {
                    "ts": envelope.get("ts_bkk", ""),
                    "author": envelope.get("author", ""),
                    "msg_id": str(msg_identifier)
                    if msg_identifier is not None
                    else "",
                    "quote": quote,
                }
            )

    # Deduplicate by msg_id
    seen = set()
    deduped = []
    for item in timeline:
        key = item.get("msg_id"), item.get("quote")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped[:limit]


def generate_actions(
    topics: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    actions = []
    topic_title = topics[0]["title"] if topics else ""
    for decision in decisions:
        text = decision.get("text", "").strip()
        if not text:
            continue
        msg_id = find_message_id_for_text(text, messages)
        status = "done" if text.lower().startswith("- [x]") else "open"
        priority_raw = (decision.get("priority") or "").upper()
        priority_map = {"P1": "high", "P2": "normal", "P3": "low"}
        priority = priority_map.get(
            priority_raw, priority_raw.lower() if priority_raw else "normal"
        )
        actions.append(
            {
                "text": clean_bullet(text),
                "owner": decision.get("owner"),
                "due_raw": decision.get("due"),
                "due": decision.get("due"),
                "priority": priority,
                "status": status,
                "msg_id": msg_id,
                "topic_title": topic_title,
            }
        )

    return actions


def generate_risks(
    topics: List[Dict[str, Any]],
    risks_text: List[str],
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    risks = []
    topic_title = topics[0]["title"] if topics else ""
    for risk in risks_text:
        cleaned = clean_bullet(risk)
        if not cleaned:
            continue
        msg_id = find_message_id_for_text(cleaned, messages)
        risks.append(
            {
                "text": cleaned,
                "likelihood": None,
                "impact": None,
                "mitigation": None,
                "msg_id": msg_id,
                "topic_title": topic_title,
            }
        )
    return risks

