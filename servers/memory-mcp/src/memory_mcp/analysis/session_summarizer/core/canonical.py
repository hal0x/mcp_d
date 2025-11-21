#!/usr/bin/env python3
"""
Методы построения канонической саммаризации для session_summarizer
"""

from typing import Any, Dict, List, Optional, Tuple

from ...utils.naming import slugify
from ...session_summarizer_constants import SESSION_SUMMARY_VERSION
from .generation import (
    generate_actions,
    generate_claims,
    generate_discussion,
    generate_risks,
    generate_topics,
)
from .quality import refresh_quality
from ..utils import (
    apply_small_session_policy,
    build_attachments,
    build_message_envelope,
    derive_rationale,
    derive_time_span,
    detect_domain_addons,
    extract_message_text,
    flatten_entities,
    format_links_artifacts,
    is_small_session,
    message_key,
)


def build_canonical_summary(
    session: Dict[str, Any],
    messages: List[Dict[str, Any]],
    legacy_summary: Dict[str, Any],
    quality_metrics,
    chat_mode: str,
    profile: str,
    entities: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Формирует каноническую структуру саммаризации (v1)."""

    keyed_messages = []
    for idx, msg in enumerate(messages):
        text_value = extract_message_text(msg)
        if not text_value or not text_value.strip():
            continue
        key = message_key(msg, fallback_index=idx)
        keyed_messages.append(
            {"index": idx, "key": key, "message": msg, "text": text_value}
        )

    valid_messages = [item["message"] for item in keyed_messages]
    message_map = [
        build_message_envelope(
            item["message"], fallback_index=item["index"], msg_key=item["key"]
        )
        for item in keyed_messages
    ]

    chat_name = legacy_summary.get("chat", session.get("chat"))
    chat_id = session.get("chat_id") or slugify(chat_name)
    time_span = legacy_summary.get("time_range_bkk") or derive_time_span(
        messages
    )
    messages_total = len(messages)
    participants = legacy_summary.get("participants", [])
    dominant_language = legacy_summary.get("dominant_language") or session.get(
        "dominant_language"
    )

    addons_info = detect_domain_addons(keyed_messages)

    topics = generate_topics(profile, legacy_summary, message_map, time_span)
    claims = generate_claims(
        profile, topics, message_map, entities, addons_info
    )
    discussion = generate_discussion(profile, message_map)
    actions = generate_actions(
        topics, legacy_summary.get("decisions_next", []), valid_messages
    )
    risks = generate_risks(
        topics, legacy_summary.get("risks_open", []), valid_messages
    )
    uncertainties = []
    entities_flat = flatten_entities(entities)
    attachments = build_attachments(legacy_summary.get("links_artifacts", []))
    rationale = derive_rationale(profile, actions, risks)

    aux_data: Dict[str, Any] = {
        "profile": profile,
        "messages_total": messages_total,
        "messages": messages,
        "message_map": message_map,
        "addons_info": addons_info,
        "legacy_summary": legacy_summary,
        "legacy_metrics": quality_metrics,
        "legacy_decisions": legacy_summary.get("decisions_next", []),
        "legacy_risks": legacy_summary.get("risks_open", []),
        "entities_flat": entities_flat,
        "time_span_default": time_span,
    }

    message_index_map: Dict[str, int] = {}
    for idx, item in enumerate(message_map):
        identifiers = []
        if item.get("id") is not None:
            identifiers.append(str(item.get("id")))
        if item.get("key") is not None:
            identifiers.append(str(item.get("key")))
        for identifier in identifiers:
            if identifier:
                message_index_map.setdefault(identifier, idx)
    aux_data["message_index_map"] = message_index_map

    small_session_info = None
    if is_small_session(messages_total, topics):
        (
            topics,
            claims,
            discussion,
            actions,
            risks,
            rationale,
            small_session_info,
        ) = apply_small_session_policy(
            topics=topics,
            claims=claims,
            discussion=discussion,
            actions=actions,
            risks=risks,
            rationale=rationale,
            message_map=message_map,
            legacy_summary=legacy_summary,
            fallback_span=time_span,
        )
    if small_session_info:
        aux_data["small_session_info"] = small_session_info

    meta = {
        "chat_name": chat_name,
        "profile": profile,
        "time_span": time_span,
        "messages_total": messages_total,
        "participants": participants,
        "dominant_language": dominant_language,
        "chat_mode": chat_mode,
        "start_time_utc": session.get("start_time_utc", ""),
        "end_time_utc": session.get("end_time_utc", ""),
        "addons": sorted(addons_info.get("addons", [])),
    }
    if small_session_info:
        meta["policy_flags"] = small_session_info.get("policy_flags", [])

    summary = {
        "version": SESSION_SUMMARY_VERSION,
        "chat_id": chat_id,
        "session_id": legacy_summary["session_id"],
        "meta": meta,
        "topics": topics,
        "claims": claims,
        "discussion": discussion,
        "actions": actions,
        "risks": risks,
        "uncertainties": uncertainties,
        "entities": entities_flat,
        "attachments": attachments,
        "rationale": rationale,
    }

    quality_context = refresh_quality(summary, aux_data)
    aux_data.update(quality_context)

    return summary, aux_data

