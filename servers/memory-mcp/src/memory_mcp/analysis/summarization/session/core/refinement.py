#!/usr/bin/env python3
"""
Методы рефайнинга для session_summarizer
"""

from typing import Any, Dict, List, Optional, Tuple

from .generation import (
    apply_addon_metadata_to_claim,
    generate_claims,
)
from .quality import refresh_quality
from ..utils import (
    build_claim_summary,
    build_topic_title,
    collect_segment_by_ids,
    create_action_from_decision,
    create_risk_entry,
    derive_rationale,
    expand_message_ids,
    lookup_message_envelope,
    message_identifier,
    message_key,
    normalize_summary,
    topic_time_span,
    truncate_text,
)

# Маппинг итераций на методы рефайнинга: (method_name, requires_profile)
REFINE_METHODS = {
    1: ("refine_pass_expand_evidence", True),
    2: ("refine_pass_claims", True),
    3: ("refine_pass_topics", False),
    4: ("refine_pass_profile_rules", True),
    5: ("refine_pass_threads", True),
    6: ("refine_pass_discussion_structure", True),
    7: ("refine_pass_context_quality", True),
}


def run_structural_pass(
    summary: Dict[str, Any],
    aux_data: Dict[str, Any],
    session: Dict[str, Any],
    iteration: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    profile = summary.get("meta", {}).get("profile") or aux_data.get("profile")
    changed = False

    # Используем словарь для выбора метода рефайнинга
    method_info = REFINE_METHODS.get(iteration)
    if method_info:
        method_name, requires_profile = method_info
        if method_name == "refine_pass_expand_evidence":
            method = refine_pass_expand_evidence
        elif method_name == "refine_pass_claims":
            method = refine_pass_claims
        elif method_name == "refine_pass_topics":
            method = refine_pass_topics
        elif method_name == "refine_pass_profile_rules":
            method = refine_pass_profile_rules
        elif method_name == "refine_pass_threads":
            method = refine_pass_threads
        elif method_name == "refine_pass_discussion_structure":
            method = refine_pass_discussion_structure
        elif method_name == "refine_pass_context_quality":
            method = refine_pass_context_quality
        else:
            method = None

        if method:
            if requires_profile:
                changed = method(summary, aux_data, profile)
            else:
                changed = method(summary, aux_data)

    previous_score = summary.get("quality", {}).get("score", 0.0)
    quality_context = refresh_quality(summary, aux_data)
    aux_data.update(quality_context)
    if summary.get("quality", {}).get("score", 0.0) != previous_score:
        changed = True

    return summary, {"changed": changed}


def refine_pass_expand_evidence(
    summary: Dict[str, Any],
    aux_data: Dict[str, Any],
    profile: str,
) -> bool:
    message_map = aux_data.get("message_map", [])
    if not message_map:
        return False

    per_topic_limit = 60 if profile == "group-project" else 40
    fallback_span = aux_data.get(
        "time_span_default", summary.get("meta", {}).get("time_span", "")
    )

    changed = False
    for topic in summary.get("topics", []):
        current_ids = [mid for mid in topic.get("message_ids", []) if mid]
        if not current_ids:
            continue
        target = min(per_topic_limit, len(message_map))
        desired = max(len(current_ids) * 2, min(len(current_ids) + 2, target))
        desired = min(desired, target)
        expanded_ids = expand_message_ids(current_ids, aux_data, desired)
        if expanded_ids != current_ids:
            topic["message_ids"] = expanded_ids
            segment = collect_segment_by_ids(expanded_ids, aux_data)
            topic["time_span"] = topic_time_span(segment, fallback_span)
            changed = True

    return changed


def refine_pass_claims(
    summary: Dict[str, Any],
    aux_data: Dict[str, Any],
    profile: str,
) -> bool:
    message_map = aux_data.get("message_map", [])
    if not message_map:
        return False

    claims = summary.get("claims", [])
    existing_by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for claim in claims:
        existing_by_topic.setdefault(claim.get("topic_title"), []).append(claim)

    modality = "media" if profile == "broadcast" else "internal"
    source = modality
    entities_pool = aux_data.get("entities_flat") or summary.get("entities", [])
    addons_info = aux_data.get("addons_info", {})
    global_asset_tags = addons_info.get("asset_tags", [])
    global_geo_tags = addons_info.get("geo_entities", [])
    addons_set = addons_info.get("addons", set())
    per_message_addons = addons_info.get("by_key", {})

    additions = 0
    for topic in summary.get("topics", []):
        title = topic.get("title")
        if not title:
            continue
        if existing_by_topic.get(title):
            continue

        for message_id in topic.get("message_ids", []):
            envelope = lookup_message_envelope(message_id, aux_data)
            if not envelope or not envelope.get("text"):
                continue
            summary_text = build_claim_summary(
                envelope.get("text", ""), topic.get("summary", "")
            )
            summary_text = summary_text[:160]
            claim = {
                "ts": envelope.get("ts_bkk", ""),
                "source": source,
                "modality": modality,
                "credibility": "medium",
                "entities": list(entities_pool[:4]),
                "summary": summary_text,
                "msg_id": message_identifier(envelope) or "",
                "topic_title": title,
            }
            apply_addon_metadata_to_claim(
                claim,
                per_message_addons.get(envelope.get("key")),
                addons_set,
                global_asset_tags,
                global_geo_tags,
            )
            claims.append(claim)
            additions += 1
            break

    if not additions:
        return False

    credibility_order = {"high": 0, "medium": 1, "low": 2}
    claims.sort(
        key=lambda item: (
            credibility_order.get(item.get("credibility", "medium"), 1),
            item.get("ts", ""),
        )
    )
    summary["claims"] = claims[:20]
    return True


def refine_pass_topics(
    summary: Dict[str, Any],
    aux_data: Dict[str, Any],
) -> bool:
    topics = summary.get("topics", [])
    if len(topics) <= 1:
        return False

    changed = False
    fallback_span = aux_data.get(
        "time_span_default",
        summary.get("meta", {}).get("time_span", ""),
    )

    i = 0
    while i < len(topics):
        if len(topics) <= 1:
            break
        topic = topics[i]
        message_ids = topic.get("message_ids", [])
        if len(message_ids) >= 3:
            i += 1
            continue
        target_idx = i - 1 if i > 0 else i + 1
        if target_idx < 0 or target_idx >= len(topics):
            i += 1
            continue
        receiver = topics[target_idx]
        combined_ids = list(
            dict.fromkeys(receiver.get("message_ids", []) + message_ids)
        )
        receiver["message_ids"] = combined_ids
        combined_summary = (
            f"{receiver.get('summary', '')} {topic.get('summary', '')}".strip()
        )
        receiver["summary"] = normalize_summary(combined_summary)
        receiver["title"] = build_topic_title(receiver["summary"])
        segment = collect_segment_by_ids(combined_ids, aux_data)
        receiver["time_span"] = topic_time_span(segment, fallback_span)
        topics.pop(i)
        changed = True
        continue
    while len(topics) > 6:
        smallest_idx = min(
            range(len(topics)),
            key=lambda idx: len(topics[idx].get("message_ids", [])),
        )
        receiver_idx = smallest_idx - 1 if smallest_idx > 0 else smallest_idx + 1
        if receiver_idx < 0 or receiver_idx >= len(topics):
            break

        if smallest_idx < receiver_idx:
            donor = topics.pop(smallest_idx)
            receiver_idx -= 1
            receiver = topics[receiver_idx]
        else:
            receiver = topics[receiver_idx]
            donor = topics.pop(smallest_idx)

        combined_ids = list(
            dict.fromkeys(
                receiver.get("message_ids", []) + donor.get("message_ids", [])
            )
        )
        receiver["message_ids"] = combined_ids
        combined_summary = (
            f"{receiver.get('summary', '')} {donor.get('summary', '')}".strip()
        )
        receiver["summary"] = normalize_summary(combined_summary)
        receiver["title"] = build_topic_title(receiver["summary"])
        segment = collect_segment_by_ids(combined_ids, aux_data)
        receiver["time_span"] = topic_time_span(segment, fallback_span)
        changed = True

    return changed


def refine_pass_profile_rules(
    summary: Dict[str, Any],
    aux_data: Dict[str, Any],
    profile: str,
) -> bool:
    changed = False

    if profile == "broadcast":
            claims_cov = aux_data.get("claims_coverage", 0.0)
            if claims_cov < 0.25:
                changed |= refine_pass_claims(summary, aux_data, profile)
            summary["rationale"] = derive_rationale(
                profile, summary.get("actions", []), summary.get("risks", [])
            )
            return changed

    # group-project specific adjustments
    actions = summary.setdefault("actions", [])
    if len(actions) < 2:
        for decision in aux_data.get("legacy_decisions", []):
            action = create_action_from_decision(decision, summary, aux_data)
            if action:
                actions.append(action)
                changed = True
            if len(actions) >= 2:
                break

    risks = summary.setdefault("risks", [])
    if len(risks) < 1:
        for risk_entry in aux_data.get("legacy_risks", []):
            risk = create_risk_entry(risk_entry, summary, aux_data)
            if risk:
                risks.append(risk)
                changed = True
                break

    summary["rationale"] = derive_rationale(profile, actions, risks)
    return changed


def refine_pass_threads(
    summary: Dict[str, Any],
    aux_data: Dict[str, Any],
    profile: str,
) -> bool:
    messages = aux_data.get("messages", [])
    if not any(msg.get("reply_to") for msg in messages):
        return False

    discussion = summary.setdefault("discussion", [])
    existing_ids = {item.get("msg_id") for item in discussion if item.get("msg_id")}

    for envelope in aux_data.get("message_map", []):
        raw = envelope.get("raw", {})
        if not raw.get("reply_to"):
            continue
        identifier = message_identifier(envelope)
        if identifier in existing_ids:
            continue
        quote = truncate_text(envelope.get("text", ""), 220)
        if not quote:
            continue
        discussion.append(
            {
                "ts": envelope.get("ts_bkk", ""),
                "author": envelope.get("author", ""),
                "msg_id": identifier or "",
                "quote": quote,
            }
        )
        return True

    return False


def refine_pass_discussion_structure(
    summary: Dict[str, Any],
    aux_data: Dict[str, Any],
    profile: str,
) -> bool:
    """Дополнительная итерация: улучшение структуры дискуссии"""
    discussion = summary.get("discussion", [])
    if not discussion:
        return False

    changed = False

    # Улучшаем качество цитат - делаем их более информативными
    for item in discussion:
        quote = item.get("quote", "")
        if len(quote) < 50:  # Слишком короткие цитаты
            # Попробуем найти более длинную цитату из того же сообщения
            msg_id = item.get("msg_id", "")
            if msg_id:
                message_map = aux_data.get("message_map", [])
                for envelope in message_map:
                    if message_identifier(envelope) == msg_id:
                        full_text = envelope.get("text", "")
                        if len(full_text) > len(quote) and len(full_text) <= 300:
                            item["quote"] = truncate_text(full_text, 250)
                            changed = True
                        break

    # Добавляем недостающие элементы дискуссии если их мало
    if len(discussion) < 3 and profile == "group-project":
        messages = aux_data.get("messages", [])
        existing_ids = {
            item.get("msg_id") for item in discussion if item.get("msg_id")
        }

        # Ищем дополнительные сообщения для дискуссии
        for msg in messages[-10:]:  # Последние 10 сообщений
            msg_id = message_key(msg)
            if msg_id not in existing_ids and len(msg.get("text", "")) > 30:
                quote = truncate_text(msg.get("text", ""), 200)
                if quote:
                    discussion.append(
                        {
                            "ts": msg.get("date_utc", ""),
                            "author": msg.get("from", {}).get("username", ""),
                            "msg_id": msg_id,
                            "quote": quote,
                        }
                    )
                    changed = True
                    if len(discussion) >= 5:  # Ограничиваем количество
                        break

    return changed


def refine_pass_context_quality(
    summary: Dict[str, Any],
    aux_data: Dict[str, Any],
    profile: str,
) -> bool:
    """Дополнительная итерация: улучшение качества контекста"""
    context = summary.get("context", "")
    if len(context) < 100:  # Слишком короткий контекст
        # Попробуем создать более информативный контекст
        messages = aux_data.get("messages", [])
        if messages:
            # Берем первые несколько сообщений для контекста
            context_parts = []
            for msg in messages[:3]:
                text = msg.get("text", "")
                if text and len(text) > 20:
                    context_parts.append(text[:100])

            if context_parts:
                new_context = " ".join(context_parts)[:300]
                if len(new_context) > len(context):
                    summary["context"] = new_context
                    return True

    # Улучшаем качество топиков
    topics = summary.get("topics", [])
    changed = False

    for topic in topics:
        title = topic.get("title", "")
        if len(title) < 10:  # Слишком короткий заголовок
            # Попробуем создать более описательный заголовок
            message_ids = topic.get("message_ids", [])
            if message_ids:
                message_map = aux_data.get("message_map", [])
                topic_texts = []
                for envelope in message_map:
                    if message_identifier(envelope) in message_ids:
                        text = envelope.get("text", "")
                        if text:
                            topic_texts.append(text[:50])

                if topic_texts:
                    # Создаем более информативный заголовок
                    combined_text = " ".join(topic_texts[:3])
                    if len(combined_text) > len(title):
                        topic["title"] = truncate_text(combined_text, 80)
                        changed = True

    return changed

