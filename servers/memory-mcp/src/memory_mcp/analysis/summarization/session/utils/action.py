#!/usr/bin/env python3
"""
Утилиты для работы с действиями и рисками для session_summarizer
"""

import re
from typing import Any, Dict, List, Optional

from .message import find_message_id_for_text
from .topic import guess_topic_title


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
    from .topic import build_minimal_topic, topic_time_span
    
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

