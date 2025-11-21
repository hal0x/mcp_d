#!/usr/bin/env python3
"""
Методы оценки качества для session_summarizer
"""

from typing import Any, Dict, List, Optional, Tuple

from ...utils import strip_markdown


def refresh_quality(
    summary: Dict[str, Any],
    aux_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Пересчитывает метрики и качество для канонической саммаризации."""
    topics = summary.get("topics", [])
    claims = summary.get("claims", [])
    actions = summary.get("actions", [])
    risks = summary.get("risks", [])
    discussion = summary.get("discussion", [])
    messages = aux_data.get("messages", [])
    messages_total = aux_data.get("messages_total", len(messages))
    message_map = aux_data.get("message_map", [])
    profile = summary.get("meta", {}).get("profile") or aux_data.get("profile")
    legacy_summary = aux_data.get("legacy_summary") or {}
    legacy_metrics = aux_data.get("legacy_metrics")
    small_session_info = aux_data.get("small_session_info")

    coverage, claims_coverage = calculate_coverage(
        topics, claims, messages_total
    )
    thread_count = count_threads(messages)
    schema_eval = evaluate_schema_requirements(
        profile=profile,
        topics=topics,
        claims=claims,
        actions=actions,
        risks=risks,
        discussion=discussion,
        messages=messages,
        thread_count=thread_count,
    )
    schema_eval["claims_coverage_ok"] = (
        claims_coverage >= schema_eval["claims_coverage_target"]
    )
    if (
        not schema_eval["claims_coverage_ok"]
        and "claims_coverage_below_threshold" not in schema_eval["issues"]
    ):
        schema_eval["issues"].append("claims_coverage_below_threshold")

    dup_rate = estimate_duplicate_rate(message_map)
    quality_score, len_penalty = compute_quality_score(
        profile=profile,
        coverage=coverage,
        claims_coverage=claims_coverage,
        schema_eval=schema_eval,
        dup_rate=dup_rate,
        claims=claims,
    )
    if small_session_info:
        quality_score = min(
            quality_score, small_session_info.get("score_cap", quality_score)
        )

    confidence = round(quality_score / 100.0, 3)
    quality_status = derive_quality_status(
        score=quality_score,
        schema_eval=schema_eval,
        legacy_summary=legacy_summary,
        legacy_metrics=legacy_metrics,
    )
    if small_session_info:
        quality_status = "needs_review"

    meta = summary.setdefault("meta", {})
    meta["confidence"] = confidence
    addons_info = aux_data.get("addons_info", {})
    meta["addons"] = sorted(addons_info.get("addons", []))
    if small_session_info:
        meta.setdefault("policy_flags", small_session_info.get("policy_flags", []))

    existing_details = summary.get("quality", {}).get("details", {})
    legacy_metrics_details = legacy_summary.get("quality_metrics", {})
    quality_details = {
        "legacy_metrics": legacy_metrics_details,
        "schema_issues": schema_eval["issues"],
        "blocking_issues": schema_eval["blocking_issues"],
        "dup_rate": dup_rate,
        "len_penalty": len_penalty,
        "claims_threshold": {
            "claims_coverage_target": schema_eval["claims_coverage_target"],
            "claims_coverage_ok": schema_eval["claims_coverage_ok"],
        },
    }
    if legacy_metrics:
        quality_details["legacy_score"] = legacy_metrics.score
    if small_session_info:
        quality_details["policy_flags"] = small_session_info.get("policy_flags", [])
    if "refinement_history" in existing_details:
        quality_details["refinement_history"] = existing_details[
            "refinement_history"
        ]

    quality = {
        "score": round(quality_score, 2),
        "status": quality_status,
        "kpi": {
            "coverage": coverage,
            "claims_coverage": claims_coverage,
            "topics": len(topics),
            "actions": len(actions),
            "risks": len(risks),
            "threads": thread_count,
        },
        "flags": {
            "topics_ok": schema_eval["topics_ok"],
            "claims_ok": schema_eval["claims_ok"],
            "actions_ok": schema_eval["actions_ok"],
            "risks_ok": schema_eval["risks_ok"],
            "threads_ok": schema_eval["threads_ok"],
            "structure_ok": schema_eval["structure_ok"],
            "claims_coverage_ok": schema_eval["claims_coverage_ok"],
            "discussion_ok": schema_eval["discussion_ok"],
        },
        "details": quality_details,
    }

    summary["quality"] = quality

    return {
        "thread_count": thread_count,
        "schema_eval": schema_eval,
        "coverage": coverage,
        "claims_coverage": claims_coverage,
    }


def evaluate_schema_requirements(
    profile: str,
    topics: List[Dict[str, Any]],
    claims: List[Dict[str, Any]],
    actions: List[Dict[str, Any]],
    risks: List[Dict[str, Any]],
    discussion: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    thread_count: int,
) -> Dict[str, Any]:
    issues: List[str] = []

    topics_count = len(topics)
    claims_count = len(claims)
    actions_count = len(actions)
    risks_count = len(risks)
    discussion_count = len(discussion)
    replies_present = any(msg.get("reply_to") for msg in messages)

    topics_ok = topics_count >= 2
    if not topics_ok:
        issues.append("topics_minimum_not_met")
    if topics_count > 6:
        issues.append("topics_exceed_maximum")

    claims_ok = claims_count >= 1
    if not claims_ok:
        issues.append("claims_minimum_not_met")

    discussion_min = 3 if profile == "broadcast" else 4
    discussion_ok = discussion_count >= discussion_min
    if not discussion_ok:
        issues.append("discussion_minimum_not_met")

    if profile == "group-project":
        actions_ok = actions_count >= 2
        if not actions_ok:
            issues.append("actions_minimum_not_met")
        risks_ok = risks_count >= 1
        if not risks_ok:
            issues.append("risks_minimum_not_met")
    else:
        actions_ok = actions_count >= 1
        risks_ok = risks_count >= 1

    threads_ok = (not replies_present) or thread_count >= 1
    if replies_present and not threads_ok:
        issues.append("threads_missing")

    structure_ok = (
        topics_ok
        and claims_ok
        and (profile != "group-project" or (actions_ok and risks_ok))
    )

    claims_coverage_target = 0.25 if profile == "broadcast" else 0.35
    # claims_coverage tracked externally; ok flag filled later when value available

    blocking_issues = [
        issue
        for issue in issues
        if issue
        in {
            "topics_minimum_not_met",
            "claims_minimum_not_met",
            "actions_minimum_not_met",
            "risks_minimum_not_met",
            "threads_missing",
        }
    ]

    return {
        "issues": issues,
        "blocking_issues": blocking_issues,
        "topics_ok": topics_ok,
        "claims_ok": claims_ok,
        "actions_ok": actions_ok,
        "risks_ok": risks_ok,
        "threads_ok": threads_ok,
        "structure_ok": structure_ok,
        "discussion_ok": discussion_ok,
        "claims_coverage_target": claims_coverage_target,
        "claims_coverage_ok": True,  # placeholder, recalculated by caller
    }


def compute_quality_score(
    profile: str,
    coverage: float,
    claims_coverage: float,
    schema_eval: Dict[str, Any],
    dup_rate: float,
    claims: List[Dict[str, Any]],
) -> Tuple[float, int]:
    coverage_component = min(1.0, max(0.0, coverage))
    claims_component = min(1.0, max(0.0, claims_coverage))

    if profile == "broadcast":
        weights = {
            "coverage": 0.35,
            "claims_coverage": 0.10,
            "actions_ok": 0.05,
            "risks_ok": 0.05,
            "threads_ok": 0.05,
            "dedup": 0.15,
            "structure_ok": 0.25,
        }
    else:
        weights = {
            "coverage": 0.20,
            "claims_coverage": 0.10,
            "actions_ok": 0.30,
            "risks_ok": 0.15,
            "threads_ok": 0.10,
            "dedup": 0.05,
            "structure_ok": 0.10,
        }

    actions_component = 1.0 if schema_eval["actions_ok"] else 0.0
    risks_component = 1.0 if schema_eval["risks_ok"] else 0.0
    threads_component = 1.0 if schema_eval["threads_ok"] else 0.0
    structure_component = 1.0 if schema_eval["structure_ok"] else 0.0
    dedup_component = 1.0 - min(1.0, max(0.0, dup_rate))

    weighted_sum = (
        coverage_component * weights["coverage"]
        + claims_component * weights["claims_coverage"]
        + actions_component * weights["actions_ok"]
        + risks_component * weights["risks_ok"]
        + threads_component * weights["threads_ok"]
        + dedup_component * weights["dedup"]
        + structure_component * weights["structure_ok"]
    )

    len_penalty = 0
    for claim in claims:
        length = len(claim.get("summary", "") or "")
        if length <= 160:
            continue
        over_soft = min(length, 220) - 160
        if over_soft > 0:
            len_penalty += ((over_soft - 1) // 30 + 1) * 1
        if length > 220:
            over_hard = min(length, 300) - 220
            if over_hard > 0:
                len_penalty += ((over_hard - 1) // 30 + 1) * 2

    score = max(0.0, min(100.0, weighted_sum * 100 - len_penalty))
    return score, len_penalty


def calculate_coverage(
    topics: List[Dict[str, Any]],
    claims: List[Dict[str, Any]],
    messages_total: int,
) -> Tuple[float, float]:
    topic_message_ids = set()
    for topic in topics:
        topic_message_ids.update(topic.get("message_ids", []))
    coverage = (
        round(len(topic_message_ids) / messages_total, 3) if messages_total else 0.0
    )

    claim_message_ids = {
        claim.get("msg_id") for claim in claims if claim.get("msg_id")
    }
    claims_coverage = (
        round(len(claim_message_ids) / len(topic_message_ids), 3)
        if topic_message_ids
        else 0.0
    )

    return coverage, claims_coverage


def count_threads(messages: List[Dict[str, Any]]) -> int:
    thread_roots = set()
    for msg in messages:
        reply = msg.get("reply_to")
        root_id: Optional[str] = None
        if isinstance(reply, dict):
            root_id = reply.get("msg_id") or reply.get("id")
        elif isinstance(reply, (str, int)):
            root_id = str(reply)

        if root_id:
            thread_roots.add(root_id)
    return len(thread_roots)


def derive_quality_status(
    score: float,
    schema_eval: Dict[str, Any],
    legacy_summary: Dict[str, Any],
    legacy_metrics,
) -> str:
    improved = legacy_summary.get("quality_metrics", {}).get("improved")

    if (
        score >= 85
        and schema_eval["structure_ok"]
        and not schema_eval["blocking_issues"]
    ):
        return "accepted"

    if improved:
        return "refined"

    return "needs_review"


def estimate_duplicate_rate(message_map: List[Dict[str, Any]]) -> float:
    if not message_map:
        return 0.0

    normalized_texts = []
    for envelope in message_map:
        text = strip_markdown(envelope.get("text", "")).lower()
        if text:
            normalized_texts.append(text)

    if not normalized_texts:
        return 0.0

    unique_texts = set(normalized_texts)
    duplicates = len(normalized_texts) - len(unique_texts)
    total = len(normalized_texts)
    if total <= 0:
        return 0.0
    return round(max(0.0, duplicates / total), 3)

