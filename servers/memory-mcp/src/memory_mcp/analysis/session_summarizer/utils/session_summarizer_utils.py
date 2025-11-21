#!/usr/bin/env python3
"""
Утилиты и вспомогательные методы для session_summarizer
Реэкспорт функций из специализированных модулей для обратной совместимости
"""

# Реэкспорт всех функций из специализированных модулей
from .session_summarizer_action_utils import (
    apply_small_session_policy,
    create_action_from_decision,
    create_risk_entry,
    derive_rationale,
    extract_action_items,
    is_small_session,
    parse_action_item,
)
from .session_summarizer_chat_utils import (
    collect_participants,
    detect_chat_mode,
    derive_time_span,
    format_message_bullet,
    map_profile,
    select_key_messages,
    select_messages_with_keywords,
)
from .session_summarizer_domain_utils import (
    build_attachments,
    detect_domain_addons,
    flatten_entities,
    format_links_artifacts,
    normalize_attachment,
    sanitize_url,
)
from .session_summarizer_message_utils import (
    build_message_envelope,
    collect_segment_by_ids,
    expand_message_ids,
    extract_message_text,
    find_message_id_for_text,
    lookup_message_envelope,
    message_identifier,
    message_key,
)
from .session_summarizer_text_utils import (
    build_claim_summary,
    build_topic_title,
    clean_bullet,
    normalize_summary,
    normalize_text_for_display,
    prepare_conversation_text,
    strip_markdown,
    truncate_text,
)
from .session_summarizer_topic_utils import (
    build_minimal_topic,
    guess_topic_title,
    split_messages_for_topics,
    topic_time_span,
)

__all__ = [
    # Message utils
    "message_key",
    "extract_message_text",
    "build_message_envelope",
    "message_identifier",
    "lookup_message_envelope",
    "expand_message_ids",
    "collect_segment_by_ids",
    "find_message_id_for_text",
    # Text utils
    "strip_markdown",
    "normalize_text_for_display",
    "truncate_text",
    "normalize_summary",
    "clean_bullet",
    "build_topic_title",
    "build_claim_summary",
    "prepare_conversation_text",
    # Chat utils
    "detect_chat_mode",
    "map_profile",
    "collect_participants",
    "select_key_messages",
    "select_messages_with_keywords",
    "format_message_bullet",
    "derive_time_span",
    # Domain utils
    "detect_domain_addons",
    "flatten_entities",
    "build_attachments",
    "format_links_artifacts",
    "normalize_attachment",
    "sanitize_url",
    # Topic utils
    "split_messages_for_topics",
    "topic_time_span",
    "guess_topic_title",
    "build_minimal_topic",
    # Action utils
    "parse_action_item",
    "extract_action_items",
    "create_action_from_decision",
    "create_risk_entry",
    "derive_rationale",
    "is_small_session",
    "apply_small_session_policy",
]
