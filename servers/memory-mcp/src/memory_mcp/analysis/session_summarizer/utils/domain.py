#!/usr/bin/env python3
"""
Утилиты для работы с доменными данными для session_summarizer
"""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from ..constants import (
    CRYPTO_EXCHANGES,
    CRYPTO_TERMS,
    CRYPTO_TICKERS,
    GEOPOLITICS_PATTERNS,
    SCI_TECH_PATTERNS,
    SCI_TECH_TERMS,
)


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

