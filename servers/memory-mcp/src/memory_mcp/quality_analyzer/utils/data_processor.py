"""Helpers for loading and normalizing chat data from JSON exports."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

__all__ = [
    "load_chat_file",
    "load_chats_from_directory",
    "normalize_message",
]


def _coerce_text(text_field) -> str:
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        parts: list[str] = []
        for part in text_field:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return ""


def normalize_message(chat_name: str, raw: dict) -> dict | None:
    """Normalize heterogeneous chat exports into a unified structure."""

    message_type = raw.get("type")

    if message_type in ("message", "text"):
        text = _coerce_text(raw.get("text", ""))
        if not text.strip():
            return None

        metadata = {
            "chat": chat_name,
            "message_id": raw.get("id"),
            "sender": raw.get("from"),
            "sender_id": raw.get("from_id"),
            "reply_to": raw.get("reply_to_message_id"),
            "has_media": bool(raw.get("media")),
            "via_bot": raw.get("via_bot"),
            "forwarded_from": raw.get("forwarded_from"),
            "message_type": message_type,
        }

        return {
            "text": text,
            "date": raw.get("date"),
            "metadata": metadata,
        }

    # Smart aggregation / NDJSON format
    if raw.get("chat") and raw.get("date_utc"):
        text = raw.get("text", "")
        if isinstance(text, list):
            text = "".join(str(part) for part in text)

        attachments = raw.get("attachments") or []
        if not text.strip() and attachments:
            text = " ".join(
                f"[{item.get('type', 'file')}: {item.get('file') or item.get('href') or ''}]"
                for item in attachments
            )

        if not text.strip():
            return None

        sender_info = raw.get("from") or {}

        metadata = {
            "chat": raw.get("chat", chat_name),
            "message_id": raw.get("id"),
            "sender": sender_info.get("display") or sender_info.get("username"),
            "sender_id": sender_info.get("id"),
            "reply_to": raw.get("reply_to"),
            "has_media": bool(attachments),
            "forwarded_from": raw.get("forwarded_from"),
            "language": raw.get("language"),
            "message_type": message_type or "smart_export",
        }

        if attachments:
            metadata["attachments"] = attachments

        return {
            "text": text,
            "date": raw.get("date_utc"),
            "metadata": metadata,
        }

    return None


def load_chat_file(path: Path) -> dict[str, list[dict]]:
    try:
        with open(path, encoding="utf-8") as fp:
            payload = json.load(fp)
        is_ndjson = False
    except json.JSONDecodeError:
        is_ndjson = True
    except Exception as exc:  # pragma: no cover - logging failure only
        logger.error("Не удалось загрузить файл чата %s: %s", path, exc)
        return {}

    messages: list[dict] = []

    if not is_ndjson:
        chat_name = payload.get("name") or path.parent.name or path.stem
        for raw in payload.get("messages", []):
            normalized = normalize_message(chat_name, raw)
            if normalized:
                messages.append(normalized)
    else:
        chat_name = path.parent.name or path.stem
        with open(path, encoding="utf-8") as fp:
            for line_number, line in enumerate(fp, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.error(
                        "Некорректная JSON-строка в %s:%d — %s",
                        path,
                        line_number,
                        exc,
                    )
                    continue

                normalized = normalize_message(chat_name, raw)
                if normalized:
                    messages.append(normalized)

    if not messages:
        logger.warning(
            "В чате %s (%s) не найдено подходящих сообщений", chat_name, path
        )

    return {chat_name: messages}


def load_chats_from_directory(
    directory: Path,
    selected_chats: Iterable[str] | None = None,
) -> dict[str, list[dict]]:
    if not directory.exists():
        logger.debug("Директория с чатами не найдена: %s", directory)
        return {}

    results: dict[str, list[dict]] = {}
    selected_set = {name.lower() for name in selected_chats} if selected_chats else None

    for file_path in directory.rglob("*.json"):
        if selected_set:
            parent_name = file_path.parent.name.lower()
            stem_name = file_path.stem.lower()
            if parent_name not in selected_set and stem_name not in selected_set:
                continue

        loaded = load_chat_file(file_path)
        for chat_name, messages in loaded.items():
            if selected_set and chat_name.lower() not in selected_set:
                continue
            results.setdefault(chat_name, []).extend(messages)

    return results
