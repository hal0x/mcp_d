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

        from_info = raw.get("from") or {}
        if isinstance(from_info, dict):
            sender_username = from_info.get("username")
            sender_display = from_info.get("display")
        else:
            sender_username = None
            sender_display = str(from_info) if from_info else None

        metadata = {
            "chat": chat_name,
            "message_id": raw.get("id"),
            "sender": sender_display or raw.get("from"),
            "sender_id": raw.get("from_id"),
            "sender_username": sender_username,
            "reply_to": raw.get("reply_to_message_id"),
            "has_media": bool(raw.get("media")),
            "via_bot": raw.get("via_bot"),
            "forwarded_from": raw.get("forwarded_from"),
            "message_type": message_type,
        }

        # Добавляем реакции, если есть
        reactions = raw.get("reactions")
        if reactions and isinstance(reactions, list):
            metadata["reactions"] = reactions

        # Добавляем дату редактирования, если есть
        edited_utc = raw.get("edited_utc")
        if edited_utc:
            metadata["edited_utc"] = edited_utc

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
            "sender": sender_info.get("display") or sender_info.get("username") if isinstance(sender_info, dict) else str(sender_info) if sender_info else None,
            "sender_id": sender_info.get("id") if isinstance(sender_info, dict) else None,
            "sender_username": sender_info.get("username") if isinstance(sender_info, dict) else None,
            "reply_to": raw.get("reply_to"),
            "has_media": bool(attachments),
            "forwarded_from": raw.get("forwarded_from"),
            "language": raw.get("language"),
            "message_type": message_type or "smart_export",
        }

        # Добавляем реакции, если есть
        reactions = raw.get("reactions")
        if reactions and isinstance(reactions, list):
            metadata["reactions"] = reactions

        # Добавляем дату редактирования, если есть
        edited_utc = raw.get("edited_utc")
        if edited_utc:
            metadata["edited_utc"] = edited_utc

        if attachments:
            metadata["attachments"] = attachments

        return {
            "text": text,
            "date": raw.get("date_utc"),
            "metadata": metadata,
        }

    return None


def load_chat_file(path: Path) -> dict[str, list[dict]]:
    """Загружает файл чата (JSON или JSONL) с нормализацией сообщений (использует общую утилиту)."""
    from ...utils.json_loader import load_json_or_jsonl

    try:
        file_messages, is_jsonl = load_json_or_jsonl(path)
    except Exception as exc:  # pragma: no cover - logging failure only
        logger.error("Не удалось загрузить файл чата %s: %s", path, exc)
        return {}

    messages: list[dict] = []

    if is_jsonl:
        chat_name = path.parent.name or path.stem
        for raw in file_messages:
            normalized = normalize_message(chat_name, raw)
            if normalized:
                messages.append(normalized)
    else:
        # Обычный JSON - может быть объект с полем "messages" или список
        if isinstance(file_messages, list) and file_messages:
            # Если это список, используем имя директории
            chat_name = path.parent.name or path.stem
            for raw in file_messages:
                normalized = normalize_message(chat_name, raw)
                if normalized:
                    messages.append(normalized)
        elif isinstance(file_messages, dict):
            # Если это словарь, ищем поле "messages"
            chat_name = file_messages.get("name") or path.parent.name or path.stem
            for raw in file_messages.get("messages", []):
                normalized = normalize_message(chat_name, raw)
                if normalized:
                    messages.append(normalized)

    if not messages:
        chat_name = path.parent.name or path.stem
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
