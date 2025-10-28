#!/usr/bin/env python3
"""Менеджер специальных инструкций для саммаризации."""

from __future__ import annotations

import json
from pathlib import Path


class InstructionManager:
    """Хранит и предоставляет специальные инструкции для саммаризации."""

    def __init__(
        self, storage_path: Path = Path("config/custom_instructions.json")
    ) -> None:
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, str]] = {
            "chats": {},
            "modes": {"group": "", "channel": ""},
        }
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_chat_instruction(self, chat: str, instruction: str) -> None:
        chat_key = chat.strip()
        if not chat_key:
            raise ValueError("chat name cannot be empty")
        self._data.setdefault("chats", {})[chat_key] = instruction.strip()
        self._save()

    def set_mode_instruction(self, mode: str, instruction: str) -> None:
        normalized = self._normalize_mode(mode)
        self._data.setdefault("modes", {})[normalized] = instruction.strip()
        self._save()

    def clear_chat_instruction(self, chat: str) -> None:
        chat_key = chat.strip()
        if chat_key in self._data.get("chats", {}):
            del self._data["chats"][chat_key]
            self._save()

    def clear_mode_instruction(self, mode: str) -> None:
        normalized = self._normalize_mode(mode)
        if normalized in self._data.get("modes", {}):
            self._data["modes"][normalized] = ""
            self._save()

    def get_instruction(self, chat: str, mode: str) -> str | None:
        chat_key = chat.strip()
        if chat_key:
            chat_instruction = self._data.get("chats", {}).get(chat_key)
            if chat_instruction:
                return chat_instruction
        normalized = self._normalize_mode(mode)
        mode_instruction = self._data.get("modes", {}).get(normalized)
        return mode_instruction or None

    def export(self) -> dict[str, dict[str, str]]:
        return {
            "chats": dict(self._data.get("chats", {})),
            "modes": dict(self._data.get("modes", {})),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                chats = raw.get("chats", {})
                modes = raw.get("modes", {})
                if isinstance(chats, dict):
                    self._data["chats"].update(
                        {str(k): str(v) for k, v in chats.items() if v}
                    )
                if isinstance(modes, dict):
                    for k, v in modes.items():
                        self._data["modes"][self._normalize_mode(k)] = str(v)
        except json.JSONDecodeError:
            # Поврежденный файл — игнорируем и оставляем дефолт
            return

    def _save(self) -> None:
        payload = {
            "chats": self._data.get("chats", {}),
            "modes": self._data.get("modes", {}),
        }
        self.storage_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        mode_lower = mode.strip().lower()
        if mode_lower not in {"group", "channel"}:
            raise ValueError("mode must be 'group' or 'channel'")
        return mode_lower


__all__ = ["InstructionManager"]
