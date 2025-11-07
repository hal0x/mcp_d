"""Persistent storage for themes and their associated chat titles."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from core.utils.json_io import load_json, save_json


def sanitize_name(name: str) -> str:
    name = (name or "").strip().replace("\\", "/")
    safe = "".join(ch if ch.isalnum() or ch in " _-." else "_" for ch in name)
    safe = safe.replace("/", "_")
    return safe or "default"


logger = logging.getLogger(__name__)


class ThemeStore:
    """Store mapping of theme -> {sanitized_chat: original_chat} in a JSON file."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        # theme -> {sanitized_chat: original_chat}
        self._themes: Dict[str, Dict[str, str]] = {}
        if self.path.exists():
            data: Dict[str, Any] = load_json(self.path, {})
            # Support legacy structures where chats were stored as lists
            for theme, chats in data.items():
                if isinstance(chats, dict):
                    self._themes[theme] = {str(k): str(v) for k, v in chats.items()}
                elif isinstance(chats, list):
                    self._themes[theme] = {str(c): str(c) for c in chats}
                else:
                    self._themes[theme] = {}

    # ----------------------------- persistence -----------------------------
    def _save(self) -> None:
        save_json(self.path, self._themes, indent=2)

    # ------------------------------- queries --------------------------------
    def list_themes(self) -> List[str]:
        return sorted(self._themes.keys())

    def get_chats(self, theme: str) -> Dict[str, str]:
        """Return mapping of sanitized chat names to original names for ``theme``."""
        theme_key = sanitize_name(theme)
        return dict(self._themes.get(theme_key, {}))

    def set_theme(self, theme: str, chats: Dict[str, str] | List[str]) -> None:
        """Persist chats mapping for ``theme``.

        ``chats`` may be a mapping ``{sanitized: original}`` or a legacy list of
        chat names.  In the latter case we store ``{name: name}``.
        """
        theme_key = sanitize_name(theme)
        if isinstance(chats, dict):
            mapping = {str(k): str(v) for k, v in chats.items()}
        else:
            mapping = {str(c): str(c) for c in chats}
        logger.info(
            "ThemeStore.set_theme: theme='%s' -> key='%s', chats=%s",
            theme,
            theme_key,
            mapping,
        )
        self._themes[theme_key] = dict(sorted(mapping.items()))
        logger.debug("Updated themes: %s", self._themes)
        self._save()
        logger.info("Theme saved to %s", self.path)

    def delete_theme(self, theme: str) -> bool:
        """Delete a theme. Returns True if theme existed and was deleted."""
        theme_key = sanitize_name(theme)
        if theme_key in self._themes:
            del self._themes[theme_key]
            self._save()
            return True
        return False

    def add_chat_to_theme(self, theme: str, chat_name: str, original_name: str) -> None:
        """Add ``chat_name`` (sanitized) with ``original_name`` to ``theme``."""
        theme_key = sanitize_name(theme)
        if theme_key in self._themes:
            chats = self._themes[theme_key]
            chats[chat_name] = original_name
            self._save()

    def remove_chat_from_theme(self, theme: str, chat_name: str) -> bool:
        """Remove sanitized ``chat_name`` from ``theme``.

        Returns ``True`` if the chat was present and removed.
        """
        theme_key = sanitize_name(theme)
        if theme_key in self._themes and chat_name in self._themes[theme_key]:
            del self._themes[theme_key][chat_name]
            self._save()
            return True
        return False

    def migrate_chat_names_to_sanitized(self) -> None:
        """Migrate chat names in all themes to sanitized versions."""
        from index.raw_storage import _sanitize_component

        migrated = False
        new_themes: Dict[str, Dict[str, str]] = {}
        for theme_key, chats in self._themes.items():
            new_map: Dict[str, str] = {}
            if isinstance(chats, dict):
                items = chats.items()
            else:  # legacy list
                items = ((c, c) for c in chats)
            for key, orig in items:
                sanitized = _sanitize_component(key)
                if sanitized != key:
                    logger.debug("Migrating chat name: '%s' -> '%s'", key, sanitized)
                    migrated = True
                new_map[sanitized] = str(orig).strip()
            new_themes[theme_key] = new_map

        if migrated or any(not isinstance(v, dict) for v in self._themes.values()):
            logger.info("Saving migrated theme names")
            self._themes = new_themes
            self._save()
