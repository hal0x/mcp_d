"""Persistence helpers for Telegram bot state.

This module defines :class:`PersistedList` and :class:`PersistedDict`, which
mirror the interface of ``list`` and ``dict`` but automatically trigger a save
callback whenever their content changes.  They are used by the Telegram bot to
persist in-memory state to disk without scattering persistence logic across the
codebase.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any, SupportsIndex

logger = logging.getLogger(__name__)


class PersistedList(list[Any]):
    """List that triggers a callback on mutation."""

    def __init__(
        self,
        iterable: Iterable[Any] = (),
        save_callback: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(iterable)
        self._save_callback = save_callback
        self._wrap_children()

    def _wrap(self, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, PersistedDict):
            return PersistedDict(value, save_callback=self._save_callback)
        if isinstance(value, list) and not isinstance(value, PersistedList):
            return PersistedList(value, save_callback=self._save_callback)
        return value

    def _wrap_children(self) -> None:
        for i in range(len(self)):
            super().__setitem__(i, self._wrap(super().__getitem__(i)))

    def _trigger(self) -> None:
        if self._save_callback:
            try:
                self._save_callback()
            except Exception:  # pragma: no cover - best effort
                logger.exception("Failed to persist state")

    # Mutation methods -------------------------------------------------
    def append(self, value: Any) -> None:
        super().append(self._wrap(value))
        self._trigger()

    def extend(self, iterable: Iterable[Any]) -> None:
        for item in iterable:
            self.append(item)

    def __setitem__(self, index: SupportsIndex | slice, value: Any) -> None:
        super().__setitem__(index, self._wrap(value))
        self._trigger()

    def __delitem__(self, index: SupportsIndex | slice) -> None:
        super().__delitem__(index)
        self._trigger()

    def insert(self, index: SupportsIndex, value: Any) -> None:
        super().insert(index, self._wrap(value))
        self._trigger()

    def pop(self, index: SupportsIndex = -1) -> Any:
        val = super().pop(index)
        self._trigger()
        return val

    def remove(self, value: Any) -> None:
        super().remove(value)
        self._trigger()

    def clear(self) -> None:
        super().clear()
        self._trigger()


class PersistedDict(dict[str, Any]):
    """Dictionary that triggers a callback on mutation."""

    def __init__(
        self,
        *args: Any,
        save_callback: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._save_callback = save_callback
        self._wrap_children()

    def _wrap(self, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, PersistedDict):
            return PersistedDict(value, save_callback=self._save_callback)
        if isinstance(value, list) and not isinstance(value, PersistedList):
            return PersistedList(value, save_callback=self._save_callback)
        return value

    def _wrap_children(self) -> None:
        for k in list(self.keys()):
            super().__setitem__(k, self._wrap(super().__getitem__(k)))

    def _trigger(self) -> None:
        if self._save_callback:
            try:
                self._save_callback()
            except Exception:  # pragma: no cover - best effort
                logger.exception("Failed to persist state")

    # Mutation methods -------------------------------------------------
    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._wrap(value))
        self._trigger()

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._trigger()

    def update(self, *args: Any, **kwargs: Any) -> None:
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def pop(self, key: str, *args: Any) -> Any:
        val = super().pop(key, *args)
        self._trigger()
        return val

    def clear(self) -> None:
        super().clear()
        self._trigger()

    def popitem(self) -> tuple[str, Any]:
        item = super().popitem()
        self._trigger()
        return item

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
            return self[key]
        return super().setdefault(key, default)


__all__ = ["PersistedList", "PersistedDict"]
