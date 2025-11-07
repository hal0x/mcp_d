"""Storage utilities for VectorIndex entries and weights."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, TypeVar

from core.utils.json_io import load_json, save_json

T = TypeVar("T")


def load_index(
    path: str | Path,
    *,
    entry_factory: Callable[[dict[str, Any]], T],
) -> tuple[List[T], Counter[str]]:
    """Load entries and weights from ``path`` using ``entry_factory``.

    The JSON file is expected to contain a list of dicts where each dict has
    ``chunk_id``, ``text``, ``embedding``, ``metadata``, ``timestamp`` and an
    optional ``weight`` (defaults to 1).
    """
    raw: List[dict[str, Any]] = load_json(path, [])
    entries: List[T] = []
    weights: Counter[str] = Counter()
    for item in raw:
        weight = int(item.get("weight", 1))
        entry = entry_factory(item)
        entries.append(entry)
        cid = item.get("chunk_id")
        if isinstance(cid, str):
            weights[cid] = weight
    return entries, weights


def save_index(
    path: str | Path,
    entries: Iterable[object],
    weights: Counter[str],
    *,
    to_dict: Callable[[object], Dict[str, Any]],
) -> None:
    """Save entries and weights to ``path`` using ``to_dict`` for each entry."""
    data: List[Dict[str, Any]] = []
    for entry in entries:
        item = to_dict(entry)
        cid = item.get("chunk_id")
        item["weight"] = weights.get(cid, 1) if isinstance(cid, str) else 1
        data.append(item)
    save_json(path, data)

