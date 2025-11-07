from __future__ import annotations

from typing import Iterable, List


def deduplicate(items: Iterable[str], *, case_sensitive: bool = False) -> List[str]:
    """Return ``items`` without duplicates while preserving order.

    When ``case_sensitive`` is ``False`` (default), comparison ignores case so
    strings that differ only by letter case are treated as duplicates.
    """
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        key = item if case_sensitive else item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
