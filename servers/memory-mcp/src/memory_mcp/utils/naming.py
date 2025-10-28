#!/usr/bin/env python3
"""Utility helpers for consistent slug/identifier generation."""

from __future__ import annotations

import re
import unicodedata

_CYRILLIC_TRANSLIT = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "j",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "h",
        "ц": "c",
        "ч": "ch",
        "ш": "sh",
        "щ": "sch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
        "А": "A",
        "Б": "B",
        "В": "V",
        "Г": "G",
        "Д": "D",
        "Е": "E",
        "Ё": "E",
        "Ж": "Zh",
        "З": "Z",
        "И": "I",
        "Й": "J",
        "К": "K",
        "Л": "L",
        "М": "M",
        "Н": "N",
        "О": "O",
        "П": "P",
        "Р": "R",
        "С": "S",
        "Т": "T",
        "У": "U",
        "Ф": "F",
        "Х": "H",
        "Ц": "C",
        "Ч": "Ch",
        "Ш": "Sh",
        "Щ": "Sch",
        "Ъ": "",
        "Ы": "Y",
        "Ь": "",
        "Э": "E",
        "Ю": "Yu",
        "Я": "Ya",
    }
)


def slugify(value: str | None, *, default: str = "unnamed") -> str:
    """Produce a path-friendly slug while preserving readability."""

    if not value:
        return default

    normalized = unicodedata.normalize("NFKD", value.strip())
    transliterated = normalized.translate(_CYRILLIC_TRANSLIT)
    ascii_only = transliterated.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower()
    compact = re.sub(r"[^a-z0-9]+", "_", lowered)
    compact = re.sub(r"_+", "_", compact).strip("_")
    return compact or default


__all__ = ["slugify"]
