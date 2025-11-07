from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypeVar, cast

T = TypeVar("T")


def load_json(path: Path, default: T) -> T:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default
    return cast(T, data)


def save_json(path: Path, data: Any, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )


_CODE_BLOCK_RE = re.compile(r"^```(?:json)?\n(?P<content>.*)\n```$", re.DOTALL)


def parse_llm_json(text: str) -> dict[str, Any] | None:
    """Parse JSON from LLM output.

    The model may wrap JSON in triple backtick fences. Surrounding fences are
    stripped before parsing. Returns ``None`` if the text is not valid JSON.
    """

    stripped = text.strip()
    match = _CODE_BLOCK_RE.match(stripped)
    if match:
        stripped = match.group("content")
    try:
        return cast(dict[str, Any], json.loads(stripped))
    except json.JSONDecodeError:
        return None
