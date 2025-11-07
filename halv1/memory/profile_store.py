"""Persistent user profile storage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_PROFILE_PATH = Path("~/.hal_assistant/profile.json").expanduser()


@dataclass
class Profile:
    preferred_name: str


class ProfileStore:
    """Simple persistent store for user profile data."""

    def __init__(self, path: str | Path = DEFAULT_PROFILE_PATH) -> None:
        self.path = Path(path).expanduser()

    def read(self) -> Profile | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid profile JSON") from exc
        user = data.get("user", {})
        name = user.get("preferred_name")
        if isinstance(name, str):
            return Profile(preferred_name=name)
        return None

    def write(self, name: str) -> None:
        payload: dict[str, Any] = {
            "user": {"preferred_name": name},
            "meta": {
                "updated_at": datetime.now(UTC).isoformat(),
                "source": "assistant",
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
