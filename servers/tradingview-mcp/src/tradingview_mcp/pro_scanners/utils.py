"""Shared utilities for professional scanners."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Iterable


def make_cache_key(namespace: str, parts: Iterable[Any]) -> str:
    material = ":".join(str(item) for item in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def utcnow_ts() -> float:
    return datetime.utcnow().timestamp()


def encode_payload(data: dict[str, Any]) -> str:
    return json.dumps(data, separators=(",", ":"), default=str)


def decode_payload(data: str) -> dict[str, Any]:
    return json.loads(data)


__all__ = ["make_cache_key", "utcnow_ts", "encode_payload", "decode_payload"]

