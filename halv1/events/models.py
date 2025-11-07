from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass(frozen=True, kw_only=True)
class Event:
    """Base class for all events.

    Generates a unique ``id`` and UTC ``timestamp`` when not provided to reduce
    boilerplate in tests and callers.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class MessageReceived(Event):
    chat_id: int
    message_id: int
    text: str


@dataclass(frozen=True)
class Indexed(Event):
    message_id: int


@dataclass(frozen=True)
class ReplyRequested(Event):
    chat_id: int
    message_id: int
    query: str


@dataclass(frozen=True)
class ReplyReady(Event):
    chat_id: int
    message_id: int
    reply: str


@dataclass(frozen=True)
class ScheduleTick(Event):
    schedule_id: str


@dataclass(frozen=True)
class ErrorOccurred(Event):
    origin: str
    error: str
    context: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ExecutionCompleted(Event):
    """Event emitted when a code/tool execution finishes.

    Attributes
    ----------
    results:
        A list of human-readable result strings produced by the execution.
    artifact:
        A structured artifact payload, commonly including keys like
        ``stdout``, ``stderr`` and ``files``.
    final:
        Flag showing whether this event represents completion of the overall
        execution plan.
    """

    results: list[str]
    artifact: Dict[str, Any]
    final: bool = False


__all__ = [
    "Event",
    "MessageReceived",
    "Indexed",
    "ReplyRequested",
    "ReplyReady",
    "ScheduleTick",
    "ErrorOccurred",
    "ExecutionCompleted",
]
