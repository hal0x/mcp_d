"""User-facing chat commands.

This module exposes a small helper ``handle_command`` which inspects an
incoming chat message and, if it begins with a supported slash command,
performs the associated action and returns a human readable response. The
function returns ``None`` when the message does not represent a command so the
caller can fall back to normal agent processing.

Implemented commands:

``/remember <text>``
    Store ``text`` in long-term memory.
``/forget <query>``
    Remove all memories containing ``query`` (case-insensitive).
``/why <id>``
    Explain relations for memory item ``id``.
``/snapshot``
    Show the contents of both short- and long-term memory.
"""

from __future__ import annotations

from typing import Any, List, Optional

from memory import MemoryService

__all__ = ["handle_command"]


def _cmd_remember(memory: MemoryService, text: str) -> str:
    if not text:
        return "Использование: /remember <текст>"
    result = memory.remember(text, frozen=True)
    if result.node_id is None:
        return f"Пропустил: {text}"
    return f"Запомнил: {text}"


def _cmd_forget(memory: MemoryService, query: str) -> str:
    if not query:
        return "Использование: /forget <запрос>"
    removed: List[str] = []
    for item in memory.search(query):
        if memory.forget(item, archive=memory.archive):
            removed.append(item)
    if not removed:
        return "Ничего не найдено"
    return "Забыл: " + "; ".join(removed)


def _cmd_why(memory: MemoryService, item_id: str) -> str:
    if not item_id or not item_id.isdigit():
        return "Использование: /why <id>"
    path = memory.explain(int(item_id))
    if not path:
        return "Не найдено"
    lines: List[str] = []
    first = path[0][1]
    lines.append(f"{first.id}: {first.content}")
    for rel, node in path[1:]:
        lines.append(f"{rel} -> {node.id}: {node.content}")
    return "\n".join(lines)


def _cmd_snapshot(memory: MemoryService) -> str:
    episodes = memory.read_events()
    short = memory.recall(long_term=False)
    long = memory.recall(long_term=True)
    schemas = memory.read_schemas()
    lines: List[str] = ["Эпизоды:"]
    lines.extend(episodes or ["<пусто>"])
    lines.append("Краткосрочная память:")
    lines.extend(short or ["<пусто>"])
    lines.append("Долгосрочная память:")
    lines.extend(long or ["<пусто>"])
    lines.append("Схемы:")
    if schemas:
        for name, eps in schemas:
            lines.append(f"{name}: {'; '.join(eps)}")
    else:
        lines.append("<пусто>")
    return "\n".join(lines)


def handle_command(
    text: str, memory: MemoryService, agent: Any | None = None
) -> Optional[str]:
    """Handle slash commands in ``text``.

    Parameters
    ----------
    text:
        Raw user input.
    memory:
        Memory store instance used by the agent.
    agent:
        Optional object reserved for future extensions.
    """

    if not text.startswith("/"):
        return None

    cmd, _, arg = text.partition(" ")
    cmd = cmd.lower()
    arg = arg.strip()

    if cmd in {"/remember", "/remember!"}:
        return _cmd_remember(memory, arg)
    if cmd == "/forget":
        return _cmd_forget(memory, arg)
    if cmd == "/why":
        return _cmd_why(memory, arg)
    if cmd == "/snapshot":
        return _cmd_snapshot(memory)
    return None
