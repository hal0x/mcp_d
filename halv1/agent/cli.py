"""Simple asynchronous CLI wrapper for agent commands.

The CLI is intentionally minimal: it forwards user input to the provided
``send`` callback unless the message is a supported slash command handled by
``handle_command``.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from memory import UnifiedMemory

from .commands import handle_command

__all__ = ["ChatCLI"]


class ChatCLI:
    """Light‑weight helper used by tests and small demos."""

    def __init__(
        self,
        send: Callable[[str], Awaitable[str]],
        memory: UnifiedMemory,
        agent: Any | None = None,
    ) -> None:
        self._send = send
        self._memory = memory
        self._agent = agent

    async def ask(self, text: str) -> str:
        """Process ``text`` and return a response."""

        cmd = handle_command(text, self._memory, self._agent)
        if cmd is not None:
            return cmd
        return await self._send(text)
