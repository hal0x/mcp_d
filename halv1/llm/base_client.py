"""Common protocol for LLM clients."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Protocol, Tuple, TypedDict, runtime_checkable


class ConversationMessage(TypedDict):
    """Single message in a conversation history."""

    role: str
    content: str


ConversationHistory = List[ConversationMessage]


@runtime_checkable
class LLMClient(Protocol):
    """Abstract client capable of generating text via a language model."""

    def generate(self, prompt: str) -> str:
        """Return a completion for ``prompt``."""

    def stream(self, prompt: str) -> Iterable[str]:
        """Yield chunks of a completion for ``prompt``."""


@runtime_checkable
class OptimizedLLMClient(Protocol):
    """Enhanced LLM client with conversation history support."""

    def generate(
        self, prompt: str, history: Optional[ConversationHistory] = None
    ) -> Tuple[str, ConversationHistory]:
        """Return a completion for ``prompt`` with conversation history.

        Returns
        -------
        Tuple[str, ConversationHistory]
            Generated response and updated history to reuse on subsequent calls.
        """

    def stream(
        self, prompt: str, history: Optional[ConversationHistory] = None
    ) -> Iterable[Tuple[str, ConversationHistory]]:
        """Yield chunks of a completion for ``prompt`` with conversation history.

        Yields
        ------
        Tuple[str, ConversationHistory]
            Response chunk and updated history after including the chunk.
        """


@runtime_checkable
class EmbeddingsClient(Protocol):
    """Client capable of producing embeddings for a given text."""

    def embed(self, text: str) -> List[float]:
        """Return an embedding vector for ``text``."""
