"""Generate summaries of ingested messages (classic and agent-style)."""

from __future__ import annotations

import logging
from typing import Iterable, Protocol

from llm.prompts import make_agent_summary_prompt
from llm.utils import unwrap_response


class LLMClient(Protocol):
    """Minimal protocol for text-generating clients."""

    def generate(self, prompt: str) -> str:
        """Return a completion for ``prompt``."""


logger = logging.getLogger(__name__)


class Summarizer:
    """Use an LLM client to summarize batches of messages.

    The client may be any object implementing ``generate(prompt: str) -> str``.
    """

    def __init__(self, client: LLMClient) -> None:
        """Initialize the summarizer with a text-generating client.

        Parameters:
            client: Any LLM client with a ``generate`` method.
        """
        self.client = client

    def summarize(self, messages: Iterable[str]) -> str:
        """Return a short summary of ``messages``."""
        joined = "\n".join(messages)
        prompt = (
            "Сформулируй краткую сводку по сообщениям ниже, сохрани ключевые факты.\n"
            "Выведи только обычный текст — без JSON, планов, шагов или кода.\n"
            + joined
        )
        result = self.client.generate(prompt)
        text, _ = unwrap_response(result)
        return text

    def summarize_cluster(self, messages: Iterable[str]) -> str:
        """Return a brief topic overview for a cluster of messages."""
        joined = "\n".join(messages)
        prompt = (
            "Опиши основную тему на основе ключевых сообщений.\n"
            "Выведи только обычный текст — без JSON, планов, шагов или кода.\n"
            + joined
        )
        result = self.client.generate(prompt)
        text, _ = unwrap_response(result)
        return text

    # ------------------------------------------------------------------
    def summarize_as_agent(
        self,
        *,
        mode: str,
        user_name: str,
        theme: str,
        timezone: str,
        window_start: str,
        window_end: str,
        now_iso: str,
        messages_block: str,
    ) -> str:
        """Agent-style prompt that highlights important items and proposes actions.

        Parameters are strings already formatted for insertion (ISO datetimes).
        ``messages_block`` contains lines like: ``[id]|[chat]|[author]|[ISO-datetime]|text``.
        """

        prompt = make_agent_summary_prompt(
            mode=mode,
            user_name=user_name,
            theme=theme,
            timezone=timezone,
            window_start=window_start,
            window_end=window_end,
            now_iso=now_iso,
            messages_block=messages_block,
        )
        logger.debug(
            "Summarizer.generate_as_agent called with mode=%s, theme=%s, messages_block length=%d",
            mode,
            theme,
            len(messages_block),
        )
        result = self.client.generate(prompt)
        text_result, extra = unwrap_response(result)
        if extra is not None:
            logger.debug(
                "Summarizer.client.generate returned tuple: text=%s... (length: %d), context length=%d",
                text_result[:100] if text_result else "None",
                len(text_result) if text_result else 0,
                len(extra) if hasattr(extra, "__len__") else -1,
            )
        else:
            logger.debug(
                "Summarizer.client.generate returned: %s... (length: %d)",
                text_result[:100] if text_result else "None",
                len(text_result) if text_result else 0,
            )

        return text_result
