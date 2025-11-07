"""Utilities for working with LLM client responses."""

from __future__ import annotations

from typing import Any, Tuple


def unwrap_response(result: Any) -> Tuple[str, Any | None]:
    """Extract the textual portion from a raw LLM response container.

    The public ``generate`` API is expected to return a plain string.  Some
    clients, such as context-aware wrappers, may return richer structures like
    ``(text, history)`` tuples.  This helper normalises those outputs so callers
    can reliably work with a string while still accessing the optional
    additional payload (e.g. updated conversation history).

    Parameters
    ----------
    result:
        Raw value returned by ``LLMClient.generate`` or similar.

    Returns
    -------
    Tuple[str, Any | None]
        A tuple with the extracted text and optional extra data (typically a
        conversation history).  The text component is guaranteed to be a
        string, defaulting to ``""`` when it cannot be determined.
    """

    history: Any | None = None
    text: Any = result

    if isinstance(result, tuple):
        # Common case for context-aware clients returning ``(text, history)``.
        if result:
            text = result[0]
            if len(result) > 1:
                history = result[1]
        else:
            text = ""
    elif isinstance(result, dict):
        # Some providers may return a dict-like payload.
        text = result.get("text") or result.get("content") or result
    elif hasattr(result, "text"):
        text = getattr(result, "text")

    if text is None:
        text_str = ""
    elif isinstance(text, str):
        text_str = text
    else:
        text_str = str(text)

    return text_str, history

