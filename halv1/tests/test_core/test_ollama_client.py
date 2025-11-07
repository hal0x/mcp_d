"""Tests for the Ollama LLM client endpoints and fallbacks."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm.ollama_client import OllamaClient


def _mock_response(payload: dict[str, Any], status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


@patch("llm.ollama_client.requests.post")
def test_generate_prefers_chat_endpoint(mock_post: MagicMock) -> None:
    mock_post.return_value = _mock_response(
        {"message": {"role": "assistant", "content": "pong"}}
    )

    client = OllamaClient(host="example.org", port=11434)
    reply, history = client.generate("ping")

    assert reply == "pong"
    assert history[-1]["content"] == "pong"

    called_url = mock_post.call_args.args[0]
    assert called_url.endswith("/api/chat")

    payload = mock_post.call_args.kwargs["json"]
    assert payload["messages"][-1] == {"role": "user", "content": "ping"}


@patch("llm.ollama_client.requests.post")
def test_generate_falls_back_to_legacy_endpoint(mock_post: MagicMock) -> None:
    chat_response = _mock_response({}, status_code=404)
    legacy_response = _mock_response({"response": "legacy"})
    mock_post.side_effect = [chat_response, legacy_response]

    client = OllamaClient(host="localhost", port=11434)

    reply, _ = client.generate("hi")

    assert reply == "legacy"
    assert mock_post.call_args_list[0].args[0].endswith("/api/chat")
    assert mock_post.call_args_list[1].args[0].endswith("/api/generate")

