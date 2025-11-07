"""Tests for LLM client factory."""

import pytest

from llm.factory import create_llm_client
from llm.ollama_client import OllamaClient


def test_create_llm_client_ollama() -> None:
    cfg = {"model": "test-model", "host": "localhost", "port": 11434}
    client = create_llm_client("ollama", cfg)
    assert isinstance(client, OllamaClient)
    assert client.model == "test-model"
    assert "localhost" in client.base_url
    assert "11434" in client.base_url


def test_create_llm_client_ollama() -> None:
    cfg = {"model": "gemma3n:e4b-it-q8_0"}
    client = create_llm_client("ollama", cfg)
    assert isinstance(client, OllamaClient)
    assert client.model == "gemma3n:e4b-it-q8_0"


def test_create_llm_client_unknown_defaults_to_ollama() -> None:
    client = create_llm_client("unknown")
    assert isinstance(client, OllamaClient)


def test_create_llm_client_with_ollama_config() -> None:
    llm_cfg = {"model": "default-model"}
    ollama_cfg = {"model": "specific-model", "host": "custom-host", "port": 9999}
    client = create_llm_client("ollama", llm_cfg, ollama_cfg)
    assert isinstance(client, OllamaClient)
    assert client.model == "specific-model"
    assert "custom-host" in client.url
    assert "9999" in client.url
