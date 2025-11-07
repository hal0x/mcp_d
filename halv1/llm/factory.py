"""Factory for creating LLM clients based on configuration."""

from __future__ import annotations

from typing import Any, Dict

from .base_client import LLMClient
from .lmstudio_client import LMStudioClient
from .ollama_client import OllamaClient


def create_llm_client(
    provider: str,
    llm_cfg: Dict[str, Any] | None = None,
    ollama_cfg: Dict[str, Any] | None = None,
) -> LLMClient:
    """Return an LLM client for ``provider``.

    Parameters
    ----------
    provider:
        Name of the backend, e.g. ``"lmstudio"`` or ``"ollama"``.
        Unknown values default to Ollama.
    llm_cfg:
        Generic configuration for the chosen provider. Used mainly for
        LMStudio but also supplies default model for Ollama.
    ollama_cfg:
        Specific configuration for the Ollama backend. Only consulted when
        ``provider`` is ``"ollama"`` or unknown.
    """

    llm_cfg = llm_cfg or {}
    ollama_cfg = ollama_cfg or {}
    provider = provider.lower()

    if provider == "lmstudio":
        # Optional advanced parameters
        stop_list = llm_cfg.get("stop") or ["```", "INCOMPLETE", "COMPLETE"]
        return LMStudioClient(
            model=llm_cfg.get("model", ""),
            host=llm_cfg.get("host", "127.0.0.1"),
            port=int(llm_cfg.get("port", 1234)),
            api_key=llm_cfg.get("api_key") or None,
            use_chat=True,
            temperature=llm_cfg.get("temperature"),
            top_p=llm_cfg.get("top_p"),
            max_tokens=llm_cfg.get("max_tokens"),
            stop=stop_list,
            seed=llm_cfg.get("seed"),
            num_ctx=llm_cfg.get("num_ctx"),
            num_keep=llm_cfg.get("num_keep"),
            max_concurrency=int(llm_cfg.get("max_concurrency", 1)),
        )

    # Default to Ollama
    return OllamaClient(
        model=ollama_cfg.get("model", llm_cfg.get("model", "gemma3n:e4b-it-q8_0")),
        host=ollama_cfg.get("host", llm_cfg.get("host", "localhost")),
        port=int(ollama_cfg.get("port", llm_cfg.get("port", 11434))),
        keep_alive=ollama_cfg.get("keep_alive", llm_cfg.get("keep_alive", "5m")),
        num_batch=ollama_cfg.get("num_batch", llm_cfg.get("num_batch", 512)),
        num_ctx=ollama_cfg.get("num_ctx", llm_cfg.get("num_ctx", 32000)),
    )
