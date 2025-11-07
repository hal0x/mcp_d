"""Core package public API.

Provides lazy accessors to avoid importing heavy dependencies at import time.
"""

__all__ = ["Agent"]

def __getattr__(name: str):  # pragma: no cover - simple import shim
    if name == "Agent":
        from .agent import Agent  # Local import to avoid side effects early
        return Agent
    raise AttributeError(f"module 'core' has no attribute {name!r}")
