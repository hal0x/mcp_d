"""Tests ensuring agent memory uses the configured persistent path."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.usefixtures("tmp_path")
def test_create_agent_memory_uses_configured_path(tmp_path: Path) -> None:
    from main import create_agent_memory  # type: ignore[attr-defined]

    cfg = {"paths": {"agent_memory": str(tmp_path / "agent_memory.json")}}

    memory = create_agent_memory(cfg, llm_client=None)

    assert getattr(memory, "path", None) == str(tmp_path / "agent_memory.json")

    memory.remember("persist me", long_term=True)
    memory.save()

    from memory import MemoryServiceAdapter

    reloaded = MemoryServiceAdapter(path=str(tmp_path / "agent_memory.json"))
    assert "persist me" in reloaded.recall(long_term=True)
