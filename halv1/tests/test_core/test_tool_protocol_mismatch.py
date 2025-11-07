from __future__ import annotations

import pytest

from agent.tool_protocol import agent_step
from planner import PlanStep
from tools import Tool
from tools.registry import ToolRegistry


class EchoLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, prompt: str) -> str:  # pragma: no cover - simple stub
        return self.reply


def test_agent_step_handler_must_return_list() -> None:
    # Registry handler returns a dict (Artifact-like), which current implementation rejects
    def bad_handler(step: PlanStep) -> dict:
        return {"stdout": step.content}

    registry = ToolRegistry()
    registry.register(Tool.CODE, bad_handler)  # type: ignore[arg-type]
    llm = EchoLLM('{"tool": "code", "arguments": {"code": "print(1)"}}')
    with pytest.raises(ValueError, match="Tool handlers must return a list of strings"):
        agent_step(llm, registry, "demo")

