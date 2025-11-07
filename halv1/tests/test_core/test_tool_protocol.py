from __future__ import annotations

from typing import List

from agent.tool_protocol import (
    FinalAnswer,
    ToolCall,
    agent_step,
    try_parse_tool_call,
)
from llm.base_client import LLMClient
from planner import PlanStep
from tools import Tool
from tools.registry import ToolRegistry


class DummyClient:
    def __init__(self, responses: List[str]) -> None:
        self._responses = responses
        self.calls = 0

    def generate(self, prompt: str) -> str:  # pragma: no cover - simple stub
        resp = self._responses[self.calls]
        self.calls += 1
        return resp


def dummy_tool(step: PlanStep) -> List[str]:  # pragma: no cover - simple stub
    return [f"echo: {step.content}"]


def test_try_parse_tool_call_variants() -> None:
    tool_json = '{"tool": "search", "arguments": {"query": "cats"}}'
    final_json = '{"final_answer": "done"}'
    fenced_tool = f"```json\n{tool_json}\n```"
    fenced_invalid = "```json\nnot json\n```"

    parsed_tool = try_parse_tool_call(tool_json)
    parsed_final = try_parse_tool_call(final_json)
    parsed_fenced = try_parse_tool_call(fenced_tool)
    invalid = try_parse_tool_call("not json")
    invalid_fenced = try_parse_tool_call(fenced_invalid)

    assert isinstance(parsed_tool, ToolCall)
    assert isinstance(parsed_fenced, ToolCall)
    assert parsed_tool.tool == "search"
    assert parsed_tool.arguments == {"query": "cats"}
    assert isinstance(parsed_final, FinalAnswer)
    assert parsed_final.final_answer == "done"
    assert invalid is None
    assert invalid_fenced is None


def test_agent_step_with_retry_and_final_answer() -> None:
    responses = [
        "not json",
        '{"tool": "code", "arguments": {"code": "1 + 1"}}',
        '{"final_answer": "2"}',
    ]
    dummy = DummyClient(responses)
    client: LLMClient = dummy  # type: ignore[assignment]
    registry = ToolRegistry()
    registry.register(Tool.CODE, dummy_tool)

    result = agent_step(client, registry, "calc")
    assert result == "2"
    assert dummy.calls == 3


def test_agent_step_argument_validation_reprompts() -> None:
    responses = [
        '{"tool": "search", "arguments": {}}',
        '{"tool": "search", "arguments": {"query": "cats"}}',
        '{"final_answer": "done"}',
    ]
    dummy = DummyClient(responses)
    client: LLMClient = dummy  # type: ignore[assignment]
    registry = ToolRegistry()
    registry.register(Tool.SEARCH, dummy_tool)

    result = agent_step(client, registry, "find cats")
    assert result == "done"
    assert dummy.calls == 3
