from planner import PlanStep
from tools import Tool
from tools.registry import ToolRegistry


def _dummy(step: PlanStep) -> list[str]:
    return ["ok"]


def test_try_get_returns_none_when_missing() -> None:
    registry = ToolRegistry()
    assert registry.try_get(Tool.SEARCH) is None
    registry.register(Tool.SEARCH, _dummy)
    assert registry.try_get(Tool.SEARCH) is _dummy
