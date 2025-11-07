import pytest

from core.utils.json_io import parse_llm_json
from planner import Plan, PlanStep, lint_plan
from tools import Tool


def test_parse_valid_json() -> None:
    assert parse_llm_json('{"a": 1}') == {"a": 1}


def test_parse_json_code_fence() -> None:
    text = """```json
{"a": 1}
```"""
    assert parse_llm_json(text) == {"a": 1}


def test_parse_invalid_json() -> None:
    assert parse_llm_json("{bad}") is None


def test_lint_plan_invalid_reference() -> None:
    plan = Plan(
        steps=[
            PlanStep(tool=Tool.CODE, content="a", id="s1", outputs={"out": "stdout"}),
            PlanStep(
                tool=Tool.CODE,
                content="b",
                id="s2",
                inputs={"x": "<from:s1.missing>"},
            ),
        ],
        context=[],
    )
    with pytest.raises(ValueError):
        lint_plan(plan)
