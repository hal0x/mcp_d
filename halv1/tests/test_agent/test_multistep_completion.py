from __future__ import annotations

import pytest

from agent.multistep import MultiStepPlanMixin
from planner.task_planner import Plan, PlanStep
from tools import Tool


class _DummyMultiStep(MultiStepPlanMixin):
    pass


@pytest.fixture
def mixin() -> MultiStepPlanMixin:
    return _DummyMultiStep()


def _make_step(expected: str) -> PlanStep:
    return PlanStep(tool=Tool.CODE, content="", expected_output=expected)


def test_step_completed_with_punctuation_variation(mixin: MultiStepPlanMixin) -> None:
    step = _make_step("Sum: 6")
    results = ["Computation done - sum = 6!"]

    assert mixin._is_step_completed(step, results)


def test_step_completed_with_synonyms(mixin: MultiStepPlanMixin) -> None:
    step = _make_step("file created")
    results = ["File has been generated successfully."]

    assert mixin._is_step_completed(step, results)


def test_step_not_completed_when_numbers_mismatch(mixin: MultiStepPlanMixin) -> None:
    step = _make_step("Sum: 10")
    results = ["Sum equals 6"]

    assert not mixin._is_step_completed(step, results)


def test_step_completed_with_case_and_synonym_variations(mixin: MultiStepPlanMixin) -> None:
    step = _make_step("Task COMPLETE")
    results = ["task finished. SUCCESS!"]

    assert mixin._is_step_completed(step, results)


def test_llm_assessment_with_tuple_response_stops_continuation(
    mixin: MultiStepPlanMixin,
) -> None:
    history = [{"role": "assistant", "content": "Done"}]

    class _TupleClient:
        def __init__(self) -> None:
            self.last_prompt: str | None = None

        def generate(self, prompt: str):  # type: ignore[override]
            self.last_prompt = prompt
            return "COMPLETE", history

    class _DummyCodeGenerator:
        def __init__(self) -> None:
            self.client = _TupleClient()
            self.conversation_history = None

    mixin.goal = "Finish the example task"
    mixin.iterations = 0
    mixin.code_generator = _DummyCodeGenerator()

    plan = Plan(steps=[PlanStep(tool=Tool.CODE, content="", is_final=True)], context=[])
    should_continue = mixin._needs_continuation(["Step result"], plan)

    assert not should_continue
    assert mixin.code_generator.conversation_history == history
    assert mixin.code_generator.client.last_prompt is not None
