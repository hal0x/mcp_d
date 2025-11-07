import asyncio
import json
from typing import Any, Iterator, cast
from unittest.mock import AsyncMock, Mock, patch

from agent.core import AgentCore, Phase, PlanGenerated
from events.models import ErrorOccurred
from executor import CodeGenerator, ExecutionResult, SimpleCodeExecutor
from llm.base_client import LLMClient
from memory import UnifiedMemory
from planner import Plan, PlanStep, TaskPlanner, Tool


class DummyBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    def subscribe(self, topic: str, handler: Any) -> None:  # pragma: no cover - stub
        return None

    async def publish(self, topic: str, event: Any) -> None:
        self.published.append((topic, event))


class EchoLLM(LLMClient):
    def generate(self, prompt: str) -> str:  # pragma: no cover - trivial
        return prompt

    def stream(self, prompt: str) -> Iterator[str]:  # pragma: no cover - trivial
        yield prompt


class SimplePlanner(TaskPlanner):
    def __init__(self) -> None:
        self.plan_calls = 0
        self.refine_calls = 0

    def plan(
        self,
        request: str,
        context: list[str] | None = None,
        previous_results: list[str] | None = None,
    ) -> Plan:
        self.plan_calls += 1
        step = PlanStep(tool=Tool.CODE, content="result = 1")
        return Plan(steps=[step], context=["code"])

    def refine(self, request: str, results: list[str]) -> str | None:
        self.refine_calls += 1
        return "next"


class DummySearch:
    async def search_and_summarize(
        self, q: str
    ) -> list[str]:  # pragma: no cover - stub
        return []

    async def fetch_async(self, url: str) -> str:  # pragma: no cover - stub
        return ""


def test_execute_plan_error_triggers_replan() -> None:
    async def main() -> None:
        bus = cast(Any, DummyBus())
        planner = SimplePlanner()
        executor = SimpleCodeExecutor()
        memory = UnifiedMemory()
        generator = CodeGenerator(EchoLLM())
        core = AgentCore(
            bus, planner, executor, cast(Any, DummySearch()), memory, generator
        )
        core.goal = "start"
        core.phase = Phase.PLAN
        core._execute_plan = cast(Any, AsyncMock(side_effect=RuntimeError("boom")))  # type: ignore[method-assign]
        plan = planner.plan("start")
        await core.exec_phase(PlanGenerated(plan=plan))
        assert planner.refine_calls == 1
        assert planner.plan_calls == 2
        topics = [t for t, _ in bus.published]
        assert "errors" in topics and "plan" in topics
        assert any(isinstance(e, ErrorOccurred) for _, e in bus.published)

    asyncio.run(main())


def test_multi_step_step_error_triggers_replan() -> None:
    async def main() -> None:
        bus = cast(Any, DummyBus())
        planner = SimplePlanner()
        executor = SimpleCodeExecutor()
        memory = UnifiedMemory()
        generator = CodeGenerator(EchoLLM())
        core = AgentCore(
            bus, planner, executor, cast(Any, DummySearch()), memory, generator
        )

        class FakeDockerExecutor:
            def execute_multi_step(self, _codes: list[str]) -> ExecutionResult:
                return ExecutionResult(
                    stdout=json.dumps(["", "ok"]),
                    stderr=json.dumps(["Step failed: boom", ""]),
                    files={},
                    returncode=0,
                )

        core.docker_executor = FakeDockerExecutor()  # type: ignore[assignment]
        core.goal = "start"
        core.phase = Phase.PLAN
        plan = Plan(
            steps=[
                PlanStep(tool=Tool.CODE, content="result = 1"),
                PlanStep(tool=Tool.CODE, content="result = 2"),
            ],
            context=["code"],
        )

        await core.exec_phase(PlanGenerated(plan=plan))

        assert planner.refine_calls == 1
        assert planner.plan_calls == 1
        topics = [t for t, _ in bus.published]
        assert "errors" in topics and "plan" in topics
        assert "execution" not in topics
        assert any(
            isinstance(event, ErrorOccurred) and "Step failed: boom" in event.error
            for topic, event in bus.published
            if topic == "errors"
        )
        assert core.phase == Phase.PLAN
        assert core.iterations == 1

    asyncio.run(main())


def test_multi_step_exception_triggers_replan() -> None:
    async def main() -> None:
        bus = cast(Any, DummyBus())
        planner = SimplePlanner()
        executor = SimpleCodeExecutor()
        memory = UnifiedMemory()
        generator = CodeGenerator(EchoLLM())
        core = AgentCore(
            bus, planner, executor, cast(Any, DummySearch()), memory, generator
        )

        class FailingDockerExecutor:
            def execute_multi_step(self, _codes: list[str]) -> ExecutionResult:
                raise RuntimeError("Step 1 raised boom")

        core.docker_executor = FailingDockerExecutor()  # type: ignore[assignment]
        core.goal = "start"
        core.phase = Phase.PLAN
        plan = Plan(
            steps=[
                PlanStep(tool=Tool.CODE, content="result = 1"),
                PlanStep(tool=Tool.CODE, content="result = 2"),
            ],
            context=["code"],
        )

        await core.exec_phase(PlanGenerated(plan=plan))

        assert planner.refine_calls == 1
        assert planner.plan_calls == 1
        topics = [t for t, _ in bus.published]
        assert "errors" in topics and "plan" in topics
        assert "execution" not in topics
        assert any(
            isinstance(event, ErrorOccurred) and "Step 1 raised boom" in event.error
            for topic, event in bus.published
            if topic == "errors"
        )
        assert core.phase == Phase.PLAN
        assert core.iterations == 1

    asyncio.run(main())


def test_incomplete_outputs_trigger_replan() -> None:
    async def main() -> None:
        bus = cast(Any, DummyBus())
        planner = SimplePlanner()
        executor = SimpleCodeExecutor()
        memory = UnifiedMemory()
        generator = CodeGenerator(EchoLLM())
        core = AgentCore(
            bus, planner, executor, cast(Any, DummySearch()), memory, generator
        )
        core.goal = "start"
        core.phase = Phase.PLAN
        artifact = {"stdout": "incomplete", "stderr": "", "files": {}}
        core._execute_plan = cast(Any, AsyncMock(return_value=([(0, artifact)], [])))  # type: ignore[method-assign]
        plan = planner.plan("start")
        await core.exec_phase(PlanGenerated(plan=plan))
        assert planner.refine_calls == 1
        assert planner.plan_calls == 2
        assert [t for t, _ in bus.published] == ["plan"]

    asyncio.run(main())


def test_memory_write_failure_no_retry() -> None:
    async def main() -> None:
        bus = cast(Any, DummyBus())
        planner = SimplePlanner()
        executor = SimpleCodeExecutor()
        memory = UnifiedMemory()
        generator = CodeGenerator(EchoLLM())
        core = AgentCore(
            bus, planner, executor, cast(Any, DummySearch()), memory, generator
        )
        remember = Mock(side_effect=RuntimeError("boom"))
        core.memory.remember = remember  # type: ignore[method-assign]
        with patch("agent.core.logger.warning") as warn_mock:
            await core._handle_plan_errors(["err"], replan=False)
        assert remember.call_count == 1
        warn_mock.assert_called_once()

    asyncio.run(main())
