import asyncio
import json
from dataclasses import dataclass
from typing import Any, Iterator, List

import pytest

from agent.core import AgentCore, ExecutionCompleted, Phase, PlanGenerated
from events.models import MessageReceived
from executor.code_executor import CodeExecutor, ExecutionResult
from llm.base_client import LLMClient
from planner import Plan, PlanStep, TaskPlanner, Tool
from services.event_bus import AsyncEventBus


class _StaticPlanner(TaskPlanner):
    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    def plan(
        self,
        request: str,
        context: List[str] | None = None,
        previous_results: List[str] | None = None,
    ) -> Plan:
        return self._plan

    def refine(self, request: str, results: List[str]) -> str | None:  # pragma: no cover - not needed
        return request


class _AcceptAllValidator:
    def validate(self, goal: str, artifact: dict | None) -> tuple[bool, str]:
        return True, "ok"


class _StubMemory:
    def __init__(self) -> None:
        self._store: list[str] = []

    def remember(self, content: str) -> None:
        self._store.append(content)

    def recall(self, long_term: bool = False) -> list[str]:
        return list(self._store)

    def semantic_search(self, _query: str) -> list[str]:
        return []


class _LLMStub(LLMClient):
    def generate(self, prompt: str) -> str:
        return "COMPLETE"

    def stream(self, prompt: str) -> Iterator[str]:  # pragma: no cover - not used
        yield prompt


@dataclass
class _CodeGeneratorStub:
    client: LLMClient


class _FakeDockerExecutor(CodeExecutor):
    def execute(self, code: str, policy: Any | None = None) -> ExecutionResult:  # pragma: no cover - not used
        return ExecutionResult(stdout=code, stderr="", files={})

    def execute_multi_step(self, codes: list[str]) -> ExecutionResult:
        outputs = [f"step:{idx}" for idx, _ in enumerate(codes)]
        errors = ["" for _ in codes]
        return ExecutionResult(
            stdout=json.dumps(outputs),
            stderr=json.dumps(errors),
            files={},
            returncode=0,
        )


class _NoopSearch:
    async def search_and_summarize(self, _q: str) -> list[str]:  # pragma: no cover - not used
        return []

    async def fetch_async(self, _url: str) -> str:  # pragma: no cover - not used
        return ""


@pytest.mark.asyncio
async def test_multi_step_plan_publishes_final_execution_event() -> None:
    plan = Plan(
        steps=[
            PlanStep(tool=Tool.CODE, content="print('a')"),
            PlanStep(tool=Tool.CODE, content="print('b')", is_final=True),
        ],
        context=[],
    )

    bus: AsyncEventBus[MessageReceived] = AsyncEventBus(workers_per_topic=1)
    planner = _StaticPlanner(plan)
    executor = _FakeDockerExecutor()
    memory = _StubMemory()
    generator = _CodeGeneratorStub(client=_LLMStub())
    validator = _AcceptAllValidator()

    core = AgentCore(
        bus=bus,
        planner=planner,
        executor=executor,
        search=_NoopSearch(),
        memory=memory,
        code_generator=generator,
        validator=validator,
    )

    core.goal = "ensure final"
    core.phase = Phase.PLAN

    captured: list[ExecutionCompleted] = []

    async def capture(event: ExecutionCompleted) -> None:
        captured.append(event)

    bus.subscribe("execution", capture)  # type: ignore[arg-type]

    await core.exec_phase(PlanGenerated(plan=plan))

    await asyncio.wait_for(bus.join(timeout=5.0), timeout=5.0)
    await bus.graceful_shutdown()

    assert captured, "execution events should be emitted"
    assert captured[-1].final, "final execution event must be marked final"
