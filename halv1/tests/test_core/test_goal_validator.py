import asyncio
import hashlib
from typing import Tuple

from agent.core import AgentCore
from core.goal_validator import GoalValidator
from events.models import MessageReceived
from executor import CodeGenerator, SimpleCodeExecutor
from llm.base_client import LLMClient
from memory import UnifiedMemory
from planner import Plan, PlanStep, TaskPlanner, Tool
from services.event_bus import AsyncEventBus
from tools.registry import ArtifactDict, ToolRegistry


class _EchoLLM(LLMClient):
    def generate(self, prompt: str) -> str:  # pragma: no cover - simple stub
        return ""

    def stream(self, prompt: str):  # pragma: no cover - simple stub
        yield ""


class _Planner(TaskPlanner):
    def __init__(self, content: str) -> None:
        self.content = content

    def plan(self, request: str, context=None, previous_results=None) -> Plan:  # type: ignore[override]
        return Plan(steps=[PlanStep(tool=Tool.CODE, content=self.content)], context=[])


class RecordingValidator(GoalValidator):
    def __init__(self) -> None:
        super().__init__()
        self.last: Tuple[bool, str] | None = None

    def validate(self, goal: str, artifacts: ArtifactDict | None) -> Tuple[bool, str]:  # type: ignore[override]
        self.last = super().validate(goal, artifacts)
        return self.last


class _Search:
    async def close(self) -> None:  # pragma: no cover - trivial stub
        return None


async def _run(
    handler, validator: RecordingValidator, content: str
) -> Tuple[Tuple[bool, str] | None, list[str]]:
    bus: AsyncEventBus = AsyncEventBus(workers_per_topic=1)
    planner = _Planner(content)
    executor = SimpleCodeExecutor()
    search = _Search()
    memory = UnifiedMemory()
    generator = CodeGenerator(_EchoLLM())
    registry = ToolRegistry()
    core = AgentCore(
        bus,
        planner,
        executor,
        search,
        memory,
        generator,
        max_iterations=1,  # Установлено в 1, чтобы агент мог выполнить код
        registry=registry,
        validator=validator,
    )
    core.registry.register(Tool.CODE, handler)
    await bus.publish(
        "incoming", MessageReceived(chat_id=1, message_id=1, text="SumNumbers")
    )
    await bus.join()
    await bus.graceful_shutdown()
    await search.close()
    return validator.last, memory.recall()


def test_sum_numbers_success() -> None:
    inputs = ["1", "2", "3"]
    total = sum(map(int, inputs))
    h = hashlib.sha256("".join(inputs).encode()).hexdigest()

    async def handler(step: PlanStep) -> ArtifactDict:
        return {
            "stdout": str(total),
            "sum": {"inputs": inputs, "total": total, "hash": h},
        }

    validator = RecordingValidator()
    result, mem = asyncio.run(_run(handler, validator, "result = 6"))
    assert result == (True, "sum validation passed")
    assert mem == [str(total)]


def test_sum_numbers_bad_hash() -> None:
    inputs = ["1", "2"]
    total = 3

    async def handler(step: PlanStep) -> ArtifactDict:
        return {
            "stdout": str(total),
            "sum": {"inputs": inputs, "total": total, "hash": "bad"},
        }

    validator = RecordingValidator()
    result, _ = asyncio.run(_run(handler, validator, "case2"))
    assert result is not None
    assert result[0] is False
    assert "hash" in result[1]


def test_sum_numbers_missing_artifact() -> None:
    async def handler(step: PlanStep) -> ArtifactDict:
        return {"stdout": "0"}

    validator = RecordingValidator()
    result, _ = asyncio.run(_run(handler, validator, "case3"))
    assert result is not None
    assert result[0] is False
    assert "missing" in result[1]
