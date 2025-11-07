import asyncio

from agent import AgentCore
from events.models import MessageReceived
from executor import CodeGenerator, SimpleCodeExecutor
from internet import SearchClient
from llm.base_client import LLMClient
from memory import UnifiedMemory
from test_memory.test_memory_persistence import DummyEmbeddings
from planner import Plan, PlanStep, TaskPlanner, Tool
from services.event_bus import AsyncEventBus


class FakeLLM(LLMClient):
    """LLM stub that handles code generation, evaluation and refinement."""

    def __init__(self) -> None:
        self.eval_calls = 0

    def generate(self, prompt: str) -> str:  # pragma: no cover - simple mapping
        if prompt.startswith("Determine if"):
            self.eval_calls += 1
            return "yes" if "result for python" in prompt else "no"
        if prompt.startswith("Refine"):
            return "search python" if "- 2" in prompt else prompt
        if "1 + 1" in prompt:
            return "result = 1 + 1"
        return prompt

    def stream(self, prompt: str):  # pragma: no cover - unused
        yield self.generate(prompt)


class IterativePlanner(TaskPlanner):
    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.calls = 0

    def plan(
        self,
        request: str,
        context: list[str] | None = None,
        previous_results: list[str] | None = None,
    ) -> Plan:
        if self.calls == 0:
            self.calls += 1
            return Plan(
                steps=[PlanStep(tool=Tool.CODE, content="result = 1 + 1")],
                context=["code"],
            )
        self.calls += 1
        return Plan(
            steps=[PlanStep(tool=Tool.SEARCH, content="python")],
            context=["search"],
        )


def test_goal_requires_multiple_iterations():
    async def main():
        bus: AsyncEventBus = AsyncEventBus(workers_per_topic=1)
        llm = FakeLLM()
        planner = IterativePlanner(llm)
        executor = SimpleCodeExecutor()
        search = SearchClient()

        async def fake_search(q: str) -> list[str]:
            return [f"result for {q}: http://example.com/{q}"]

        search.search_and_summarize = fake_search
        memory = UnifiedMemory(embeddings_client=DummyEmbeddings())
        generator = CodeGenerator(llm)
        AgentCore(bus, planner, executor, search, memory, generator, max_iterations=3)

        await bus.publish(
            "incoming",
            MessageReceived(
                chat_id=1,
                message_id=1,
                text="compute 1+1 then search python",
            ),
        )
        await bus.join()
        await bus.graceful_shutdown()
        await search.close()
        return memory.recall(), planner.calls, llm.eval_calls

    results, plan_calls, eval_calls = asyncio.run(main())
    assert results == ["2", "result for python: http://example.com/python"]
    assert plan_calls == 3  # Планировщик вызывается 3 раза
    assert eval_calls == 2


class DoneLLM(LLMClient):
    def generate(self, prompt: str) -> str:  # pragma: no cover - simple logic
        if prompt.startswith("Determine if"):
            return "yes" if "done" in prompt else "no"
        if "'incomplete'" in prompt:
            return "result = 'incomplete'"
        if "'done'" in prompt:
            return "result = 'done'"
        return prompt


class LoopPlanner(TaskPlanner):
    def __init__(self) -> None:
        self.plan_calls = 0
        self.refine_calls = 0

    def plan(
        self,
        request: str,
        context: list[str] | None = None,
        previous_results: list[str] | None = None,
    ) -> Plan:
        if self.plan_calls == 0:
            self.plan_calls += 1
            return Plan(
                steps=[PlanStep(tool=Tool.CODE, content="result = 'incomplete'")],
                context=["code"],
            )
        self.plan_calls += 1
        return Plan(
            steps=[PlanStep(tool=Tool.CODE, content="result = 'done'")],
            context=["code"],
        )

    def refine(self, request: str, results: list[str]) -> str | None:  # type: ignore[override]
        self.refine_calls += 1
        return "next"


def test_incomplete_step_triggers_refine():
    async def main():
        bus: AsyncEventBus = AsyncEventBus(workers_per_topic=1)
        llm = DoneLLM()
        planner = LoopPlanner()
        executor = SimpleCodeExecutor()
        search = SearchClient()
        memory = UnifiedMemory(embeddings_client=DummyEmbeddings())
        generator = CodeGenerator(llm)
        AgentCore(bus, planner, executor, search, memory, generator, max_iterations=2)

        await bus.publish(
            "incoming", MessageReceived(chat_id=1, message_id=1, text="start")
        )
        await bus.join()
        await bus.graceful_shutdown()
        await search.close()
        return memory.recall(), planner.plan_calls, planner.refine_calls

    results, plan_calls, refine_calls = asyncio.run(main())
    assert results == ["done"]
    assert (
        plan_calls >= 2
    ), f"Планировщик должен быть вызван минимум 2 раза, получено: {plan_calls}"
    assert (
        refine_calls >= 1
    ), f"Refine должен быть вызван минимум 1 раз, получено: {refine_calls}"
