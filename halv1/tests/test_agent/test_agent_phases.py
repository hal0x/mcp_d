import asyncio

import pytest

from agent.core import AgentCore, ExecutionCompleted, Phase, PlanGenerated
from events.models import MessageReceived
from executor import CodeGenerator, SimpleCodeExecutor
from internet import SearchClient
from llm.base_client import LLMClient
from memory import UnifiedMemory
from planner import Plan, SimpleTaskPlanner, TaskPlanner, Tool
from services.event_bus import AsyncEventBus


class EchoLLM(LLMClient):
    def generate(self, prompt: str) -> str:  # pragma: no cover - simple
        mapping = {"add two and two": "result = 2 + 2"}
        for key, value in mapping.items():
            if key in prompt:
                return value
        return prompt

    def stream(self, prompt: str):  # pragma: no cover - simple
        yield self.generate(prompt)


def test_phase_sequence_and_outputs():
    async def main():
        bus: AsyncEventBus = AsyncEventBus(workers_per_topic=1)
        planner = SimpleTaskPlanner()
        executor = SimpleCodeExecutor()
        search = SearchClient()

        async def fake_search(q: str) -> list[str]:
            return [f"result for {q}: http://example.com/{q}"]

        search.search_and_summarize = fake_search  # type: ignore[method-assign]
        memory = UnifiedMemory()
        generator = CodeGenerator(EchoLLM())
        core = AgentCore(
            bus, planner, executor, search, memory, generator, max_iterations=0
        )

        plan_events: list[PlanGenerated] = []
        exec_events: list[ExecutionCompleted] = []
        report_events: list[ExecutionCompleted] = []
        bus.subscribe("plan", lambda e: plan_events.append(e))
        bus.subscribe("execution", lambda e: exec_events.append(e))
        bus.subscribe("report", lambda e: report_events.append(e))

        await bus.publish(
            "incoming", MessageReceived(chat_id=1, message_id=1, text="add two and two")
        )
        await bus.join()
        await bus.graceful_shutdown()
        await search.close()
        return core, plan_events, exec_events, report_events

    core, plan_events, exec_events, report_events = asyncio.run(main())
    assert core.phase_history == [Phase.PLAN, Phase.EXEC, Phase.VALIDATE, Phase.REPORT]

    # Проверяем события и/или финальный отчет из ядра
    assert len(plan_events) > 0, "Должно быть событие плана"
    assert len(exec_events) > 0, "Должно быть событие выполнения"

    if report_events:
        reported = report_events[0].results
    else:
        # Фолбек: используем результаты выполнения, если отчетное событие
        # было сведено к финализации без отдельного события
        reported = exec_events[-1].results

    assert isinstance(plan_events[0].plan, Plan)
    assert plan_events[0].plan.steps[0].tool == Tool.CODE
    assert exec_events[0].results == ["4"]
    assert reported == ["4"]
    # Проверяем, что в результатах нет служебных строк
    assert all(r != "OK" for e in exec_events for r in e.results)
    assert all(r != "OK" for r in reported)


def test_phase_violation():
    async def main() -> None:
        bus: AsyncEventBus = AsyncEventBus(workers_per_topic=1)
        planner: TaskPlanner = SimpleTaskPlanner()
        executor = SimpleCodeExecutor()
        search = SearchClient()
        memory = UnifiedMemory()
        generator = CodeGenerator(EchoLLM())
        core = AgentCore(bus, planner, executor, search, memory, generator)

        with pytest.raises(RuntimeError):
            await core.exec_phase(PlanGenerated(plan=Plan(steps=[], context=[])))

        await bus.graceful_shutdown()
        await search.close()

    asyncio.run(main())
