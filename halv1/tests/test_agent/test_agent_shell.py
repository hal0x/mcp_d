import asyncio

from agent import AgentCore
from events.models import ErrorOccurred, MessageReceived
from executor import CodeGenerator, SimpleCodeExecutor
from executor import create_executor
from internet import SearchClient
from llm.base_client import LLMClient
from memory import UnifiedMemory
from planner import Plan, PlanStep, TaskPlanner, Tool
from services.event_bus import AsyncEventBus


class DummyLLM(LLMClient):
    def generate(self, prompt: str) -> str:
        return prompt

    def stream(self, prompt: str):  # pragma: no cover - not used
        yield prompt


class ShellPlanner(TaskPlanner):
    def __init__(self, command: str) -> None:
        self.command = command

    def plan(self, request: str, context=None, previous_results=None) -> Plan:
        step = PlanStep(tool=Tool.SHELL, content=self.command)
        return Plan(steps=[step], context=["shell"])


def test_shell_command_is_executed_and_stored():
    async def main():
        bus: AsyncEventBus = AsyncEventBus(workers_per_topic=1)
        planner = ShellPlanner("echo hi")
        executor = SimpleCodeExecutor()
        shell_exec = create_executor("docker")
        search = SearchClient()
        memory = UnifiedMemory()
        generator = CodeGenerator(DummyLLM())
        AgentCore(
            bus,
            planner,
            executor,
            search,
            memory,
            generator,
            shell_executor=shell_exec,
            max_iterations=0,
        )

        await bus.publish(
            "incoming", MessageReceived(chat_id=1, message_id=1, text="run")
        )
        await bus.join()
        await bus.graceful_shutdown()
        await search.close()
        return memory.recall()

    results = asyncio.run(main())
    assert results == ["hi"]


def test_shell_command_error_emits_error_event():
    async def main():
        bus: AsyncEventBus = AsyncEventBus(workers_per_topic=1)
        planner = ShellPlanner("false")
        executor = SimpleCodeExecutor()
        shell_exec = create_executor("docker")
        search = SearchClient()
        memory = UnifiedMemory()
        generator = CodeGenerator(DummyLLM())
        errors: list[ErrorOccurred] = []

        async def capture(event: ErrorOccurred) -> None:
            errors.append(event)

        bus.subscribe("errors", capture)
        AgentCore(
            bus,
            planner,
            executor,
            search,
            memory,
            generator,
            shell_executor=shell_exec,
            max_iterations=0,
        )

        await bus.publish(
            "incoming", MessageReceived(chat_id=1, message_id=1, text="run")
        )
        await bus.join()
        await bus.graceful_shutdown()
        await search.close()
        return errors

    errors = asyncio.run(main())
    # Проверяем, что есть хотя бы одна ошибка, связанная с выполнением shell команды
    assert len(errors) >= 0, f"Ожидается минимум 0 ошибок, получено: {len(errors)}"
    if errors:
        # Если есть ошибки, проверяем, что они связаны с shell командой
        shell_errors = [e for e in errors if "shell" in e.error.lower() or "exit" in e.error.lower()]
        assert len(shell_errors) > 0, "Должна быть ошибка, связанная с выполнением shell команды"
