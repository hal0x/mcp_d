import asyncio
from pathlib import Path
from typing import Iterable

from agent import AgentCore
from events.models import Event, MessageReceived
from executor import CodeGenerator, SimpleCodeExecutor
from internet import SearchClient
from llm.base_client import LLMClient
from memory import UnifiedMemory
from planner import Plan, PlanStep, TaskPlanner, Tool
from services.event_bus import AsyncEventBus
import pytest


class EchoLLM(LLMClient):
    def generate(self, prompt: str) -> str:  # pragma: no cover - trivial
        return prompt

    def stream(self, prompt: str) -> Iterable[str]:  # pragma: no cover - unused
        yield self.generate(prompt)


class LoopPlanner(TaskPlanner):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.calls = 0

    def plan(
        self,
        request: str,
        context: list[str] | None = None,
        previous_results: list[str] | None = None,
    ) -> Plan:
        self.calls += 1
        return Plan(
            steps=[
                PlanStep(tool=Tool.FILE_IO, content=f"write {self.path}\nhello"),
                PlanStep(
                    tool=Tool.FILE_IO,
                    content=f"read {self.path}",
                    is_final=True,
                    expected_output="bye",
                ),
            ],
            context=[],
        )


def test_file_io_loop_stops_after_max_iterations(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    
    async def main() -> tuple[int, int, str]:
        bus: AsyncEventBus[Event] = AsyncEventBus(workers_per_topic=1)
        planner = LoopPlanner(path)
        executor = SimpleCodeExecutor()
        search = SearchClient()
        memory = UnifiedMemory()
        generator = CodeGenerator(EchoLLM())
        core = AgentCore(bus, planner, executor, search, memory, generator, max_iterations=1)

        await bus.publish(
            "incoming",
            MessageReceived(chat_id=1, message_id=1, text="start"),
        )
        await bus.join()
        await bus.graceful_shutdown()
        await search.close()
        return planner.calls, core.iterations, path.read_text()

    plan_calls, iterations, content = asyncio.run(main())
    
    # Проверяем, что планировщик был вызван нужное количество раз
    assert plan_calls == 2, f"Планировщик должен быть вызван 2 раза, но был вызван {plan_calls} раз"
    
    # Проверяем, что агент выполнил нужное количество итераций
    assert iterations == 2, f"Агент должен выполнить 2 итерации, но выполнил {iterations}"
    
    # Проверяем, что файл был создан агентом (не вручную)
    assert path.exists(), "Файл должен быть создан агентом"
    
    # Проверяем содержимое файла
    assert content == "hello", f"Файл должен содержать 'hello', но содержит '{content}'"
    
    # Дополнительная проверка: файл не должен быть пустым
    assert len(content) > 0, "Файл не должен быть пустым"
    
    print(f"✅ Тест прошел: файл создан, содержимое: '{content}', размер: {len(content)} байт")
