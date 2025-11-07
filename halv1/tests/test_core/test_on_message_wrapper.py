import asyncio

from agent.core import AgentCore
from events.models import MessageReceived
from executor import CodeGenerator, SimpleCodeExecutor
from internet import SearchClient
from llm.base_client import LLMClient
from memory import UnifiedMemory
from planner import LLMTaskPlanner
from services.event_bus import AsyncEventBus


class DummyLLM(LLMClient):
    """Minimal LLM stub aligned with current prompts.

    - For planning prompts (LLMTaskPlanner), returns a strict JSON contract
      with a single code step that sets ``result``.
    - For code generation prompts (CodeGenerator), returns valid Python code
      that assigns ``result = 1 + 1``.
    """

    def generate(self, prompt: str) -> str:  # pragma: no cover - deterministic
        if "Create a STRICT CONTRACT for executing the user request." in prompt:
            return (
                '{"steps": ['
                '{"tool": "code", '
                ' "content": "result = 1 + 1", '
                ' "expected_output": "2", '
                ' "is_final": true'
                '}]}'
            )
        # Fallback for CodeGenerator prompt
        return "result = 1 + 1"

    def stream(self, prompt: str):  # pragma: no cover - not used here
        yield self.generate(prompt)


def test_on_message_wrapper_integration() -> None:
    async def run() -> str:
        llm = DummyLLM()
        bus = AsyncEventBus(workers_per_topic=1)
        planner = LLMTaskPlanner(llm)
        executor = SimpleCodeExecutor()
        search = SearchClient()  # lightweight stub if full client unavailable
        memory = UnifiedMemory()
        generator = CodeGenerator(llm)
        AgentCore(
            bus,
            planner,
            executor,
            search,
            memory,
            generator,
            max_iterations=0,  # avoid any continuation loops
        )

        counter = 0

        async def wrapper(text: str, chat_id: int) -> str:
            nonlocal counter
            counter += 1
            await bus.publish(
                "incoming",
                MessageReceived(chat_id=chat_id, message_id=counter, text=text),
            )
            # Wait for all topics to drain
            await bus.join()
            return memory.recall()[-1]

        try:
            result = await wrapper("result = 1 + 1", 1)
        finally:
            # Ensure clean shutdown even if assertion fails
            await bus.graceful_shutdown()
        return result

    assert asyncio.run(run()) == "2"
