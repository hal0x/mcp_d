import pytest

from agent.actions import ActionHandlersMixin
from executor import ExecutionError, ExecutionResult
from planner import PlanStep, Tool


class _StubExecutor:
    def __init__(self) -> None:
        self.executed: list[str] = []

    def execute(self, code: str, policy=None):
        self.executed.append(code)
        if "plaftform" in code:
            raise ExecutionError(
                "Traceback (most recent call last):\nModuleNotFoundError: No module named 'plaftform'"
            )
        return ExecutionResult(stdout="[1, 1, 3, 4, 5]", stderr="", files={})


class _StubCodeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate(self, description: str, max_attempts: int = 3, *, error_reason: str = ""):
        self.calls.append((description, error_reason))
        if error_reason:
            return "result = sorted([3, 1, 4, 1, 5])\nprint(result)"
        return "import plaftform\nresult = sorted([3, 1, 4, 1, 5])\nprint(result)"


class _StubSearch:
    async def fetch_async(self, _query: str):  # pragma: no cover - unused
        return ""

    async def search_and_summarize(self, _query: str):  # pragma: no cover - unused
        return [""]


class _ActionHost(ActionHandlersMixin):
    def __init__(self) -> None:
        self.executor = _StubExecutor()
        self.shell_executor = self.executor
        self.code_generator = _StubCodeGenerator()
        self.search = _StubSearch()


@pytest.mark.asyncio
async def test_retries_with_runtime_feedback():
    host = _ActionHost()
    step = PlanStep(tool=Tool.CODE, content="Отсортируй список [3,1,4,1,5]")

    artifact = await host._execute_code(step)

    assert artifact["stdout"] == "[1, 1, 3, 4, 5]"
    assert len(host.executor.executed) == 2
    assert "plaftform" not in host.executor.executed[-1]
    # Вторая попытка должна использовать сообщение об ошибке
    assert host.code_generator.calls[1][1] == "ModuleNotFoundError: No module named 'plaftform'"
