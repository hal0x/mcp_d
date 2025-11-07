import pytest
import mcp.types as mcp_types

from executor.code_executor import ExecutionError
from executor.mcp_executor import MCPCodeExecutor


def _patch_call(monkeypatch, result: mcp_types.CallToolResult):
    async def fake_call(self, code: str) -> mcp_types.CallToolResult:  # noqa: D401 - simple stub
        return result

    monkeypatch.setattr(MCPCodeExecutor, "_call_tool", fake_call)


def test_execute_returns_text_stdout(monkeypatch):
    executor = MCPCodeExecutor(config={})
    call_result = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text="42")]
    )
    _patch_call(monkeypatch, call_result)

    result = executor.execute("print(42)")

    assert result.stdout == "42"
    assert result.stderr == ""
    assert result.files == {}


def test_execute_uses_structured_content(monkeypatch):
    executor = MCPCodeExecutor(config={})
    call_result = mcp_types.CallToolResult(
        content=[],
        structuredContent={
            "stdout": "done",
            "stderr": "warn",
            "files": {"output.txt": "payload"},
        },
    )
    _patch_call(monkeypatch, call_result)

    result = executor.execute("print('ok')")

    assert result.stdout == "done"
    assert result.stderr == "warn"
    assert result.files == {"output.txt": b"payload"}


def test_execute_raises_on_error(monkeypatch):
    executor = MCPCodeExecutor(config={})
    async def fake_call(self, code: str) -> mcp_types.CallToolResult:
        raise ExecutionError("boom")

    monkeypatch.setattr(MCPCodeExecutor, "_call_tool", fake_call)

    with pytest.raises(ExecutionError):
        executor.execute("raise ValueError")


def test_build_arguments_honours_code_argument():
    executor = MCPCodeExecutor(config={"code_argument": "source", "extra_arguments": {"packages": ["numpy"]}})
    tool = mcp_types.Tool(
        name="run",
        description="Run python",
        inputSchema={"type": "object", "properties": {"source": {"type": "string"}}},
    )

    arguments = executor._build_arguments(tool, "print('hello')")

    assert arguments == {"packages": ["numpy"], "source": "print('hello')"}
