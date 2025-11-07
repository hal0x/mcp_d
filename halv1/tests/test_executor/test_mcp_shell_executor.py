import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import mcp.types as mcp_types

from executor.code_executor import ExecutionError, ToolPolicy
from executor.mcp_shell_executor import MCPShellExecutor


@asynccontextmanager
async def _stub_session(call_result: mcp_types.CallToolResult, capture: dict):
    class _Session:
        async def call_tool(self, name: str, arguments: dict):
            capture["name"] = name
            capture["arguments"] = arguments
            return call_result

    yield _Session()


def _patch_session(monkeypatch, call_result: mcp_types.CallToolResult, capture: dict):
    @asynccontextmanager
    async def fake_open_session(self):
        async with _stub_session(call_result, capture) as session:
            yield session

    monkeypatch.setattr(MCPShellExecutor, "_open_session", fake_open_session)


def test_execute_multi_step_collects_artifacts(monkeypatch, tmp_path):
    response = {
        "stdout": ["ok"],
        "stderr": ["warn"],
        "exit_code": 0,
    }
    call_result = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=json.dumps(response))]
    )
    capture: dict = {}
    _patch_session(monkeypatch, call_result, capture)

    artifact_dir = tmp_path / "artifacts" / "run"

    def fake_create_artifact_dir(self) -> Path:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def fake_collect_artifacts(self, path: Path) -> dict[str, bytes]:
        return {"data.txt": b"payload"}

    monkeypatch.setattr(MCPShellExecutor, "_create_artifact_dir", fake_create_artifact_dir)
    monkeypatch.setattr(MCPShellExecutor, "_collect_artifacts", fake_collect_artifacts)
    monkeypatch.setattr(MCPShellExecutor, "_cleanup_artifact_dir", lambda self, path: None)

    executor = MCPShellExecutor(
        config={"max_wall_time_s": 10, "artifacts_path": str(tmp_path / "artifacts")}
    )
    result = executor.execute_multi_step(["print('ok')"])

    assert capture["name"] == "run_code_multi_step"
    assert capture["arguments"]["steps"] == ["print('ok')"]
    assert capture["arguments"]["out_artifacts_path"] == str(artifact_dir)
    assert result.stdout == json.dumps(["ok"])
    assert result.stderr == json.dumps(["warn"])
    assert result.files == {"data.txt": b"payload"}
    assert result.returncode == 0


def test_execute_applies_policy(monkeypatch, tmp_path):
    response = {"stdout": "[]", "stderr": "[]", "files": {}}
    call_result = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=json.dumps(response))]
    )
    capture: dict = {}
    _patch_session(monkeypatch, call_result, capture)

    artifact_dir = tmp_path / "artifacts" / "run"

    monkeypatch.setattr(MCPShellExecutor, "_create_artifact_dir", lambda self: artifact_dir)
    monkeypatch.setattr(MCPShellExecutor, "_collect_artifacts", lambda self, path: {})
    monkeypatch.setattr(MCPShellExecutor, "_cleanup_artifact_dir", lambda self, path: None)

    executor = MCPShellExecutor(
        config={
            "max_mem_mb": 512,
            "cpu_quota": 1.5,
            "network_mode": "none",
            "artifacts_path": str(tmp_path / "artifacts"),
        }
    )
    policy = ToolPolicy(max_wall_time_s=5, max_mem_mb=256, cpu_quota=2.0, network_mode="bridge")

    executor.execute_multi_step(["print('hi')"], policy=policy)

    args = capture["arguments"]
    assert args["timeout_seconds"] == 5
    assert args["memory"] == "256m"
    assert args["cpus"] == "2.0"
    assert args["network_enabled"] is True


def test_execute_raises_on_tool_error(monkeypatch, tmp_path):
    call_result = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text="failure")],
        structuredContent={"error": "boom"},
        isError=True,
    )
    capture: dict = {}
    _patch_session(monkeypatch, call_result, capture)

    executor = MCPShellExecutor(config={"artifacts_path": str(tmp_path / "artifacts")})

    with pytest.raises(ExecutionError):
        executor.execute_multi_step(["print('x')"])


@pytest.mark.asyncio
async def test_execute_multi_step_inside_event_loop(monkeypatch, tmp_path):
    response = {"stdout": [], "stderr": [], "files": {}}
    call_result = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=json.dumps(response))]
    )
    capture: dict = {}
    _patch_session(monkeypatch, call_result, capture)

    artifact_dir = tmp_path / "artifacts" / "run"

    monkeypatch.setattr(MCPShellExecutor, "_create_artifact_dir", lambda self: artifact_dir)
    monkeypatch.setattr(MCPShellExecutor, "_collect_artifacts", lambda self, path: {})
    monkeypatch.setattr(MCPShellExecutor, "_cleanup_artifact_dir", lambda self, path: None)

    executor = MCPShellExecutor(config={"artifacts_path": str(tmp_path / "artifacts")})

    # Direct call from within an event loop should succeed (uses a worker thread internally).
    result = executor.execute_multi_step(["print('ok')"])

    assert result.stdout == "[]"
