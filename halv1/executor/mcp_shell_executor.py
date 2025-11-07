"""Execute code via the shell-mcp server using the MCP protocol."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Coroutine, Dict, List, Mapping, Optional

import mcp.types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .code_executor import CodeExecutor, ExecutionError, ExecutionResult, ToolPolicy


class MCPShellExecutor(CodeExecutor):
    """Adapter that forwards code execution to the shell-mcp MCP server."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        shell_cfg = dict(cfg.get("shell_mcp", {}))

        self._command = shell_cfg.get("command", "uv")
        self._args = list(shell_cfg.get("args", ["run", "shell-mcp"]))
        self._env = dict(shell_cfg.get("env", {})) or None
        self._cwd = shell_cfg.get("cwd")

        self._default_timeout = int(cfg.get("max_wall_time_s", 120) or 120)
        self._default_mem_mb = _coerce_int(cfg.get("max_mem_mb"))
        self._default_cpu_quota = _coerce_float(cfg.get("cpu_quota"))
        self._default_network_mode = str(cfg.get("network_mode", "") or "").lower()
        artifact_root = cfg.get("artifacts_path", "runs/mcp-artifacts")
        self._artifact_root = Path(artifact_root).expanduser()
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        self._artifact_root = self._artifact_root.resolve()
        self._cleanup_artifacts = bool(cfg.get("cleanup_artifacts", False))

    def execute(self, code: str, policy: ToolPolicy | None = None) -> ExecutionResult:
        """Execute a single code snippet by delegating to run_code_multi_step."""
        return self._run_sync(self._execute_async([code], policy))

    def execute_multi_step(self, steps: List[str], policy: ToolPolicy | None = None) -> ExecutionResult:
        """Execute many code snippets sequentially inside one container."""
        if not steps:
            raise ExecutionError("No steps supplied for multi-step execution")
        return self._run_sync(self._execute_async(list(steps), policy))

    def _run_sync(self, coro: Coroutine[Any, Any, ExecutionResult]) -> ExecutionResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)  # No running loop, execute directly

        result_box: Dict[str, ExecutionResult] = {}
        error_box: Dict[str, BaseException] = {}

        def _worker() -> None:
            try:
                result_box["value"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001 - propagate original error
                error_box["error"] = exc

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join()

        if "error" in error_box:
            raise error_box["error"]
        return result_box["value"]

    async def _execute_async(self, steps: List[str], policy: ToolPolicy | None) -> ExecutionResult:
        artifact_dir = self._create_artifact_dir()
        try:
            params = self._build_arguments(steps, policy, artifact_dir)
            async with self._open_session() as session:
                result = await session.call_tool("run_code_multi_step", params)
            if result.isError:
                raise ExecutionError(self._extract_error_message(result))
            response = self._extract_response(result)
            files = self._collect_artifacts(artifact_dir)
            return self._build_result(response, files)
        finally:
            self._cleanup_artifact_dir(artifact_dir)

    def _build_arguments(self, steps: List[str], policy: ToolPolicy | None, artifact_dir: Path) -> Dict[str, Any]:
        timeout = self._resolve_timeout(policy)
        params: Dict[str, Any] = {
            "steps": steps,
            "language": "python",
            "timeout_seconds": timeout,
        }

        memory = self._resolve_memory(policy)
        if memory:
            params["memory"] = memory

        cpus = self._resolve_cpus(policy)
        if cpus:
            params["cpus"] = cpus

        network_enabled = self._resolve_network_enabled(policy)
        if network_enabled is not None:
            params["network_enabled"] = network_enabled
        params["out_artifacts_path"] = str(artifact_dir)

        return params

    def _resolve_timeout(self, policy: ToolPolicy | None) -> int:
        if policy and policy.max_wall_time_s:
            return max(1, int(policy.max_wall_time_s))
        return max(1, int(self._default_timeout))

    def _resolve_memory(self, policy: ToolPolicy | None) -> Optional[str]:
        if policy and policy.max_mem_mb:
            return f"{int(policy.max_mem_mb)}m"
        if self._default_mem_mb:
            return f"{self._default_mem_mb}m"
        return None

    def _resolve_cpus(self, policy: ToolPolicy | None) -> Optional[str]:
        if policy and policy.cpu_quota:
            return str(policy.cpu_quota)
        if self._default_cpu_quota:
            return str(self._default_cpu_quota)
        return None

    def _resolve_network_enabled(self, policy: ToolPolicy | None) -> Optional[bool]:
        if policy and policy.network_mode:
            return policy.network_mode.lower() != "none"
        if self._default_network_mode:
            return self._default_network_mode != "none"
        return None

    @asynccontextmanager
    async def _open_session(self) -> AsyncIterator[ClientSession]:
        params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
            cwd=self._cwd,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    def _extract_response(self, result: mcp_types.CallToolResult) -> Dict[str, Any]:
        structured = result.structuredContent
        if isinstance(structured, dict):
            return structured

        for block in result.content:
            if isinstance(block, mcp_types.TextContent):
                text = block.text.strip()
                if not text:
                    continue
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    continue

        raise ExecutionError("shell-mcp returned an unexpected response format")

    def _extract_error_message(self, result: mcp_types.CallToolResult) -> str:
        chunks: list[str] = []
        structured = result.structuredContent
        if isinstance(structured, dict):
            for key in ("stderr", "error", "message"):
                value = structured.get(key)
                if value:
                    chunks.append(str(value))

        for block in result.content:
            if isinstance(block, mcp_types.TextContent):
                chunks.append(block.text)
        return "\n".join(chunk for chunk in chunks if chunk).strip() or "shell-mcp tool reported an error"

    def _build_result(self, data: Mapping[str, Any], files: Dict[str, bytes]) -> ExecutionResult:
        stdout_val = data.get("stdout", "")
        stderr_val = data.get("stderr", "")

        stdout = json.dumps(stdout_val) if isinstance(stdout_val, list) else str(stdout_val or "")
        stderr = json.dumps(stderr_val) if isinstance(stderr_val, list) else str(stderr_val or "")

        exit_code = data.get("exit_code")
        try:
            returncode = int(exit_code) if exit_code is not None else 0
        except Exception:
            returncode = 0

        return ExecutionResult(stdout=stdout, stderr=stderr, files=files, returncode=returncode)

    def _create_artifact_dir(self) -> Path:
        run_dir = self._artifact_root / f"run-{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _collect_artifacts(self, artifact_dir: Path) -> Dict[str, bytes]:
        files: Dict[str, bytes] = {}
        if not artifact_dir.exists():
            return files
        for path in artifact_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                files[str(path.relative_to(artifact_dir))] = path.read_bytes()
            except Exception:
                continue
        return files

    def _cleanup_artifact_dir(self, artifact_dir: Path) -> None:
        if not self._cleanup_artifacts:
            return
        import shutil

        if artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


__all__ = ["MCPShellExecutor"]
