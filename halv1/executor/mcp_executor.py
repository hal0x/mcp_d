"""MCP-based executor that proxies code execution via a Model Context Protocol server."""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, AsyncIterator, Dict, Iterable, Mapping, Tuple

import anyio
import mcp.types as mcp_types

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client

from .code_executor import CodeExecutor, ExecutionError, ExecutionResult, ToolPolicy


@dataclass(slots=True)
class MCPTransportConfig:
    """Runtime configuration for connecting to an MCP server."""

    transport: str = "stdio"
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    session_timeout: float = 120.0
    call_timeout: float = 120.0
    tool_name: str | None = None
    code_argument: str | None = None
    extra_arguments: dict[str, Any] | None = None


class MCPCodeExecutor(CodeExecutor):
    """Execute Python code by forwarding requests to an MCP server."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        cfg = MCPTransportConfig(**(config or {}))
        self._cfg = cfg

        transport = (cfg.transport or "stdio").lower()
        if transport not in {"stdio", "sse"}:
            raise ValueError(f"Unsupported MCP transport: {cfg.transport}")
        self._transport = transport

        if self._transport == "stdio":
            self._command = cfg.command or "deno"
            self._args = cfg.args or [
                "run",
                "-N",
                "-R=node_modules",
                "-W=node_modules",
                "--node-modules-dir=auto",
                "jsr:@pydantic/mcp-run-python",
                "stdio",
            ]
        else:
            self._command = None
            self._args = None

        self._env = cfg.env or {}
        self._cwd = cfg.cwd
        self._url = cfg.url
        self._headers = cfg.headers or {}
        self._session_timeout = max(cfg.session_timeout, 1.0)
        self._call_timeout = max(cfg.call_timeout, 1.0)
        self._tool_name = cfg.tool_name
        self._code_argument = cfg.code_argument
        self._extra_arguments = dict(cfg.extra_arguments or {})

        if self._transport == "sse" and not self._url:
            raise ValueError("MCP SSE transport requires 'url' in configuration")

    def execute(self, code: str, policy: ToolPolicy | None = None) -> ExecutionResult:
        """Synchronously execute *code* via the configured MCP server."""

        try:
            return anyio.run(self._execute_async, code)
        except ExecutionError:
            raise
        except FileNotFoundError as exc:
            command = self._command or ""
            raise ExecutionError(
                f"MCP server command not found: {command!s}. Install the runtime or update settings."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - propagate as execution failure
            raise ExecutionError(str(exc)) from exc

    async def _execute_async(self, code: str) -> ExecutionResult:
        result = await self._call_tool(code)
        return self._convert_result(result)

    async def _call_tool(self, code: str) -> mcp_types.CallToolResult:
        async with self._open_transport() as (read_stream, write_stream):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=self._session_timeout),
            ) as session:
                await session.initialize()
                tool = await self._resolve_tool(session)
                arguments = self._build_arguments(tool, code)
                call_result = await session.call_tool(
                    tool.name,
                    arguments,
                    read_timeout_seconds=timedelta(seconds=self._call_timeout),
                )
                if call_result.isError:
                    raise ExecutionError(self._extract_error_message(call_result))
                return call_result

    async def _resolve_tool(self, session: ClientSession) -> mcp_types.Tool:
        tools_response = await session.list_tools()
        tools = {tool.name: tool for tool in tools_response.tools}

        tool_name = self._tool_name
        if tool_name is None:
            if len(tools) == 1:
                tool_name = next(iter(tools))
            elif not tools:
                raise ExecutionError("MCP server does not expose any tools")
            else:
                raise ExecutionError(
                    "Multiple MCP tools available; specify 'tool_name' in executor configuration"
                )

        tool = tools.get(tool_name)
        if tool is None:
            raise ExecutionError(f"MCP tool '{tool_name}' not found on server")
        return tool

    def _build_arguments(self, tool: mcp_types.Tool, code: str) -> dict[str, Any]:
        arguments: dict[str, Any] = dict(self._extra_arguments)
        property_name = self._code_argument or self._guess_code_property(tool)
        arguments[property_name] = code
        return arguments

    def _guess_code_property(self, tool: mcp_types.Tool) -> str:
        schema = tool.inputSchema or {}
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}

        if self._code_argument and self._code_argument in properties:
            return self._code_argument

        for candidate in ("code", "source", "python", "script"):
            if candidate in properties:
                return candidate

        if properties:
            return next(iter(properties))

        return "code"

    def _extract_error_message(self, result: mcp_types.CallToolResult) -> str:
        chunks: list[str] = []
        if result.structuredContent and isinstance(result.structuredContent, dict):
            for key in ("stderr", "error", "message"):
                value = result.structuredContent.get(key)
                if value:
                    chunks.append(str(value))
        for block in result.content:
            if isinstance(block, mcp_types.TextContent):
                chunks.append(block.text)
        return "\n".join(chunks) or "MCP tool returned an error"

    def _convert_result(self, result: mcp_types.CallToolResult) -> ExecutionResult:
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        files: Dict[str, bytes] = {}

        structured = result.structuredContent
        if isinstance(structured, dict):
            stdout_val = structured.get("stdout")
            if stdout_val:
                stdout_chunks.append(str(stdout_val))
            stderr_val = structured.get("stderr")
            if stderr_val:
                stderr_chunks.append(str(stderr_val))
            for name, content in self._iter_files(structured.get("files")):
                files[name] = content

        for block in result.content:
            if isinstance(block, mcp_types.TextContent):
                stdout_chunks.append(block.text)
            elif isinstance(block, mcp_types.EmbeddedResource):
                resource = block.resource
                if hasattr(resource, "text"):
                    text_value = getattr(resource, "text")
                    stdout_chunks.append(text_value)
                    uri = getattr(resource, "uri", None)
                    filename = uri or f"resource_{len(files) + 1}.txt"
                    files[filename] = text_value.encode("utf-8")
                elif hasattr(resource, "blob"):
                    blob_value = getattr(resource, "blob")
                    uri = getattr(resource, "uri", None)
                    filename = uri or f"resource_{len(files) + 1}.bin"
                    if isinstance(blob_value, str):
                        try:
                            files[filename] = base64.b64decode(blob_value)
                        except Exception:  # noqa: BLE001 - best effort decoding
                            files[filename] = blob_value.encode("utf-8")

        stdout = "\n".join(chunk for chunk in stdout_chunks if chunk)
        stderr = "\n".join(chunk for chunk in stderr_chunks if chunk)
        return ExecutionResult(stdout=stdout, stderr=stderr, files=files, returncode=0)

    def _iter_files(self, files_payload: Any) -> Iterable[tuple[str, bytes]]:
        if not isinstance(files_payload, Mapping):
            return []
        results: list[tuple[str, bytes]] = []
        for name, value in files_payload.items():
            if isinstance(value, bytes):
                results.append((name, value))
            elif isinstance(value, str):
                results.append((name, value.encode("utf-8")))
            elif isinstance(value, Mapping):
                data = value.get("data") or value.get("text") or value.get("blob")
                if isinstance(data, bytes):
                    results.append((name, data))
                elif isinstance(data, str):
                    if value.get("encoding") == "base64":
                        try:
                            results.append((name, base64.b64decode(data)))
                            continue
                        except Exception:  # noqa: BLE001 - fall back to utf-8
                            pass
                    try:
                        results.append((name, data.encode("utf-8")))
                    except Exception:  # noqa: BLE001 - skip invalid entry
                        continue
        return results

    @asynccontextmanager
    async def _open_transport(
        self,
    ) -> AsyncIterator[Tuple[Any, Any]]:
        if self._transport == "stdio":
            server = StdioServerParameters(
                command=self._command or "",
                args=self._args or [],
                env=self._env or None,
                cwd=self._cwd,
            )
            async with stdio_client(server) as streams:
                yield streams
        else:
            assert self._url is not None  # guarded in __init__
            async with sse_client(self._url, headers=self._headers or None) as streams:
                yield streams
