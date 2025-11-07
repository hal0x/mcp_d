"""Bright Data powered search client using the MCP protocol."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import mcp.types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from llm.base_client import LLMClient
from llm.prompts import make_web_summary_prompt
from llm.utils import unwrap_response
from metrics import ERRORS

logger = logging.getLogger(__name__)


class SearchError(RuntimeError):
    """Raised when the Bright Data MCP server returns an error."""


@dataclass(slots=True)
class BrightDataConfig:
    command: str = "npx"
    args: List[str] = field(default_factory=lambda: ["@brightdata/mcp"])
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    engine: str = "google"
    call_timeout: int = 120
    session_timeout: int = 120


class BrightDataTransport:
    """Minimal stdio transport for the Bright Data MCP server."""

    def __init__(self, config: BrightDataConfig) -> None:
        self._cfg = config

    async def call(self, tool_name: str, arguments: Mapping[str, Any]) -> str:
        params = StdioServerParameters(
            command=self._cfg.command,
            args=self._cfg.args,
            env=self._cfg.env or None,
            cwd=self._cfg.cwd,
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(
                read_stream,
                write_stream,
            ) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, dict(arguments))
        if result.isError:
            raise SearchError(_extract_error_message(result))
        return _extract_text(result)


class SearchClient:
    """Performs web searches via the Bright Data MCP server."""

    def __init__(
        self,
        session: Any | None = None,  # legacy compatibility
        logger: logging.Logger | None = None,
        *,
        llm: LLMClient | None = None,
        transport: BrightDataTransport | None = None,
        config: Mapping[str, Any] | None = None,
        max_results: int = 5,
        **_: Any,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.session = session
        self.llm = llm
        if transport is None:
            cfg = self._build_config(config)
            self._transport = BrightDataTransport(cfg)
            self._engine = cfg.engine
        else:
            self._transport = transport
            candidate_cfg = config or {}
            self._engine = getattr(transport, "engine", candidate_cfg.get("engine", "google"))
        self._max_results = max(1, max_results)

    # ------------------------------------------------------------------ Public API

    def search(self, query: str, max_results: int | None = None) -> List[Tuple[str, str]]:
        """Synchronously search for ``query``."""
        return _run_sync(self.search_async(query, max_results=max_results))

    async def search_async(
        self, query: str, max_results: int | None = None
    ) -> List[Tuple[str, str]]:
        """Asynchronously search for ``query`` using Bright Data."""

        results = await self._call_search_engine(query, max_results)
        return results

    def fetch(self, url: str) -> str:
        """Synchronously fetch ``url`` and return markdown content."""
        return _run_sync(self.fetch_async(url))

    async def fetch_async(self, url: str) -> str:
        """Asynchronously fetch ``url`` via the Bright Data scraper."""
        if not url:
            return ""
        try:
            content = await self._transport.call("scrape_as_markdown", {"url": url})
        except Exception as exc:  # pragma: no cover - network failures
            self.logger.error("scrape failed for %s: %s", url, exc)
            raise
        return content.strip()

    async def search_and_summarize(
        self,
        query: str,
        max_results: int | None = None,
    ) -> List[str]:
        """Search for ``query`` and summarize each result via the configured LLM."""

        pairs = await self.search_async(query, max_results=max_results)
        if not pairs:
            return []
        summaries: List[str] = []
        for title, url in pairs:
            try:
                content = await self.fetch_async(url)
            except Exception:
                continue
            if not content:
                continue
            summaries.append(await self._summarize_result(title, url, content))
        return summaries

    async def close(self) -> None:
        """Compatibility hook for old interface. Nothing to close."""
        return None

    # ------------------------------------------------------------------ Internals

    async def _call_search_engine(
        self, query: str, max_results: int | None
    ) -> List[Tuple[str, str]]:
        limit = max_results or self._max_results
        payload = {"query": query, "engine": self._engine}
        try:
            markdown = await self._transport.call("search_engine", payload)
        except Exception as exc:
            self.logger.error("search failed for %s: %s", query, exc)
            raise
        pairs = _parse_markdown_results(markdown, limit)
        return pairs

    async def _summarize_result(self, title: str, url: str, markdown: str) -> str:
        if self.llm is None:
            return f"{title}\n{url}\n{markdown[:500]}"

        prompt = make_web_summary_prompt(f"{title}\n{url}\n{markdown}")
        try:
            response, _ = self.llm.generate(prompt)
        except Exception as exc:  # pragma: no cover - LLM failures
            ERRORS.labels(component="brightdata_search", etype=type(exc).__name__).inc()
            self.logger.error("LLM summary failed for %s: %s", url, exc)
            return f"{title}\n{url}\n{markdown[:500]}"
        summary = unwrap_response(response)
        return f"{title}\n{url}\n{summary}"

    @staticmethod
    def _build_config(config: Mapping[str, Any] | None) -> BrightDataConfig:
        source = dict(config or {})
        if not source:
            source = _load_default_config()

        env = dict(source.get("env") or {})
        api_token = env.get("API_TOKEN") or os.getenv("BRIGHTDATA_API_TOKEN")
        if not api_token:
            raise SearchError("API_TOKEN is required for Bright Data MCP")
        env["API_TOKEN"] = api_token

        return BrightDataConfig(
            command=source.get("command", "npx"),
            args=list(source.get("args", ["@brightdata/mcp"])),
            env=env,
            cwd=source.get("cwd"),
            engine=source.get("engine", "google"),
            call_timeout=int(source.get("call_timeout", 120) or 120),
            session_timeout=int(source.get("session_timeout", 120) or 120),
        )


# ---------------------------------------------------------------------- Helpers


def _run_sync(coro: "asyncio.Future[Any] | asyncio.coroutines.coroutine") -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box: Dict[str, Any] = {}
    error_box: Dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            result_box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            error_box["error"] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()
    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("value")


def _extract_text(result: mcp_types.CallToolResult) -> str:
    parts: List[str] = []
    for block in result.content:
        if isinstance(block, mcp_types.TextContent) and block.text:
            parts.append(block.text)
        elif isinstance(block, mcp_types.EmbeddedResource):
            text = getattr(block.resource, "text", None)
            if text:
                parts.append(text)
    if result.structuredContent:
        if isinstance(result.structuredContent, str):
            parts.append(result.structuredContent)
        elif isinstance(result.structuredContent, Mapping):
            text = result.structuredContent.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def _extract_error_message(result: mcp_types.CallToolResult) -> str:
    chunks: List[str] = []
    if isinstance(result.structuredContent, Mapping):
        for key in ("error", "stderr", "message"):
            value = result.structuredContent.get(key)
            if value:
                chunks.append(str(value))
    for block in result.content:
        if isinstance(block, mcp_types.TextContent) and block.text:
            chunks.append(block.text)
    return "\n".join(chunks) or "Bright Data MCP tool returned an error"


def _parse_markdown_results(markdown: str, limit: int) -> List[Tuple[str, str]]:
    if not markdown:
        return []
    pattern = re.compile(r"\[(?P<title>[^\]]+)]\((?P<url>https?://[^\)]+)\)")
    matches = list(pattern.finditer(markdown))
    pairs: List[Tuple[str, str]] = []
    for match in matches:
        title = match.group("title").strip()
        url = match.group("url").strip()
        if not title or not url:
            continue
        pairs.append((title, url))
        if len(pairs) >= limit:
            break
    if not pairs:
        lines = [line.strip() for line in markdown.splitlines() if line.strip()]
        for line in lines:
            if line.startswith("http"):
                pairs.append((line, line))
            elif "http" in line:
                url_match = re.search(r"https?://\\S+", line)
                if url_match:
                    pairs.append((line[: url_match.start()].strip() or line, url_match.group()))
            if len(pairs) >= limit:
                break
    return pairs[:limit]


def _load_default_config() -> Mapping[str, Any]:
    try:
        from executor.config_loader import config_loader  # type: ignore
    except Exception:
        return {}
    try:
        settings = config_loader.load_settings()
    except Exception:
        return {}
    internet_cfg = settings.get("internet", {})
    return internet_cfg.get("mcp", {})
